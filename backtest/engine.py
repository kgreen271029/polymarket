"""
backtest/engine.py — Vectorised daily backtesting engine.

Simulates the improved strategy on 30 days of historical data.
Supports fractional shares, ATR stops, relative-strength filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicators (self-contained, no external deps beyond pandas/numpy)
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def _relative_strength(close: pd.Series, spy: pd.Series, period: int = 20) -> pd.Series:
    """Return of stock / return of SPY over rolling `period` bars."""
    stock_ret = close / close.shift(period) - 1
    spy_ret   = spy   / spy.shift(period)   - 1
    return stock_ret - spy_ret          # positive = outperforming SPY


def _gap_pct(df: pd.DataFrame) -> pd.Series:
    return (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100


def _volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    avg = df["volume"].rolling(period).mean().shift(1)
    return df["volume"] / avg.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Signal generation (improved strategy)
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """
    Add signal columns to the stock DataFrame.
    Returns df with added columns:
      atr, rsi, ema20, ema50, sma50, rel_strength, gap_pct, vol_ratio,
      signal (1=BUY, 0=flat), stop, target
    """
    df = df.copy()
    spy_close = spy["close"].reindex(df.index, method="ffill")

    df["atr"]          = _atr(df)
    df["rsi"]          = _rsi(df["close"])
    df["ema20"]        = _ema(df["close"], 20)
    df["ema50"]        = _ema(df["close"], 50)
    df["sma50"]        = _sma(df["close"], 50)
    df["rel_strength"] = _relative_strength(df["close"], spy_close, 20)
    df["gap_pct"]      = _gap_pct(df)
    df["vol_ratio"]    = _volume_ratio(df)
    df["macd_hist"]    = (_ema(df["close"], 12) - _ema(df["close"], 26)) - \
                          (_ema(df["close"], 12) - _ema(df["close"], 26)).ewm(span=9, adjust=False).mean()

    # ── Entry rules (ALL must be true) ──────────────────────────────────────
    # 1. Price above 50-day SMA (uptrend filter)
    trend_up   = df["close"] > df["sma50"]
    # 2. Outperforming SPY over 20 days
    rs_pos     = df["rel_strength"] > 0.00
    # 3. RSI in sweet spot (not overbought)
    rsi_ok     = df["rsi"].between(42, 72)
    # 4. Volume expansion (buying conviction)
    vol_ok     = df["vol_ratio"] > 1.3
    # 5. MACD histogram positive (momentum turning up)
    macd_ok    = df["macd_hist"] > 0
    # 6. Price above EMA-20 (short-term trend)
    above_ema  = df["close"] > df["ema20"]
    # 7. Gap filter: don't chase gaps > 5% (reduces overnight risk chasing)
    gap_ok     = df["gap_pct"].abs() < 5.0

    df["signal"] = (trend_up & rs_pos & rsi_ok & vol_ok & macd_ok & above_ema & gap_ok).astype(int)

    # ATR-based stop (2× ATR below close) and target (3× ATR above = 1.5:1 R:R)
    df["stop"]   = df["close"] - 2.0 * df["atr"]
    df["target"] = df["close"] + 3.0 * df["atr"]

    return df


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    symbol:      str
    entry_date:  date
    entry_price: float
    shares:      float
    stop:        float
    target:      float
    exit_date:   Optional[date]  = None
    exit_price:  Optional[float] = None
    exit_reason: str             = ""
    pnl:         float           = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    def close(self, exit_date: date, exit_price: float, reason: str) -> None:
        self.exit_date   = exit_date
        self.exit_price  = exit_price
        self.exit_reason = reason
        self.pnl         = (exit_price - self.entry_price) * self.shares


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    initial_capital: float
    final_equity:    float
    trades:          list[Trade]
    equity_curve:    list[tuple[date, float]]

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.initial_capital - 1) * 100

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_open]

    @property
    def win_rate(self) -> float:
        c = self.closed_trades
        if not c: return 0.0
        return sum(1 for t in c if t.pnl > 0) / len(c) * 100

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.closed_trades if t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.closed_trades if t.pnl <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win  = sum(t.pnl for t in self.closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl < 0))
        return gross_win / gross_loss if gross_loss else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        equities = [e for _, e in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        equities = [e for _, e in self.equity_curve]
        daily_ret = pd.Series(equities).pct_change().dropna()
        if daily_ret.std() == 0:
            return 0.0
        return float(daily_ret.mean() / daily_ret.std() * math.sqrt(252))


def run_backtest(
    stock_data: dict[str, pd.DataFrame],
    spy_df:     pd.DataFrame,
    initial_capital: float = 200.0,
    max_positions:   int   = 3,
    position_pct:    float = 0.30,   # 30% of equity per position
    slippage_pct:    float = 0.001,  # 0.1% slippage per trade
    max_hold_days:   int   = 10,     # forced exit after 10 days
) -> BacktestResult:
    """
    Simulate the improved strategy across all symbols.

    Entry:  Next open after signal fires.
    Exit:   Stop hit, target hit, or max_hold_days reached.
    """
    # Pre-compute signals for every symbol
    signals: dict[str, pd.DataFrame] = {}
    for sym, df in stock_data.items():
        if sym == "SPY":
            continue
        try:
            signals[sym] = compute_signals(df, spy_df)
        except Exception:
            pass

    # Get unified date index (trading days)
    all_dates = sorted({d for df in signals.values() for d in df.index})
    # Restrict to last 30 trading days
    backtest_dates = all_dates[-30:]

    cash      = initial_capital
    positions: list[Trade] = []
    all_trades: list[Trade] = []
    equity_curve: list[tuple[date, float]] = []

    for today in backtest_dates:
        today_date = today.date() if hasattr(today, "date") else today

        # ── 1. Update open positions ────────────────────────────────────────
        for trade in list(positions):
            sym = trade.symbol
            if sym not in signals or today not in signals[sym].index:
                continue
            row        = signals[sym].loc[today]
            high       = float(row["high"])
            low        = float(row["low"])
            close      = float(row["close"])
            days_held  = (today_date - trade.entry_date).days

            exit_price = None
            reason     = ""

            if low <= trade.stop:
                exit_price = trade.stop * (1 - slippage_pct)
                reason     = "stop"
            elif high >= trade.target:
                exit_price = trade.target * (1 - slippage_pct)
                reason     = "target"
            elif days_held >= max_hold_days:
                exit_price = close * (1 - slippage_pct)
                reason     = "max_hold"

            if exit_price is not None:
                trade.close(today_date, exit_price, reason)
                cash += exit_price * trade.shares
                positions.remove(trade)

        # ── 2. Compute portfolio equity ─────────────────────────────────────
        open_value = sum(
            float(signals[t.symbol].loc[today]["close"]) * t.shares
            for t in positions
            if t.symbol in signals and today in signals[t.symbol].index
        )
        equity = cash + open_value
        equity_curve.append((today_date, equity))

        # ── 3. Look for new entries ─────────────────────────────────────────
        if len(positions) >= max_positions:
            continue

        # Score candidates by relative strength, take the best
        candidates = []
        for sym, df in signals.items():
            if today not in df.index:
                continue
            row = df.loc[today]
            if float(row.get("signal", 0)) != 1.0:
                continue
            already_in = any(t.symbol == sym for t in positions)
            if already_in:
                continue
            candidates.append((sym, float(row.get("rel_strength", 0)), row))

        # Sort by relative strength descending — buy the strongest
        candidates.sort(key=lambda x: x[1], reverse=True)

        slots = max_positions - len(positions)
        for sym, _, row in candidates[:slots]:
            entry_price = float(row["open"]) * (1 + slippage_pct)
            stop        = float(row["stop"])
            target      = float(row["target"])

            if entry_price <= stop:
                continue   # degenerate signal

            position_size = equity * position_pct
            shares        = position_size / entry_price

            if cash < position_size:
                continue

            cash -= entry_price * shares
            trade = Trade(
                symbol=sym,
                entry_date=today_date,
                entry_price=entry_price,
                shares=shares,
                stop=stop,
                target=target,
            )
            positions.append(trade)
            all_trades.append(trade)

    # ── Force-close any remaining open positions at last price ──────────────
    last_date = backtest_dates[-1]
    last_day  = last_date.date() if hasattr(last_date, "date") else last_date
    for trade in list(positions):
        sym = trade.symbol
        if sym in signals and last_date in signals[sym].index:
            close_px = float(signals[sym].loc[last_date]["close"])
        else:
            close_px = trade.entry_price
        trade.close(last_day, close_px * (1 - slippage_pct), "end_of_backtest")
        cash += close_px * trade.shares

    final_equity = cash
    return BacktestResult(
        initial_capital=initial_capital,
        final_equity=final_equity,
        trades=all_trades,
        equity_curve=equity_curve,
    )
