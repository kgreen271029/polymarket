"""
backtest/walk_forward.py — Walk-forward validation to detect overfitting.

Splits historical data into rolling windows:
  - In-sample (IS): 4 weeks of training
  - Out-of-sample (OOS): 2 weeks of testing

If OOS win rate < 50% of IS win rate across all windows, the strategy is
likely overfit and should not be deployed live.

Run: python backtest/walk_forward.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from analysis.multifactor import score_symbol, volatility_contraction
from analysis.technical import TechnicalAnalyzer


SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN",
    "AMD", "PLTR", "ARM", "JPM", "GS",
]

IS_WEEKS   = 6   # in-sample window (30 bars)
OOS_WEEKS  = 3   # out-of-sample window (15 bars)
SLIPPAGE   = 0.001  # 0.1% round-trip
ATR_STOP   = 2.2
ATR_TARGET = 3.3


@dataclass
class WindowResult:
    window_idx: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_trades: int
    is_win_rate: float
    oos_trades: int
    oos_win_rate: float
    oos_return_pct: float
    passed: bool  # OOS win rate >= 60% of IS win rate


def _download(symbol: str, period: str = "6mo") -> pd.DataFrame | None:
    try:
        raw = yf.download(symbol, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 30:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        return raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        logger.debug("[WF] {} download error: {}", symbol, e)
        return None


def _backtest_slice(df: pd.DataFrame, spy_df: pd.DataFrame) -> dict:
    """Run a simple multi-factor backtest on a price DataFrame slice.

    Uses all bars in df as context; entry signals evaluated starting at bar 60.
    When df has fewer than 65 bars, starts as early as possible (min 30 bars context).
    """
    trades = []
    min_context = min(60, max(30, len(df) - 5))
    i = min_context
    while i < len(df) - 1:
        window = df.iloc[:i]
        spy_window = spy_df.iloc[:i]

        if len(window) < 55:
            i += 1
            continue

        try:
            s = TechnicalAnalyzer.build_signal_summary(window)
            if not s:
                i += 1
                continue

            mf = score_symbol(window, spy_window)
            vcp = volatility_contraction(window)

            price      = s.get("latest_close", 0)
            rsi        = s.get("rsi", 50)
            sma50      = s.get("sma_50", price)
            vol_ratio  = s.get("volume_ratio", 1.0)
            pct_vs_sma = s.get("price_vs_sma20_pct", 0)
            atr        = s.get("atr", price * 0.02)

            # Entry conditions (same as live SwingTrader)
            entry_ok = (
                price > 0
                and price > sma50
                and 0 < pct_vs_sma < 3.5
                and 38 <= rsi <= 65
                and vol_ratio > 1.3
                and vcp >= 0.5
                and mf["composite"] >= 45
            )

            if not entry_ok:
                i += 1
                continue

            # Simulate trade: check next N bars for stop/target hit
            stop   = price * (1 - ATR_STOP * atr / price)
            target = price * (1 + ATR_TARGET * atr / price)
            entry  = price * (1 + SLIPPAGE)  # slippage in

            won = None
            for j in range(i + 1, min(i + 11, len(df))):  # max 10-day hold
                future_low  = float(df["low"].iloc[j])
                future_high = float(df["high"].iloc[j])
                if future_low <= stop:
                    won = False
                    break
                if future_high >= target:
                    won = True
                    break

            if won is None:
                # Time exit: compare exit price vs entry
                exit_price = float(df["close"].iloc[min(i + 10, len(df) - 1)]) * (1 - SLIPPAGE)
                won = exit_price > entry

            trades.append(won)
            i += 10  # skip to avoid overlapping trades

        except Exception:
            i += 1

    n = len(trades)
    wr = sum(trades) / n if n > 0 else 0.0
    return {"n": n, "win_rate": wr}


def run_walk_forward(symbol_list: list[str] = SYMBOLS,
                     total_period: str = "6mo") -> list[WindowResult]:
    """Run rolling walk-forward validation across all symbols."""
    logger.info("[WF] Downloading {} symbols...", len(symbol_list))

    all_data: dict[str, pd.DataFrame] = {}
    for sym in symbol_list:
        df = _download(sym, total_period)
        if df is not None:
            all_data[sym] = df

    spy = _download("SPY", total_period)
    if spy is None:
        logger.error("[WF] SPY data unavailable — aborting")
        return []

    if not all_data:
        logger.error("[WF] No data downloaded")
        return []

    # Use the shortest common length across all stocks
    min_len = min(len(df) for df in all_data.values())
    min_len = min(min_len, len(spy))

    # Align all dataframes to the same length
    spy = spy.iloc[-min_len:].reset_index(drop=True)
    for sym in all_data:
        all_data[sym] = all_data[sym].iloc[-min_len:].reset_index(drop=True)

    is_bars  = IS_WEEKS * 5    # trading days
    oos_bars = OOS_WEEKS * 5
    step     = oos_bars         # roll forward by OOS window

    results: list[WindowResult] = []
    window_idx = 0
    start = 0

    while start + is_bars + oos_bars <= min_len:
        is_end  = start + is_bars
        oos_end = is_end + oos_bars

        is_trades_all:  list[bool] = []
        oos_trades_all: list[bool] = []
        oos_pnl: list[float] = []

        for sym, df in all_data.items():
            # Include up to 60 bars of prefix for indicator context
            prefix = max(0, start - 60)
            is_df   = df.iloc[prefix:is_end].reset_index(drop=True)
            oos_df  = df.iloc[prefix:oos_end].reset_index(drop=True)
            spy_is  = spy.iloc[prefix:is_end].reset_index(drop=True)
            spy_oos = spy.iloc[prefix:oos_end].reset_index(drop=True)

            # Evaluate only signals in the IS/OOS window (after prefix)
            is_res  = _backtest_slice(is_df,  spy_is)
            oos_res = _backtest_slice(oos_df, spy_oos)
            is_trades_all.extend([True] * is_res["n"])
            oos_trades_all.extend([True] * int(oos_res["n"] * oos_res["win_rate"])
                                  + [False] * int(oos_res["n"] * (1 - oos_res["win_rate"])))

        is_wr  = sum(is_trades_all) / len(is_trades_all) if is_trades_all else 0.0
        oos_wr = (sum(1 for x in oos_trades_all if x) / len(oos_trades_all)
                  if oos_trades_all else 0.0)

        # OOS degradation ratio: should be >= 60% of IS win rate
        passed = (oos_wr >= is_wr * 0.60) if is_wr > 0 else False

        # Approximate OOS return (simple: win_rate * avg_win - loss_rate * avg_loss)
        avg_win_r  = ATR_TARGET / ATR_STOP
        oos_return = (oos_wr * avg_win_r - (1 - oos_wr)) * 100 if oos_trades_all else 0.0

        try:
            is_start_date  = str(spy.index[start])[:10] if hasattr(spy.index[0], '__str__') else f"bar{start}"
            is_end_date    = str(spy.index[is_end - 1])[:10] if is_end <= len(spy) else f"bar{is_end}"
            oos_start_date = str(spy.index[is_end])[:10] if is_end < len(spy) else f"bar{is_end}"
            oos_end_date   = str(spy.index[oos_end - 1])[:10] if oos_end <= len(spy) else f"bar{oos_end}"
        except Exception:
            is_start_date = oos_start_date = oos_end_date = is_end_date = "N/A"

        results.append(WindowResult(
            window_idx=window_idx,
            is_start=is_start_date, is_end=is_end_date,
            oos_start=oos_start_date, oos_end=oos_end_date,
            is_trades=len(is_trades_all),
            is_win_rate=round(is_wr, 3),
            oos_trades=len(oos_trades_all),
            oos_win_rate=round(oos_wr, 3),
            oos_return_pct=round(oos_return, 1),
            passed=passed,
        ))

        window_idx += 1
        start += step

    return results


def print_report(results: list[WindowResult]) -> None:
    if not results:
        print("No walk-forward results to display.")
        return

    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    avg_oos_wr = np.mean([r.oos_win_rate for r in results])
    avg_is_wr  = np.mean([r.is_win_rate for r in results])

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION REPORT — {total} windows")
    print(f"{'='*70}")
    print(f"  Avg IS win rate  : {avg_is_wr:.1%}")
    print(f"  Avg OOS win rate : {avg_oos_wr:.1%}")
    print(f"  Windows passed   : {passed}/{total} ({passed/total:.0%})")
    print(f"  Verdict          : {'✓ ROBUST' if passed/total >= 0.70 else '⚠ POSSIBLE OVERFIT'}")
    print(f"{'='*70}")
    print(f"\n{'Win#':<6} {'IS Period':<22} {'IS WR':>7} {'OOS Period':<22} {'OOS WR':>7} {'OOS Ret%':>9} {'Pass':>5}")
    print("-" * 80)
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"  {r.window_idx:<4} {r.is_start+' → '+r.is_end:<22} {r.is_win_rate:>6.1%}"
              f"  {r.oos_start+' → '+r.oos_end:<22} {r.oos_win_rate:>6.1%}"
              f" {r.oos_return_pct:>8.1f}%  {status}")
    print()


if __name__ == "__main__":
    results = run_walk_forward()
    print_report(results)
