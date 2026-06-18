"""
backtest/compare_strategies.py — Compare baseline vs improved strategy on 30-day window.

Baseline: price > SMA50, RSI 38-65, volume > 1.3x (original)
Improved: + VCP filter (ATR contracting) + tighter SMA band (3.5% vs 4.0%) + chandelier exit

Run: python backtest/compare_strategies.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from analysis.multifactor import score_symbol, volatility_contraction, chandelier_exit
from analysis.technical import TechnicalAnalyzer


SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN",
    "AMD", "PLTR", "ARM", "JPM", "GS", "COIN", "MSTR",
]
STARTING_CAPITAL = 200.0
MAX_POSITIONS    = 3
POSITION_SIZE    = STARTING_CAPITAL * 0.25  # 25% per trade
SLIPPAGE         = 0.001


def _download(symbol: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(symbol, period="4mo", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 40:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        return raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception:
        return None


def _run_strategy(all_data: dict[str, pd.DataFrame], spy: pd.DataFrame,
                  use_vcp: bool = True, label: str = "") -> dict:
    """Simulate 30-day live trading window (last 30 bars of each symbol)."""
    trades = []
    equity = STARTING_CAPITAL
    open_positions: dict[str, dict] = {}

    for sym, full_df in all_data.items():
        if sym == "SPY" or full_df is None or len(full_df) < 65:
            continue

        # Use last 30 bars as "live" window; preceding bars as history
        live_start = len(full_df) - 30
        spy_full = spy

        for bar_idx in range(live_start, len(full_df) - 1):
            # Check exits for this symbol first
            if sym in open_positions:
                pos = open_positions[sym]
                cur_price = float(full_df["close"].iloc[bar_idx])

                # Update highest price
                if cur_price > pos["highest"]:
                    pos["highest"] = cur_price

                # Chandelier trailing stop from recent highs
                trail_stop = 0.0
                if bar_idx >= 22:
                    df_for_trail = full_df.iloc[:bar_idx + 1]
                    try:
                        trail_stop = chandelier_exit(df_for_trail, atr_mult=2.5)
                    except Exception:
                        pass

                effective_stop = max(pos["stop"], trail_stop)
                days_held = bar_idx - pos["entry_bar"]

                exit_reason = None
                if float(full_df["low"].iloc[bar_idx]) <= effective_stop:
                    exit_reason = "stop"
                    exit_price = effective_stop * (1 - SLIPPAGE)
                elif pos["target"] and float(full_df["high"].iloc[bar_idx]) >= pos["target"]:
                    exit_reason = "target"
                    exit_price = pos["target"] * (1 - SLIPPAGE)
                elif days_held >= 10 and cur_price < pos["entry_price"]:
                    exit_reason = "max_hold"
                    exit_price = cur_price * (1 - SLIPPAGE)

                if exit_reason:
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    equity += pnl
                    trades.append({
                        "symbol": sym, "exit": exit_reason,
                        "entry": pos["entry_price"], "exit_price": exit_price,
                        "pnl": pnl, "pct": (exit_price / pos["entry_price"] - 1) * 100,
                        "days": days_held, "win": pnl > 0,
                    })
                    del open_positions[sym]

            # Entry check (only if not already in this symbol and capacity available)
            if sym in open_positions or len(open_positions) >= MAX_POSITIONS:
                continue

            # Need at least 60 bars of history for indicators
            if bar_idx < 60:
                continue

            window   = full_df.iloc[:bar_idx + 1].reset_index(drop=True)
            spy_win  = spy_full.iloc[:bar_idx + 1].reset_index(drop=True) if len(spy_full) > bar_idx else spy_full

            try:
                s = TechnicalAnalyzer.build_signal_summary(window)
                if not s:
                    continue

                mf = score_symbol(window, spy_win)

                price      = s.get("latest_close", 0)
                rsi        = s.get("rsi", 50)
                sma50      = s.get("sma_50", price)
                vol_ratio  = s.get("volume_ratio", 1.0)
                pct_vs_sma = s.get("price_vs_sma20_pct", 0)
                atr        = s.get("atr", price * 0.02)

                # Core conditions (both strategies share these)
                base_ok = (
                    price > 0
                    and price > sma50
                    and 38 <= rsi <= 65
                    and vol_ratio > 1.3
                )

                if use_vcp:
                    # Improved: tighter SMA band + VCP filter
                    sma_ok = 0 < pct_vs_sma < 3.5
                    vcp_score = volatility_contraction(window) if len(window) >= 55 else 0.3
                    entry_ok = base_ok and sma_ok and vcp_score >= 0.5
                    atr_stop_mult   = 2.2
                    atr_target_mult = 3.3
                else:
                    # Baseline: wider SMA band, no VCP filter
                    sma_ok = 0 < pct_vs_sma < 4.0
                    entry_ok = base_ok and sma_ok
                    atr_stop_mult   = 2.0
                    atr_target_mult = 3.0

                if not entry_ok:
                    continue

                entry_price = float(full_df["open"].iloc[bar_idx + 1]) * (1 + SLIPPAGE)
                stop   = entry_price - atr_stop_mult * atr
                target = entry_price + atr_target_mult * atr
                shares = POSITION_SIZE / entry_price

                open_positions[sym] = {
                    "entry_price": entry_price,
                    "stop": stop, "target": target,
                    "shares": shares, "entry_bar": bar_idx + 1,
                    "highest": entry_price,
                }

            except Exception:
                continue

    # Force-close any remaining positions at last price
    for sym, pos in open_positions.items():
        if sym in all_data and all_data[sym] is not None:
            exit_price = float(all_data[sym]["close"].iloc[-1]) * (1 - SLIPPAGE)
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            equity += pnl
            trades.append({
                "symbol": sym, "exit": "eod",
                "entry": pos["entry_price"], "exit_price": exit_price,
                "pnl": pnl, "pct": (exit_price / pos["entry_price"] - 1) * 100,
                "days": 30, "win": pnl > 0,
            })

    n       = len(trades)
    wins    = sum(1 for t in trades if t["win"])
    wr      = wins / n if n > 0 else 0.0
    pnl_all = [t["pnl"] for t in trades]
    avg_win = np.mean([t["pnl"] for t in trades if t["win"]]) if wins > 0 else 0
    avg_los = abs(np.mean([t["pnl"] for t in trades if not t["win"]])) if wins < n else 0

    return {
        "label": label, "trades": trades, "n": n, "wins": wins,
        "win_rate": wr, "equity": equity,
        "return_pct": (equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100,
        "avg_win": avg_win, "avg_loss": avg_los,
        "profit_factor": (avg_win * wins) / (avg_los * (n - wins) + 1e-9) if n > 0 else 0,
        "max_dd": _max_drawdown(STARTING_CAPITAL, [t["pnl"] for t in trades]),
    }


def _max_drawdown(start: float, pnl_list: list[float]) -> float:
    equity = start
    peak = start
    max_dd = 0.0
    for p in pnl_list:
        equity += p
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
    return max_dd


def print_comparison(base: dict, improved: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  STRATEGY COMPARISON — 30-day backtest on ${STARTING_CAPITAL:.0f}")
    print(f"{'='*65}")
    hdr = f"  {'Metric':<25} {'Baseline':>12} {'Improved':>12} {'Change':>10}"
    print(hdr)
    print("-" * 65)

    metrics = [
        ("Total Trades",      f"{base['n']}",       f"{improved['n']}",       ""),
        ("Win Rate",          f"{base['win_rate']:.1%}",  f"{improved['win_rate']:.1%}",
         f"{(improved['win_rate']-base['win_rate'])*100:+.1f}pp"),
        ("Final Equity",      f"${base['equity']:.2f}",   f"${improved['equity']:.2f}",
         f"{improved['return_pct']-base['return_pct']:+.1f}%"),
        ("Return %",          f"{base['return_pct']:.2f}%", f"{improved['return_pct']:.2f}%",
         f"{improved['return_pct']-base['return_pct']:+.2f}pp"),
        ("Profit Factor",     f"{base['profit_factor']:.2f}x",  f"{improved['profit_factor']:.2f}x", ""),
        ("Max Drawdown",      f"{base['max_dd']:.1%}",   f"{improved['max_dd']:.1%}",
         f"{(improved['max_dd']-base['max_dd'])*100:+.1f}pp"),
        ("Avg Win",           f"${base['avg_win']:.2f}",  f"${improved['avg_win']:.2f}", ""),
        ("Avg Loss",          f"${base['avg_loss']:.2f}", f"${improved['avg_loss']:.2f}", ""),
    ]
    for name, bv, iv, chg in metrics:
        print(f"  {name:<25} {bv:>12} {iv:>12} {chg:>10}")
    print(f"{'='*65}\n")

    if improved["trades"]:
        print("  Improved strategy trades:")
        for t in improved["trades"]:
            icon = "✓" if t["win"] else "✗"
            print(f"  {icon} {t['symbol']:<6} {t['exit']:<10} "
                  f"${t['entry']:.2f} → ${t['exit_price']:.2f}  "
                  f"{t['pct']:+.2f}%  ({t['days']}d)")
    print()


if __name__ == "__main__":
    print("Downloading market data...")
    all_data = {}
    for sym in SYMBOLS:
        df = _download(sym)
        if df is not None:
            all_data[sym] = df
            print(f"  {sym}: {len(df)} bars")

    spy = _download("SPY")
    if spy is None:
        print("ERROR: SPY data unavailable")
        sys.exit(1)

    print(f"\nRunning baseline strategy on {len(all_data)} symbols...")
    baseline = _run_strategy(all_data, spy, use_vcp=False, label="Baseline")

    print("Running improved strategy (VCP + tighter bands + chandelier)...")
    improved = _run_strategy(all_data, spy, use_vcp=True, label="Improved")

    print_comparison(baseline, improved)
