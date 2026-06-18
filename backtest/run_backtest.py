"""
backtest/run_backtest.py — Download data and run 30-day backtest.

Usage:
    python -m backtest.run_backtest
    python backtest/run_backtest.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from backtest.engine import run_backtest

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN",
    "TSLA", "AMD", "ARM", "JPM", "GS", "BAC",
    "PLTR", "SMCI", "COIN", "MSTR",
]
INITIAL_CAPITAL = 200.0


def _fetch(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """Fetch OHLCV bars from yfinance and normalise column names."""
    try:
        end   = datetime.now()
        start = end - timedelta(days=days)
        raw   = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return None
        raw.index = pd.to_datetime(raw.index)
        # yfinance ≥0.2.x returns MultiIndex (Price, Ticker) columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        raw = raw.rename(columns={"adj_close": "close"})
        for col in ("open", "high", "low", "close", "volume"):
            if col not in raw.columns:
                return None
        return raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        print(f"  [warn] {symbol}: {e}")
        return None


def main() -> None:
    print("=" * 60)
    print("  30-DAY IMPROVED STRATEGY BACKTEST")
    print(f"  Starting capital: ${INITIAL_CAPITAL:.2f}")
    print(f"  Symbols:          {len(SYMBOLS)}")
    print("=" * 60)
    print("Downloading price history…")

    stock_data: dict[str, pd.DataFrame] = {}
    spy_df = _fetch("SPY", days=120)
    if spy_df is None:
        print("ERROR: Could not download SPY data.")
        sys.exit(1)
    stock_data["SPY"] = spy_df

    for sym in SYMBOLS:
        df = _fetch(sym, days=120)
        if df is not None:
            stock_data[sym] = df
            print(f"  {sym:<6} {len(df)} bars")
        else:
            print(f"  {sym:<6} SKIPPED (no data)")

    print(f"\nRunning backtest on last 30 trading days…")
    result = run_backtest(
        stock_data=stock_data,
        spy_df=spy_df,
        initial_capital=INITIAL_CAPITAL,
        max_positions=3,
        position_pct=0.30,
        slippage_pct=0.001,
        max_hold_days=10,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Starting capital : ${result.initial_capital:>9.2f}")
    print(f"  Final equity     : ${result.final_equity:>9.2f}")
    pct = result.total_return_pct
    sign = "+" if pct >= 0 else ""
    print(f"  Total return     : {sign}{pct:.2f}%")
    print(f"  Total trades     : {len(result.closed_trades)}")
    print(f"  Win rate         : {result.win_rate:.1f}%")
    print(f"  Avg win  ($ P&L) : ${result.avg_win:>8.2f}")
    print(f"  Avg loss ($ P&L) : ${result.avg_loss:>8.2f}")
    print(f"  Profit factor    : {result.profit_factor:.2f}x")
    print(f"  Max drawdown     : {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe ratio     : {result.sharpe:.2f}")
    print("=" * 60)

    # ── Trade log ─────────────────────────────────────────────────────────────
    if not result.closed_trades:
        print("\nNo closed trades in the backtest window.")
        print("(Signal filters may be too strict for this 30-day window.)")
        return

    print("\n  TRADE LOG")
    print(f"  {'SYM':<6} {'ENTRY':>10} {'EXIT':>10} {'ENTRY $':>8} {'EXIT $':>8} {'P&L $':>8} {'REASON'}")
    print("  " + "-" * 66)
    for t in sorted(result.closed_trades, key=lambda x: x.entry_date):
        pnl_str = f"{'+'if t.pnl>=0 else ''}{t.pnl:.2f}"
        print(
            f"  {t.symbol:<6} {str(t.entry_date):>10} {str(t.exit_date):>10} "
            f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
            f"{pnl_str:>8} {t.exit_reason}"
        )

    # ── Equity curve summary ───────────────────────────────────────────────────
    print("\n  EQUITY CURVE (weekly snapshots)")
    print(f"  {'DATE':>12}  {'EQUITY':>10}")
    curve = result.equity_curve
    step  = max(1, len(curve) // 6)
    for i in range(0, len(curve), step):
        d, eq = curve[i]
        print(f"  {str(d):>12}  ${eq:>9.2f}")
    if curve:
        d, eq = curve[-1]
        print(f"  {str(d):>12}  ${eq:>9.2f}  ← final")

    print("\n  KEY STRATEGY IMPROVEMENTS TESTED:")
    print("  ✓ Relative strength filter vs SPY (only outperformers)")
    print("  ✓ ATR-based stops (2×ATR) and targets (3×ATR)")
    print("  ✓ RSI sweet spot 42–72 (avoids overbought entries)")
    print("  ✓ Volume conviction filter (1.3× average)")
    print("  ✓ MACD histogram momentum confirmation")
    print("  ✓ Gap filter: skip chasing gaps > 5%")
    print("  ✓ Max 10-day hold (forces discipline)")
    print("  ✓ Fractional shares + 0.1% slippage simulation")
    print("=" * 60)


if __name__ == "__main__":
    main()
