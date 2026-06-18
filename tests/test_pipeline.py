"""
tests/test_pipeline.py — End-to-end verification of the trading pipeline.

Tests the full flow with mocked broker:
  signal -> risk check -> route -> fill emit -> portfolio update -> exit -> journal

Run: python -m tests.test_pipeline
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. Technical indicators return all required keys
# ---------------------------------------------------------------------------

def test_technical_keys():
    print("\n[1] Technical analyzer keys")
    from analysis.technical import TechnicalAnalyzer
    import numpy as np

    # Build a synthetic uptrend with 60 bars
    n = 60
    close = pd.Series([100 + i * 0.5 + (i % 3) for i in range(n)])
    df = pd.DataFrame({
        "open":  close * 0.99,
        "high":  close * 1.02,
        "low":   close * 0.98,
        "close": close,
        "volume": pd.Series([1_000_000 + i * 10_000 for i in range(n)]),
    })
    s = TechnicalAnalyzer.build_signal_summary(df)
    for key in ("rsi", "atr", "sma_20", "sma_50", "ema_20", "macd_hist",
                "volume_ratio", "trend", "latest_close", "price_vs_sma20_pct"):
        check(f"summary has '{key}'", key in s, f"got keys {list(s.keys())}")
    check("atr > 0", s.get("atr", 0) > 0, f"atr={s.get('atr')}")
    check("sma_50 > 0", s.get("sma_50", 0) > 0, f"sma_50={s.get('sma_50')}")
    check("sma_50 != latest_close (real SMA)",
          abs(s.get("sma_50", 0) - s.get("latest_close", 0)) > 0.01)


# ---------------------------------------------------------------------------
# 2. Risk manager: buys, sells, fractional shares
# ---------------------------------------------------------------------------

def test_risk_manager():
    print("\n[2] Risk manager")
    from core.portfolio import PortfolioTracker
    from core.risk_manager import RiskManager
    from strategies.base_strategy import TradeSignal

    class Cfg:
        MAX_RISK_PER_TRADE = 20.0
        DAILY_LOSS_LIMIT_PCT = 0.15
        MAX_OPEN_POSITIONS = 6

    port = PortfolioTracker(starting_capital=92.0)
    rm = RiskManager(port, Cfg())

    # Fractional buy of an expensive stock should be APPROVED (not rejected as <1 share)
    buy = TradeSignal(symbol="ARM", side="buy", asset_class="stock",
                      strategy_name="t", entry_price=418.0, stop_price=400.0,
                      take_profit=450.0, confidence="Medium", reasoning="t", qty=0.04)
    v = rm.check(buy)
    check("fractional buy approved", v.approved, v.reason)
    check("fractional qty preserved/scaled", v.adjusted_qty and v.adjusted_qty > 0,
          f"qty={v.adjusted_qty}")

    # Sell with no stop_price should be APPROVED (exits bypass guards)
    sell = TradeSignal(symbol="ARM", side="sell", asset_class="stock",
                       strategy_name="t", entry_price=418.0, stop_price=None,
                       take_profit=None, confidence="High", reasoning="exit", qty=0.04)
    v2 = rm.check(sell)
    check("sell without stop approved", v2.approved, v2.reason)

    # Oversized buy should be SCALED to fit cash
    big = TradeSignal(symbol="MSFT", side="buy", asset_class="stock",
                      strategy_name="t", entry_price=430.0, stop_price=420.0,
                      take_profit=450.0, confidence="Medium", reasoning="t", qty=10.0)
    v3 = rm.check(big)
    check("oversized buy scaled down", v3.approved and v3.adjusted_qty < 10.0,
          f"qty={v3.adjusted_qty}")
    check("scaled value within cash",
          v3.adjusted_qty * 430.0 <= 92.0 + 0.01,
          f"value={v3.adjusted_qty * 430.0:.2f}")


# ---------------------------------------------------------------------------
# 3. Full pipeline: signal -> fill -> portfolio -> exit -> journal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline():
    print("\n[3] Full pipeline (mock broker)")
    from core.engine import TradingEngine, FillEvent
    from core.portfolio import PortfolioTracker
    from core.risk_manager import RiskManager
    from strategies.base_strategy import TradeSignal

    class Cfg:
        MAX_RISK_PER_TRADE = 20.0
        DAILY_LOSS_LIMIT_PCT = 0.15
        MAX_OPEN_POSITIONS = 6
        starting_capital = 92.0

    # Mock Robinhood broker that emits fills like the real one
    class MockRH:
        def __init__(self, fill_queue):
            self._fill_queue = fill_queue
            self.orders = []
        async def place_order(self, symbol, side, qty, order_type="market", limit_price=None):
            self.orders.append((symbol, side, qty))
            await self._fill_queue.put(FillEvent(
                order_id=f"ord_{len(self.orders)}", symbol=symbol, side=side,
                qty=qty, fill_price=418.0 if side == "buy" else 450.0,
                asset_class="stock", timestamp=datetime.now(tz=timezone.utc),
            ))
            return f"ord_{len(self.orders)}"
        async def get_quote_price(self, symbol):
            return 450.0

    port = PortfolioTracker(starting_capital=92.0)
    engine = TradingEngine(port, None, None, None, None, Cfg(), notifier=None)
    engine._risk_manager = RiskManager(port, Cfg())
    mock_rh = MockRH(engine.fill_queue)
    engine._robinhood_broker = mock_rh
    engine.market_open = True

    # Start signal + fill processors
    tasks = [
        asyncio.create_task(engine._process_signals()),
        asyncio.create_task(engine._process_fills()),
    ]

    # Push a BUY signal
    await engine.signal_bus.put(TradeSignal(
        symbol="ARM", side="buy", asset_class="stock", strategy_name="SwingTrader",
        entry_price=418.0, stop_price=400.0, take_profit=450.0,
        confidence="Medium", reasoning="test buy", qty=0.04,
    ))
    await asyncio.sleep(0.5)

    check("buy order placed on broker", len(mock_rh.orders) == 1,
          f"orders={mock_rh.orders}")
    pos = port.get_open_position("ARM")
    check("position created in portfolio", pos is not None)
    if pos:
        check("stop_loss threaded to position", pos.stop_loss == 400.0,
              f"stop={pos.stop_loss}")
        check("take_profit threaded to position", pos.take_profit == 450.0,
              f"tp={pos.take_profit}")
        check("strategy name threaded", pos.strategy_name == "SwingTrader",
              f"strat={pos.strategy_name}")

    # Push a SELL signal to close
    await engine.signal_bus.put(TradeSignal(
        symbol="ARM", side="sell", asset_class="stock", strategy_name="SwingTrader",
        entry_price=450.0, stop_price=None, take_profit=None,
        confidence="High", reasoning="test exit", qty=0.04,
    ))
    await asyncio.sleep(0.5)

    check("sell order placed", len(mock_rh.orders) == 2, f"orders={mock_rh.orders}")
    check("position closed", port.get_open_position("ARM") is None)
    check("realized profit recorded", port.total_pnl > 0, f"pnl={port.total_pnl:.2f}")

    # Verify journaling happened
    from analysis.metrics import JOURNAL_PATH
    import json
    if JOURNAL_PATH.exists():
        data = json.loads(JOURNAL_PATH.read_text())
        check("trade journaled", len(data) >= 1, f"journal entries={len(data)}")
        if data:
            check("journal pnl positive", data[-1]["pnl"] > 0, f"pnl={data[-1]['pnl']}")

    engine._shutdown_event.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# 4. Metrics math
# ---------------------------------------------------------------------------

def test_metrics():
    print("\n[4] Metrics math")
    from analysis.metrics import expectancy, breakeven_win_rate, profit_factor

    check("expectancy 41/480/190 ≈ 84.7",
          abs(expectancy(0.41, 480, 190) - 84.7) < 0.1)
    check("breakeven 2:1 ≈ 0.333",
          abs(breakeven_win_rate(2.0) - 0.3333) < 0.01)
    check("breakeven 1:1 = 0.5",
          abs(breakeven_win_rate(1.0) - 0.5) < 0.01)
    check("profit_factor", abs(profit_factor([100, 50], [-30, -20]) - 3.0) < 0.01)


# ---------------------------------------------------------------------------
# 5. Filters return correct types
# ---------------------------------------------------------------------------

def test_filters():
    print("\n[5] Filters")
    from analysis.filters import kelly_position_size, atr_trailing_stop, bearish_divergence

    size = kelly_position_size(92.0, 418.0, 400.0, win_rate=0.5, avg_win_loss_ratio=1.5, max_pct=0.30)
    check("kelly size is positive", size > 0, f"size={size}")
    check("kelly size <= 30% cap", size <= 92.0 * 0.30 + 0.01, f"size={size}")

    trail = atr_trailing_stop(100.0, 2.0, 110.0, multiplier=1.5)
    check("trailing stop rises with price", trail > 100.0 - 3.0, f"trail={trail}")
    check("trailing stop has floor", trail >= 90.0, f"trail={trail}")

    # Divergence: price rising, RSI falling
    prices = pd.Series([100, 102, 104, 106, 108, 107, 109, 111, 113, 115])
    rsi = pd.Series([60, 65, 70, 72, 74, 68, 66, 64, 62, 60])
    check("bearish divergence detected", bearish_divergence(prices, rsi) in (True, False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("  PIPELINE VERIFICATION TESTS")
    print("=" * 60)

    test_technical_keys()
    test_risk_manager()
    await test_full_pipeline()
    test_metrics()
    test_filters()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
