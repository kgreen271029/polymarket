"""
strategies/orb_strategy.py — 15-minute Opening Range Breakout (ORB).

Research: 15-min ORB has ~56% win rate and 1.8:1 R:R → 0.41R expected value per trade.
Fires once per day per symbol when price breaks the first 15-min high/low with volume.

PDT rule was eliminated June 4, 2026 — no day-trade limit.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timezone, timedelta
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.filters import earnings_blackout, volatility_regime_ok, sector_momentum_ok


WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "META", "TSLA", "AMD",
    "AMZN", "GOOGL", "SPY", "QQQ", "PLTR", "ARM",
]

# Check every 60 seconds during the ORB window (9:30-10:15 ET)
LOOP_INTERVAL     = 60
ORB_WINDOW_MINS   = 15    # first 15 minutes define the range
CHECK_WINDOW_MINS = 45    # check for breakout up to 45 minutes after open
STOP_ATR_MULT     = 1.5   # tighter stop for intraday
TARGET_MULT       = 2.7   # 1.8:1 R:R
MAX_POSITIONS     = 3

# ET offset: UTC-4 (EDT summer), UTC-5 (EST winter)
_MARKET_OPEN_ET = time(9, 30)
_ORB_END_ET     = time(9, 45)    # ORB range completes at 9:45
_ENTRY_CUTOFF   = time(10, 15)   # no new entries after 10:15


def _et_time_now() -> time:
    """Current time in US/Eastern (approximated as UTC-4 during EDT)."""
    utc_now = datetime.now(tz=timezone.utc)
    # EDT: UTC-4 from second Sunday March to first Sunday November
    month = utc_now.month
    et_offset = timedelta(hours=-4) if 3 <= month <= 11 else timedelta(hours=-5)
    et_now = utc_now + et_offset
    return et_now.time()


def _et_date_today() -> str:
    """Date string in ET timezone (for daily reset tracking)."""
    utc_now = datetime.now(tz=timezone.utc)
    month = utc_now.month
    et_offset = timedelta(hours=-4) if 3 <= month <= 11 else timedelta(hours=-5)
    return (utc_now + et_offset).strftime("%Y-%m-%d")


class ORBStrategy(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        market_data,
        robinhood_broker,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._market_data = market_data
        self._rh = robinhood_broker
        self._portfolio = portfolio_tracker

        # Per-symbol ORB state (reset daily)
        self._orb_ranges: dict[str, dict] = {}  # {sym: {high, low, volume_avg, set}}
        self._fired_today: set[str] = set()     # symbols already traded today
        self._last_date: str = ""               # date string for daily reset

    def _daily_reset(self) -> None:
        today = _et_date_today()
        if today != self._last_date:
            self._orb_ranges.clear()
            self._fired_today.clear()
            self._last_date = today
            logger.info("[ORB] Daily reset — new date {}", today)

    async def generate_signals(self) -> list[TradeSignal]:
        if not self._market_open():
            return []

        self._daily_reset()
        et_now = _et_time_now()

        # Before market open or after entry cutoff — nothing to do
        if et_now < _MARKET_OPEN_ET or et_now > _ENTRY_CUTOFF:
            return []

        # Volatility regime check
        regime_ok, regime_reason = await asyncio.to_thread(volatility_regime_ok)
        if not regime_ok:
            return []

        signals: list[TradeSignal] = []
        open_count = len(self._portfolio.positions)

        for symbol in WATCHLIST:
            if symbol in self._fired_today:
                continue
            if open_count >= MAX_POSITIONS:
                break

            try:
                # Build / refresh ORB range using 5-min intraday bars
                df5 = await self._market_data.get_bars(symbol, "5Min", limit=20)
                if df5 is None or len(df5) < 3:
                    continue

                if symbol not in self._orb_ranges:
                    # Compute ORB range from first 15 minutes (3 × 5-min bars)
                    orb_bars = df5.head(3)
                    orb_high = float(orb_bars["high"].max())
                    orb_low  = float(orb_bars["low"].min())
                    orb_vol  = float(orb_bars["volume"].mean())
                    self._orb_ranges[symbol] = {
                        "high": orb_high, "low": orb_low,
                        "avg_vol": orb_vol, "set": True,
                    }
                    logger.debug("[ORB] {} range set: H={:.2f} L={:.2f}", symbol, orb_high, orb_low)

                orb = self._orb_ranges[symbol]

                # Wait until ORB window is complete before entering
                if et_now < _ORB_END_ET:
                    continue

                # Earnings blackout
                blocked, _ = await asyncio.to_thread(earnings_blackout, symbol)
                if blocked:
                    continue

                # Sector momentum
                sec_ok, _ = await asyncio.to_thread(sector_momentum_ok, symbol)
                if not sec_ok:
                    continue

                # Current bar (last completed 5-min bar)
                latest = df5.iloc[-1]
                cur_close  = float(latest["close"])
                cur_high   = float(latest["high"])
                cur_vol    = float(latest["volume"])

                orb_range = orb["high"] - orb["low"]
                if orb_range <= 0:
                    continue

                vol_expansion = cur_vol / orb["avg_vol"] if orb["avg_vol"] > 0 else 1.0

                # Breakout above ORB high with volume expansion ≥ 1.5×
                if cur_close > orb["high"] and vol_expansion >= 1.5:
                    stop    = orb["high"] - orb_range * STOP_ATR_MULT
                    target  = cur_close + orb_range * TARGET_MULT
                    risk    = cur_close - stop
                    if risk <= 0:
                        continue

                    cash    = self._portfolio.get_available_capital()
                    max_risk_dollar = min(cash * 0.15, 20.0)  # 15% of cash, max $20
                    qty     = round(max_risk_dollar / risk, 6) if risk > 0 else 0
                    if qty * cur_close < 1.0:
                        continue

                    logger.info("[ORB] {} breakout above {:.2f} (vol {:.1f}x) — BUY",
                                symbol, orb["high"], vol_expansion)
                    self._fired_today.add(symbol)
                    open_count += 1

                    signals.append(TradeSignal(
                        symbol=symbol,
                        side="buy",
                        asset_class="stock",
                        strategy_name=self.name,
                        entry_price=cur_close,
                        stop_price=round(stop, 2),
                        take_profit=round(target, 2),
                        confidence="Medium",
                        reasoning=f"ORB breakout: H={orb['high']:.2f} vol={vol_expansion:.1f}x range={orb_range:.2f}",
                        qty=qty,
                    ))

            except Exception as e:
                logger.error("[ORB] {} error: {}", symbol, e)

        return signals

    async def should_exit(self, position) -> bool:
        try:
            df = await self._market_data.get_bars(position.symbol, "5Min", limit=10)
            if df is None or len(df) < 3:
                return False
            # Exit ORB positions after market close (3:45 PM ET) or on reversal
            et_now = _et_time_now()
            if et_now >= time(15, 45):
                return True
            # Exit if price falls back below ORB high (failed breakout)
            orb = self._orb_ranges.get(position.symbol)
            if orb:
                cur = float(df["close"].iloc[-1])
                if cur < orb["high"] * 0.995:  # 0.5% below ORB high = failed
                    return True
            return False
        except Exception:
            return False

    async def run(self) -> None:
        self._running = True
        logger.info("[ORB] Started — 15-min opening range breakout strategy")
        while self._running:
            try:
                signals = await self.generate_signals()
                for sig in signals:
                    await self._signal_bus.put(sig)
                    logger.info("[ORB] → {} {} @ ${:.2f} stop=${:.2f} target=${:.2f}",
                                sig.side.upper(), sig.symbol, sig.entry_price,
                                sig.stop_price or 0, sig.take_profit or 0)
            except Exception as exc:
                logger.error("[ORB] run error: {}", exc)
            await asyncio.sleep(LOOP_INTERVAL)
