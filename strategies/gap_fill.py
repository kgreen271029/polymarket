"""
strategies/gap_fill.py — Gap-fill mean reversion (Robinhood stocks).

Research basis: ~76% of gaps fill; downward gaps fill more reliably than up gaps.
Strategy: when a liquid stock gaps DOWN 1-4% at open and shows stabilisation,
buy expecting a fill back toward the prior close. Tight ATR stop, target = gap fill.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.technical import TechnicalAnalyzer
from analysis.filters import earnings_blackout, volatility_regime_ok, kelly_position_size
from analysis.metrics import vix_kelly_fraction, get_adaptive_stats

# Liquid large-caps only — tight spreads matter for gap fills
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "AMD", "JPM", "GS", "TSLA", "SPY", "QQQ",
]
LOOP_INTERVAL = 600  # 10 min — gap fills happen intraday


class GapFillTrader(BaseStrategy):
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

    async def generate_signals(self) -> list[TradeSignal]:
        if not self._market_open():
            return []

        cash = self._portfolio.get_available_capital()
        if cash < 5.0:
            return []

        # Regime + VIX gates
        regime_ok, _ = await asyncio.to_thread(volatility_regime_ok)
        if not regime_ok:
            return []
        kelly_frac, vix_note = await asyncio.to_thread(vix_kelly_fraction)
        if kelly_frac <= 0:
            logger.info("[GapFill] {} — no new trades", vix_note)
            return []

        stats = get_adaptive_stats()
        signals: list[TradeSignal] = []
        open_syms = {p.split(":")[0] for p in self._portfolio.positions}

        for symbol in WATCHLIST:
            if symbol in open_syms:
                continue
            try:
                df = await self._market_data.get_bars(symbol, "1Day", limit=30)
                if df is None or len(df) < 20:
                    continue
                if "close" not in df.columns or "open" not in df.columns:
                    continue

                prev_close = float(df["close"].iloc[-2])
                today_open = float(df["open"].iloc[-1])
                today_low  = float(df["low"].iloc[-1])
                cur_price  = float(df["close"].iloc[-1])
                if prev_close <= 0:
                    continue

                gap_pct = (today_open - prev_close) / prev_close * 100

                # Only DOWN gaps of 1-4% (sweet spot for fills)
                if not (-4.0 <= gap_pct <= -1.0):
                    continue

                # Earnings filter — gaps on earnings often DON'T fill
                blocked, _ = await asyncio.to_thread(earnings_blackout, symbol, 3)
                if blocked:
                    continue

                # Stabilisation: price has bounced off the low (not free-falling)
                if cur_price <= today_low * 1.002:
                    continue

                s = TechnicalAnalyzer.build_signal_summary(df)
                atr = s.get("atr", cur_price * 0.02)
                rsi = s.get("rsi", 50)

                # Want oversold-ish but recovering
                if rsi > 55:
                    continue

                stop   = round(min(today_low, cur_price - 1.5 * atr), 2)
                target = round(prev_close, 2)   # the gap fill
                if target <= cur_price or stop >= cur_price:
                    continue

                # Reward:risk must be >= 1.2
                rr = (target - cur_price) / (cur_price - stop)
                if rr < 1.2:
                    continue

                dollar_size = kelly_position_size(
                    account_cash=cash,
                    entry_price=cur_price,
                    stop_price=stop,
                    win_rate=stats["win_rate"],
                    avg_win_loss_ratio=max(stats["avg_win_loss_ratio"], rr),
                    max_pct=kelly_frac * 0.6,   # gap fills are mean-reversion, size smaller
                )
                qty = round(dollar_size / cur_price, 6)
                if qty <= 0:
                    continue

                reason = (f"Gap-fill: {symbol} gapped {gap_pct:.1f}% down, "
                          f"recovering off low, target prior close ${target:.2f} "
                          f"(R:R {rr:.1f}). {vix_note}")
                logger.info("[GapFill] → {}", reason)
                signals.append(TradeSignal(
                    symbol=symbol,
                    side="buy",
                    asset_class="stock",
                    strategy_name=self.name,
                    entry_price=cur_price,
                    stop_price=stop,
                    take_profit=target,
                    confidence="Medium",
                    reasoning=reason,
                    qty=qty,
                ))
            except Exception as e:
                logger.error("[GapFill] Error on {}: {}", symbol, e)

        return signals

    async def should_exit(self, position) -> bool:
        return False  # exits handled by SwingTrader's stop/target monitor

    async def run(self, market_just_opened: "asyncio.Event | None" = None) -> None:
        self._running = True
        logger.info("[GapFill] Started (interval={}s)", LOOP_INTERVAL)
        while self._running:
            try:
                signals = await self.generate_signals()
                for sig in signals:
                    await self._signal_bus.put(sig)
                    logger.info("[GapFill] → {} {} @ ${:.2f} stop=${:.2f} tgt=${:.2f}",
                                sig.side.upper(), sig.symbol, sig.entry_price,
                                sig.stop_price or 0, sig.take_profit or 0)
            except Exception as exc:
                logger.error("[GapFill] run error: {}", exc)

            remaining = LOOP_INTERVAL
            while remaining > 0 and self._running:
                chunk = min(60, remaining)
                if market_just_opened is not None:
                    try:
                        await asyncio.wait_for(market_just_opened.wait(), timeout=float(chunk))
                        logger.info("[GapFill] Market-open signal — immediate scan")
                        break
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(chunk)
                remaining -= chunk
