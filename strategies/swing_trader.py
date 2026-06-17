"""strategies/swing_trader.py — Multi-day stock swing trades via Robinhood MCP (PDT-safe)."""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.technical import TechnicalAnalyzer

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ",
    "AMZN", "META", "GOOGL", "AMD", "COIN", "MSTR",
]
LOOP_INTERVAL = 900
MIN_HOLD_DAYS = 2


class SwingTrader(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        market_data,
        robinhood_broker,
        ai_analyzer,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._market_data = market_data
        self._rh = robinhood_broker
        self._ai = ai_analyzer
        self._portfolio = portfolio_tracker

    async def generate_signals(self) -> list[TradeSignal]:
        if not self._market_open():
            return []

        signals: list[TradeSignal] = []
        open_stocks = sum(1 for p in self._portfolio.positions.values() if p.asset_class == "stock")
        if open_stocks >= 2:
            return signals

        for symbol in WATCHLIST:
            if symbol in {p.split(":")[0] for p in self._portfolio.positions}:
                continue
            try:
                df = await self._market_data.get_bars(symbol, "1Day", limit=55)
                if df is None or len(df) < 30:
                    continue

                s = TechnicalAnalyzer.build_signal_summary(df)
                if not s:
                    continue

                price = s.get("latest_close", 0)
                rsi = s.get("rsi", 50)
                sma20 = s.get("sma_20", 0)
                volume_ratio = s.get("volume_ratio", 1.0)
                pct_vs_sma = s.get("price_vs_sma20_pct", 0)

                if (
                    price > 0
                    and sma20 > 0
                    and 0 < pct_vs_sma < 3.0   # just reclaimed SMA from below
                    and 38 <= rsi <= 60
                    and volume_ratio > 1.2
                ):
                    # Stop at recent 5-day low
                    recent_low = float(df["low"].tail(5).min())
                    stop = round(recent_low * 0.995, 2)
                    risk = price - stop
                    target = round(price + risk * 2.0, 2)

                    from analysis.ai_analyzer import AnalysisContext
                    context = AnalysisContext(
                        symbol=symbol,
                        asset_class="stock",
                        strategy_name=self.name,
                        proposed_action="BUY",
                        signal_summary=s,
                        news_headlines=[],
                        available_capital=self._portfolio.get_available_capital(),
                        open_position_count=len(self._portfolio.positions),
                        daily_pnl_pct=self._portfolio.daily_pnl,
                    )
                    decision = await self._ai.analyze(context)

                    if decision.recommendation == "BUY":
                        signals.append(TradeSignal(
                            symbol=symbol,
                            side="buy",
                            asset_class="stock",
                            strategy_name=self.name,
                            entry_price=price,
                            stop_price=decision.stop_price or stop,
                            take_profit=decision.take_profit or target,
                            confidence=decision.confidence,
                            reasoning=decision.reasoning,
                            metadata={"min_hold_days": MIN_HOLD_DAYS},
                        ))
            except Exception as e:
                logger.error("[SwingTrader] Error on {}: {}", symbol, e)

        return signals

    async def should_exit(self, position: object) -> bool:
        try:
            df = await self._market_data.get_bars(position.symbol, "1Day", limit=25)  # type: ignore[attr-defined]
            if df is None or len(df) < 20:
                return False
            s = TechnicalAnalyzer.build_signal_summary(df)
            rsi = s.get("rsi", 50)
            pct_vs_sma = s.get("price_vs_sma20_pct", 0)
            return rsi > 72 or pct_vs_sma < -2.0
        except Exception:
            return False

    async def run(self) -> None:
        await self.run_loop(LOOP_INTERVAL)
