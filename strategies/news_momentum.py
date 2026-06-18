"""strategies/news_momentum.py — News-driven stock momentum via Robinhood MCP."""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.sentiment import SentimentScorer
from data.market_data import HeadlineEvent


class NewsMomentum(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        robinhood_broker,
        ai_analyzer,
        market_data,
        news_queue: asyncio.Queue,
        breaking_queue: asyncio.Queue,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._rh = robinhood_broker
        self._ai = ai_analyzer
        self._market_data = market_data
        self._news_queue = news_queue
        self._breaking_queue = breaking_queue
        self._portfolio = portfolio_tracker
        self._scorer = SentimentScorer()

    async def generate_signals(self) -> list[TradeSignal]:
        return []

    async def should_exit(self, position: object) -> bool:
        return False

    async def run_loop(self, _interval: int = 0) -> None:  # type: ignore[override]
        self._running = True
        logger.info("[NewsMomentum] Event-driven loop started")
        while self._running:
            try:
                try:
                    event: HeadlineEvent = self._breaking_queue.get_nowait()
                except asyncio.QueueEmpty:
                    try:
                        event = await asyncio.wait_for(self._news_queue.get(), timeout=10.0)
                    except asyncio.TimeoutError:
                        continue

                if not self._market_open():
                    continue

                await self._process_event(event)
            except Exception as e:
                logger.error("[NewsMomentum] Unhandled error: {}", e)
                await asyncio.sleep(1)

    async def _process_event(self, event: HeadlineEvent) -> None:
        stock_symbols = [s for s in event.symbols_mentioned if "/" not in s and len(s) <= 5]
        if not stock_symbols:
            return

        score = self._scorer.score_headline(event)
        if score.score < 0.3 or event.urgency_score < 0.4:
            return

        open_stocks = sum(1 for p in self._portfolio.positions.values() if p.asset_class == "stock")
        if open_stocks >= 3:
            return

        for symbol in stock_symbols[:2]:
            try:
                quote = await self._rh.get_quote(symbol)
                if not quote:
                    continue
                price = 0.0
                for field in ("last_trade_price", "last_non_reg_trade_price", "ask_price"):
                    val = quote.get(field)
                    if val:
                        try:
                            price = float(val)
                            break
                        except (TypeError, ValueError):
                            pass
                if price <= 0:
                    continue

                from analysis.ai_analyzer import AnalysisContext
                context = AnalysisContext(
                    symbol=symbol,
                    asset_class="stock",
                    strategy_name=self.name,
                    proposed_action="BUY",
                    signal_summary={"latest_close": price, "rsi": 50},
                    news_headlines=[event.title],
                    available_capital=self._portfolio.get_available_capital(),
                    open_position_count=len(self._portfolio.positions),
                    daily_pnl_pct=self._portfolio.daily_pnl,
                )
                decision = await self._ai.analyze(context)

                if decision.recommendation != "BUY":
                    continue

                stop = decision.stop_price or round(price * 0.96, 2)
                target = decision.take_profit or round(price * 1.08, 2)

                cash = self._portfolio.get_available_capital()
                max_dollars = min(cash * 0.20, 20.0)
                qty = round(max_dollars / price, 6) if price > 0 else 0
                if qty <= 0:
                    continue

                force_swing = self._portfolio.get_day_trade_count() >= 3
                signal = TradeSignal(
                    symbol=symbol,
                    side="buy",
                    asset_class="stock",
                    strategy_name=self.name,
                    entry_price=price,
                    stop_price=stop,
                    take_profit=target,
                    confidence=decision.confidence,
                    reasoning=f"{event.title[:80]} | {decision.reasoning}",
                    qty=qty,
                    metadata={"force_swing": force_swing, "news_url": event.raw_url},
                )
                await self._signal_bus.put(signal)
                logger.info("[NewsMomentum] Signal: BUY {} @ {} | {}", symbol, price, event.title[:60])
            except Exception as e:
                logger.error("[NewsMomentum] Error processing {}: {}", symbol, e)

    async def run(self) -> None:
        await self.run_loop()
