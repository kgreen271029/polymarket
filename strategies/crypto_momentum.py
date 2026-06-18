"""strategies/crypto_momentum.py — News-driven crypto momentum breakouts."""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.technical import TechnicalAnalyzer
from analysis.sentiment import SentimentScorer
from data.market_data import HeadlineEvent

WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]
LOOP_INTERVAL = 300
STOP_PCT = 0.006
TARGET_PCT = 0.015

_CRYPTO_ALIASES = {
    "BTC": "BTC/USD", "BITCOIN": "BTC/USD",
    "ETH": "ETH/USD", "ETHEREUM": "ETH/USD",
    "SOL": "SOL/USD", "SOLANA": "SOL/USD",
    "DOGE": "DOGE/USD", "DOGECOIN": "DOGE/USD",
    "AVAX": "AVAX/USD", "AVALANCHE": "AVAX/USD",
}


class CryptoMomentum(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        market_data,
        ai_analyzer,
        news_queue: asyncio.Queue,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._market_data = market_data
        self._ai = ai_analyzer
        self._news_queue = news_queue
        self._portfolio = portfolio_tracker
        self._scorer = SentimentScorer()
        self._recent_news: list[tuple] = []

    def _drain_news(self) -> list[HeadlineEvent]:
        events: list[HeadlineEvent] = []
        while True:
            try:
                events.append(self._news_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    async def generate_signals(self) -> list[TradeSignal]:
        signals: list[TradeSignal] = []

        news_events = self._drain_news()
        crypto_news: dict[str, list[str]] = {}
        for event in news_events:
            score = self._scorer.score_headline(event)
            if score.score > 0.3:
                for sym in event.symbols_mentioned:
                    mapped = _CRYPTO_ALIASES.get(sym.upper())
                    if mapped:
                        crypto_news.setdefault(mapped, []).append(event.title)

        open_symbols = {p.split(":")[0] for p in self._portfolio.positions}
        open_crypto = sum(1 for p in self._portfolio.positions.values() if p.asset_class == "crypto")
        if open_crypto >= 4:
            return signals

        targets = list(crypto_news.keys()) if crypto_news else WATCHLIST[:3]

        for symbol in targets:
            if symbol in open_symbols or symbol.replace("/", "") in open_symbols:
                continue
            try:
                df = await self._market_data.get_bars(symbol, "5Min", limit=50)
                if df is None or len(df) < 20:
                    continue

                s = TechnicalAnalyzer.build_signal_summary(df)
                if not s:
                    continue

                price = s.get("latest_close", 0)
                volume_ratio = s.get("volume_ratio", 1.0)
                rsi = s.get("rsi", 50)
                vwap_delta = s.get("vwap_delta_pct", 0)

                if volume_ratio < 2.5 or vwap_delta < 0 or rsi > 70 or price == 0:
                    continue

                from analysis.ai_analyzer import AnalysisContext
                context = AnalysisContext(
                    symbol=symbol,
                    asset_class="crypto",
                    strategy_name=self.name,
                    proposed_action="BUY",
                    signal_summary=s,
                    news_headlines=crypto_news.get(symbol, [])[:3],
                    available_capital=self._portfolio.get_available_capital(),
                    open_position_count=len(self._portfolio.positions),
                    daily_pnl_pct=self._portfolio.daily_pnl,
                )
                decision = await self._ai.analyze(context)

                if decision.recommendation == "BUY":
                    stop = decision.stop_price or round(price * (1 - STOP_PCT), 6)
                    target = decision.take_profit or round(price * (1 + TARGET_PCT), 6)
                    cash = self._portfolio.get_available_capital()
                    max_dollars = min(cash * 0.15, 20.0)
                    qty = round(max_dollars / price, 6) if price > 0 else 0
                    if qty <= 0:
                        continue
                    signals.append(TradeSignal(
                        symbol=symbol,
                        side="buy",
                        asset_class="crypto",
                        strategy_name=self.name,
                        entry_price=price,
                        stop_price=stop,
                        take_profit=target,
                        confidence=decision.confidence,
                        reasoning=decision.reasoning,
                        qty=qty,
                    ))
            except Exception as e:
                logger.error("[CryptoMomentum] Error on {}: {}", symbol, e)

        return signals

    async def should_exit(self, position: object) -> bool:
        return False

    async def run(self) -> None:
        await self.run_loop(LOOP_INTERVAL)
