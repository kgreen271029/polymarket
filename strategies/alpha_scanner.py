"""strategies/alpha_scanner.py — Pre-breakout signal scanner across multiple data sources."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal
from analysis.technical import TechnicalAnalyzer
from analysis.sentiment import SentimentScorer
from data.market_data import HeadlineEvent

LOOP_INTERVAL = 180

CRYPTO_WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]
STOCK_WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "COIN", "MSTR"]


class AlphaScanner(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        market_data,
        social_feed,
        news_queue: asyncio.Queue,
        ai_analyzer,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._market_data = market_data
        self._social = social_feed
        self._news_queue = news_queue
        self._ai = ai_analyzer
        self._portfolio = portfolio_tracker
        self._scorer = SentimentScorer()
        self._news_mention_count: dict[str, int] = defaultdict(int)

    async def generate_signals(self) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        candidates: dict[str, int] = defaultdict(int)  # symbol → alpha score

        # --- Source 1: social velocity ---
        try:
            social_signals = self._social.get_recent_signals() if hasattr(self._social, "get_recent_signals") else []
            for sig in social_signals:
                if sig.velocity_ratio > 3.0:
                    mapped = sig.symbol
                    candidates[mapped] += int(sig.velocity_ratio)
        except Exception as e:
            logger.debug("[AlphaScanner] Social source error: {}", e)

        # --- Source 2: news mention velocity ---
        new_mentions: dict[str, int] = defaultdict(int)
        while True:
            try:
                event: HeadlineEvent = self._news_queue.get_nowait()
                for sym in event.symbols_mentioned:
                    new_mentions[sym] += 1
            except asyncio.QueueEmpty:
                break
        for sym, count in new_mentions.items():
            self._news_mention_count[sym] = self._news_mention_count.get(sym, 0) + count
            if self._news_mention_count[sym] >= 3:
                candidates[sym] += self._news_mention_count[sym] * 2
                self._news_mention_count[sym] = 0

        # --- Source 3: crypto range expansion ---
        for symbol in CRYPTO_WATCHLIST:
            try:
                df = await self._market_data.get_bars(symbol, "5Min", limit=30)
                if df is None or len(df) < 20:
                    continue
                s = TechnicalAnalyzer.build_signal_summary(df)
                vol_ratio = s.get("volume_ratio", 1.0)
                rsi = s.get("rsi", 50)
                if vol_ratio > 3.0 and 45 <= rsi <= 65:
                    candidates[symbol] += int(vol_ratio * 2)
            except Exception:
                pass

        # --- Source 4: stock breakout screen (market hours only) ---
        if self._market_open():
            for symbol in STOCK_WATCHLIST:
                try:
                    df = await self._market_data.get_bars(symbol, "1Day", limit=25)
                    if df is None or len(df) < 22:
                        continue
                    s = TechnicalAnalyzer.build_signal_summary(df)
                    if s.get("breakout_20") and s.get("volume_ratio", 1.0) > 1.4:
                        candidates[symbol] += 8
                except Exception:
                    pass

        # Send top 3 candidates to AI
        open_symbols = {p.split(":")[0] for p in self._portfolio.positions}
        top = sorted(
            [(sym, score) for sym, score in candidates.items() if sym not in open_symbols],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        for symbol, score in top:
            if score < 6:
                continue
            try:
                is_crypto = "/" in symbol or (symbol.upper().endswith("USD") and len(symbol) > 4)
                tf = "5Min" if is_crypto else "1Day"
                df = await self._market_data.get_bars(symbol, tf, limit=40)
                if df is None or len(df) < 20:
                    continue
                sig_summary = TechnicalAnalyzer.build_signal_summary(df)
                price = sig_summary.get("latest_close", 0)
                if not price:
                    continue

                from analysis.ai_analyzer import AnalysisContext
                context = AnalysisContext(
                    symbol=symbol,
                    asset_class="crypto" if is_crypto else "stock",
                    strategy_name=self.name,
                    proposed_action="BUY",
                    signal_summary=sig_summary,
                    news_headlines=[],
                    available_capital=self._portfolio.get_available_capital(),
                    open_position_count=len(self._portfolio.positions),
                    daily_pnl_pct=self._portfolio.daily_pnl,
                )
                decision = await self._ai.analyze(context)

                if decision.recommendation == "BUY" and decision.confidence in ("Medium", "High"):
                    stop = decision.stop_price or round(price * 0.96, 4)
                    target = decision.take_profit or round(price * 1.08, 4)
                    signals.append(TradeSignal(
                        symbol=symbol,
                        side="buy",
                        asset_class="crypto" if is_crypto else "stock",
                        strategy_name=self.name,
                        entry_price=price,
                        stop_price=stop,
                        take_profit=target,
                        confidence=decision.confidence,
                        reasoning=f"Alpha score={score} | {decision.reasoning}",
                    ))
            except Exception as e:
                logger.error("[AlphaScanner] AI analysis failed for {}: {}", symbol, e)

        return signals

    async def should_exit(self, position: object) -> bool:
        return False

    async def run(self) -> None:
        await self.run_loop(LOOP_INTERVAL)
