"""strategies/polymarket_strategy.py — Prediction market mispricing detector."""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal

LOOP_INTERVAL = 600
MIN_LIQUIDITY = 50_000.0
EDGE_THRESHOLD = 0.08
MAX_POSITION = 10.0


class PolymarketStrategy(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        polymarket_broker,
        ai_analyzer,
        market_open_fn: Callable[[], bool],
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._broker = polymarket_broker
        self._ai = ai_analyzer

    async def generate_signals(self) -> list[TradeSignal]:
        signals: list[TradeSignal] = []

        try:
            markets = await self._broker.get_open_markets(limit=50)
        except Exception as e:
            logger.error("[Polymarket] get_open_markets failed: {}", e)
            return signals

        for market in markets:
            try:
                liquidity = float(market.get("liquidity", 0) or 0)
                if liquidity < MIN_LIQUIDITY:
                    continue

                yes_price = float(market.get("bestBid", market.get("yes_price", 0.5)) or 0.5)
                condition_id = market.get("conditionId") or market.get("condition_id", "")
                token_id = market.get("tokens", [{}])[0].get("token_id", "") if market.get("tokens") else ""
                question = market.get("question", market.get("title", "Unknown market"))

                if not condition_id or not token_id:
                    continue

                from analysis.ai_analyzer import AnalysisContext
                context = AnalysisContext(
                    symbol=condition_id[:12],
                    asset_class="polymarket",
                    strategy_name=self.name,
                    proposed_action="BUY",
                    signal_summary={
                        "implied_probability": yes_price,
                        "question": question,
                        "liquidity": liquidity,
                        "latest_close": yes_price,
                    },
                    news_headlines=[question],
                    available_capital=MAX_POSITION,
                    open_position_count=0,
                    daily_pnl_pct=0.0,
                )
                decision = await self._ai.analyze(context)

                if decision.recommendation not in ("BUY", "SELL"):
                    continue

                ai_prob = decision.stop_price if decision.stop_price else None
                if ai_prob is None:
                    continue

                edge = abs(ai_prob - yes_price)
                if edge < EDGE_THRESHOLD:
                    continue

                trade_yes = ai_prob > yes_price
                # qty = number of contracts = $10 / price
                qty = round(MAX_POSITION / yes_price, 4) if yes_price > 0 else 0
                if qty <= 0:
                    continue
                signals.append(TradeSignal(
                    symbol=condition_id[:16],
                    side="buy",
                    asset_class="polymarket",
                    strategy_name=self.name,
                    entry_price=yes_price,
                    stop_price=None,  # Polymarket: max loss = position size (no stop needed)
                    take_profit=1.0,
                    confidence=decision.confidence,
                    reasoning=f"Edge={edge:.2%} on '{question[:60]}' | {decision.reasoning}",
                    qty=qty,
                    metadata={
                        "token_id": token_id,
                        "condition_id": condition_id,
                        "question": question,
                        "trade_yes": trade_yes,
                        "size_usdc": MAX_POSITION,
                    },
                ))
            except Exception as e:
                logger.error("[Polymarket] Error on market {}: {}", market.get("conditionId", "?"), e)

        return signals

    async def should_exit(self, position: object) -> bool:
        # Polymarket positions resolve automatically
        return False

    async def run(self) -> None:
        await self.run_loop(LOOP_INTERVAL)
