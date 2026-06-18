"""strategies/new_coin_hunter.py — Early entry on trending new crypto coins."""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from strategies.base_strategy import BaseStrategy, TradeSignal

LOOP_INTERVAL = 120
MAX_COIN_POSITION = 10.0
TRAILING_STOP_PCT = 0.08
MIN_AI_SCORE = 7
MAX_CONCURRENT_NEW_COINS = 2

# Alpaca-tradeable symbols (new coins must be mappable to these)
_ALPACA_CRYPTO = {
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "MATIC", "LINK", "UNI",
    "AAVE", "LTC", "BCH", "XLM", "XRP", "DOT", "ATOM", "ALGO",
}


class NewCoinHunter(BaseStrategy):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        new_coins_scanner,
        social_feed,
        ai_analyzer,
        portfolio_tracker,
        market_open_fn: Callable[[], bool],
        market_data=None,
    ) -> None:
        super().__init__(signal_bus, market_open_fn)
        self._scanner = new_coins_scanner
        self._social = social_feed
        self._ai = ai_analyzer
        self._portfolio = portfolio_tracker
        self._market_data = market_data
        self._max_price_seen: dict[str, float] = {}

    async def generate_signals(self) -> list[TradeSignal]:
        signals: list[TradeSignal] = []

        new_coin_positions = sum(
            1 for p in self._portfolio.positions.values()
            if p.metadata.get("is_new_coin")  # type: ignore[attr-defined]
        )
        if new_coin_positions >= MAX_CONCURRENT_NEW_COINS:
            return signals

        try:
            events = await self._scanner.get_new_coin_events()
        except Exception as e:
            logger.error("[NewCoinHunter] Failed to get events: {}", e)
            return signals

        for event in events:
            if event.days_old > 30:
                continue
            if event.social_velocity < 5.0:
                continue
            if event.trending_rank is None or event.trending_rank > 15:
                continue

            sym = event.symbol.upper()
            tradeable_sym = sym if sym in _ALPACA_CRYPTO else None
            if not tradeable_sym:
                continue

            alpaca_symbol = f"{tradeable_sym}/USD"
            if alpaca_symbol in {p.split(":")[0] for p in self._portfolio.positions}:
                continue

            try:
                decision = await self._ai.analyze_new_coin(
                    coin_data={
                        "id": event.coin_id,
                        "symbol": sym,
                        "name": event.name,
                        "days_old": event.days_old,
                        "trending_rank": event.trending_rank,
                        "coingecko_url": event.coingecko_url,
                    },
                    social_data={
                        "social_velocity": event.social_velocity,
                    },
                )

                score = decision.get("score", 0)
                if score < MIN_AI_SCORE:
                    logger.info("[NewCoinHunter] {} scored {}/10 — skipping", sym, score)
                    continue

                # Fetch current price — required by risk manager
                price = 0.0
                if self._market_data is not None:
                    try:
                        df = await self._market_data.get_bars(alpaca_symbol, "1Min", limit=5)
                        if df is not None and len(df) > 0:
                            price = float(df["close"].iloc[-1])
                    except Exception:
                        pass
                if price <= 0:
                    logger.debug("[NewCoinHunter] No price for {} — skipping", alpaca_symbol)
                    continue

                stop = round(price * (1 - TRAILING_STOP_PCT), 6)
                qty = round(MAX_COIN_POSITION / price, 6)
                if qty <= 0:
                    continue

                signals.append(TradeSignal(
                    symbol=alpaca_symbol,
                    side="buy",
                    asset_class="crypto",
                    strategy_name=self.name,
                    entry_price=price,
                    stop_price=stop,
                    take_profit=None,
                    confidence="Medium" if score >= 8 else "Low",
                    reasoning=decision.get("reasoning", f"New coin score {score}/10"),
                    qty=qty,
                    metadata={
                        "is_new_coin": True,
                        "ai_score": score,
                        "trending_rank": event.trending_rank,
                        "social_velocity": event.social_velocity,
                        "max_position_usd": MAX_COIN_POSITION,
                    },
                ))
            except Exception as e:
                logger.error("[NewCoinHunter] AI analysis failed for {}: {}", sym, e)

        return signals

    async def should_exit(self, position: object) -> bool:
        sym = position.symbol  # type: ignore[attr-defined]
        current = position.current_price  # type: ignore[attr-defined]
        prev_max = self._max_price_seen.get(sym, current)
        self._max_price_seen[sym] = max(prev_max, current)
        # Trailing stop: exit if price fell 8% from all-time high since entry
        if current < self._max_price_seen[sym] * (1 - TRAILING_STOP_PCT):
            return True
        return False

    async def run(self) -> None:
        await self.run_loop(LOOP_INTERVAL)
