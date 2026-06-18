"""strategies/base_strategy.py — Abstract base class and shared TradeSignal type."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from loguru import logger


@dataclass
class TradeSignal:
    symbol: str
    side: str                  # "buy" | "sell"
    asset_class: str           # "crypto" | "stock" | "polymarket"
    strategy_name: str
    entry_price: float
    stop_price: float | None
    take_profit: float | None
    confidence: str            # "High" | "Medium" | "Low"
    reasoning: str
    qty: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    def __init__(
        self,
        signal_bus: asyncio.Queue,
        market_open_fn: Callable[[], bool],
    ) -> None:
        self._signal_bus = signal_bus
        self._market_open = market_open_fn
        self._running = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def generate_signals(self) -> list[TradeSignal]:
        """Analyse market conditions and return any trade signals."""

    @abstractmethod
    async def should_exit(self, position: object) -> bool:
        """Return True if the given position should be closed now."""

    async def run_loop(self, interval_seconds: int) -> None:
        self._running = True
        logger.info("[{}] Started (interval={}s)", self.name, interval_seconds)
        while self._running:
            try:
                signals = await self.generate_signals()
                for sig in signals:
                    await self._signal_bus.put(sig)
                    logger.info(
                        "[{}] → {} {} {} @ {:.4f} | conf={}",
                        self.name, sig.side.upper(), sig.symbol,
                        sig.asset_class, sig.entry_price, sig.confidence,
                    )
            except Exception as exc:
                logger.error("[{}] Error in run_loop: {}", self.name, exc)
            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        self._running = False
