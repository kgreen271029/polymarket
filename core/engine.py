from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from core.portfolio import PortfolioTracker
    from core.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Shared event types (imported by brokers, strategies, and data feeds)
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    symbol: str
    side: str           # "buy" or "sell"
    qty: float
    entry_price: float
    asset_class: str    # "crypto" | "stock" | "polymarket"
    strategy_name: str
    confidence: str     # "low" | "medium" | "high"
    stop_price: float | None = None
    take_profit: float | None = None
    order_id: str | None = None
    timestamp: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(tz=timezone.utc)


@dataclass
class FillEvent:
    order_id: str
    symbol: str
    side: str           # "buy" or "sell"
    qty: float
    fill_price: float
    asset_class: str
    timestamp: datetime


@dataclass
class HeadlineEvent:
    headline: str
    source: str
    sentiment: float    # -1.0 to +1.0
    symbols: list[str]
    timestamp: datetime
    urgency: str = "normal"  # "normal" | "breaking"


@dataclass
class SocialSignal:
    platform: str
    content: str
    sentiment: float
    symbols: list[str]
    engagement_score: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TradingEngine:
    def __init__(
        self,
        portfolio: "PortfolioTracker",
        risk_manager: "RiskManager",
        alpaca_broker: Any,
        robinhood_broker: Any,
        polymarket_broker: Any,
        config: Any,
        notifier: Any = None,
    ) -> None:
        self._portfolio = portfolio
        self._risk_manager = risk_manager
        self._alpaca_broker = alpaca_broker
        self._robinhood_broker = robinhood_broker
        self._polymarket_broker = polymarket_broker
        self._config = config
        self._notifier = notifier

        self.signal_bus: asyncio.Queue[TradeSignal] = asyncio.Queue()
        self.news_queue: asyncio.Queue[HeadlineEvent] = asyncio.Queue()
        self.social_queue: asyncio.Queue[SocialSignal] = asyncio.Queue()
        self.breaking_queue: asyncio.Queue[HeadlineEvent] = asyncio.Queue()
        self.fill_queue: asyncio.Queue[FillEvent] = asyncio.Queue()

        self.market_open: bool = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._shutdown_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, *extra_tasks: "asyncio.coroutines._CoroutineType[Any, Any, Any]") -> None:
        """Start the engine and all passed-in strategy/feed coroutines."""
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self._shutdown_handler)
        loop.add_signal_handler(signal.SIGTERM, self._shutdown_handler)

        core_coros = [
            self._process_signals(),
            self._process_fills(),
            self._market_open_gate(),
            self._daily_reset(),
        ]
        all_coros = list(core_coros) + list(extra_tasks)

        logger.info("TradingEngine starting — spawning {} coroutines", len(all_coros))
        self._tasks = [asyncio.create_task(c, name=getattr(c, "__name__", repr(c))) for c in all_coros]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("TradingEngine main gather cancelled — shutdown complete")

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    async def _process_signals(self) -> None:
        logger.info("Signal processor started")
        while not self._shutdown_event.is_set():
            try:
                signal_obj: TradeSignal = await asyncio.wait_for(
                    self.signal_bus.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if signal_obj.asset_class == "stock" and not self.market_open:
                logger.warning(
                    "[ENGINE] Dropping stock signal for {} — market closed", signal_obj.symbol
                )
                self.signal_bus.task_done()
                continue

            verdict = self._risk_manager.check(signal_obj)
            if not verdict.approved:
                logger.warning(
                    "[ENGINE] Signal REJECTED: {} {} {} — {}",
                    signal_obj.side,
                    signal_obj.symbol,
                    signal_obj.asset_class,
                    verdict.reason,
                )
                self.signal_bus.task_done()
                continue

            # Apply any qty scaling the risk manager decided on
            if verdict.adjusted_qty is not None:
                signal_obj.qty = verdict.adjusted_qty

            await self._route_signal(signal_obj)
            self.signal_bus.task_done()

    async def _route_signal(self, signal_obj: TradeSignal) -> None:
        broker_name = {
            "crypto": "alpaca",
            "stock": "robinhood",
            "polymarket": "polymarket",
        }.get(signal_obj.asset_class, "unknown")

        logger.info(
            "[ENGINE] Routing {} {} {} qty={:.4f} → {} broker",
            signal_obj.side,
            signal_obj.symbol,
            signal_obj.asset_class,
            signal_obj.qty,
            broker_name,
        )

        try:
            if signal_obj.asset_class == "crypto" and self._alpaca_broker:
                await asyncio.to_thread(self._alpaca_broker.submit_order, signal_obj)
            elif signal_obj.asset_class == "stock" and self._robinhood_broker:
                await self._robinhood_broker.place_order(
                    symbol=signal_obj.symbol,
                    side=signal_obj.side,
                    qty=signal_obj.qty,
                    order_type="market",
                )
            elif signal_obj.asset_class == "polymarket" and self._polymarket_broker:
                await asyncio.to_thread(self._polymarket_broker.submit_order, signal_obj)
            else:
                logger.error("[ENGINE] No broker available for asset_class '{}' — signal dropped", signal_obj.asset_class)
                return

            # Send trade alert notification
            if self._notifier:
                asyncio.create_task(self._notifier.trade_alert(
                    symbol=signal_obj.symbol,
                    side=signal_obj.side,
                    qty=signal_obj.qty,
                    price=signal_obj.entry_price,
                    strategy=signal_obj.strategy_name,
                    reasoning=signal_obj.reasoning or "",
                    stop=signal_obj.stop_price,
                    target=signal_obj.take_profit,
                ))
        except Exception as exc:
            logger.exception(
                "[ENGINE] Broker error routing {} {}: {}", signal_obj.symbol, signal_obj.asset_class, exc
            )

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    async def _process_fills(self) -> None:
        logger.info("Fill processor started")
        while not self._shutdown_event.is_set():
            try:
                fill: FillEvent = await asyncio.wait_for(self.fill_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if fill.side == "buy":
                # Stop/TP come from the originating signal; broker should embed them in the fill
                # if available, otherwise None is acceptable — risk guard already validated stop
                self._portfolio.record_fill(
                    symbol=fill.symbol,
                    side="long",
                    qty=fill.qty,
                    fill_price=fill.fill_price,
                    order_id=fill.order_id,
                    asset_class=fill.asset_class,
                    strategy_name="unknown",
                    stop_loss=None,
                    take_profit=None,
                )
            else:
                self._portfolio.record_close(fill.symbol, fill.fill_price, fill.qty)

            self.fill_queue.task_done()

    # ------------------------------------------------------------------
    # Market hours gate
    # ------------------------------------------------------------------

    async def _market_open_gate(self) -> None:
        """Polls Alpaca's clock endpoint every 60 s and updates self.market_open."""
        logger.info("Market gate started")
        while not self._shutdown_event.is_set():
            try:
                clock = await asyncio.to_thread(self._alpaca_broker.get_clock)
                self.market_open = bool(clock.is_open)
                logger.debug("Market status: {}", "OPEN" if self.market_open else "CLOSED")
            except Exception as exc:
                logger.warning("[ENGINE] Could not fetch market clock: {}", exc)
                # Fail closed for stock signals on API error
                self.market_open = False

            await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    async def _daily_reset(self) -> None:
        """Fires at the next midnight UTC, then every 24 h."""
        logger.info("Daily reset task started")
        while not self._shutdown_event.is_set():
            now = datetime.now(tz=timezone.utc)
            next_midnight = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            seconds_until_midnight = (next_midnight - now).total_seconds()

            try:
                await asyncio.sleep(seconds_until_midnight)
            except asyncio.CancelledError:
                break

            prev_daily = self._portfolio.daily_pnl
            self._portfolio.daily_pnl = 0.0
            # day_trades rolling window is maintained automatically by the deque logic
            # in PortfolioTracker — no explicit roll needed here.
            logger.info(
                "[ENGINE] Daily reset fired — yesterday daily_pnl={:+.2f}, total_pnl={:+.2f}",
                prev_daily,
                self._portfolio.total_pnl,
            )
            logger.remove()        # drop current log sink
            logger.add(            # re-add with a date-stamped file
                f"logs/trading_{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}.log",
                rotation="1 day",
                retention="30 days",
                level="DEBUG",
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown_handler(self) -> None:
        logger.warning("[ENGINE] Shutdown signal received — cancelling all tasks")
        self._shutdown_event.set()

        total_pnl = self._portfolio.total_pnl
        daily_pnl = self._portfolio.daily_pnl
        open_count = len(self._portfolio.positions)
        logger.info(
            "[ENGINE] Final P&L summary: total={:+.2f}, today={:+.2f}, open_positions={}",
            total_pnl,
            daily_pnl,
            open_count,
        )

        for task in self._tasks:
            task.cancel()
