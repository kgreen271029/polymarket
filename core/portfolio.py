from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from loguru import logger
from rich.table import Table
from rich import box

if TYPE_CHECKING:
    pass


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    current_price: float
    stop_loss: float | None
    take_profit: float | None
    asset_class: str  # "crypto", "stock", "polymarket"
    strategy_name: str
    opened_at: datetime
    order_id: str

    @property
    def unrealized_pnl(self) -> float:
        multiplier = 1.0 if self.side == "long" else -1.0
        return multiplier * (self.current_price - self.entry_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        multiplier = 1.0 if self.side == "long" else -1.0
        return multiplier * (self.current_price - self.entry_price) / self.entry_price * 100

    @property
    def notional_value(self) -> float:
        return self.qty * self.current_price


@dataclass
class _DayTradeRecord:
    symbol: str
    opened_at: datetime
    closed_at: datetime


class PortfolioTracker:
    def __init__(self, starting_capital: float) -> None:
        self.cash: float = starting_capital
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.positions: dict[str, Position] = {}
        # Each entry is (symbol, open_date) for same-day round trips
        self._day_trade_records: list[_DayTradeRecord] = []
        # Tracks when each symbol was opened today so we can detect PDT violations
        self._intraday_opens: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def record_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        order_id: str,
        asset_class: str,
        strategy_name: str,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> None:
        key = f"{symbol}:{side}"
        if key in self.positions:
            # Average into existing position rather than overwrite
            existing = self.positions[key]
            total_qty = existing.qty + qty
            avg_price = (existing.qty * existing.entry_price + qty * fill_price) / total_qty
            existing.qty = total_qty
            existing.entry_price = avg_price
            existing.current_price = fill_price
            logger.info(f"Averaged into {key}: total_qty={total_qty:.4f} avg={avg_price:.4f}")
        else:
            now = datetime.now(tz=timezone.utc)
            self.positions[key] = Position(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=fill_price,
                current_price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                asset_class=asset_class,
                strategy_name=strategy_name,
                opened_at=now,
                order_id=order_id,
            )
            self._intraday_opens[symbol] = now
            cost = qty * fill_price
            self.cash -= cost
            logger.info(f"Opened position {key}: qty={qty} @ {fill_price:.4f}, cash={self.cash:.2f}")

    def record_close(self, symbol: str, exit_price: float, qty: float) -> float:
        """Close (or partially close) a position. Returns realised PnL."""
        key_long = f"{symbol}:long"
        key_short = f"{symbol}:short"
        key = key_long if key_long in self.positions else key_short

        if key not in self.positions:
            logger.warning(f"record_close: no open position for {symbol}")
            return 0.0

        pos = self.positions[key]
        close_qty = min(qty, pos.qty)
        multiplier = 1.0 if pos.side == "long" else -1.0
        pnl = multiplier * (exit_price - pos.entry_price) * close_qty

        self.cash += close_qty * exit_price
        self.daily_pnl += pnl
        self.total_pnl += pnl

        if self.is_day_trade(symbol):
            now = datetime.now(tz=timezone.utc)
            opened_at = self._intraday_opens.get(symbol, pos.opened_at)
            self._day_trade_records.append(
                _DayTradeRecord(symbol=symbol, opened_at=opened_at, closed_at=now)
            )
            logger.debug(f"Day trade recorded for {symbol}")

        if close_qty >= pos.qty:
            del self.positions[key]
            self._intraday_opens.pop(symbol, None)
        else:
            pos.qty -= close_qty

        logger.info(f"Closed {close_qty} of {key} @ {exit_price:.4f}, pnl={pnl:+.2f}")
        return pnl

    def update_price(self, symbol: str, price: float) -> None:
        for key in (f"{symbol}:long", f"{symbol}:short"):
            if key in self.positions:
                self.positions[key].current_price = price

    # ------------------------------------------------------------------
    # PDT helpers
    # ------------------------------------------------------------------

    def is_day_trade(self, symbol: str) -> bool:
        """True if we have an intraday open for this symbol (same calendar day UTC)."""
        opened_at = self._intraday_opens.get(symbol)
        if opened_at is None:
            return False
        today = datetime.now(tz=timezone.utc).date()
        return opened_at.date() == today

    def get_day_trade_count(self) -> int:
        """Number of day trades in the rolling 5-calendar-day window."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=5)
        return sum(1 for r in self._day_trade_records if r.closed_at >= cutoff)

    # ------------------------------------------------------------------
    # Capital
    # ------------------------------------------------------------------

    def get_available_capital(self) -> float:
        """Cash minus a conservative stop-order margin reserve for open positions."""
        reserved = 0.0
        for pos in self.positions.values():
            if pos.stop_loss is not None:
                # Reserve the worst-case exit value at the stop price
                reserved += pos.qty * pos.stop_loss
        # Never let reserved exceed actual cash
        return max(0.0, self.cash - reserved)

    def get_open_position(self, symbol: str) -> Position | None:
        return self.positions.get(f"{symbol}:long") or self.positions.get(f"{symbol}:short")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def to_rich_table(self) -> Table:
        table = Table(
            title="Open Positions",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            highlight=True,
        )
        table.add_column("Symbol", style="bold cyan")
        table.add_column("Side", style="white")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L ($)", justify="right")
        table.add_column("P&L (%)", justify="right")
        table.add_column("Stop", justify="right")
        table.add_column("Strategy", style="dim")

        for pos in self.positions.values():
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            stop_str = f"{pos.stop_loss:.4f}" if pos.stop_loss else "—"
            table.add_row(
                pos.symbol,
                pos.side,
                f"{pos.qty:.4f}",
                f"{pos.entry_price:.4f}",
                f"{pos.current_price:.4f}",
                f"[{pnl_color}]{pos.unrealized_pnl:+.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pos.unrealized_pnl_pct:+.2f}%[/{pnl_color}]",
                stop_str,
                pos.strategy_name,
            )

        return table
