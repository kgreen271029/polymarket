from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from core.portfolio import PortfolioTracker

if TYPE_CHECKING:
    from core.engine import TradeSignal


_CONFIDENCE_SCALAR: dict[str, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5,
}

# Minimum lot sizes per asset class to avoid sub-penny / dust trades
_MIN_LOT: dict[str, float] = {
    "crypto": 0.0001,   # BTC-denominated; crypto qty is in dollars so this is $0.0001
    "stock": 1.0,       # 1 share minimum
    "polymarket": 1.0,  # 1 contract minimum
}

# Maximum allowable stop distance as a fraction of entry price
_MAX_STOP_DISTANCE = 0.10


@dataclass
class RiskVerdict:
    approved: bool
    reason: str
    adjusted_qty: float | None = None


class RiskManager:
    def __init__(self, portfolio: PortfolioTracker, config: object) -> None:
        self._portfolio = portfolio
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, signal: "TradeSignal") -> RiskVerdict:
        """Run all five risk guards in sequence. Returns first failure or approval."""

        verdict = self._guard_daily_loss(signal)
        if not verdict.approved:
            return verdict

        verdict = self._guard_max_positions(signal)
        if not verdict.approved:
            return verdict

        verdict = self._guard_position_size(signal)
        if not verdict.approved:
            return verdict

        verdict = self._guard_pdt(signal)
        if not verdict.approved:
            return verdict

        verdict = self._guard_stop_loss(signal)
        if not verdict.approved:
            return verdict

        return RiskVerdict(approved=True, reason="all checks passed", adjusted_qty=signal.qty)

    def calculate_position_size(self, signal: "TradeSignal", confidence: str) -> float:
        scalar = _CONFIDENCE_SCALAR.get(confidence.lower(), 1.0)
        available = self._portfolio.get_available_capital()
        base_size = min(self._config.MAX_RISK_PER_TRADE, available * 0.20)
        risk_amount = base_size * scalar

        if signal.asset_class == "crypto":
            # For crypto the qty IS the dollar amount — broker converts to coins
            qty = risk_amount
        else:
            if signal.entry_price <= 0:
                logger.warning(f"calculate_position_size: entry_price={signal.entry_price} <= 0")
                return 0.0
            qty = risk_amount / signal.entry_price

        min_lot = _MIN_LOT.get(signal.asset_class, 1.0)
        return max(qty, min_lot)

    def get_portfolio_value(self) -> float:
        position_value = sum(
            pos.qty * pos.current_price for pos in self._portfolio.positions.values()
        )
        return self._portfolio.cash + position_value

    # ------------------------------------------------------------------
    # Individual guards
    # ------------------------------------------------------------------

    def _guard_daily_loss(self, signal: "TradeSignal") -> RiskVerdict:
        portfolio_value = self.get_portfolio_value()
        limit = portfolio_value * self._config.DAILY_LOSS_LIMIT_PCT
        if self._portfolio.daily_pnl <= -limit:
            reason = (
                f"daily loss limit hit: daily_pnl={self._portfolio.daily_pnl:.2f} "
                f">= -{limit:.2f} ({self._config.DAILY_LOSS_LIMIT_PCT*100:.1f}% of {portfolio_value:.2f})"
            )
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)
        return RiskVerdict(approved=True, reason="daily loss OK")

    def _guard_max_positions(self, signal: "TradeSignal") -> RiskVerdict:
        n = len(self._portfolio.positions)
        if n >= self._config.MAX_OPEN_POSITIONS:
            reason = f"max open positions reached: {n}/{self._config.MAX_OPEN_POSITIONS}"
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)
        return RiskVerdict(approved=True, reason="position count OK")

    def _guard_position_size(self, signal: "TradeSignal") -> RiskVerdict:
        if signal.entry_price <= 0:
            reason = f"invalid entry_price={signal.entry_price}"
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)

        position_value = signal.qty * signal.entry_price
        max_value = self._config.MAX_RISK_PER_TRADE

        if position_value <= max_value:
            return RiskVerdict(approved=True, reason="position size OK", adjusted_qty=signal.qty)

        # Scale down rather than outright reject when possible
        scaled_qty = max_value / signal.entry_price
        min_lot = _MIN_LOT.get(signal.asset_class, 1.0)
        if scaled_qty < min_lot:
            reason = (
                f"position value {position_value:.2f} > max {max_value:.2f} "
                f"and scaled qty {scaled_qty:.4f} < min lot {min_lot}"
            )
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)

        reason = f"position size scaled from {signal.qty:.4f} to {scaled_qty:.4f}"
        logger.info(f"[RISK SCALE] {signal.symbol} — {reason}")
        return RiskVerdict(approved=True, reason=reason, adjusted_qty=scaled_qty)

    def _guard_pdt(self, signal: "TradeSignal") -> RiskVerdict:
        if signal.asset_class != "stock":
            return RiskVerdict(approved=True, reason="PDT N/A for non-stock")

        day_trade_count = self._portfolio.get_day_trade_count()
        would_be_day_trade = self._portfolio.is_day_trade(signal.symbol)

        if day_trade_count >= 3 and would_be_day_trade:
            reason = (
                f"PDT limit: {day_trade_count} day trades in rolling 5-day window; "
                f"{signal.symbol} was opened today and closing would be day trade #4+"
            )
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)

        return RiskVerdict(approved=True, reason="PDT OK")

    def _guard_stop_loss(self, signal: "TradeSignal") -> RiskVerdict:
        if signal.stop_price is None:
            reason = "stop_price is required but was None"
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)

        if signal.entry_price <= 0:
            return RiskVerdict(approved=True, reason="stop check skipped — invalid entry")

        stop_distance = abs(signal.entry_price - signal.stop_price) / signal.entry_price
        if stop_distance > _MAX_STOP_DISTANCE:
            reason = (
                f"stop distance {stop_distance*100:.1f}% > max {_MAX_STOP_DISTANCE*100:.0f}% "
                f"(entry={signal.entry_price:.4f}, stop={signal.stop_price:.4f})"
            )
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)

        return RiskVerdict(approved=True, reason="stop loss OK")
