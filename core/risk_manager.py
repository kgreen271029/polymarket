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

# Minimum lot sizes per asset class to avoid sub-penny / dust trades.
# Robinhood supports fractional shares down to ~$1, so stocks allow fractions.
_MIN_LOT: dict[str, float] = {
    "crypto": 0.0001,   # BTC-denominated; crypto qty is in dollars so this is $0.0001
    "stock": 0.0,       # fractional shares allowed (min enforced by dollar value)
    "polymarket": 1.0,  # 1 contract minimum
}

# Minimum position dollar value (avoid dust trades / broker rejection)
_MIN_POSITION_USD = 1.0

# Maximum allowable stop distance as a fraction of entry price
# 2.2×ATR for high-vol stocks (NVDA, MSTR) can exceed 10% — allow up to 15%
_MAX_STOP_DISTANCE = 0.15


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
        """Run all risk guards in sequence. Returns first failure or approval."""

        # SELL/exit orders reduce risk — never block them on entry guards.
        # (Blocking exits would trap the bot in losing positions.)
        if signal.side.lower() == "sell":
            return RiskVerdict(approved=True, reason="exit order — guards bypassed",
                               adjusted_qty=signal.qty)

        verdict = self._guard_daily_loss(signal)
        if not verdict.approved:
            return verdict

        verdict = self._guard_max_positions(signal)
        if not verdict.approved:
            return verdict

        size_verdict = self._guard_position_size(signal)
        if not size_verdict.approved:
            return size_verdict
        # Preserve any qty scaling from the size guard through the rest of the checks
        final_qty = size_verdict.adjusted_qty if size_verdict.adjusted_qty is not None else signal.qty

        verdict = self._guard_pdt(signal)
        if not verdict.approved:
            return verdict

        verdict = self._guard_stop_loss(signal)
        if not verdict.approved:
            return verdict

        return RiskVerdict(approved=True, reason="all checks passed", adjusted_qty=final_qty)

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
        # Cap at MAX_RISK_PER_TRADE OR available cash, whichever is smaller
        available = self._portfolio.get_available_capital()
        max_value = min(self._config.MAX_RISK_PER_TRADE, available)

        # Reject dust trades
        if position_value < _MIN_POSITION_USD and max_value < _MIN_POSITION_USD:
            reason = f"insufficient capital: only ${available:.2f} available"
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)

        if position_value <= max_value:
            if position_value < _MIN_POSITION_USD:
                reason = f"position value ${position_value:.2f} below ${_MIN_POSITION_USD} minimum"
                logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
                return RiskVerdict(approved=False, reason=reason)
            return RiskVerdict(approved=True, reason="position size OK", adjusted_qty=signal.qty)

        # Scale down to fit the cap (fractional shares OK for stocks)
        scaled_qty = max_value / signal.entry_price
        scaled_value = scaled_qty * signal.entry_price
        if scaled_value < _MIN_POSITION_USD:
            reason = f"scaled position ${scaled_value:.2f} below ${_MIN_POSITION_USD} minimum"
            logger.warning(f"[RISK REJECT] {signal.symbol} — {reason}")
            return RiskVerdict(approved=False, reason=reason)

        reason = f"position size scaled from {signal.qty:.6f} to {scaled_qty:.6f} (${scaled_value:.2f})"
        logger.info(f"[RISK SCALE] {signal.symbol} — {reason}")
        return RiskVerdict(approved=True, reason=reason, adjusted_qty=round(scaled_qty, 6))

    def _guard_pdt(self, signal: "TradeSignal") -> RiskVerdict:
        # PDT rule eliminated by FINRA on June 4, 2026 — no longer enforced
        # Keeping the guard as a no-op to log activity for audit purposes only
        if signal.asset_class != "stock":
            return RiskVerdict(approved=True, reason="PDT N/A for non-stock")

        day_trade_count = self._portfolio.get_day_trade_count()
        would_be_day_trade = self._portfolio.is_day_trade(signal.symbol)
        if would_be_day_trade:
            logger.info(f"[RISK] {signal.symbol} day-trade #{day_trade_count+1} (PDT rule removed Jun 2026)")

        return RiskVerdict(approved=True, reason="PDT OK (rule removed Jun 2026)")

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
