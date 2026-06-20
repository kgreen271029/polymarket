"""Position sizing and risk management."""
import logging
from bot.config import MAX_RISK_PER_TRADE, DAILY_LOSS_LIMIT_PCT, MAX_OPEN_POSITIONS, STARTING_CAPITAL

logger = logging.getLogger(__name__)


def calc_position_size(equity: float, buying_power: float, confidence: int) -> float:
    """
    Return dollar amount to invest in a trade.
    Scales with LLM confidence (50-100 → 40%-100% of MAX_RISK_PER_TRADE).
    Never exceeds buying_power or MAX_RISK_PER_TRADE.
    """
    if confidence < 55:
        return 0.0
    scale = min(1.0, (confidence - 50) / 50)
    amount = MAX_RISK_PER_TRADE * scale
    amount = min(amount, buying_power * 0.95)
    amount = max(amount, 1.0)
    logger.debug(f"Position size: ${amount:.2f} (confidence={confidence}, scale={scale:.2f})")
    return round(amount, 2)


def check_daily_loss_limit(equity: float, starting_equity: float) -> bool:
    """Return True if we've hit the daily loss limit and should stop trading."""
    if starting_equity <= 0:
        return False
    loss_pct = (starting_equity - equity) / starting_equity
    if loss_pct >= DAILY_LOSS_LIMIT_PCT:
        logger.warning(
            f"Daily loss limit hit: {loss_pct*100:.1f}% >= {DAILY_LOSS_LIMIT_PCT*100:.1f}%"
        )
        return True
    return False


def should_stop_loss(position: dict, stop_loss_pct: float = 0.05) -> bool:
    """Return True if position P&L% has breached the stop loss threshold."""
    pnl_pct = position.get("pnl_pct", 0)
    if pnl_pct <= -(stop_loss_pct * 100):
        logger.info(f"Stop loss triggered for {position['symbol']}: {pnl_pct:.1f}%")
        return True
    return False


def should_take_profit(position: dict, take_profit_pct: float = 0.12) -> bool:
    """Return True if position has hit take profit."""
    pnl_pct = position.get("pnl_pct", 0)
    if pnl_pct >= (take_profit_pct * 100):
        logger.info(f"Take profit triggered for {position['symbol']}: {pnl_pct:.1f}%")
        return True
    return False


def can_open_position(current_positions: list[dict]) -> bool:
    """Return True if we can open another position."""
    count = len(current_positions)
    if count >= MAX_OPEN_POSITIONS:
        logger.info(f"Max positions reached ({count}/{MAX_OPEN_POSITIONS})")
        return False
    return True


def already_in_position(symbol: str, positions: list[dict]) -> bool:
    return any(p["symbol"] == symbol for p in positions)
