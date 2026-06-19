"""Risk management and position sizing."""

import os
from datetime import datetime
import logging


class RiskManager:
    """Manages trading risk and position sizing."""

    def __init__(self, logger):
        self.logger = logger
        self.starting_capital = float(os.getenv("STARTING_CAPITAL", 92.65))
        self.max_risk_per_trade = float(os.getenv("MAX_RISK_PER_TRADE", 20.0))
        self.daily_loss_limit_pct = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 0.15))
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", 6))
        self.daily_trades = []
        self.daily_pnl = 0.0

    def reset_daily_stats(self):
        """Reset daily trading statistics at market open."""
        today = datetime.now().date()
        if self.daily_trades and self.daily_trades[0].date() != today:
            self.daily_trades = []
            self.daily_pnl = 0.0

    def can_trade(self, portfolio_value, num_open_positions):
        """Check if we can execute a trade based on risk limits."""
        # Check position limit
        if num_open_positions >= self.max_open_positions:
            self.logger.warning(f"Max open positions ({self.max_open_positions}) reached")
            return False

        # Check daily loss limit
        loss_limit = self.starting_capital * self.daily_loss_limit_pct
        if self.daily_pnl < -loss_limit:
            self.logger.warning(
                f"Daily loss limit exceeded. "
                f"Current PnL: ${self.daily_pnl:.2f}, Limit: ${-loss_limit:.2f}"
            )
            return False

        return True

    def calculate_position_size(self, portfolio_value, entry_price, stop_loss_price):
        """Calculate position size based on risk management rules."""
        if entry_price <= 0 or stop_loss_price >= entry_price:
            return 0

        # Risk amount per trade
        risk_amount = self.max_risk_per_trade

        # Risk per share
        risk_per_share = entry_price - stop_loss_price

        # Position size
        position_size = int(risk_amount / risk_per_share)

        # Ensure we have enough capital
        required_capital = position_size * entry_price
        if required_capital > portfolio_value * 0.5:  # Max 50% of portfolio per trade
            position_size = int(portfolio_value * 0.5 / entry_price)

        return max(1, position_size)

    def update_daily_pnl(self, trade_result):
        """Update daily P&L with trade result."""
        if trade_result:
            self.daily_pnl += trade_result.get("pnl", 0)
            self.daily_trades.append(datetime.now())

    def get_risk_summary(self, portfolio_value):
        """Get current risk metrics."""
        return {
            "portfolio_value": portfolio_value,
            "daily_pnl": self.daily_pnl,
            "daily_trades": len(self.daily_trades),
            "max_daily_loss_usd": self.starting_capital * self.daily_loss_limit_pct,
        }
