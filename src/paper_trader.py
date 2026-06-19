"""Paper trading - virtual trading account with real prices."""

import os
import json
import logging
from datetime import datetime


class PaperTrader:
    """Virtual trading account - tracks trades as if they're real."""

    def __init__(self, logger, starting_capital=92.65):
        self.logger = logger
        self.starting_capital = float(starting_capital)
        self.cash = self.starting_capital
        self.positions = {}  # {symbol: {quantity, avg_price, current_price}}
        self.trades = []
        self.portfolio_file = "paper_trading.json"
        self.load_state()

    def load_state(self):
        """Load previous trading state."""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    state = json.load(f)
                    self.cash = state.get('cash', self.starting_capital)
                    self.positions = state.get('positions', {})
                    self.trades = state.get('trades', [])
                    self.logger.info(f"📄 Loaded paper trading state: ${self.cash:.2f} cash")
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")

    def save_state(self):
        """Save trading state to file."""
        state = {
            'cash': self.cash,
            'positions': self.positions,
            'trades': self.trades,
            'updated': datetime.now().isoformat()
        }
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def buy(self, symbol, quantity, price):
        """Execute a buy order (paper trading)."""
        cost = quantity * price
        if cost > self.cash:
            self.logger.warning(f"❌ Insufficient cash for {symbol}: need ${cost:.2f}, have ${self.cash:.2f}")
            return False

        # Deduct cash
        self.cash -= cost

        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}

        pos = self.positions[symbol]
        total_qty = pos['quantity'] + quantity
        total_cost = (pos['quantity'] * pos['avg_price']) + cost
        pos['avg_price'] = total_cost / total_qty if total_qty > 0 else price
        pos['quantity'] = total_qty

        # Record trade
        trade = {
            'type': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'timestamp': datetime.now().isoformat(),
            'cash_remaining': self.cash
        }
        self.trades.append(trade)
        self.save_state()

        self.logger.info(f"✅ [PAPER TRADE] BUY {quantity} {symbol} @ ${price:.2f} | Cash: ${self.cash:.2f}")
        return True

    def sell(self, symbol, quantity, price):
        """Execute a sell order (paper trading)."""
        if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
            self.logger.warning(f"❌ Cannot sell {quantity} {symbol} - position too small")
            return False

        proceeds = quantity * price
        self.cash += proceeds

        pos = self.positions[symbol]
        pos['quantity'] -= quantity

        if pos['quantity'] == 0:
            del self.positions[symbol]

        # Record trade
        trade = {
            'type': 'SELL',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds,
            'timestamp': datetime.now().isoformat(),
            'cash_remaining': self.cash
        }
        self.trades.append(trade)
        self.save_state()

        self.logger.info(f"✅ [PAPER TRADE] SELL {quantity} {symbol} @ ${price:.2f} | Cash: ${self.cash:.2f}")
        return True

    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value."""
        holdings_value = sum(
            pos['quantity'] * current_prices.get(symbol, pos['avg_price'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + holdings_value

    def get_summary(self, current_prices):
        """Get portfolio summary."""
        total_value = self.get_portfolio_value(current_prices)
        pnl = total_value - self.starting_capital

        return {
            'cash': self.cash,
            'total_value': total_value,
            'positions': len(self.positions),
            'pnl': pnl,
            'pnl_pct': (pnl / self.starting_capital * 100) if self.starting_capital > 0 else 0,
            'trades_executed': len(self.trades)
        }
