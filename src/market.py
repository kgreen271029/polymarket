"""Market data and trading operations."""

import os
import logging
from datetime import datetime
import pytz
import requests


class MarketManager:
    """Manages market data and trading operations via Robinhood."""

    def __init__(self, logger):
        self.logger = logger
        self.authenticated = False
        self.account_info = None
        self.rh = None
        self._init_robinhood()
        self.authenticate()

    def _init_robinhood(self):
        """Lazy load robin_stocks to avoid import errors."""
        try:
            import robin_stocks.robinhood as rh
            self.rh = rh
            self.logger.info("robin_stocks module loaded")
        except (ImportError, RuntimeError, Exception) as e:
            self.logger.warning(f"robin_stocks not available, using dry-run mode")
            self.rh = None

    def authenticate(self):
        """Authenticate with Robinhood API."""
        if self.rh is None:
            self.logger.warning("Robinhood API not available, using dry-run mode")
            return

        try:
            mcp_token = os.getenv("ROBINHOOD_MCP_TOKEN")
            client_id = os.getenv("ROBINHOOD_CLIENT_ID")

            if mcp_token and client_id:
                self.rh.login(
                    username=None,
                    password=None,
                    mfa_code=None,
                    authorization_token=mcp_token,
                    client_id=client_id
                )
                self.authenticated = True
                self.logger.info("Successfully authenticated with Robinhood")
            else:
                self.logger.warning("Robinhood credentials not provided")

        except Exception as e:
            self.logger.error(f"Failed to authenticate with Robinhood: {e}")

    def is_market_open(self):
        """Check if market is currently open."""
        now = datetime.now(pytz.timezone("US/Eastern"))
        weekday = now.weekday()

        # Market is closed on weekends (5, 6)
        if weekday >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_account_info(self):
        """Get account information."""
        if not self.authenticated or self.rh is None:
            return None

        try:
            self.account_info = self.rh.account.get_account()
            return self.account_info
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return None

    def get_portfolio_value(self):
        """Get total portfolio value."""
        if not self.authenticated or self.rh is None:
            return float(os.getenv("STARTING_CAPITAL", 92.65))

        try:
            account = self.get_account_info()
            if account and "portfolio_equity" in account:
                return float(account["portfolio_equity"])
        except Exception as e:
            self.logger.error(f"Failed to get portfolio value: {e}")

        return float(os.getenv("STARTING_CAPITAL", 92.65))

    def get_stock_price(self, symbol):
        """Get current price for a stock."""
        if not self.authenticated or self.rh is None:
            return None

        try:
            quote = self.rh.stocks.get_quotes(symbol)[0]
            if quote and "last_trade_price" in quote:
                return float(quote["last_trade_price"])
        except Exception as e:
            self.logger.error(f"Failed to get price for {symbol}: {e}")

        return None

    def get_holdings(self):
        """Get current stock holdings."""
        if not self.authenticated or self.rh is None:
            return []

        try:
            positions = self.rh.account.get_positions()
            return [
                {
                    "symbol": pos.get("symbol"),
                    "quantity": float(pos.get("quantity", 0)),
                    "average_buy_price": float(pos.get("average_buy_price", 0)),
                }
                for pos in positions if float(pos.get("quantity", 0)) > 0
            ]
        except Exception as e:
            self.logger.error(f"Failed to get holdings: {e}")
            return []

    def buy_stock(self, symbol, quantity):
        """Execute a buy order."""
        if not self.authenticated or self.rh is None:
            self.logger.warning(f"[DRY RUN] Would buy {quantity} shares of {symbol}")
            return True

        try:
            result = self.rh.stocks.order_buy_market(symbol, quantity)
            self.logger.info(f"Buy order placed: {quantity} shares of {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to buy {symbol}: {e}")
            return False

    def sell_stock(self, symbol, quantity):
        """Execute a sell order."""
        if not self.authenticated or self.rh is None:
            self.logger.warning(f"[DRY RUN] Would sell {quantity} shares of {symbol}")
            return True

        try:
            result = self.rh.stocks.order_sell_market(symbol, quantity)
            self.logger.info(f"Sell order placed: {quantity} shares of {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to sell {symbol}: {e}")
            return False

    def get_market_news(self, symbols):
        """Get news for provided symbols."""
        news = {}
        api_key = os.getenv("NEWS_API_KEY")

        if not api_key:
            return news

        try:
            for symbol in symbols:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    "q": symbol,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 5,
                    "apiKey": api_key
                }
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    news[symbol] = data.get("articles", [])
        except Exception as e:
            self.logger.warning(f"Failed to fetch news: {e}")

        return news
