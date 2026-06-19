"""Main trading bot logic."""

import asyncio
import logging
from datetime import datetime, timedelta
import pytz

from src.analyzer import MarketAnalyzer


class TradingBot:
    """Main trading bot that coordinates market analysis and execution."""

    def __init__(self, market_manager, risk_manager, notification_manager, dry_run=False, logger=None):
        self.market_manager = market_manager
        self.risk_manager = risk_manager
        self.notification_manager = notification_manager
        self.dry_run = dry_run
        self.logger = logger or logging.getLogger(__name__)
        self.analyzer = MarketAnalyzer(self.logger)
        self.watchlist = self._load_watchlist()

    def _load_watchlist(self):
        """Load trading watchlist."""
        # Default watchlist of popular stocks
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "META", "NVDA", "SPY", "QQQ", "IWM"
        ]

    async def run(self):
        """Main trading loop during market hours."""
        self.logger.info("Trading bot market hours session started")

        et = pytz.timezone("US/Eastern")
        market_open = datetime.now(et).replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = datetime.now(et).replace(hour=16, minute=0, second=0, microsecond=0)

        while True:
            now = datetime.now(et)

            # Check if market is still open
            if now >= market_close:
                self.logger.info("Market closed. Running EOD analysis...")
                await self.run_eod_analysis()
                break

            # Run analysis every 5 minutes
            await self.analyze_and_trade()

            # Wait before next analysis
            await asyncio.sleep(300)

    async def analyze_and_trade(self):
        """Analyze market and execute trades."""
        try:
            portfolio_value = self.market_manager.get_portfolio_value()
            holdings = self.market_manager.get_holdings()
            num_open_positions = len(holdings)

            self.logger.info(
                f"Portfolio: ${portfolio_value:.2f} | "
                f"Open positions: {num_open_positions} | "
                f"Daily P&L: ${self.risk_manager.daily_pnl:.2f}"
            )

            # Check if we can trade
            if not self.risk_manager.can_trade(portfolio_value, num_open_positions):
                self.logger.warning("Trading conditions not met, skipping analysis")
                return

            # Analyze each stock in watchlist
            for symbol in self.watchlist:
                if num_open_positions >= self.risk_manager.max_open_positions:
                    break

                try:
                    # Get price and news
                    price = self.market_manager.get_stock_price(symbol)
                    if not price:
                        continue

                    news = self.market_manager.get_market_news([symbol])

                    # Perform analysis
                    analysis = self.analyzer.analyze_stock(
                        symbol=symbol,
                        price_history=[price],  # Simplified for demo
                        news=news.get(symbol, []),
                        technical_indicators={}
                    )

                    # Check recommendation
                    if analysis.get("recommendation") == "BUY":
                        confidence = analysis.get("confidence", 0)
                        if confidence > 70:
                            await self._execute_buy(symbol, price, analysis)
                            num_open_positions += 1

                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"Error in analyze_and_trade: {e}")

    async def _execute_buy(self, symbol, price, analysis):
        """Execute a buy order."""
        try:
            portfolio_value = self.market_manager.get_portfolio_value()
            stop_loss = analysis.get("stop_loss", price * 0.95)

            quantity = self.risk_manager.calculate_position_size(
                portfolio_value, price, stop_loss
            )

            if quantity < 1:
                self.logger.warning(f"Position size too small for {symbol}")
                return

            success = self.market_manager.buy_stock(symbol, quantity)

            if success:
                self.logger.info(
                    f"BUY {quantity} {symbol} @ ${price:.2f} "
                    f"(SL: ${stop_loss:.2f}, Confidence: {analysis.get('confidence')}%)"
                )

                await self.notification_manager.send_trade_alert(
                    symbol, "BUY", quantity, price,
                    f"Confidence: {analysis.get('confidence')}%"
                )

        except Exception as e:
            self.logger.error(f"Failed to execute buy for {symbol}: {e}")

    async def run_eod_analysis(self):
        """Run end-of-day analysis and generate summary."""
        try:
            self.logger.info("Running end-of-day analysis...")

            portfolio_value = self.market_manager.get_portfolio_value()
            holdings = self.market_manager.get_holdings()

            # Get EOD summary from analyzer
            summary = self.analyzer.get_eod_summary(
                self.risk_manager.daily_trades,
                portfolio_value,
                self.risk_manager.daily_pnl
            )

            self.logger.info(f"EOD Summary:\n{summary}")

            # Send summary notification
            await self.notification_manager.send_eod_summary(summary)

            # Log holdings
            if holdings:
                self.logger.info("Current holdings:")
                for h in holdings:
                    self.logger.info(
                        f"  {h['symbol']}: {h['quantity']} shares @ ${h['average_buy_price']:.2f}"
                    )

        except Exception as e:
            self.logger.error(f"Error in EOD analysis: {e}")
