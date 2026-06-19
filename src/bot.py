"""Main trading bot logic.

Each cycle the bot:
  1. Re-prices and manages open positions (stop-loss / take-profit / exit signal).
  2. Scans the watchlist for new BUY candidates using the multi-signal analyzer.
  3. Ranks candidates by confidence and opens positions within risk limits.
"""

import os
import asyncio
import logging
from datetime import datetime
import pytz

from src.analyzer import MarketAnalyzer
from src.paper_trader import PaperTrader


class TradingBot:
    """Coordinates market analysis, risk, and execution."""

    def __init__(self, market_manager, risk_manager, notification_manager, dry_run=False, logger=None):
        self.market_manager = market_manager
        self.risk_manager = risk_manager
        self.notification_manager = notification_manager
        self.dry_run = dry_run
        self.logger = logger or logging.getLogger(__name__)
        self.analyzer = MarketAnalyzer(self.logger)
        self.paper_trader = PaperTrader(self.logger)
        self.live = market_manager.authenticated  # True only with real broker auth
        self.min_confidence = int(os.getenv("MIN_CONFIDENCE", 50))
        self.scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", 300))
        self.watchlist = self._load_watchlist()
        self.logger.info(
            f"Bot ready | mode={'LIVE' if self.live else 'PAPER'} | "
            f"watchlist={len(self.watchlist)} symbols | min_confidence={self.min_confidence}"
        )

    def _load_watchlist(self):
        """Load scan universe from watchlist.txt if present, else a liquid default set."""
        path = os.path.join(os.path.dirname(__file__), "..", "watchlist.txt")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    syms = [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]
                if syms:
                    return syms
            except Exception as e:
                self.logger.warning(f"Could not read watchlist.txt: {e}")
        # Default: large-cap + high-liquidity ETFs across sectors
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD",
            "NFLX", "AVGO", "COST", "PEP", "ADBE", "CRM", "INTC", "QCOM",
            "JPM", "BAC", "V", "MA", "DIS", "PYPL", "UBER", "SHOP",
            "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH",
        ]

    # ---------- main loop ----------

    async def run(self):
        """Main trading loop during market hours."""
        et = pytz.timezone("US/Eastern")
        market_close = datetime.now(et).replace(hour=16, minute=0, second=0, microsecond=0)
        self.logger.info("Trading session started")

        while True:
            now = datetime.now(et)
            if now >= market_close:
                self.logger.info("Market closed. Running EOD analysis.")
                await self.run_eod_analysis()
                break
            await self.analyze_and_trade()
            await asyncio.sleep(self.scan_interval)

    # ---------- per-cycle work ----------

    async def analyze_and_trade(self):
        try:
            # 1. Manage existing positions first
            await self._manage_open_positions()

            # 2. Portfolio snapshot
            prices = self._current_prices(self.paper_trader.positions.keys())
            portfolio_value = self.paper_trader.get_portfolio_value(prices)
            num_open = len(self.paper_trader.positions)

            self.logger.info(
                f"Portfolio: ${portfolio_value:.2f} | Cash: ${self.paper_trader.cash:.2f} | "
                f"Open: {num_open} | Daily P&L: ${self.risk_manager.daily_pnl:.2f}"
            )

            if not self.risk_manager.can_trade(portfolio_value, num_open):
                return

            # 3. Scan watchlist for BUY candidates
            candidates = []
            for symbol in self.watchlist:
                if symbol in self.paper_trader.positions:
                    continue  # already holding
                bars = self.market_manager.get_bars(symbol)
                if not bars:
                    continue
                decision = self.analyzer.analyze(symbol, bars)
                if decision["recommendation"] == "BUY" and decision["confidence"] >= self.min_confidence:
                    candidates.append(decision)

            # 4. Best signals first
            candidates.sort(key=lambda d: d["confidence"], reverse=True)
            self.logger.info(f"Scan complete: {len(candidates)} BUY candidate(s)")

            for decision in candidates:
                if len(self.paper_trader.positions) >= self.risk_manager.max_open_positions:
                    break
                await self._execute_buy(decision)

        except Exception as e:
            self.logger.error(f"analyze_and_trade error: {e}", exc_info=True)

    async def _manage_open_positions(self):
        """Check each held position against stop-loss, take-profit, and exit signals."""
        for symbol in list(self.paper_trader.positions.keys()):
            pos = self.paper_trader.positions[symbol]
            bars = self.market_manager.get_bars(symbol)
            if not bars:
                continue
            price = bars["price"]
            stop = pos.get("stop_loss")
            target = pos.get("target_price")

            reason = None
            if stop and price <= stop:
                reason = f"stop-loss hit (${price:.2f} <= ${stop:.2f})"
            elif target and price >= target:
                reason = f"target reached (${price:.2f} >= ${target:.2f})"
            else:
                decision = self.analyzer.analyze(symbol, bars)
                if decision["recommendation"] == "SELL":
                    reason = f"exit signal ({decision['reasoning']})"

            if reason:
                qty = pos["quantity"]
                if self.market_manager.sell_stock(symbol, qty, self.paper_trader):
                    self.logger.info(f"SELL {qty} {symbol} @ ${price:.2f} — {reason}")
                    await self.notification_manager.send_trade_alert(
                        symbol, "SELL", qty, price, reason
                    )

    async def _execute_buy(self, decision):
        symbol = decision["symbol"]
        price = decision["entry_price"]
        stop = decision.get("stop_loss", price * 0.95)
        target = decision.get("target_price", price * 1.10)
        try:
            prices = self._current_prices(self.paper_trader.positions.keys())
            portfolio_value = self.paper_trader.get_portfolio_value(prices)
            quantity = self.risk_manager.calculate_position_size(portfolio_value, price, stop)
            if quantity < 1 or quantity * price > self.paper_trader.cash:
                # fall back to what cash allows
                quantity = int(self.paper_trader.cash // price)
            if quantity < 1:
                self.logger.info(f"Skip {symbol}: insufficient cash for 1 share (${price:.2f})")
                return

            if self.market_manager.buy_stock(symbol, quantity, self.paper_trader):
                # attach risk levels to the position for later management
                self.paper_trader.positions[symbol]["stop_loss"] = stop
                self.paper_trader.positions[symbol]["target_price"] = target
                self.paper_trader.save_state()
                self.logger.info(
                    f"BUY {quantity} {symbol} @ ${price:.2f} "
                    f"(conf {decision['confidence']}%, stop ${stop:.2f}, target ${target:.2f}) "
                    f"— {decision['reasoning']}"
                )
                await self.notification_manager.send_trade_alert(
                    symbol, "BUY", quantity, price,
                    f"Conf {decision['confidence']}% | {decision['reasoning']}"
                )
        except Exception as e:
            self.logger.error(f"Buy failed for {symbol}: {e}")

    def _current_prices(self, symbols):
        """Build a {symbol: price} map for valuation (uses cached bars)."""
        out = {}
        for s in symbols:
            bars = self.market_manager.get_bars(s)
            if bars:
                out[s] = bars["price"]
        return out

    # ---------- end of day ----------

    async def run_eod_analysis(self):
        try:
            prices = self._current_prices(self.paper_trader.positions.keys())
            s = self.paper_trader.get_summary(prices)
            text = (
                f"END OF DAY\n"
                f"Trades: {s['trades_executed']} | Open: {s['positions']}\n"
                f"Value: ${s['total_value']:.2f} | Cash: ${s['cash']:.2f}\n"
                f"P&L: ${s['pnl']:.2f} ({s['pnl_pct']:+.2f}%) — "
                f"{'GREEN' if s['pnl'] >= 0 else 'RED'}"
            )
            self.logger.info(text)
            await self.notification_manager.send_eod_summary(text)
            for symbol, pos in self.paper_trader.positions.items():
                self.logger.info(
                    f"  Holding {pos['quantity']} {symbol} @ ${pos['avg_price']:.2f} avg"
                )
        except Exception as e:
            self.logger.error(f"EOD error: {e}")
