"""main.py — Entry point for the agentic trading bot."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from loguru import logger

from config import get_config
from utils.logger import setup_logging
from core.engine import TradingEngine
from core.portfolio import PortfolioTracker
from core.risk_manager import RiskManager
from brokers.alpaca_broker import AlpacaBroker
from brokers.robinhood_broker import RobinhoodBroker
from brokers.polymarket_broker import PolymarketBroker
from data.market_data import MarketDataFeed
from data.news_feed import NewsFeed
from data.social_feed import SocialFeed
from data.new_coins import NewCoinsScanner
from analysis.ai_analyzer import AIAnalyzer
from strategies.crypto_scalper import CryptoScalper
from strategies.crypto_momentum import CryptoMomentum
from strategies.news_momentum import NewsMomentum
from strategies.swing_trader import SwingTrader
from strategies.new_coin_hunter import NewCoinHunter
from strategies.alpha_scanner import AlphaScanner
from strategies.polymarket_strategy import PolymarketStrategy


async def main(dry_run: bool = False) -> None:
    cfg = get_config()
    setup_logging(cfg.log_level)

    missing = cfg.validate()
    if missing:
        logger.error("Missing required environment variables: {}", missing)
        logger.error("Copy .env.example to .env and fill in your API keys")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  AGENTIC TRADING BOT STARTING")
    logger.info("  Capital: ${:.2f}", cfg.starting_capital)
    logger.info("  Robinhood: {}", "CONNECTED" if cfg.has_robinhood() else "NOT CONFIGURED")
    logger.info("  Polymarket: {}", "CONNECTED" if cfg.has_polymarket() else "NOT CONFIGURED")
    logger.info("  Mode: {}", "DRY RUN" if dry_run else "LIVE TRADING")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN: Config loaded and validated. Exiting without placing orders.")
        logger.info("  Alpaca key: {}...", cfg.alpaca_api_key[:8])
        logger.info("  Claude key: {}...", cfg.anthropic_api_key[:8])
        logger.info("  Robinhood: {}", "set" if cfg.has_robinhood() else "not set")
        logger.info("  Polymarket: {}", "set" if cfg.has_polymarket() else "not set")
        return

    # --- Core components ---
    portfolio = PortfolioTracker(starting_capital=cfg.starting_capital)
    engine = TradingEngine(
        portfolio=portfolio,
        risk_manager=None,  # injected below
        alpaca_broker=None,
        robinhood_broker=None,
        polymarket_broker=None,
        config=cfg,
    )
    risk_mgr = RiskManager(portfolio=portfolio, config=cfg)
    engine._risk_manager = risk_mgr

    # --- Brokers ---
    alpaca = AlpacaBroker(cfg.alpaca_api_key, cfg.alpaca_api_secret, engine.fill_queue)
    robinhood = RobinhoodBroker(cfg.robinhood_mcp_token, engine.fill_queue)
    polymarket_broker = PolymarketBroker(
        cfg.polymarket_private_key,
        cfg.polymarket_api_key,
        cfg.polymarket_api_secret,
        cfg.polymarket_api_passphrase,
    )
    engine._alpaca_broker = alpaca
    engine._robinhood_broker = robinhood
    engine._polymarket_broker = polymarket_broker

    if cfg.has_polymarket():
        await polymarket_broker.initialize()

    # --- Data feeds ---
    market_data = MarketDataFeed(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    news_feed = NewsFeed(cfg.news_api_key, engine.news_queue, engine.breaking_queue)
    social_feed = SocialFeed(engine.social_queue)
    new_coins = NewCoinsScanner(social_feed)

    # --- AI ---
    ai = AIAnalyzer(cfg.anthropic_api_key)

    # --- Strategies ---
    market_open_fn = lambda: engine.market_open  # noqa: E731

    crypto_scalper = CryptoScalper(engine.signal_bus, market_data, portfolio, market_open_fn)
    crypto_momentum = CryptoMomentum(engine.signal_bus, market_data, ai, engine.news_queue, portfolio, market_open_fn)
    news_momentum = NewsMomentum(engine.signal_bus, robinhood, ai, market_data, engine.news_queue, engine.breaking_queue, portfolio, market_open_fn)
    swing_trader = SwingTrader(engine.signal_bus, market_data, robinhood, ai, portfolio, market_open_fn)
    new_coin_hunter = NewCoinHunter(engine.signal_bus, new_coins, social_feed, ai, portfolio, market_open_fn)
    alpha_scanner = AlphaScanner(engine.signal_bus, market_data, social_feed, engine.news_queue, ai, portfolio, market_open_fn)

    strategy_coros = [
        crypto_scalper.run(),
        crypto_momentum.run(),
        news_momentum.run(),
        swing_trader.run(),
        new_coin_hunter.run(),
        alpha_scanner.run(),
        news_feed.poll_loop(),
        social_feed.poll_loop(),
        new_coins.poll_loop(),
    ]

    if cfg.has_polymarket():
        poly_strategy = PolymarketStrategy(engine.signal_bus, polymarket_broker, ai, market_open_fn)
        strategy_coros.append(poly_strategy.run())

    logger.info("Launching {} coroutines...", len(strategy_coros) + 4)  # +4 for engine core tasks
    await engine.run(*strategy_coros)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry_run))
