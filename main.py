"""main.py — Entry point for the agentic trading bot."""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from config import get_config
from utils.logger import setup_logging
from utils.notifier import Notifier
from core.engine import TradingEngine
from core.portfolio import PortfolioTracker
from core.risk_manager import RiskManager
from brokers.robinhood_broker import RobinhoodBroker
from brokers.polymarket_broker import PolymarketBroker
from data.news_feed import NewsFeed
from data.social_feed import SocialFeed
from data.yfinance_data import YFinanceDataFeed
from analysis.ai_analyzer import AIAnalyzer
from strategies.news_momentum import NewsMomentum
from strategies.swing_trader import SwingTrader
from strategies.alpha_scanner import AlphaScanner
from strategies.gap_fill import GapFillTrader
from strategies.orb_strategy import ORBStrategy
from strategies.short_scanner import ShortScanner
from strategies.polymarket_strategy import PolymarketStrategy
from strategies.eod_analyzer import run_once_and_print, format_ideas_report, run_eod_scan


async def eod_mode(cfg) -> None:
    """Run end-of-day research and print tomorrow's trade ideas."""
    setup_logging(cfg.log_level)
    ai       = AIAnalyzer(groq_api_key=cfg.groq_api_key, api_key=cfg.anthropic_api_key)
    notifier = Notifier(cfg.telegram_bot_token, cfg.telegram_chat_id)

    logger.info("Running end-of-day market analysis...")
    ideas  = await run_eod_scan(ai)
    report = format_ideas_report(ideas)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    if cfg.has_telegram():
        await notifier.eod_report(report)
        logger.info("Report sent to Telegram.")
    else:
        logger.info("Telegram not configured — report printed to console only.")
        print("\nTo get this sent to your phone nightly, see the Telegram setup note in .env.example")

    await notifier.close()


async def main(dry_run: bool = False, eod: bool = False) -> None:
    cfg = get_config()
    setup_logging(cfg.log_level)

    if eod:
        await eod_mode(cfg)
        return

    missing = cfg.validate()
    if missing:
        logger.error("Missing required environment variables: {}", missing)
        logger.error("Copy .env.example to .env and fill in your API keys")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  AGENTIC TRADING BOT STARTING")
    logger.info("  Capital: ${:.2f}", cfg.starting_capital)
    logger.info("  Alpaca:     {}", "ENABLED" if cfg.has_alpaca() else "disabled (crypto/stock data via yfinance)")
    logger.info("  Robinhood:  {}", "CONNECTED" if cfg.has_robinhood() else "NOT CONFIGURED")
    logger.info("  Polymarket: {}", "CONNECTED" if cfg.has_polymarket() else "NOT CONFIGURED")
    logger.info("  Telegram:   {}", "ENABLED" if cfg.has_telegram() else "not set")
    logger.info("  Mode: {}", "DRY RUN" if dry_run else "LIVE TRADING")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN: Config loaded and validated. Exiting without placing orders.")
        logger.info("  Claude key:   {}...", cfg.anthropic_api_key[:8])
        logger.info("  Robinhood:    {}", "set" if cfg.has_robinhood() else "not set")
        logger.info("  Telegram:     {}", "set" if cfg.has_telegram() else "not set")
        logger.info("  Polymarket:   {}", "set" if cfg.has_polymarket() else "not set")
        return

    # --- Core components ---
    notifier  = Notifier(cfg.telegram_bot_token, cfg.telegram_chat_id)
    portfolio = PortfolioTracker(starting_capital=cfg.starting_capital)
    engine    = TradingEngine(
        portfolio=portfolio,
        risk_manager=None,
        alpaca_broker=None,
        robinhood_broker=None,
        polymarket_broker=None,
        config=cfg,
        notifier=notifier,
    )
    risk_mgr = RiskManager(portfolio=portfolio, config=cfg)
    engine._risk_manager = risk_mgr

    # --- Brokers ---
    robinhood = RobinhoodBroker(
        cfg.robinhood_mcp_token,
        engine.fill_queue,
        refresh_token=cfg.robinhood_refresh_token,
        client_id=cfg.robinhood_client_id,
    )
    engine._robinhood_broker = robinhood

    # Sync real cash + existing positions from Robinhood on startup
    await portfolio.sync_from_robinhood(robinhood)

    if cfg.has_alpaca():
        from brokers.alpaca_broker import AlpacaBroker
        from data.market_data import MarketDataFeed
        alpaca       = AlpacaBroker(cfg.alpaca_api_key, cfg.alpaca_api_secret, engine.fill_queue)
        market_data  = MarketDataFeed(cfg.alpaca_api_key, cfg.alpaca_api_secret)
        engine._alpaca_broker = alpaca
    else:
        market_data = None
        logger.info("Alpaca not configured — skipping crypto strategies, using yfinance for stock data")

    # Use yfinance as free market data fallback when Alpaca is not configured
    if market_data is None:
        market_data = YFinanceDataFeed()
        logger.info("Using yfinance for market data (free, no API key needed)")

    polymarket_broker = None
    if cfg.has_polymarket():
        polymarket_broker = PolymarketBroker(
            cfg.polymarket_private_key, cfg.polymarket_api_key,
            cfg.polymarket_api_secret, cfg.polymarket_api_passphrase,
        )
        engine._polymarket_broker = polymarket_broker
        await polymarket_broker.initialize()

    # --- Data feeds ---
    news_feed   = NewsFeed(cfg.news_api_key, engine.news_queue, engine.breaking_queue)
    social_feed = SocialFeed(engine.social_queue)

    # --- AI ---
    ai = AIAnalyzer(groq_api_key=cfg.groq_api_key, api_key=cfg.anthropic_api_key)

    market_open_fn = lambda: engine.market_open  # noqa: E731

    # --- Strategies (stocks only when no Alpaca) ---
    strategy_coros = [
        news_feed.poll_loop(),
        social_feed.poll_loop(),
    ]

    # Stock strategies — use yfinance-backed alpha scanner when no Alpaca
    news_momentum = NewsMomentum(
        engine.signal_bus, robinhood, ai, market_data,
        engine.news_queue, engine.breaking_queue, portfolio, market_open_fn,
    )
    swing_trader = SwingTrader(
        engine.signal_bus, market_data, robinhood, ai, portfolio, market_open_fn,
    )
    alpha_scanner = AlphaScanner(
        engine.signal_bus, market_data, social_feed,
        engine.news_queue, ai, portfolio, market_open_fn,
    )
    gap_fill = GapFillTrader(
        engine.signal_bus, market_data, robinhood, portfolio, market_open_fn,
    )
    orb = ORBStrategy(
        engine.signal_bus, market_data, robinhood, portfolio, market_open_fn,
    )
    short_scanner = ShortScanner(
        engine.signal_bus, market_data, robinhood, portfolio, market_open_fn,
    )
    strategy_coros += [
        news_momentum.run(),
        swing_trader.run(),
        alpha_scanner.run(),
        gap_fill.run(),
        orb.run(),
        short_scanner.run(),
    ]

    # Crypto strategies — only if Alpaca is configured
    if cfg.has_alpaca():
        from strategies.crypto_scalper import CryptoScalper
        from strategies.crypto_momentum import CryptoMomentum
        from data.new_coins import NewCoinsScanner
        from strategies.new_coin_hunter import NewCoinHunter
        new_coins        = NewCoinsScanner(social_feed)
        crypto_scalper   = CryptoScalper(engine.signal_bus, market_data, portfolio, market_open_fn)
        crypto_momentum  = CryptoMomentum(engine.signal_bus, market_data, ai, engine.news_queue, portfolio, market_open_fn)
        new_coin_hunter  = NewCoinHunter(engine.signal_bus, new_coins, social_feed, ai, portfolio, market_open_fn, market_data)
        strategy_coros  += [crypto_scalper.run(), crypto_momentum.run(), new_coin_hunter.run(), new_coins.poll_loop()]

    if cfg.has_polymarket():
        poly_strategy = PolymarketStrategy(engine.signal_bus, polymarket_broker, ai, market_open_fn)
        strategy_coros.append(poly_strategy.run())

    if cfg.has_telegram():
        await notifier.send("🤖 <b>Trading bot started</b>\nRobinhood connected. Watching for setups...")

    logger.info("Launching {} coroutines...", len(strategy_coros) + 4)
    await engine.run(*strategy_coros)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    eod     = "--eod" in sys.argv
    asyncio.run(main(dry_run=dry_run, eod=eod))
