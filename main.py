#!/usr/bin/env python3
"""
Agentic Trading Bot - Main Entry Point
Executes trades based on AI-driven market analysis
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import asyncio
from dotenv import load_dotenv

from src.bot import TradingBot
from src.market import MarketManager
from src.risk_manager import RiskManager
from src.notifications import NotificationManager
from src.logger import setup_logger

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Agentic Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no real trades)")
    parser.add_argument("--eod", action="store_true", help="Run end-of-day analysis only")
    args = parser.parse_args()

    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger = setup_logger(log_level)

    logger.info(f"Starting Trading Bot - Dry Run: {args.dry_run}, EOD Mode: {args.eod}")

    # Validate environment
    validate_environment(logger)

    # Initialize managers
    try:
        market_manager = MarketManager(logger)
        risk_manager = RiskManager(logger)
        notification_manager = NotificationManager(logger)

        bot = TradingBot(
            market_manager=market_manager,
            risk_manager=risk_manager,
            notification_manager=notification_manager,
            dry_run=args.dry_run,
            logger=logger
        )

        # Check if market is open
        if not market_manager.is_market_open() and not args.eod:
            logger.info("Market is closed. Exiting.")
            return

        if args.eod:
            # Run EOD analysis
            logger.info("Running end-of-day analysis...")
            asyncio.run(bot.run_eod_analysis())
        else:
            # Run main trading loop
            logger.info("Starting trading session...")
            asyncio.run(bot.run())

        logger.info("Trading bot completed successfully")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def validate_environment(logger):
    """Validate that all required environment variables are set."""
    required_vars = [
        "ROBINHOOD_MCP_TOKEN",
        "ROBINHOOD_REFRESH_TOKEN",
        "ROBINHOOD_CLIENT_ID",
        "ANTHROPIC_API_KEY",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.warning(f"Missing environment variables: {', '.join(missing)}")
        logger.warning("Bot will attempt to run but may fail on trade execution")


if __name__ == "__main__":
    main()
