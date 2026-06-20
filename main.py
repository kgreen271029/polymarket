#!/usr/bin/env python3
"""
Robinhood AI Trading Bot — powered by Claude.
Runs in two modes:
  python main.py           # market-hours loop (Mon-Fri 9:30-16:00 ET)
  python main.py --dry-run # same loop but no real orders placed
  python main.py --eod     # end-of-day summary only
"""
import argparse
import logging
import time
from datetime import datetime, time as dtime

import pytz
import schedule

from bot import robinhood, llm, telegram
from bot.config import SCAN_INTERVAL_MINUTES, LOG_LEVEL
from bot.scanner import run_scan

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

ET = pytz.timezone("US/Eastern")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def is_market_open() -> bool:
    now_et = datetime.now(ET).time()
    return MARKET_OPEN <= now_et <= MARKET_CLOSE


def run_eod_mode():
    logger.info("=== EOD Mode ===")
    if not robinhood.login():
        logger.critical("Cannot login to Robinhood")
        return

    account = robinhood.get_account_info()
    positions = robinhood.get_positions()
    orders = robinhood.get_todays_orders()

    summary = llm.analyze_eod(positions, orders, account)
    logger.info(f"EOD Summary:\n{summary}")

    from bot.config import STARTING_CAPITAL
    telegram.send_eod_summary(summary, account["equity"], STARTING_CAPITAL)


def run_market_hours_loop(dry_run: bool):
    mode = "DRY RUN" if dry_run else "LIVE"
    logger.info(f"=== Trading Bot Starting [{mode}] ===")

    if not robinhood.login():
        logger.critical("Cannot login to Robinhood, aborting")
        telegram.send_error("Startup", "Robinhood login failed")
        return

    account = robinhood.get_account_info()
    positions = robinhood.get_positions()
    starting_equity = account["equity"]

    logger.info(
        f"Account: equity=${starting_equity:.2f}, "
        f"buying_power=${account['buying_power']:.2f}, "
        f"open_positions={len(positions)}"
    )

    telegram.send_startup(mode, starting_equity, len(positions))

    scan_count = 0

    def scan_job():
        nonlocal scan_count
        if not is_market_open():
            logger.info("Market closed, skipping scan")
            return
        scan_count += 1
        logger.info(f"--- Scan #{scan_count} ---")
        try:
            result = run_scan(dry_run=dry_run, starting_equity=starting_equity)
            if result.get("halted"):
                logger.warning("Trading halted by risk manager")
        except Exception as e:
            logger.error(f"Scan #{scan_count} failed: {e}")
            telegram.send_error(f"Scan #{scan_count}", str(e))

    # Run immediately on start, then every SCAN_INTERVAL_MINUTES
    scan_job()
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(scan_job)

    # Main loop — runs until market closes or process is killed
    while True:
        now_et = datetime.now(ET)
        if now_et.time() > MARKET_CLOSE:
            logger.info("Market closed. Bot shutting down.")
            telegram.send(f"🔔 Market closed. Final equity: ${robinhood.get_account_info()['equity']:.2f}")
            break
        schedule.run_pending()
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="Robinhood AI Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run without placing real orders")
    parser.add_argument("--eod", action="store_true", help="Run EOD analysis only")
    args = parser.parse_args()

    if args.eod:
        run_eod_mode()
    else:
        run_market_hours_loop(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
