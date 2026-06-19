"""Logging configuration for the trading bot."""

import logging
import sys


def setup_logger(log_level_str="INFO"):
    """Configure logger with both file and console output."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    logger = logging.getLogger("trading_bot")
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler("trading_bot.log")
    file_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
