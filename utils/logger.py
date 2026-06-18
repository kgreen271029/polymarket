"""utils/logger.py — Structured logging setup and trade audit CSV."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from loguru import logger


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    strategy: str
    ai_confidence: str
    reasoning_summary: str
    asset_class: str


_TRADES_CSV: Path | None = None
_CSV_HEADERS = list(TradeRecord.__dataclass_fields__.keys())


def setup_logging(log_level: str = "INFO") -> None:
    """Configure loguru sinks: colorized stderr + rotating JSON file."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )

    log_file = logs_dir / "trading_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="1 day",
        retention="14 days",
        serialize=True,
        enqueue=True,
    )

    global _TRADES_CSV
    _TRADES_CSV = logs_dir / "trades.csv"
    _ensure_csv_header()


def _ensure_csv_header() -> None:
    if _TRADES_CSV and not _TRADES_CSV.exists():
        with open(_TRADES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
            writer.writeheader()


def log_trade(record: TradeRecord) -> None:
    """Append a completed trade to the audit CSV."""
    if _TRADES_CSV is None:
        return
    with open(_TRADES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
        writer.writerow(asdict(record))


def make_trade_record(
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    strategy: str,
    ai_confidence: str = "",
    reasoning: str = "",
    asset_class: str = "crypto",
) -> TradeRecord:
    pnl = (exit_price - entry_price) * qty if side == "buy" else (entry_price - exit_price) * qty
    pnl_pct = (exit_price - entry_price) / entry_price * 100 if side == "buy" else 0.0
    return TradeRecord(
        timestamp=datetime.utcnow().isoformat(),
        symbol=symbol,
        side=side,
        qty=round(qty, 8),
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=round(pnl, 4),
        pnl_pct=round(pnl_pct, 2),
        strategy=strategy,
        ai_confidence=ai_confidence,
        reasoning_summary=reasoning[:120],
        asset_class=asset_class,
    )
