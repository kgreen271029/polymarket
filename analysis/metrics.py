"""
analysis/metrics.py — Trading edge metrics + adaptive trade journal.

Implements the math from deep research:
  - Expectancy (positive EV check)
  - Break-even win rate vs R:R
  - Sharpe / Sortino ratios
  - Profit factor
  - VIX-based Kelly fraction adjustment

Also keeps a persistent trade journal (logs/trade_journal.json) so the bot
learns its REAL win rate / avg win-loss and feeds that back into Kelly sizing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

JOURNAL_PATH = Path("logs/trade_journal.json")


# ---------------------------------------------------------------------------
# Pure-math edge metrics
# ---------------------------------------------------------------------------

def expectancy(win_rate: float, avg_win: float, avg_loss: float,
               cost: float = 0.0) -> float:
    """Expected dollar value per trade. avg_loss passed as a positive number."""
    return (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) - cost


def breakeven_win_rate(rr_ratio: float) -> float:
    """Win rate needed to break even at a given reward:risk ratio.
    rr_ratio = avg_win / avg_loss. e.g. 2.0 → 0.333."""
    if rr_ratio <= 0:
        return 1.0
    return 1.0 / (1.0 + rr_ratio)


def profit_factor(wins: list[float], losses: list[float]) -> float:
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return gross_win / gross_loss if gross_loss else float("inf")


def sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.04) -> float:
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    ann_ret = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * math.sqrt(252)
    return float((ann_ret - risk_free) / ann_vol)


def sortino_ratio(daily_returns: pd.Series, risk_free: float = 0.04) -> float:
    if len(daily_returns) < 2:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    dd = downside.std() * math.sqrt(252)
    if dd == 0:
        return 0.0
    ann_ret = daily_returns.mean() * 252
    return float((ann_ret - risk_free) / dd)


# ---------------------------------------------------------------------------
# VIX-based Kelly fraction (research: scale down sizing as VIX rises)
# ---------------------------------------------------------------------------

def vix_kelly_fraction() -> tuple[float, str]:
    """
    Returns (kelly_fraction, note) based on current VIX level.
      VIX < 20  → 0.50 (half Kelly)
      VIX 20-30 → 0.35
      VIX 30-40 → 0.25 (quarter Kelly)
      VIX > 40  → 0.0  (cash / A-setups only)
    """
    try:
        import yfinance as yf
        import warnings
        warnings.filterwarnings("ignore")
        h = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
        if h.empty:
            return 0.35, "VIX unavailable — default 0.35"
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0].lower() for c in h.columns]
        else:
            h.columns = [c.lower() for c in h.columns]
        vix = float(h["close"].iloc[-1])
        if vix < 20:
            return 0.50, f"VIX {vix:.1f} (calm) — half Kelly"
        elif vix < 30:
            return 0.35, f"VIX {vix:.1f} (elevated) — 0.35 Kelly"
        elif vix < 40:
            return 0.25, f"VIX {vix:.1f} (high) — quarter Kelly"
        else:
            return 0.0, f"VIX {vix:.1f} (extreme) — no new trades"
    except Exception as e:
        logger.debug("[metrics] vix_kelly_fraction error: {}", e)
        return 0.35, "VIX check failed — default 0.35"


# ---------------------------------------------------------------------------
# Persistent trade journal → adaptive win rate
# ---------------------------------------------------------------------------

@dataclass
class JournalEntry:
    symbol: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    strategy: str
    opened_at: str
    closed_at: str
    reason: str


def record_trade(entry: JournalEntry) -> None:
    """Append a closed trade to the journal."""
    try:
        JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = []
        if JOURNAL_PATH.exists():
            data = json.loads(JOURNAL_PATH.read_text())
        data.append(asdict(entry))
        JOURNAL_PATH.write_text(json.dumps(data, indent=2))
        logger.info("[metrics] Trade journaled: {} pnl={:+.2f}", entry.symbol, entry.pnl)
    except Exception as e:
        logger.error("[metrics] record_trade failed: {}", e)


def get_adaptive_stats(min_trades: int = 10) -> dict:
    """
    Read the journal and compute real win rate, avg win/loss, expectancy.
    Falls back to sensible defaults until min_trades are recorded.
    """
    defaults = {
        "win_rate": 0.50, "avg_win": 0.0, "avg_loss": 0.0,
        "avg_win_loss_ratio": 1.5, "expectancy": 0.0,
        "profit_factor": 0.0, "n_trades": 0, "adaptive": False,
    }
    try:
        if not JOURNAL_PATH.exists():
            return defaults
        data = json.loads(JOURNAL_PATH.read_text())
        if len(data) < min_trades:
            defaults["n_trades"] = len(data)
            return defaults

        wins   = [t["pnl"] for t in data if t["pnl"] > 0]
        losses = [t["pnl"] for t in data if t["pnl"] <= 0]
        n = len(data)
        win_rate = len(wins) / n if n else 0.5
        avg_win  = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        ratio    = abs(avg_win / avg_loss) if avg_loss else 1.5
        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_win_loss_ratio": ratio,
            "expectancy": expectancy(win_rate, avg_win, abs(avg_loss)),
            "profit_factor": profit_factor(wins, [l for l in losses]),
            "n_trades": n,
            "adaptive": True,
        }
    except Exception as e:
        logger.error("[metrics] get_adaptive_stats failed: {}", e)
        return defaults
