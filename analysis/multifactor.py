"""
analysis/multifactor.py — Multi-factor stock ranking (research Tier 1, #1 impact).

Combines momentum, trend, relative strength, and volatility-contraction into a
single composite score to rank candidates. Only the strongest stocks trade.

Academic basis: 12-month momentum ~17% annual (Antonacci); multi-factor
confirmation lifts win rate from ~35% (single signal) to 55%+.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def momentum_factor(close: pd.Series) -> float:
    """Blended 1/3/6-month momentum, skipping the most recent week (reversal)."""
    if len(close) < 130:
        return 0.0
    c = close.values
    # skip last 5 days to avoid short-term reversal
    ret_1m = c[-5] / c[-26] - 1 if len(c) > 26 else 0
    ret_3m = c[-5] / c[-68] - 1 if len(c) > 68 else 0
    ret_6m = c[-5] / c[-131] - 1 if len(c) > 131 else 0
    return ret_1m * 0.20 + ret_3m * 0.50 + ret_6m * 0.30


def trend_factor(df: pd.DataFrame) -> float:
    """0-1 score: price above rising 50/200 SMAs (Stage-2 advancing)."""
    close = df["close"]
    if len(close) < 60:
        return 0.0
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(min(200, len(close) - 1)).mean()
    price = float(close.iloc[-1])
    s50 = float(sma50.iloc[-1])
    s200 = float(sma200.iloc[-1])
    if np.isnan(s50) or np.isnan(s200):
        return 0.0

    score = 0.0
    if price > s50:  score += 0.4
    if s50 > s200:   score += 0.3
    if price > s200: score += 0.3
    # slope bonus
    if len(sma50) > 20:
        slope = (s50 - float(sma50.iloc[-20])) / s50
        score += max(-0.2, min(0.2, slope * 5))
    return max(0.0, min(1.0, score))


def relative_strength(close: pd.Series, spy: pd.Series, period: int = 63) -> float:
    """Stock 3-month return minus SPY 3-month return (positive = leader)."""
    if len(close) < period + 1 or len(spy) < period + 1:
        return 0.0
    stock_ret = float(close.iloc[-1]) / float(close.iloc[-period]) - 1
    spy_ret = float(spy.iloc[-1]) / float(spy.iloc[-period]) - 1
    return stock_ret - spy_ret


def volatility_contraction(df: pd.DataFrame) -> float:
    """1.0 if recent ATR is contracting vs longer ATR (VCP setup), else lower."""
    if len(df) < 55:
        return 0.3
    atr20 = _atr(df, 20).iloc[-1]
    atr50 = _atr(df, 50).iloc[-1]
    if np.isnan(atr20) or np.isnan(atr50) or atr50 == 0:
        return 0.3
    ratio = atr20 / atr50
    # tighter recent range = higher score
    if ratio < 0.7:   return 1.0
    if ratio < 0.85:  return 0.7
    if ratio < 1.0:   return 0.5
    return 0.3


def score_symbol(df: pd.DataFrame, spy: pd.DataFrame) -> dict:
    """Return component + composite scores for a single symbol."""
    close = df["close"]
    spy_close = spy["close"]

    mom = momentum_factor(close)
    trn = trend_factor(df)
    rs  = relative_strength(close, spy_close)
    vcp = volatility_contraction(df)

    # Normalise momentum/RS into 0-1-ish via simple scaling
    mom_n = max(0.0, min(1.0, (mom + 0.1) / 0.4))   # -10%..+30% → 0..1
    rs_n  = max(0.0, min(1.0, (rs + 0.1) / 0.3))    # -10%..+20% → 0..1

    composite = mom_n * 0.35 + trn * 0.25 + vcp * 0.20 + rs_n * 0.20
    return {
        "momentum": mom, "trend": trn, "rel_strength": rs,
        "vcp": vcp, "composite": composite * 100,
    }


def rank_universe(stock_data: dict[str, pd.DataFrame], spy: pd.DataFrame,
                  top_n: int = 6) -> list[tuple[str, dict]]:
    """Rank all symbols by composite score; return top_n (symbol, scores)."""
    scored = []
    for sym, df in stock_data.items():
        if sym == "SPY" or df is None or len(df) < 60:
            continue
        try:
            scored.append((sym, score_symbol(df, spy)))
        except Exception as e:
            logger.debug("[multifactor] {} error: {}", sym, e)
    scored.sort(key=lambda x: x[1]["composite"], reverse=True)
    return scored[:top_n]


def chandelier_exit(df: pd.DataFrame, atr_mult: float = 2.5,
                    lookback: int = 22) -> float:
    """
    Chandelier Exit: highest-high(lookback) - atr_mult * ATR.
    Volatility-adjusted trailing stop that beats fixed % stops by 15-25%.
    """
    if len(df) < lookback:
        return float(df["close"].iloc[-1]) * 0.92
    hh = float(df["high"].tail(lookback).max())
    atr = float(_atr(df, 14).iloc[-1])
    if np.isnan(atr):
        atr = float(df["close"].iloc[-1]) * 0.02
    return hh - atr_mult * atr
