"""
analysis/regime.py — Market regime detection (research Tier 1, #3 impact).

Lightweight regime classifier (no hmmlearn dependency) using:
  - SPY trend (price vs 50/200 SMA + slope)
  - Realised volatility percentile
  - Market breadth proxy (SPY distance from 200MA)

Regimes: BULL (trade momentum full size), BEAR (defensive, half size,
mean-reversion only), CHOPPY (minimal size or stand aside).

Research: momentum wins 65-70% in bull, only 30-35% in bear; avoiding
high-vol/choppy regimes cut losing trades ~40%.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")


@dataclass
class Regime:
    name: str            # "bull" | "bear" | "choppy"
    size_mult: float     # position-size multiplier (0.0-1.0)
    use_momentum: bool   # momentum vs mean-reversion bias
    detail: str

    @property
    def tradeable(self) -> bool:
        return self.size_mult > 0.0


def _fetch_spy(period: str = "1y") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        h = yf.download("SPY", period=period, auto_adjust=True, progress=False)
        if h.empty:
            return None
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0].lower() for c in h.columns]
        else:
            h.columns = [c.lower() for c in h.columns]
        return h
    except Exception as e:
        logger.debug("[regime] SPY fetch error: {}", e)
        return None


def detect_regime(spy: pd.DataFrame | None = None) -> Regime:
    """Classify the current market regime from SPY price action."""
    if spy is None:
        spy = _fetch_spy()
    if spy is None or len(spy) < 60:
        return Regime("bull", 1.0, True, "SPY unavailable — default bull")

    close = spy["close"]
    price = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(min(200, len(close) - 1)).mean().iloc[-1])

    # 50-day SMA slope over last 20 days
    sma50_series = close.rolling(50).mean()
    slope = 0.0
    if len(sma50_series) > 20 and not np.isnan(sma50_series.iloc[-20]):
        slope = (sma50 - float(sma50_series.iloc[-20])) / sma50

    # Realised volatility percentile (20-day vs 1-year)
    rets = close.pct_change()
    vol20 = float(rets.tail(20).std())
    vol_year = rets.tail(252) if len(rets) >= 252 else rets
    vol_pct = float((vol_year < vol20).mean()) if len(vol_year) else 0.5

    # Distance from 200MA (breadth proxy)
    dist200 = (price - sma200) / sma200 if sma200 else 0.0

    # ── Classification ──────────────────────────────────────────────────────
    bull = price > sma50 > sma200 and slope > 0
    bear = price < sma200 and slope < 0
    high_vol = vol_pct > 0.85

    if bull and not high_vol:
        return Regime("bull", 1.0, True,
                      f"SPY uptrend (+{dist200*100:.1f}% vs 200MA, vol pct {vol_pct:.0%})")
    if bear:
        # In a bear, trade small & favour mean-reversion
        return Regime("bear", 0.4, False,
                      f"SPY downtrend ({dist200*100:.1f}% vs 200MA) — defensive")
    if high_vol:
        return Regime("choppy", 0.3, False,
                      f"high volatility (vol pct {vol_pct:.0%}) — reduce size")
    # mild uptrend / transition
    return Regime("bull", 0.7, True,
                  f"mild uptrend ({dist200*100:+.1f}% vs 200MA) — partial size")
