"""
analysis/filters.py — Smart pre-trade filters and position sizing.

All functions are free-data only (yfinance, no API keys needed).
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sector ETF map
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    # Tech
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "GOOGL": "XLK", "META": "XLK", "AMZN": "XLK", "ARM": "XLK",
    # Finance
    "JPM": "XLF", "GS": "XLF", "BAC": "XLF", "SOFI": "XLF", "COIN": "XLF",
    # Healthcare
    "GILD": "XLV",
    # Industrials / Energy
    "SMR": "XLI", "FLNC": "XLE",
    # Misc large cap
    "TSLA": "XLY", "PLTR": "XLK", "SMCI": "XLK", "MSTR": "XLK",
}

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLC", "XLB", "XLRE"]


# ---------------------------------------------------------------------------
# 1. Earnings blackout
# ---------------------------------------------------------------------------

def earnings_blackout(symbol: str, window_days: int = 14) -> tuple[bool, str]:
    """
    Returns (blocked, reason). Blocks trades within window_days of earnings.
    Free via yfinance calendar.
    """
    try:
        import yfinance as yf
        cal = yf.Ticker(symbol).calendar
        if cal is None or cal.empty:
            return False, "no earnings date found"
        # calendar returns a DataFrame; earnings date is in the index or columns
        if hasattr(cal, "columns"):
            # Newer yfinance: DataFrame with dates in columns
            dates = []
            for col in cal.columns:
                try:
                    d = pd.to_datetime(col)
                    dates.append(d)
                except Exception:
                    pass
        else:
            dates = [pd.to_datetime(cal)]

        today = pd.Timestamp.now()
        for d in dates:
            days_away = (d - today).days
            if -3 <= days_away <= window_days:
                return True, f"earnings in {days_away}d ({d.date()})"
        return False, "clear of earnings"
    except Exception as e:
        logger.debug("[filters] earnings_blackout({}) error: {}", symbol, e)
        return False, "could not check"


# ---------------------------------------------------------------------------
# 2. Sector momentum
# ---------------------------------------------------------------------------

def sector_momentum_ok(symbol: str, period: int = 20) -> tuple[bool, str]:
    """
    Returns (ok, reason). Rejects trade if the stock's sector ETF is below its
    20-day SMA (don't fight a down sector).
    """
    try:
        import yfinance as yf
        etf = SECTOR_MAP.get(symbol.upper(), "SPY")
        hist = yf.download(etf, period="40d", auto_adjust=True, progress=False)
        if hist.empty or len(hist) < period:
            return True, f"{etf} data unavailable — passing"
        # Flatten MultiIndex
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0].lower() for c in hist.columns]
        else:
            hist.columns = [c.lower() for c in hist.columns]
        close = hist["close"]
        sma = float(close.rolling(period).mean().iloc[-1])
        price = float(close.iloc[-1])
        ok = price >= sma
        pct = (price / sma - 1) * 100
        return ok, f"{etf} {'above' if ok else 'BELOW'} SMA20 ({pct:+.1f}%)"
    except Exception as e:
        logger.debug("[filters] sector_momentum_ok({}) error: {}", symbol, e)
        return True, "could not check — passing"


# ---------------------------------------------------------------------------
# 3. Volatility regime
# ---------------------------------------------------------------------------

def volatility_regime_ok(symbol: str = "SPY") -> tuple[bool, str]:
    """
    Returns (ok, reason). Skip trading when market volatility is very low
    AND falling (choppy, whipsaw regime). Uses ATR trend on SPY.
    """
    try:
        import yfinance as yf
        hist = yf.download(symbol, period="40d", auto_adjust=True, progress=False)
        if hist.empty or len(hist) < 20:
            return True, "data unavailable — passing"
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0].lower() for c in hist.columns]
        else:
            hist.columns = [c.lower() for c in hist.columns]

        h, l, c = hist["high"], hist["low"], hist["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(com=13, min_periods=14).mean()

        current_atr = float(atr.iloc[-1])
        ma_atr      = float(atr.tail(15).mean())
        rising      = float(atr.iloc[-1]) > float(atr.iloc[-5])

        choppy = current_atr < (ma_atr * 0.75) and not rising
        if choppy:
            return False, f"choppy regime — ATR {current_atr:.2f} < MA {ma_atr:.2f}"
        return True, f"ATR {current_atr:.2f} (MA {ma_atr:.2f}, {'rising' if rising else 'stable'})"
    except Exception as e:
        logger.debug("[filters] volatility_regime_ok error: {}", e)
        return True, "could not check — passing"


# ---------------------------------------------------------------------------
# 4. Top sectors (for EOD scan prioritisation)
# ---------------------------------------------------------------------------

def top_sectors(n: int = 3, lookback_days: int = 20) -> list[str]:
    """Return the top-N performing sector ETF tickers over lookback_days."""
    try:
        import yfinance as yf
        results = {}
        for etf in SECTOR_ETFS:
            try:
                h = yf.download(etf, period=f"{lookback_days + 5}d",
                                auto_adjust=True, progress=False)
                if h.empty or len(h) < 5:
                    continue
                if isinstance(h.columns, pd.MultiIndex):
                    h.columns = [c[0].lower() for c in h.columns]
                else:
                    h.columns = [c.lower() for c in h.columns]
                ret = float(h["close"].iloc[-1]) / float(h["close"].iloc[-lookback_days]) - 1
                results[etf] = ret
            except Exception:
                pass
        ranked = sorted(results, key=results.get, reverse=True)
        return ranked[:n]
    except Exception as e:
        logger.debug("[filters] top_sectors error: {}", e)
        return ["XLK", "XLF", "XLV"]


def symbol_in_top_sectors(symbol: str, top: list[str]) -> bool:
    """True if symbol belongs to one of the top-performing sectors."""
    etf = SECTOR_MAP.get(symbol.upper())
    return etf in top if etf else True   # unknown sectors pass through


# ---------------------------------------------------------------------------
# 5. News catalyst scoring
# ---------------------------------------------------------------------------

_POSITIVE_CATALYSTS = {
    "beats earnings": 15, "beat earnings": 15, "earnings beat": 15,
    "fda approval": 20, "fda approved": 20,
    "partnership": 12, "acquisition": 18, "acquires": 18, "deal": 8,
    "buyback": 10, "share repurchase": 10,
    "dividend increase": 8, "raised guidance": 12, "guidance raised": 12,
    "analyst upgrade": 12, "price target raised": 10, "outperform": 8,
    "record revenue": 12, "revenue growth": 8, "contract win": 12,
    "breakthrough": 10, "launch": 7, "ipo": 12,
}
_NEGATIVE_CATALYSTS = {
    "misses earnings": -15, "earnings miss": -15,
    "fda rejection": -20, "recall": -15, "lawsuit": -10,
    "guidance cut": -12, "lowers guidance": -12,
    "analyst downgrade": -12, "sell rating": -12,
    "investigation": -10, "subpoena": -10, "fraud": -18,
    "bankruptcy": -25, "default": -20,
}
_NOISE = {
    "market tumbles": -8, "sector falls": -8, "stocks slip": -5,
    "concerns over": -5, "fears": -5, "uncertainty": -3,
}


def score_catalyst(headline: str, symbol: str, published_at: datetime | None = None) -> int:
    """
    Score a news headline 0–100 for actionability.
    >60 = strong catalyst worth acting on.
    """
    hl = headline.lower()
    score = 40  # neutral baseline

    for kw, pts in _POSITIVE_CATALYSTS.items():
        if kw in hl:
            score += pts
    for kw, pts in _NEGATIVE_CATALYSTS.items():
        if kw in hl:
            score += pts
    for kw, pts in _NOISE.items():
        if kw in hl:
            score += pts

    # Recency bonus
    if published_at:
        hours_old = (datetime.now() - published_at.replace(tzinfo=None)).total_seconds() / 3600
        if hours_old < 1:
            score += 20
        elif hours_old < 4:
            score += 10
        elif hours_old > 48:
            score -= 20

    # Symbol directly named in headline (not just sector noise)
    if symbol.upper() in headline.upper():
        score += 10

    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# 6. Kelly / ATR position sizing
# ---------------------------------------------------------------------------

def kelly_position_size(
    account_cash: float,
    entry_price: float,
    stop_price: float,
    win_rate: float = 0.50,
    avg_win_loss_ratio: float = 2.0,
    max_pct: float = 0.30,
) -> float:
    """
    Fractional Kelly position sizing.
    Returns dollar amount to invest (capped at max_pct of account).
    """
    if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        return account_cash * 0.10  # minimal default

    risk_per_share = entry_price - stop_price
    risk_pct       = risk_per_share / entry_price

    # Kelly: f = (b*p - q) / b  where b = win/loss ratio
    b = avg_win_loss_ratio
    p = win_rate
    q = 1 - p
    kelly_f = max(0.0, (b * p - q) / b)

    # Use half-Kelly for safety, cap at max_pct
    fraction = min(kelly_f * 0.5, max_pct)
    dollar_size = account_cash * fraction

    # Also cap so max risk is 2% of account
    max_risk_dollars = account_cash * 0.02
    if risk_pct > 0:
        dollar_size = min(dollar_size, max_risk_dollars / risk_pct)

    return max(0.0, dollar_size)


def atr_trailing_stop(
    entry_price: float,
    atr: float,
    highest_since_entry: float,
    multiplier: float = 1.5,
) -> float:
    """
    Returns current trailing stop price.
    Trails up as price rises; never moves down.
    """
    trail = highest_since_entry - (atr * multiplier)
    initial = entry_price - (atr * multiplier)
    return max(trail, initial, entry_price * 0.90)   # floor: -10%


# ---------------------------------------------------------------------------
# 7. RSI divergence exit signal
# ---------------------------------------------------------------------------

def bearish_divergence(prices: pd.Series, rsi: pd.Series, lookback: int = 10) -> bool:
    """
    True if price makes a new high but RSI fails to confirm (bearish divergence).
    Reliable swing exit signal.
    """
    if len(prices) < lookback or len(rsi) < lookback:
        return False
    p = prices.iloc[-lookback:].values
    r = rsi.iloc[-lookback:].values
    if len(p) < 4:
        return False

    # Price at new high but RSI lower than at previous high
    prev_high_idx = np.argmax(p[:-2])
    current_price = p[-1]
    current_rsi   = r[-1]
    prev_high_rsi = r[prev_high_idx]

    return bool(current_price >= p[prev_high_idx] and current_rsi < prev_high_rsi - 3)
