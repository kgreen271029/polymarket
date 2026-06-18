"""
data/yfinance_data.py — yfinance-backed market data feed.

Drop-in replacement for MarketDataFeed when Alpaca keys are not configured.
Provides the same get_bars() / get_snapshot() interface.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

_INTERVAL_MAP = {
    "1Min":  ("1m",  8),    # yfinance: (interval, max_days)
    "5Min":  ("5m",  60),
    "15Min": ("15m", 60),
    "1Hour": ("1h",  730),
    "1Day":  ("1d",  None),
}

# Cache TTLs by timeframe (seconds) — intraday bars expire faster than daily
_CACHE_TTL: dict[str, int] = {
    "1Min":  30,
    "5Min":  60,
    "15Min": 120,
    "1Hour": 300,
    "1Day":  300,
}


class YFinanceDataFeed:
    """yfinance-backed market data (free, no API key) with per-symbol TTL cache."""

    def __init__(self) -> None:
        # {(symbol, timeframe, limit): (DataFrame, expiry_ts)}
        self._cache: dict[tuple, tuple[Any, float]] = {}

    def _cache_get(self, key: tuple) -> pd.DataFrame | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        df, expiry = entry
        if time.monotonic() > expiry:
            del self._cache[key]
            return None
        return df

    def _cache_set(self, key: tuple, df: pd.DataFrame, ttl: int) -> None:
        self._cache[key] = (df, time.monotonic() + ttl)
        # Evict stale entries if cache grows large
        if len(self._cache) > 200:
            now = time.monotonic()
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 100,
    ) -> pd.DataFrame | None:
        key = (symbol, timeframe, limit)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        df = await asyncio.to_thread(self._get_bars_sync, symbol, timeframe, limit)
        if df is not None:
            ttl = _CACHE_TTL.get(timeframe, 60)
            self._cache_set(key, df, ttl)
        return df

    def _get_bars_sync(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        try:
            import yfinance as yf
            import warnings
            warnings.filterwarnings("ignore")

            interval, max_days = _INTERVAL_MAP.get(timeframe, ("1d", None))
            if max_days is not None:
                days = min(limit * 2, max_days)
            else:
                days = max(limit * 2, 90)

            end   = datetime.now()
            start = end - timedelta(days=days)
            raw   = yf.download(symbol, start=start, end=end,
                                 interval=interval, auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                return None

            # Flatten MultiIndex columns (yfinance ≥0.2)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]

            raw = raw.rename(columns={"adj_close": "close"})
            needed = ["open", "high", "low", "close", "volume"]
            if not all(c in raw.columns for c in needed):
                return None

            df = raw[needed].dropna().tail(limit).copy()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.warning("[YFinanceDataFeed] get_bars({}, {}): {}", symbol, timeframe, e)
            return None

    async def get_snapshot(self, symbol: str) -> dict:
        return await asyncio.to_thread(self._get_snapshot_sync, symbol)

    def _get_snapshot_sync(self, symbol: str) -> dict:
        try:
            import yfinance as yf
            import warnings
            warnings.filterwarnings("ignore")

            ticker = yf.Ticker(symbol)
            info   = ticker.fast_info
            price  = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            return {
                "symbol":     symbol,
                "last_price": float(price) if price else None,
                "bid":        None,
                "ask":        None,
                "timestamp":  datetime.now(),
            }
        except Exception as e:
            logger.warning("[YFinanceDataFeed] get_snapshot({}): {}", symbol, e)
            return {"symbol": symbol, "last_price": None, "bid": None, "ask": None, "timestamp": None}

    # stream_loop stub — not needed for daily strategies, prevents AttributeError
    async def stream_loop(self) -> None:
        while True:
            await asyncio.sleep(3600)
