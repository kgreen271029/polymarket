"""
Free market data provider using Yahoo Finance.
No API key required. Returns live prices and historical OHLCV bars.
Includes simple in-memory caching and retry/backoff to respect rate limits.
"""

import time
import logging
import requests


class DataProvider:
    """Fetches free market data from Yahoo Finance (no key needed)."""

    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"
    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}

    def __init__(self, logger, cache_ttl=60):
        self.logger = logger
        self.cache_ttl = cache_ttl          # seconds to cache a symbol's bars
        self._cache = {}                    # symbol -> (timestamp, payload)

    def _fetch(self, symbol, interval, rng, retries=3):
        """Raw fetch of a Yahoo chart payload with retry/backoff."""
        url = f"{self.BASE}{symbol}?interval={interval}&range={rng}"
        delay = 1.0
        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                if resp.status_code == 429:
                    self.logger.debug(f"{symbol}: rate limited, backing off {delay}s")
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                result = data.get("chart", {}).get("result")
                if result:
                    return result[0]
                return None
            except Exception as e:
                self.logger.debug(f"{symbol}: fetch attempt {attempt+1} failed: {e}")
                time.sleep(delay)
                delay *= 2
        return None

    def get_bars(self, symbol, interval="1d", rng="3mo"):
        """
        Return a dict with aligned OHLCV lists and the latest price.
        Cached for cache_ttl seconds per symbol+interval+range.
        """
        key = (symbol, interval, rng)
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]

        result = self._fetch(symbol, interval, rng)
        if not result:
            return None

        try:
            quote = result["indicators"]["quote"][0]
            timestamps = result.get("timestamp", [])
            opens, highs, lows, closes, volumes = [], [], [], [], []
            ts = []
            for i in range(len(timestamps)):
                c = quote["close"][i]
                if c is None:
                    continue  # skip incomplete bars (holidays, halts)
                ts.append(timestamps[i])
                opens.append(quote["open"][i] if quote["open"][i] is not None else c)
                highs.append(quote["high"][i] if quote["high"][i] is not None else c)
                lows.append(quote["low"][i] if quote["low"][i] is not None else c)
                closes.append(c)
                volumes.append(quote["volume"][i] if quote["volume"][i] is not None else 0)

            if not closes:
                return None

            meta = result.get("meta", {})
            payload = {
                "symbol": symbol,
                "timestamp": ts,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "price": meta.get("regularMarketPrice", closes[-1]),
                "prev_close": meta.get("chartPreviousClose", closes[-2] if len(closes) > 1 else closes[-1]),
            }
            self._cache[key] = (now, payload)
            return payload
        except Exception as e:
            self.logger.debug(f"{symbol}: parse failed: {e}")
            return None

    def get_price(self, symbol):
        """Latest price for a symbol, or None."""
        bars = self.get_bars(symbol, interval="1d", rng="5d")
        return bars["price"] if bars else None
