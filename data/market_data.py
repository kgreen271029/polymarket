"""
data/market_data.py

Wraps Alpaca market data APIs (historical + streaming) and defines
the shared dataclasses used across the data ingestion layer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Shared dataclasses (re-exported from data/__init__.py)
# ---------------------------------------------------------------------------


@dataclass
class HeadlineEvent:
    title: str
    summary: str
    source: str
    published_at: datetime
    symbols_mentioned: list[str]
    raw_url: str
    urgency_score: float  # 0.0 to 1.0


@dataclass
class SocialSignal:
    symbol: str
    mention_count: int
    velocity_ratio: float  # current_30min / previous_30min
    avg_sentiment: float   # -1.0 to 1.0
    top_post_title: str
    platform: str          # "reddit"


@dataclass
class NewCoinEvent:
    coin_id: str           # CoinGecko ID
    symbol: str            # e.g. "BONK"
    name: str
    days_old: int
    trending_rank: Optional[int]
    social_velocity: float
    coingecko_url: str


# ---------------------------------------------------------------------------
# Alpaca timeframe mapping
# ---------------------------------------------------------------------------

_TIMEFRAME_MAP: dict[str, object] = {}  # populated lazily after imports


def _resolve_timeframe(timeframe: str) -> object:
    """Return the Alpaca TimeFrame enum value for a string like '1Min'."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    mapping = {
        "1Min":  TimeFrame(1, TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day":  TimeFrame(1, TimeFrameUnit.Day),
    }
    if timeframe not in mapping:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Valid options: {list(mapping.keys())}"
        )
    return mapping[timeframe]


# ---------------------------------------------------------------------------
# Helper: detect crypto vs stock symbol
# ---------------------------------------------------------------------------


def _is_crypto_symbol(symbol: str) -> bool:
    """Return True when the symbol looks like a crypto pair."""
    return "/" in symbol or symbol.upper().endswith("USD")


def _normalize_alpaca_crypto(symbol: str) -> str:
    """
    Convert user-facing crypto symbol to Alpaca's format.

    'BTC/USD' → 'BTC/USD'   (Alpaca crypto uses '/' natively for streams)
    'BTCUSD'  → 'BTC/USD'

    For historical requests Alpaca accepts both; for streams it wants 'BTC/USD'.
    """
    symbol = symbol.upper()
    if "/" in symbol:
        return symbol
    # Try to split trailing USD / USDT / USDC
    for quote in ("USDT", "USDC", "USD"):
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            return f"{base}/{quote}"
    return symbol


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MarketDataFeed:
    """
    Unified interface to Alpaca market data (historical + streaming).

    Parameters
    ----------
    api_key : str
        Alpaca API key ID.
    api_secret : str
        Alpaca secret key.
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
        from alpaca.data.live import CryptoDataStream, StockDataStream

        self._api_key = api_key
        self._api_secret = api_secret

        self.stock_client = StockHistoricalDataClient(
            api_key=api_key, secret_key=api_secret
        )
        self.crypto_client = CryptoHistoricalDataClient(
            api_key=api_key, secret_key=api_secret
        )
        self.crypto_stream = CryptoDataStream(
            api_key=api_key, secret_key=api_secret
        )
        self.stock_stream = StockDataStream(
            api_key=api_key, secret_key=api_secret
        )

        logger.info("MarketDataFeed initialised (Alpaca)")

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a stock or crypto symbol.

        Parameters
        ----------
        symbol : str
            e.g. "AAPL", "BTC/USD", "BTCUSD"
        timeframe : str
            One of "1Min", "5Min", "15Min", "1Hour", "1Day"
        limit : int
            Maximum number of bars to return.

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume, vwap, timestamp
        """
        tf = _resolve_timeframe(timeframe)
        is_crypto = _is_crypto_symbol(symbol)

        try:
            if is_crypto:
                df = await self._get_crypto_bars(symbol, tf, limit)
            else:
                df = await self._get_stock_bars(symbol, tf, limit)
        except Exception as exc:
            logger.error(f"get_bars({symbol}, {timeframe}) failed: {exc}")
            raise

        return df

    async def _get_crypto_bars(
        self, symbol: str, timeframe: object, limit: int
    ) -> pd.DataFrame:
        from alpaca.data.requests import CryptoBarsRequest

        alpaca_symbol = _normalize_alpaca_crypto(symbol)
        request = CryptoBarsRequest(
            symbol_or_symbols=alpaca_symbol,
            timeframe=timeframe,
            limit=limit,
        )
        bars = await asyncio.to_thread(
            self.crypto_client.get_crypto_bars, request
        )
        return self._bars_to_df(bars, alpaca_symbol)

    async def _get_stock_bars(
        self, symbol: str, timeframe: object, limit: int
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest

        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=timeframe,
            limit=limit,
        )
        bars = await asyncio.to_thread(
            self.stock_client.get_stock_bars, request
        )
        return self._bars_to_df(bars, symbol.upper())

    def _bars_to_df(self, bars: object, symbol: str) -> pd.DataFrame:
        """Normalise an Alpaca BarSet/DataFrame into our standard schema."""
        # alpaca-py returns a dict-like BarSet; convert to DataFrame
        try:
            df = bars.df
        except AttributeError:
            # Some versions return a dict; handle that
            raw = bars[symbol] if hasattr(bars, "__getitem__") else bars
            records = []
            for bar in raw:
                records.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "vwap": getattr(bar, "vwap", None),
                    }
                )
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
            return df

        # If multi-level index (symbol, timestamp) → drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)

        df.index.name = "timestamp"

        # Rename to standard column names
        rename_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "vwap": "vwap",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure vwap column exists
        if "vwap" not in df.columns:
            df["vwap"] = None

        # Keep only our standard columns
        keep = [c for c in ["open", "high", "low", "close", "volume", "vwap"] if c in df.columns]
        df = df[keep].copy()
        df["timestamp"] = df.index

        # Preserve DatetimeIndex so strategies can filter by time (ORB session filter, etc.)
        return df

    # ------------------------------------------------------------------
    # Snapshot (latest quote + trade)
    # ------------------------------------------------------------------

    async def get_snapshot(self, symbol: str) -> dict:
        """
        Return latest snapshot data for a symbol.

        Returns
        -------
        dict with keys: symbol, bid, ask, last_price, timestamp
        """
        is_crypto = _is_crypto_symbol(symbol)
        try:
            if is_crypto:
                return await self._get_crypto_snapshot(symbol)
            else:
                return await self._get_stock_snapshot(symbol)
        except Exception as exc:
            logger.error(f"get_snapshot({symbol}) failed: {exc}")
            raise

    async def _get_crypto_snapshot(self, symbol: str) -> dict:
        from alpaca.data.requests import CryptoSnapshotRequest

        alpaca_symbol = _normalize_alpaca_crypto(symbol)
        request = CryptoSnapshotRequest(symbol_or_symbols=alpaca_symbol)
        result = await asyncio.to_thread(
            self.crypto_client.get_crypto_snapshot, request
        )
        snap = result[alpaca_symbol] if isinstance(result, dict) else result
        return {
            "symbol": alpaca_symbol,
            "bid": getattr(snap.latest_quote, "bid_price", None),
            "ask": getattr(snap.latest_quote, "ask_price", None),
            "last_price": getattr(snap.latest_trade, "price", None),
            "timestamp": getattr(snap.latest_trade, "timestamp", None),
        }

    async def _get_stock_snapshot(self, symbol: str) -> dict:
        from alpaca.data.requests import StockSnapshotRequest

        sym = symbol.upper()
        request = StockSnapshotRequest(symbol_or_symbols=sym)
        result = await asyncio.to_thread(
            self.stock_client.get_stock_snapshot, request
        )
        snap = result[sym] if isinstance(result, dict) else result
        return {
            "symbol": sym,
            "bid": getattr(snap.latest_quote, "bid_price", None),
            "ask": getattr(snap.latest_quote, "ask_price", None),
            "last_price": getattr(snap.latest_trade, "price", None),
            "timestamp": getattr(snap.latest_trade, "timestamp", None),
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def subscribe_crypto(
        self,
        symbols: list[str],
        bar_callback: Callable,
    ) -> None:
        """
        Subscribe to real-time crypto bars via WebSocket.

        Reconnects on disconnect with exponential backoff (1s → 60s).

        Parameters
        ----------
        symbols : list[str]
            e.g. ["BTC/USD", "ETH/USD"]
        bar_callback : async callable
            Called with each incoming bar object.
        """
        alpaca_symbols = [_normalize_alpaca_crypto(s) for s in symbols]

        delay = 1.0
        max_delay = 60.0

        while True:
            try:
                logger.info(f"Subscribing to crypto bars: {alpaca_symbols}")
                self.crypto_stream.subscribe_bars(bar_callback, *alpaca_symbols)
                await self.crypto_stream.run()  # blocks until disconnect
            except asyncio.CancelledError:
                logger.info("Crypto stream cancelled — shutting down")
                return
            except Exception as exc:
                logger.warning(
                    f"Crypto stream disconnected ({exc}). "
                    f"Reconnecting in {delay:.0f}s …"
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                # Recreate stream object to get fresh connection
                from alpaca.data.live import CryptoDataStream

                self.crypto_stream = CryptoDataStream(
                    api_key=self._api_key, secret_key=self._api_secret
                )
            else:
                # Clean exit — reset backoff
                delay = 1.0

    async def subscribe_stocks(
        self,
        symbols: list[str],
        bar_callback: Callable,
    ) -> None:
        """
        Subscribe to real-time stock bars via WebSocket.

        Only meaningful during market hours; reconnects with exponential backoff.

        Parameters
        ----------
        symbols : list[str]
            e.g. ["AAPL", "TSLA"]
        bar_callback : async callable
            Called with each incoming bar object.
        """
        syms = [s.upper() for s in symbols]

        delay = 1.0
        max_delay = 60.0

        while True:
            try:
                logger.info(f"Subscribing to stock bars: {syms}")
                self.stock_stream.subscribe_bars(bar_callback, *syms)
                await self.stock_stream.run()
            except asyncio.CancelledError:
                logger.info("Stock stream cancelled — shutting down")
                return
            except Exception as exc:
                logger.warning(
                    f"Stock stream disconnected ({exc}). "
                    f"Reconnecting in {delay:.0f}s …"
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                from alpaca.data.live import StockDataStream

                self.stock_stream = StockDataStream(
                    api_key=self._api_key, secret_key=self._api_secret
                )
            else:
                delay = 1.0

    # ------------------------------------------------------------------
    # Symbol helpers (public)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """
        Convert 'BTC/USD' → 'BTCUSD' (Alpaca's concatenated crypto format).

        Most Alpaca historical endpoints accept both; this is kept for
        callers that need the concatenated form explicitly.
        """
        return symbol.replace("/", "").upper()
