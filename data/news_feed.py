"""
data/news_feed.py

Multi-source news aggregator (NewsAPI + RSS feeds) that polls every 3 minutes,
deduplicates events, scores urgency, and routes breaking news to a separate queue.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import aiohttp
import feedparser
from loguru import logger

from data.market_data import HeadlineEvent

# ---------------------------------------------------------------------------
# Known ticker / crypto symbol dictionary used for extraction
# ---------------------------------------------------------------------------

# A representative set — expand as needed
_STOCK_TICKERS: set[str] = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "AMD",
    "INTC", "NFLX", "PYPL", "SQ", "SHOP", "SNAP", "TWTR", "UBER", "LYFT",
    "COIN", "HOOD", "SOFI", "PLTR", "ROKU", "ZM", "DOCU", "CRWD", "DDOG",
    "NET", "SNOW", "ABNB", "DASH", "RIVN", "LCID", "NIO", "BIDU", "JD",
    "BABA", "PDD", "TSM", "ASML", "QCOM", "MU", "AMAT", "LRCX", "KLAC",
    "TXN", "AVGO", "MRVL", "MCHP", "ON", "STX", "WDC", "PSTG", "PANW",
    "FTNT", "ZS", "OKTA", "CYBR", "S", "VRNS", "QLYS", "TENB", "RPM",
    "JPM", "BAC", "GS", "MS", "C", "WFC", "USB", "PNC", "TFC", "COF",
    "AXP", "V", "MA", "DIS", "CMCSA", "T", "VZ", "TMUS", "GME", "AMC",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "XLE", "XLF",
}

_CRYPTO_NAMES_AND_SYMBOLS: dict[str, str] = {
    # name → symbol (both will be matched)
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "cardano": "ADA",
    "ada": "ADA",
    "ripple": "XRP",
    "xrp": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "polkadot": "DOT",
    "dot": "DOT",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "chainlink": "LINK",
    "link": "LINK",
    "polygon": "MATIC",
    "matic": "MATIC",
    "litecoin": "LTC",
    "ltc": "LTC",
    "shiba inu": "SHIB",
    "shib": "SHIB",
    "uniswap": "UNI",
    "uni": "UNI",
    "aave": "AAVE",
    "binance coin": "BNB",
    "bnb": "BNB",
    "tron": "TRX",
    "trx": "TRX",
    "near": "NEAR",
    "cosmos": "ATOM",
    "atom": "ATOM",
    "algo": "ALGO",
    "algorand": "ALGO",
    "stellar": "XLM",
    "xlm": "XLM",
    "monero": "XMR",
    "xmr": "XMR",
    "filecoin": "FIL",
    "fil": "FIL",
    "aptos": "APT",
    "apt": "APT",
    "arbitrum": "ARB",
    "arb": "ARB",
    "optimism": "OP",
    "injective": "INJ",
    "inj": "INJ",
    "sui": "SUI",
    "bonk": "BONK",
    "pepe": "PEPE",
    "floki": "FLOKI",
}

# All crypto symbols (upper-case) for fast lookup
_CRYPTO_SYMBOLS: set[str] = {v.upper() for v in _CRYPTO_NAMES_AND_SYMBOLS.values()}

# ---------------------------------------------------------------------------
# RSS feed sources
# ---------------------------------------------------------------------------

_RSS_FEEDS: list[tuple[str, str]] = [
    # Verified live as of June 2026 — tested with HTTP 200 and real content
    ("Bloomberg Markets",    "https://feeds.bloomberg.com/markets/news.rss"),
    ("Bloomberg Technology", "https://feeds.bloomberg.com/technology/news.rss"),
    ("Benzinga",             "https://www.benzinga.com/feed"),
    ("Seeking Alpha",        "https://seekingalpha.com/market_currents.xml"),
    ("CoinDesk",             "https://www.coindesk.com/arc/outboundfeeds/rss"),
    ("CoinTelegraph",        "https://cointelegraph.com/rss"),
    ("Decrypt",              "https://decrypt.co/feed"),
    ("PR Newswire",          "https://www.prnewswire.com/rss/news-releases-list.rss"),
    # Reuters discontinued open RSS — removed
]

# ---------------------------------------------------------------------------
# Urgency keyword lists
# ---------------------------------------------------------------------------

_HIGH_URGENCY_KEYWORDS: list[str] = [
    "breaking",
    "halt",
    "halted",
    "fda",
    "sec",
    "earnings",
    "miss",
    "beat",
    "crash",
    "ban",
    "lawsuit",
    "recall",
    "investigation",
    "acquisition",
    "merger",
]

# ---------------------------------------------------------------------------
# Ticker pattern: 1-5 uppercase letters optionally preceded by $ sign
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\$?([A-Z]{1,5})(?:[^A-Z/]|$)")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NewsFeed:
    """
    Aggregates news from NewsAPI and multiple RSS feeds every 3 minutes.

    Parameters
    ----------
    api_key : str
        NewsAPI key (leave empty string to skip NewsAPI).
    news_queue : asyncio.Queue
        Receives all new HeadlineEvent objects.
    breaking_queue : asyncio.Queue
        Receives only high-urgency HeadlineEvent objects (score > 0.7).
    """

    POLL_INTERVAL: int = 180  # seconds

    def __init__(
        self,
        api_key: str,
        news_queue: asyncio.Queue,
        breaking_queue: asyncio.Queue,
    ) -> None:
        self._api_key = api_key
        self._news_queue = news_queue
        self._breaking_queue = breaking_queue
        self._seen: set[str] = set()

        # Keywords passed to NewsAPI
        self._keywords: list[str] = [
            "crypto", "bitcoin", "ethereum", "stock market",
            "SEC", "FDA", "earnings", "acquisition", "IPO",
        ]

        logger.info("NewsFeed initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def poll_loop(self) -> None:
        """Run forever: poll all sources, deduplicate, score, and enqueue."""
        logger.info("NewsFeed poll_loop started")
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                logger.info("NewsFeed poll_loop cancelled")
                return
            except Exception as exc:
                logger.error(f"NewsFeed poll_loop unexpected error: {exc}")

            await asyncio.sleep(self.POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    async def _poll_once(self) -> None:
        """Single poll cycle: gather → deduplicate → enqueue."""
        tasks = [self._fetch_rss()]
        if self._api_key:
            tasks.append(self._fetch_newsapi(self._keywords))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events: list[HeadlineEvent] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"NewsFeed fetch task failed: {result}")
            elif isinstance(result, list):
                all_events.extend(result)

        new_events = self._deduplicate(all_events)
        logger.info(f"NewsFeed: {len(all_events)} fetched, {len(new_events)} new")

        for event in new_events:
            await self._news_queue.put(event)
            if event.urgency_score > 0.7:
                await self._breaking_queue.put(event)
                logger.warning(
                    f"BREAKING: [{event.source}] {event.title[:80]}"
                )

    # ------------------------------------------------------------------
    # NewsAPI
    # ------------------------------------------------------------------

    async def _fetch_newsapi(self, keywords: list[str]) -> list[HeadlineEvent]:
        """
        Fetch recent articles from NewsAPI.

        Returns an empty list (with a warning) if the API key is missing or invalid.
        """
        if not self._api_key:
            logger.warning("NewsAPI key not set — skipping NewsAPI fetch")
            return []

        query = " OR ".join(f'"{kw}"' for kw in keywords)
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": self._api_key,
        }
        url = "https://newsapi.org/v2/everything"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 401:
                        logger.warning("NewsAPI: invalid API key (401) — skipping")
                        return []
                    resp.raise_for_status()
                    data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.error(f"NewsAPI request failed: {exc}")
            return []
        except Exception as exc:
            logger.error(f"NewsAPI unexpected error: {exc}")
            return []

        events: list[HeadlineEvent] = []
        for article in data.get("articles", []):
            try:
                title: str = article.get("title") or ""
                summary: str = article.get("description") or ""
                source: str = (article.get("source") or {}).get("name") or "NewsAPI"
                raw_url: str = article.get("url") or ""
                pub_str: str = article.get("publishedAt") or ""

                published_at = self._parse_iso_datetime(pub_str)
                urgency = self._score_urgency(title, summary)
                symbols = self._extract_symbols(f"{title} {summary}")

                events.append(
                    HeadlineEvent(
                        title=title,
                        summary=summary,
                        source=source,
                        published_at=published_at,
                        symbols_mentioned=symbols,
                        raw_url=raw_url,
                        urgency_score=urgency,
                    )
                )
            except Exception as exc:
                logger.debug(f"NewsAPI article parse error: {exc}")

        return events

    # ------------------------------------------------------------------
    # RSS feeds
    # ------------------------------------------------------------------

    async def _fetch_rss(self) -> list[HeadlineEvent]:
        """Fetch all RSS feeds in parallel and merge results."""
        tasks = [self._fetch_single_rss(name, url) for name, url in _RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        events: list[HeadlineEvent] = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"RSS fetch error: {result}")
            elif isinstance(result, list):
                events.extend(result)
        return events

    async def _fetch_single_rss(self, source_name: str, url: str) -> list[HeadlineEvent]:
        """Fetch and parse a single RSS feed using feedparser (via asyncio.to_thread)."""
        try:
            feed = await asyncio.to_thread(feedparser.parse, url)
        except Exception as exc:
            logger.warning(f"RSS parse error [{source_name}]: {exc}")
            return []

        events: list[HeadlineEvent] = []
        for entry in feed.get("entries", []):
            try:
                title: str = entry.get("title") or ""
                summary: str = entry.get("summary") or entry.get("description") or ""
                raw_url: str = entry.get("link") or ""

                # Parse published date — feedparser gives 'published' or 'updated'
                published_at = self._parse_feedparser_date(entry)

                urgency = self._score_urgency(title, summary)
                symbols = self._extract_symbols(f"{title} {summary}")

                events.append(
                    HeadlineEvent(
                        title=title,
                        summary=summary,
                        source=source_name,
                        published_at=published_at,
                        symbols_mentioned=symbols,
                        raw_url=raw_url,
                        urgency_score=urgency,
                    )
                )
            except Exception as exc:
                logger.debug(f"RSS entry parse error [{source_name}]: {exc}")

        logger.debug(f"RSS [{source_name}]: {len(events)} entries")
        return events

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, events: list[HeadlineEvent]) -> list[HeadlineEvent]:
        """
        Remove duplicate events using SHA-256 of (title[:50] + source + date).

        Already-seen hashes are stored in self._seen across poll cycles.
        Capped at 10,000 entries to prevent unbounded memory growth over long runs.
        """
        unique: list[HeadlineEvent] = []
        for event in events:
            date_str = event.published_at.date().isoformat()
            key = f"{event.title[:50]}|{event.source}|{date_str}"
            digest = hashlib.sha256(key.encode()).hexdigest()
            if digest not in self._seen:
                self._seen.add(digest)
                unique.append(event)
        # Evict oldest entries when set grows too large (approximate FIFO via rebuild)
        if len(self._seen) > 10_000:
            self._seen = set(list(self._seen)[-5_000:])
        return unique

    # ------------------------------------------------------------------
    # Urgency scoring
    # ------------------------------------------------------------------

    def _score_urgency(self, title: str, summary: str) -> float:
        """
        Return 0.9 if high-urgency keywords are found, otherwise 0.3.

        Search is case-insensitive over the combined title + summary text.
        """
        combined = f"{title} {summary}".lower()
        for kw in _HIGH_URGENCY_KEYWORDS:
            if kw in combined:
                return 0.9
        return 0.3

    # ------------------------------------------------------------------
    # Symbol extraction
    # ------------------------------------------------------------------

    def _extract_symbols(self, text: str) -> list[str]:
        """
        Extract stock tickers and crypto symbols from free-form text.

        Matches:
        - Uppercase 1-5 char words (with optional leading $) that appear in
          the known stock ticker set.
        - Crypto full names (e.g. "bitcoin", "ethereum") and their symbols.
        """
        found: set[str] = set()

        # Uppercase version of text for ticker matching
        upper_text = text.upper()

        # Match $TICKER or all-caps words
        for match in _TICKER_RE.finditer(upper_text):
            token = match.group(1)
            if token in _STOCK_TICKERS:
                found.add(token)
            if token in _CRYPTO_SYMBOLS:
                found.add(token)

        # Match crypto names (lowercase search)
        lower_text = text.lower()
        for name, symbol in _CRYPTO_NAMES_AND_SYMBOLS.items():
            if name in lower_text:
                found.add(symbol)

        return sorted(found)

    # ------------------------------------------------------------------
    # Date parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_iso_datetime(dt_str: str) -> datetime:
        """Parse ISO 8601 string (NewsAPI format) to timezone-aware datetime."""
        if not dt_str:
            return datetime.now(tz=timezone.utc)
        try:
            # Python 3.11+ fromisoformat handles 'Z'; for older, replace manually
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(tz=timezone.utc)

    @staticmethod
    def _parse_feedparser_date(entry: dict) -> datetime:
        """Extract a datetime from a feedparser entry dict."""
        # feedparser provides 'published_parsed' as a time.struct_time
        for field in ("published_parsed", "updated_parsed"):
            parsed = entry.get(field)
            if parsed:
                import time
                ts = time.mktime(parsed)
                return datetime.fromtimestamp(ts, tz=timezone.utc)

        # Try raw string fields
        for field in ("published", "updated"):
            raw = entry.get(field)
            if raw:
                try:
                    return parsedate_to_datetime(raw)
                except Exception:
                    pass

        return datetime.now(tz=timezone.utc)
