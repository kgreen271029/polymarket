"""
data/social_feed.py

Reddit social sentiment tracker using the public JSON API (no OAuth).
Also cross-checks with CoinGecko trending for crypto signals.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from loguru import logger

from data.market_data import SocialSignal

# ---------------------------------------------------------------------------
# Known tickers / crypto names (shared subset — same as news_feed)
# ---------------------------------------------------------------------------

_STOCK_TICKERS: set[str] = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "AMD",
    "INTC", "NFLX", "PYPL", "SQ", "SHOP", "SNAP", "COIN", "HOOD", "SOFI",
    "PLTR", "ROKU", "ZM", "DOCU", "CRWD", "DDOG", "NET", "SNOW", "ABNB",
    "DASH", "RIVN", "LCID", "NIO", "BIDU", "JD", "BABA", "PDD", "TSM",
    "ASML", "QCOM", "MU", "AMAT", "TXN", "AVGO", "MRVL", "PANW", "FTNT",
    "ZS", "OKTA", "JPM", "BAC", "GS", "MS", "C", "WFC", "COF", "AXP",
    "V", "MA", "DIS", "CMCSA", "T", "VZ", "TMUS", "GME", "AMC",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO",
}

_CRYPTO_NAMES_AND_SYMBOLS: dict[str, str] = {
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

_CRYPTO_SYMBOLS: set[str] = {v.upper() for v in _CRYPTO_NAMES_AND_SYMBOLS.values()}

# Ticker regex: $TICKER or bare ALL-CAPS word
_TICKER_RE = re.compile(r"\$?([A-Z]{1,6})(?:[^A-Z/]|$)")

# ---------------------------------------------------------------------------
# Sentiment word sets (simple lexicon-based)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS: frozenset[str] = frozenset([
    "moon", "mooning", "bull", "bullish", "pump", "pumping", "surge", "rally",
    "buy", "long", "gain", "profit", "winner", "undervalued", "cheap", "hodl",
    "hold", "breakout", "ath", "accumulate", "growth", "green", "up", "rise",
    "flying", "explode", "skyrocket", "opportunity", "gem", "dip", "dca",
])

_NEGATIVE_WORDS: frozenset[str] = frozenset([
    "crash", "dump", "bear", "bearish", "sell", "short", "loss", "scam",
    "rug", "rug pull", "dead", "worthless", "overvalued", "fraud", "exit",
    "down", "fall", "collapse", "plunge", "capitulate", "rekt", "liquidated",
    "ponzi", "bubble", "fear", "warning", "caution", "avoid", "red",
])

# Window for velocity history: store (timestamp, count) tuples
_VELOCITY_WINDOW_SECONDS: int = 1800  # 30 minutes


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SocialFeed:
    """
    Reddit social sentiment tracker.

    Polls subreddits every 300 seconds, extracts symbol mentions, calculates
    velocity ratios, and emits SocialSignal objects to social_queue.

    Also cross-checks with CoinGecko trending coins.

    Parameters
    ----------
    social_queue : asyncio.Queue
        Receives SocialSignal objects for symbols with velocity_ratio > 2.0
        and for all CoinGecko trending coins.
    """

    POLL_INTERVAL: int = 300  # seconds

    SUBREDDITS: list[str] = [
        "CryptoCurrency",
        "wallstreetbets",
        "investing",
        "stocks",
        "Bitcoin",
        "ethereum",
    ]

    # Minimum posts mentioning a symbol to be considered a real signal
    _MIN_MENTION_COUNT: int = 2

    def __init__(self, social_queue: asyncio.Queue) -> None:
        self._social_queue = social_queue

        # symbol → deque of (unix_timestamp, mention_count) snapshots
        self._mention_history: dict[str, deque] = {}

        # symbol → most recent baseline count (previous 30-min window)
        self._baseline: dict[str, float] = {}

        # Cache of CoinGecko trending coins for velocity lookup by external callers
        self._trending_coins: list[dict] = []

        logger.info("SocialFeed initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def poll_loop(self) -> None:
        """Run forever: poll Reddit + CoinGecko, then sleep 300s."""
        logger.info("SocialFeed poll_loop started")
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                logger.info("SocialFeed poll_loop cancelled")
                return
            except Exception as exc:
                logger.error(f"SocialFeed poll_loop unexpected error: {exc}")

            await asyncio.sleep(self.POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    async def _poll_once(self) -> None:
        """Single poll cycle: fetch Reddit posts + CoinGecko trending."""
        # Fetch all subreddits in parallel
        reddit_tasks = [self._fetch_subreddit(sub) for sub in self.SUBREDDITS]
        trending_task = self._fetch_coingecko_trending()

        results = await asyncio.gather(*reddit_tasks, trending_task, return_exceptions=True)

        trending_result = results[-1]
        reddit_results = results[:-1]

        # Merge all reddit posts
        all_posts: list[dict] = []
        for result in reddit_results:
            if isinstance(result, Exception):
                logger.debug(f"SocialFeed subreddit fetch error: {result}")
            elif isinstance(result, list):
                all_posts.extend(result)

        # Handle trending
        if isinstance(trending_result, Exception):
            logger.warning(f"CoinGecko trending fetch error: {trending_result}")
            trending_coins: list[dict] = []
        else:
            trending_coins = trending_result or []

        self._trending_coins = trending_coins

        # Extract symbol mentions from all posts
        mentions = self._extract_mentions(all_posts)

        logger.info(
            f"SocialFeed: {len(all_posts)} posts, "
            f"{len(mentions)} symbols mentioned, "
            f"{len(trending_coins)} trending coins"
        )

        now_ts = time.time()

        # Emit signals for high-velocity symbols
        for symbol, posts in mentions.items():
            count = len(posts)
            if count < self._MIN_MENTION_COUNT:
                continue

            # Update history
            if symbol not in self._mention_history:
                self._mention_history[symbol] = deque()

            self._mention_history[symbol].append((now_ts, count))
            self._prune_history(symbol, now_ts)

            velocity = self._calculate_velocity(symbol, count)

            if velocity > 2.0:
                sentiment = self._calculate_sentiment(posts)
                top_title = self._top_post_title(posts)

                signal = SocialSignal(
                    symbol=symbol,
                    mention_count=count,
                    velocity_ratio=velocity,
                    avg_sentiment=sentiment,
                    top_post_title=top_title,
                    platform="reddit",
                )
                await self._social_queue.put(signal)
                logger.info(
                    f"SocialFeed signal: {symbol} "
                    f"velocity={velocity:.2f}x mentions={count}"
                )

        # Emit signals for all CoinGecko trending coins regardless of velocity
        for idx, coin in enumerate(trending_coins):
            coin_symbol = coin.get("symbol", "").upper()
            if not coin_symbol:
                continue

            current_count = len(mentions.get(coin_symbol, []))
            velocity = self._calculate_velocity(coin_symbol, current_count) if current_count else 1.0
            posts_for_coin = mentions.get(coin_symbol, [])
            sentiment = self._calculate_sentiment(posts_for_coin) if posts_for_coin else 0.0
            top_title = self._top_post_title(posts_for_coin) if posts_for_coin else coin.get("name", "")

            signal = SocialSignal(
                symbol=coin_symbol,
                mention_count=current_count,
                velocity_ratio=velocity,
                avg_sentiment=sentiment,
                top_post_title=top_title,
                platform="reddit",
            )
            await self._social_queue.put(signal)

    # ------------------------------------------------------------------
    # Reddit fetching
    # ------------------------------------------------------------------

    async def _fetch_subreddit(self, subreddit: str) -> list[dict]:
        """
        GET hot posts from a subreddit using the public JSON API.

        Returns a list of post dicts containing title, score, num_comments,
        created_utc, and selftext.
        """
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=50"
        headers = {"User-Agent": "TradingBot/1.0"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 429:
                        logger.warning(f"Reddit rate-limited on r/{subreddit}")
                        return []
                    resp.raise_for_status()
                    data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning(f"Reddit fetch error r/{subreddit}: {exc}")
            return []
        except Exception as exc:
            logger.warning(f"Reddit unexpected error r/{subreddit}: {exc}")
            return []

        posts: list[dict] = []
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            posts.append(
                {
                    "title": post_data.get("title") or "",
                    "selftext": post_data.get("selftext") or "",
                    "score": post_data.get("score") or 0,
                    "num_comments": post_data.get("num_comments") or 0,
                    "created_utc": post_data.get("created_utc") or 0.0,
                    "subreddit": subreddit,
                    "url": post_data.get("url") or "",
                }
            )

        logger.debug(f"Reddit r/{subreddit}: {len(posts)} posts fetched")
        return posts

    # ------------------------------------------------------------------
    # Mention extraction
    # ------------------------------------------------------------------

    def _extract_mentions(self, posts: list[dict]) -> dict[str, list[dict]]:
        """
        Scan post titles and bodies for ticker/crypto symbols.

        Returns dict mapping symbol → list of posts that mention it.
        """
        mentions: dict[str, list[dict]] = {}

        for post in posts:
            text = f"{post.get('title', '')} {post.get('selftext', '')}"
            symbols_found = self._find_symbols_in_text(text)

            for symbol in symbols_found:
                mentions.setdefault(symbol, []).append(post)

        return mentions

    def _find_symbols_in_text(self, text: str) -> set[str]:
        """Return set of known symbols found in text."""
        found: set[str] = set()
        upper_text = text.upper()
        lower_text = text.lower()

        # Match $TICKER or bare caps words against known sets
        for match in _TICKER_RE.finditer(upper_text):
            token = match.group(1)
            if token in _STOCK_TICKERS:
                found.add(token)
            if token in _CRYPTO_SYMBOLS:
                found.add(token)

        # Match crypto names
        for name, symbol in _CRYPTO_NAMES_AND_SYMBOLS.items():
            if name in lower_text:
                found.add(symbol)

        return found

    # ------------------------------------------------------------------
    # Velocity calculation
    # ------------------------------------------------------------------

    def _calculate_velocity(self, symbol: str, current_count: int) -> float:
        """
        Compare current mention count to previous 30-min baseline.

        Returns ratio of current / baseline (1.0 = normal, 5.0 = 5x spike).
        If no history exists, returns 1.0.
        """
        history = self._mention_history.get(symbol)

        if not history or len(history) < 2:
            # Not enough data to compare — store baseline for next time
            self._baseline[symbol] = max(float(current_count), 1.0)
            return 1.0

        # Baseline = average of all historical counts except the latest
        historical_counts = [count for _, count in list(history)[:-1]]
        baseline = sum(historical_counts) / len(historical_counts) if historical_counts else 1.0
        baseline = max(baseline, 1.0)

        self._baseline[symbol] = baseline
        return current_count / baseline

    def _prune_history(self, symbol: str, now_ts: float) -> None:
        """Remove history entries older than the velocity window (30 min)."""
        history = self._mention_history.get(symbol)
        if not history:
            return
        cutoff = now_ts - _VELOCITY_WINDOW_SECONDS
        while history and history[0][0] < cutoff:
            history.popleft()

    # ------------------------------------------------------------------
    # Sentiment calculation
    # ------------------------------------------------------------------

    def _calculate_sentiment(self, posts: list[dict]) -> float:
        """
        Simple lexicon-based sentiment score averaged over posts.

        Returns a value in [-1.0, 1.0].
        """
        if not posts:
            return 0.0

        scores: list[float] = []
        for post in posts:
            text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
            words = set(re.findall(r"\b\w+\b", text))
            pos = len(words & _POSITIVE_WORDS)
            neg = len(words & _NEGATIVE_WORDS)
            total = pos + neg
            if total == 0:
                scores.append(0.0)
            else:
                scores.append((pos - neg) / total)

        return sum(scores) / len(scores)

    @staticmethod
    def _top_post_title(posts: list[dict]) -> str:
        """Return the title of the highest-scored post."""
        if not posts:
            return ""
        top = max(posts, key=lambda p: p.get("score", 0))
        return top.get("title", "")[:200]

    # ------------------------------------------------------------------
    # CoinGecko trending
    # ------------------------------------------------------------------

    async def _fetch_coingecko_trending(self) -> list[dict]:
        """
        GET https://api.coingecko.com/api/v3/search/trending

        Returns list of coin dicts: {coin_id, symbol, name, market_cap_rank}
        Ranked by search interest (index 0 = most trending).
        """
        url = "https://api.coingecko.com/api/v3/search/trending"
        headers = {"Accept": "application/json"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning(f"CoinGecko trending fetch error: {exc}")
            return []
        except Exception as exc:
            logger.warning(f"CoinGecko trending unexpected error: {exc}")
            return []

        coins: list[dict] = []
        for item in data.get("coins", []):
            coin = item.get("item", {})
            coins.append(
                {
                    "coin_id": coin.get("id", ""),
                    "symbol": coin.get("symbol", "").upper(),
                    "name": coin.get("name", ""),
                    "market_cap_rank": coin.get("market_cap_rank"),
                }
            )

        logger.debug(f"CoinGecko trending: {len(coins)} coins")
        return coins

    # ------------------------------------------------------------------
    # Public accessor for other modules
    # ------------------------------------------------------------------

    def get_trending_coins(self) -> list[dict]:
        """Return the most recently fetched CoinGecko trending coins list."""
        return list(self._trending_coins)

    def get_symbol_velocity(self, symbol: str) -> float:
        """
        Return the current velocity ratio for a symbol.

        Returns 1.0 if no history is available.
        """
        history = self._mention_history.get(symbol.upper())
        if not history:
            return 1.0

        baseline = self._baseline.get(symbol.upper(), 1.0)
        latest_count = history[-1][1] if history else 0
        return latest_count / max(baseline, 1.0)
