"""
data/new_coins.py

New crypto coin detector via CoinGecko free API.
Identifies newly listed coins (< 30 days old) and trending coins,
cross-references with Reddit velocity from SocialFeed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING

import aiohttp
from loguru import logger

from data.market_data import NewCoinEvent

if TYPE_CHECKING:
    from data.social_feed import SocialFeed

# ---------------------------------------------------------------------------
# CoinGecko API constants
# ---------------------------------------------------------------------------

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Rate limit: 30 req/min on free tier → enforce 2s between calls
_RATE_LIMIT_DELAY = 2.0  # seconds

# How many days old a coin can be to qualify as "new"
_NEW_COIN_DAYS_THRESHOLD = 30

# How many coins to fetch per page (CoinGecko max = 250)
_COINS_PER_PAGE = 250

# How many pages to scan for new coins
_COINS_PAGES_TO_SCAN = 4

# Poll interval for the main loop
_POLL_INTERVAL_SECONDS = 900  # 15 minutes

# Snapshot refresh interval for known-coins list
_KNOWN_COINS_REFRESH_SECONDS = 86400  # 24 hours


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
) -> Optional[dict | list]:
    """Perform a GET request and return parsed JSON, or None on error."""
    try:
        async with session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if resp.status == 429:
                logger.warning(f"CoinGecko rate limited: {url}")
                return None
            resp.raise_for_status()
            return await resp.json()
    except aiohttp.ClientError as exc:
        logger.warning(f"CoinGecko HTTP error ({url}): {exc}")
        return None
    except Exception as exc:
        logger.warning(f"CoinGecko unexpected error ({url}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NewCoinsScanner:
    """
    Detects newly listed crypto coins via CoinGecko and cross-references
    social velocity from SocialFeed.

    Parameters
    ----------
    social_feed : SocialFeed
        Used to cross-check mention velocity for detected new coins.
    """

    def __init__(self, social_feed: "SocialFeed") -> None:
        self._social_feed = social_feed

        # Snapshot of all known coin IDs (populated on first run, refreshed daily)
        self._known_coins: set[str] = set()
        self._known_coins_snapshot_ts: float = 0.0  # unix timestamp of last refresh

        # Cache of last computed new-coin events (returned to strategies on demand)
        self._current_events: list[NewCoinEvent] = []

        logger.info("NewCoinsScanner initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def poll_loop(self) -> None:
        """Run forever: scan for new/trending coins every 900 seconds."""
        logger.info("NewCoinsScanner poll_loop started")
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                logger.info("NewCoinsScanner poll_loop cancelled")
                return
            except Exception as exc:
                logger.error(f"NewCoinsScanner poll_loop unexpected error: {exc}")

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def get_new_coin_events(self) -> list[NewCoinEvent]:
        """
        Return the most recently computed new+trending coin events,
        sorted by social_velocity descending.

        Called by strategy modules on demand.
        """
        return sorted(
            self._current_events,
            key=lambda e: e.social_velocity,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    async def _poll_once(self) -> None:
        """Single scan: refresh known-coins list if stale, then scan for new ones."""
        async with aiohttp.ClientSession(
            headers={"Accept": "application/json"}
        ) as session:
            # Refresh known-coins list if older than 24 hours
            now_ts = datetime.now(tz=timezone.utc).timestamp()
            if now_ts - self._known_coins_snapshot_ts > _KNOWN_COINS_REFRESH_SECONDS:
                await self._load_known_coins(session)

            # Fetch new coins from market data
            new_coins = await self._fetch_new_coins(session)

            # Fetch trending coins
            trending = await self._fetch_trending(session)
            trending_ids = {coin.get("id", ""): idx for idx, coin in enumerate(trending)}

            logger.info(
                f"NewCoinsScanner: {len(new_coins)} candidate new coins, "
                f"{len(trending)} trending"
            )

            # Build events
            events: list[NewCoinEvent] = []
            seen_ids: set[str] = set()

            for coin in new_coins:
                coin_id = coin.get("id", "")
                if not coin_id or coin_id in seen_ids:
                    continue
                seen_ids.add(coin_id)

                if not self._is_new_coin(coin):
                    continue

                trending_rank = trending_ids.get(coin_id)
                event = self._build_event(coin, trending_rank)
                events.append(event)

            # Also include trending coins not already captured
            for idx, coin in enumerate(trending):
                coin_id = coin.get("id", "")
                if not coin_id or coin_id in seen_ids:
                    continue
                seen_ids.add(coin_id)
                event = self._build_event(coin, idx)
                events.append(event)

            self._current_events = events
            logger.info(f"NewCoinsScanner: {len(events)} new/trending coin events")

    # ------------------------------------------------------------------
    # CoinGecko: load known coins list
    # ------------------------------------------------------------------

    async def _load_known_coins(self, session: aiohttp.ClientSession) -> None:
        """
        Fetch /coins/list (all coins ever listed) to populate _known_coins.

        This acts as a reference snapshot; coins absent from this set are
        truly new to CoinGecko.
        """
        url = f"{_COINGECKO_BASE}/coins/list"
        logger.info("NewCoinsScanner: refreshing known-coins list from CoinGecko")

        data = await _get_json(session, url)
        await asyncio.sleep(_RATE_LIMIT_DELAY)

        if not isinstance(data, list):
            logger.warning("Failed to load known-coins list from CoinGecko")
            return

        self._known_coins = {coin["id"] for coin in data if "id" in coin}
        self._known_coins_snapshot_ts = datetime.now(tz=timezone.utc).timestamp()
        logger.info(f"NewCoinsScanner: known-coins list loaded ({len(self._known_coins)} coins)")

    # ------------------------------------------------------------------
    # CoinGecko: fetch new coins via markets endpoint
    # ------------------------------------------------------------------

    async def _fetch_new_coins(self, session: aiohttp.ClientSession) -> list[dict]:
        """
        Scan CoinGecko /coins/markets pages for recently listed coins.

        Strategy:
        1. Fetch pages ordered by id_asc (stable ordering, covers full listing).
        2. Also fetch newest pages ordered by market_cap_asc (small new listings).
        3. Filter via _is_new_coin().

        Rate limit: 2s delay between page fetches.
        """
        url = f"{_COINGECKO_BASE}/coins/markets"
        base_params = {
            "vs_currency": "usd",
            "per_page": _COINS_PER_PAGE,
            "sparkline": "false",
            "price_change_percentage": "24h",
        }

        candidates: list[dict] = []

        # Scan first N pages sorted by id_asc
        for page in range(1, _COINS_PAGES_TO_SCAN + 1):
            params = {**base_params, "order": "id_asc", "page": page}
            data = await _get_json(session, url, params=params)
            await asyncio.sleep(_RATE_LIMIT_DELAY)

            if not isinstance(data, list):
                logger.warning(f"NewCoinsScanner: markets page {page} returned unexpected data")
                break

            candidates.extend(data)
            logger.debug(f"NewCoinsScanner: markets page {page} fetched ({len(data)} coins)")

            if len(data) < _COINS_PER_PAGE:
                # Last page
                break

        # Also scan a page of low market-cap coins (often newer listings)
        params_low_cap = {**base_params, "order": "market_cap_asc", "page": 1}
        data_low_cap = await _get_json(session, url, params=params_low_cap)
        await asyncio.sleep(_RATE_LIMIT_DELAY)

        if isinstance(data_low_cap, list):
            candidates.extend(data_low_cap)

        return candidates

    # ------------------------------------------------------------------
    # CoinGecko: fetch trending coins
    # ------------------------------------------------------------------

    async def _fetch_trending(self, session: aiohttp.ClientSession) -> list[dict]:
        """
        GET /search/trending — returns coins ranked by search interest.

        Returns list of dicts with keys: id, symbol, name, market_cap_rank, etc.
        """
        url = f"{_COINGECKO_BASE}/search/trending"
        data = await _get_json(session, url)
        await asyncio.sleep(_RATE_LIMIT_DELAY)

        if not isinstance(data, dict):
            logger.warning("NewCoinsScanner: trending endpoint returned unexpected data")
            return []

        coins: list[dict] = []
        for item in data.get("coins", []):
            coin = item.get("item", {})
            coins.append(
                {
                    "id": coin.get("id", ""),
                    "symbol": coin.get("symbol", "").upper(),
                    "name": coin.get("name", ""),
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "atl_date": None,  # Not provided by trending endpoint
                }
            )

        return coins

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_new_coin(self, coin: dict) -> bool:
        """
        Return True if coin qualifies as 'new'.

        Criteria (either/both):
        1. atl_date (all-time-low date) is within the last 30 days
           — used as a proxy for listing date since brand-new coins hit ATL near launch.
        2. coin_id is NOT in self._known_coins (absent from our daily snapshot,
           meaning it was listed after our last snapshot was taken).
        """
        coin_id: str = coin.get("id", "")

        # Check if the coin wasn't in our known-coins snapshot
        if coin_id and coin_id not in self._known_coins:
            return True

        # Check atl_date as a proxy for listing date
        atl_date_str: Optional[str] = coin.get("atl_date")
        if atl_date_str:
            try:
                # CoinGecko returns ISO 8601, e.g. "2024-01-15T00:00:00.000Z"
                atl_date = datetime.fromisoformat(
                    atl_date_str.replace("Z", "+00:00")
                )
                cutoff = datetime.now(tz=timezone.utc) - timedelta(days=_NEW_COIN_DAYS_THRESHOLD)
                if atl_date >= cutoff:
                    return True
            except Exception as exc:
                logger.debug(f"NewCoinsScanner: atl_date parse error for {coin_id}: {exc}")

        return False

    def _build_event(self, coin: dict, trending_rank: Optional[int]) -> NewCoinEvent:
        """
        Construct a NewCoinEvent from a CoinGecko coin dict.

        Social velocity is looked up from SocialFeed's mention history.
        """
        coin_id: str = coin.get("id", "")
        symbol: str = (coin.get("symbol") or "").upper()
        name: str = coin.get("name") or ""

        # Estimate days_old from atl_date if available
        days_old = self._estimate_days_old(coin)

        # Cross-reference social velocity
        social_velocity = self._social_feed.get_symbol_velocity(symbol)

        coingecko_url = f"https://www.coingecko.com/en/coins/{coin_id}"

        return NewCoinEvent(
            coin_id=coin_id,
            symbol=symbol,
            name=name,
            days_old=days_old,
            trending_rank=trending_rank,
            social_velocity=social_velocity,
            coingecko_url=coingecko_url,
        )

    @staticmethod
    def _estimate_days_old(coin: dict) -> int:
        """
        Estimate coin age in days using atl_date as a proxy for listing date.

        Returns -1 if the date is unavailable.
        """
        atl_date_str: Optional[str] = coin.get("atl_date")
        if not atl_date_str:
            return -1

        try:
            atl_date = datetime.fromisoformat(
                atl_date_str.replace("Z", "+00:00")
            )
            delta = datetime.now(tz=timezone.utc) - atl_date
            return max(0, delta.days)
        except Exception:
            return -1
