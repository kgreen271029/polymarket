"""brokers/polymarket_broker.py — Polymarket CLOB order execution via py-clob-client."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137


class PolymarketBroker:
    def __init__(
        self,
        private_key: str,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
    ) -> None:
        self._private_key = private_key
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._client: Any = None
        self._initialized = False

    def _available(self) -> bool:
        if not self._private_key:
            logger.warning("[POLY] No Polymarket private key — broker inactive")
            return False
        return True

    async def initialize(self) -> None:
        if not self._available():
            return
        try:
            from py_clob_client.client import ClobClient
            self._client = await asyncio.to_thread(
                ClobClient,
                host=CLOB_HOST,
                key=self._private_key,
                chain_id=POLYGON_CHAIN_ID,
            )
            if self._api_key:
                await asyncio.to_thread(
                    self._client.set_api_creds,
                    self._api_key,
                    self._api_secret,
                    self._api_passphrase,
                )
            self._initialized = True
            logger.info("[POLY] ClobClient initialized")
        except Exception as e:
            logger.error("[POLY] Failed to initialize ClobClient: {}", e)

    # ------------------------------------------------------------------
    # Market data queries
    # ------------------------------------------------------------------

    async def get_open_markets(self, limit: int = 50) -> list[dict]:
        if not self._initialized:
            return []
        try:
            result = await asyncio.to_thread(
                self._client.get_markets,
                next_cursor="",
            )
            markets = result.get("data", []) if isinstance(result, dict) else []
            active = [
                m for m in markets
                if m.get("active") and not m.get("closed")
                and float(m.get("volume", 0) or 0) > 0
            ]
            active.sort(key=lambda m: float(m.get("volume", 0) or 0), reverse=True)
            return active[:limit]
        except Exception as e:
            logger.error("[POLY] get_open_markets failed: {}", e)
            return []

    async def get_order_book(self, token_id: str) -> dict:
        if not self._initialized:
            return {}
        try:
            book = await asyncio.to_thread(self._client.get_order_book, token_id)
            return book if isinstance(book, dict) else {}
        except Exception as e:
            logger.error("[POLY] get_order_book({}) failed: {}", token_id, e)
            return {}

    async def get_positions(self) -> list[dict]:
        if not self._initialized:
            return []
        try:
            positions = await asyncio.to_thread(self._client.get_positions)
            return positions if isinstance(positions, list) else []
        except Exception as e:
            logger.error("[POLY] get_positions failed: {}", e)
            return []

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_usdc: float,
    ) -> str | None:
        """
        Place a limit order on the Polymarket CLOB.
        price: 0.01 to 0.99 (probability as decimal)
        size_usdc: dollar amount of the order (minimum $1)
        side: "yes" | "no"
        """
        if not self._initialized:
            return None
        if size_usdc < 1.0:
            logger.warning("[POLY] Order size ${} below $1 minimum — skipping", size_usdc)
            return None
        if not (0.01 <= price <= 0.99):
            logger.warning("[POLY] Invalid price {} — must be 0.01-0.99", price)
            return None

        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY, SELL

            clob_side = BUY if side.lower() == "yes" else SELL
            order_args = OrderArgs(
                price=price,
                size=size_usdc,
                side=clob_side,
                token_id=token_id,
            )
            result = await asyncio.shield(
                asyncio.to_thread(self._client.create_and_post_order, order_args)
            )
            order_id = result.get("orderID") or result.get("id") or str(result)
            logger.info("[POLY] Order placed: {} @ {} size=${} id={}", side.upper(), price, size_usdc, order_id)
            return str(order_id)
        except Exception as e:
            logger.error("[POLY] place_order failed token={} side={}: {}", token_id, side, e)
            return None

    async def check_order_status(self, order_id: str) -> str:
        if not self._initialized:
            return "unknown"
        try:
            order = await asyncio.to_thread(self._client.get_order, order_id)
            status = order.get("status", "unknown") if isinstance(order, dict) else "unknown"
            return status.lower()
        except Exception as e:
            logger.error("[POLY] check_order_status({}) failed: {}", order_id, e)
            return "unknown"
