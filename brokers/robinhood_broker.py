"""brokers/robinhood_broker.py — Robinhood Agentic Trading via MCP HTTP endpoint."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
from loguru import logger

MCP_URL = "https://agent.robinhood.com/mcp/trading"
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)


class RobinhoodBroker:
    """
    Calls the Robinhood Agentic MCP server using JSON-RPC 2.0 over HTTP.

    Setup: Robinhood app → Settings → Agentic Trading → get Bearer token.
    Reference: https://robinhood.com/us/en/support/articles/agentic-trading-overview/
    """

    def __init__(self, mcp_token: str, fill_queue: asyncio.Queue) -> None:
        self._token = mcp_token
        self._fill_queue = fill_queue
        self._session: aiohttp.ClientSession | None = None
        self._req_id = 0

    def _available(self) -> bool:
        if not self._token:
            logger.warning("[RH] No Robinhood MCP token — broker inactive")
            return False
        return True

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
        return self._session

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a single MCP tool call via JSON-RPC 2.0."""
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": self._req_id,
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        session = await self._get_session()
        async with session.post(MCP_URL, json=payload, headers=headers) as resp:
            data = await resp.json()

        if "error" in data:
            raise RuntimeError(f"MCP error calling {tool_name}: {data['error']}")
        return data.get("result", {})

    # ------------------------------------------------------------------
    # Read-only account tools
    # ------------------------------------------------------------------

    async def get_portfolio(self) -> dict:
        if not self._available():
            return {}
        try:
            result = await self._call_tool("get_portfolio", {})
            return result
        except Exception as e:
            logger.error("[RH] get_portfolio failed: {}", e)
            return {}

    async def get_quote(self, symbol: str) -> dict:
        if not self._available():
            return {}
        try:
            result = await self._call_tool("get_equity_quotes", {"symbols": [symbol]})
            quotes = result.get("quotes", result)
            if isinstance(quotes, list) and quotes:
                return quotes[0]
            return quotes if isinstance(quotes, dict) else {}
        except Exception as e:
            logger.error("[RH] get_quote({}) failed: {}", symbol, e)
            return {}

    async def get_orders(self) -> list[dict]:
        if not self._available():
            return []
        try:
            result = await self._call_tool("get_equity_orders", {})
            return result.get("orders", result) if isinstance(result, dict) else []
        except Exception as e:
            logger.error("[RH] get_orders failed: {}", e)
            return []

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def review_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict:
        """ALWAYS call this before place_order. Logs the review for audit."""
        if not self._available():
            return {}
        try:
            args: dict[str, Any] = {
                "symbol": symbol,
                "side": side.lower(),
                "quantity": qty,
                "type": order_type,
            }
            if limit_price is not None:
                args["limit_price"] = limit_price
            result = await self._call_tool("review_equity_order", args)
            logger.info("[RH] Order review for {} {} qty={}: {}", side.upper(), symbol, qty, result)
            return result
        except Exception as e:
            logger.error("[RH] review_order({}) failed: {}", symbol, e)
            return {}

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> str | None:
        """Place equity order. Always calls review_order first."""
        if not self._available():
            return None

        # Safety: always review before placing
        review = await self.review_order(symbol, side, qty, order_type, limit_price)
        if not review:
            logger.warning("[RH] Skipping order — review returned empty for {} {}", side, symbol)
            return None

        try:
            args: dict[str, Any] = {
                "symbol": symbol,
                "side": side.lower(),
                "quantity": qty,
                "type": order_type,
            }
            if limit_price is not None:
                args["limit_price"] = limit_price

            result = await asyncio.shield(self._call_tool("place_equity_order", args))
            order_id = result.get("id") or result.get("order_id") or str(result)
            logger.info("[RH] Order placed: {} {} qty={} id={}", side.upper(), symbol, qty, order_id)
            return str(order_id)
        except Exception as e:
            logger.error("[RH] place_order({}) failed: {}", symbol, e)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        if not self._available():
            return False
        try:
            await self._call_tool("cancel_equity_order", {"order_id": order_id})
            logger.info("[RH] Order cancelled: {}", order_id)
            return True
        except Exception as e:
            logger.error("[RH] cancel_order({}) failed: {}", order_id, e)
            return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
