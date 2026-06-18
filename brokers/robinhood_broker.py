"""brokers/robinhood_broker.py — Robinhood Agentic Trading via MCP HTTP endpoint."""

from __future__ import annotations

import asyncio
import urllib.parse
import urllib.request
from typing import Any

import aiohttp
from loguru import logger

MCP_URL      = "https://agent.robinhood.com/mcp/trading"
TOKEN_URL    = "https://api.robinhood.com/oauth2/token/"
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)


class RobinhoodBroker:
    """
    Calls the Robinhood Agentic MCP server using JSON-RPC 2.0 over HTTP.

    Setup: run get_robinhood_token.py locally once to get ROBINHOOD_MCP_TOKEN.
    If the token expires, the broker auto-refreshes using ROBINHOOD_REFRESH_TOKEN.
    Reference: https://robinhood.com/us/en/support/articles/agentic-trading-overview/
    """

    def __init__(
        self,
        mcp_token: str,
        fill_queue: asyncio.Queue,
        refresh_token: str = "",
        client_id: str = "",
    ) -> None:
        self._token          = mcp_token
        self._refresh        = refresh_token
        self._client_id      = client_id
        self._fill_queue     = fill_queue
        self._session: aiohttp.ClientSession | None = None
        self._req_id         = 0
        self._account_number = ""

    def _available(self) -> bool:
        if not self._token:
            logger.warning("[RH] No Robinhood MCP token — broker inactive")
            return False
        return True

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
        return self._session

    async def _refresh_token(self) -> bool:
        """Attempt to refresh the access token using the stored refresh token."""
        if not self._refresh or not self._client_id:
            return False
        try:
            payload = urllib.parse.urlencode({
                "grant_type":    "refresh_token",
                "refresh_token": self._refresh,
                "client_id":     self._client_id,
            }).encode()
            req  = urllib.request.Request(
                TOKEN_URL, data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            import json as _json
            data = _json.loads(resp.read())
            new_token = data.get("access_token")
            if new_token:
                self._token   = new_token
                new_refresh   = data.get("refresh_token")
                if new_refresh:
                    self._refresh = new_refresh
                logger.info("[RH] Token refreshed successfully")
                return True
        except Exception as e:
            logger.error("[RH] Token refresh failed: {}", e)
        return False

    @staticmethod
    async def _parse_sse(resp: aiohttp.ClientResponse) -> dict:
        """Read a Server-Sent Events response and return the first JSON-RPC result."""
        import json as _json
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        return _json.loads(payload)
                    except Exception:
                        pass
        return {}

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a single MCP tool call. Handles both JSON and SSE (streaming) responses."""
        self._req_id += 1
        rpc_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": self._req_id,
        }

        async def _do_request(token: str) -> dict:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            session = await self._get_session()
            async with session.post(MCP_URL, json=rpc_payload, headers=headers) as resp:
                if resp.status == 401:
                    return {"__status": 401}
                ct = resp.content_type or ""
                if "event-stream" in ct:
                    return await self._parse_sse(resp)
                return await resp.json(content_type=None)

        data = await _do_request(self._token)

        if data.get("__status") == 401:
            logger.warning("[RH] 401 Unauthorized — attempting token refresh")
            if await self._refresh_token():
                data = await _do_request(self._token)
            else:
                logger.error("[RH] Token expired and refresh failed. Re-run get_robinhood_token.py")
                raise RuntimeError("Robinhood token expired — re-run get_robinhood_token.py")

        if "error" in data:
            raise RuntimeError(f"MCP error calling {tool_name}: {data['error']}")

        # MCP wraps results in content array for tool calls
        result = data.get("result", data)
        if isinstance(result, dict) and "content" in result:
            import json as _json
            content = result["content"]
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                try:
                    return _json.loads(text)
                except Exception:
                    return {"text": text}
        return result

    # ------------------------------------------------------------------
    # Read-only account tools
    # ------------------------------------------------------------------

    async def get_account_number(self) -> str:
        """Fetch the agentic brokerage account number (cached after first call)."""
        if self._account_number:
            return self._account_number
        try:
            result = await self._call_tool("get_accounts", {})
            data = result.get("data", result) if isinstance(result, dict) else {}
            accounts = data.get("accounts", []) if isinstance(data, dict) else []
            if not isinstance(accounts, list):
                accounts = []
            # Prefer agentic_allowed account, fall back to default
            agentic = next((a for a in accounts if a.get("agentic_allowed")), None)
            chosen = agentic or next((a for a in accounts if a.get("is_default")), None) or (accounts[0] if accounts else None)
            if chosen:
                self._account_number = chosen.get("account_number", "")
                logger.info("[RH] Using account {} ({})", self._account_number, chosen.get("nickname") or chosen.get("type", ""))
            return self._account_number
        except Exception as e:
            logger.error("[RH] get_accounts failed: {}", e)
            return ""

    async def get_portfolio(self) -> dict:
        if not self._available():
            return {}
        try:
            acct = await self.get_account_number()
            args = {"account_number": acct} if acct else {}
            return await self._call_tool("get_portfolio", args)
        except Exception as e:
            logger.error("[RH] get_portfolio failed: {}", e)
            return {}

    async def get_positions(self) -> list[dict]:
        if not self._available():
            return []
        try:
            acct = await self.get_account_number()
            if not acct:
                return []
            result = await self._call_tool("get_equity_positions", {"account_number": acct})
            return result.get("positions", result) if isinstance(result, dict) else []
        except Exception as e:
            logger.error("[RH] get_positions failed: {}", e)
            return []

    async def get_quote(self, symbol: str) -> dict:
        if not self._available():
            return {}
        try:
            result = await self._call_tool("get_equity_quotes", {"symbols": [symbol]})
            data = result.get("data", result)
            results = data.get("results", []) if isinstance(data, dict) else []
            if results:
                q = results[0].get("quote", results[0])
                return q
            return {}
        except Exception as e:
            logger.error("[RH] get_quote({}) failed: {}", symbol, e)
            return {}

    async def get_quote_price(self, symbol: str) -> float:
        """Return the current price as a float, 0.0 on failure."""
        q = await self.get_quote(symbol)
        for field in ("last_trade_price", "last_non_reg_trade_price", "ask_price"):
            val = q.get(field)
            if val:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 0.0

    async def get_orders(self) -> list[dict]:
        if not self._available():
            return []
        try:
            acct = await self.get_account_number()
            args = {"account_number": acct} if acct else {}
            result = await self._call_tool("get_equity_orders", args)
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
            acct = await self.get_account_number()
            args: dict[str, Any] = {
                "account_number": acct,
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

        review = await self.review_order(symbol, side, qty, order_type, limit_price)
        if not review:
            logger.warning("[RH] Skipping order — review returned empty for {} {}", side, symbol)
            return None

        try:
            acct = await self.get_account_number()
            args: dict[str, Any] = {
                "account_number": acct,
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

            # Emit a FillEvent so the portfolio + journal update.
            # Market orders fill near the live quote; use it as the fill price.
            try:
                fill_price = await self.get_quote_price(symbol)
                if fill_price <= 0 and limit_price:
                    fill_price = float(limit_price)
                await self._emit_fill(str(order_id), symbol, side.lower(), float(qty), fill_price)
            except Exception as fe:
                logger.warning("[RH] could not emit fill for {}: {}", symbol, fe)

            return str(order_id)
        except Exception as e:
            logger.error("[RH] place_order({}) failed: {}", symbol, e)
            return None

    async def _emit_fill(self, order_id: str, symbol: str, side: str,
                         qty: float, fill_price: float) -> None:
        """Push a FillEvent onto the engine fill queue so state stays in sync."""
        if self._fill_queue is None:
            return
        from datetime import datetime, timezone
        from core.engine import FillEvent
        await self._fill_queue.put(FillEvent(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            fill_price=fill_price,
            asset_class="stock",
            timestamp=datetime.now(tz=timezone.utc),
        ))
        logger.info("[RH] Fill emitted: {} {} qty={:.4f} @ ${:.2f}", side.upper(), symbol, qty, fill_price)

    async def cancel_order(self, order_id: str) -> bool:
        if not self._available():
            return False
        try:
            acct = await self.get_account_number()
            args: dict[str, Any] = {"order_id": order_id}
            if acct:
                args["account_number"] = acct
            await self._call_tool("cancel_equity_order", args)
            logger.info("[RH] Order cancelled: {}", order_id)
            return True
        except Exception as e:
            logger.error("[RH] cancel_order({}) failed: {}", order_id, e)
            return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
