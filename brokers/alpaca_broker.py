"""brokers/alpaca_broker.py — Alpaca live order execution (stocks + crypto)."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger


class AlpacaBroker:
    def __init__(self, api_key: str, api_secret: str, fill_queue: asyncio.Queue) -> None:
        from alpaca.trading.client import TradingClient
        self._client = TradingClient(api_key, api_secret, paper=False)
        self._fill_queue = fill_queue
        self._api_key = api_key
        self._api_secret = api_secret

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize(self, symbol: str) -> str:
        """'BTC/USD' → 'BTCUSD' for Alpaca trading endpoints."""
        return symbol.replace("/", "")

    def _is_crypto(self, symbol: str) -> bool:
        return "/" in symbol or symbol.upper().endswith("USD") and len(symbol) > 4

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        take_profit: float,
        strategy_name: str = "",
    ) -> str | None:
        """Submit a bracket (entry + stop-loss + take-profit) order. Returns order_id or None."""
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        from alpaca.common.exceptions import APIError

        normalized = self._normalize(symbol)
        alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=normalized,
            qty=qty,
            side=alpaca_side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(stop_price, 4)),
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 4)),
        )

        for attempt in range(2):
            try:
                order = await asyncio.to_thread(self._client.submit_order, order_data=request)
                logger.info(
                    "[ALPACA] Bracket order submitted: {} {} {} qty={} stop={} tp={} id={}",
                    side.upper(), normalized, strategy_name, qty, stop_price, take_profit, order.id
                )
                return str(order.id)
            except APIError as e:
                if "429" in str(e) and attempt == 0:
                    await asyncio.sleep(5)
                    continue
                logger.error("[ALPACA] place_bracket_order failed for {}: {}", symbol, e)
                return None
        return None

    async def place_market_order(self, symbol: str, side: str, qty: float) -> str | None:
        """Simple market order with no bracket legs."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.common.exceptions import APIError

        normalized = self._normalize(symbol)
        alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=normalized,
            qty=qty,
            side=alpaca_side,
            time_in_force=TimeInForce.GTC,
        )
        try:
            order = await asyncio.to_thread(self._client.submit_order, order_data=request)
            logger.info("[ALPACA] Market order: {} {} qty={} id={}", side.upper(), normalized, qty, order.id)
            return str(order.id)
        except APIError as e:
            logger.error("[ALPACA] place_market_order failed for {}: {}", symbol, e)
            return None

    async def close_position(self, symbol: str) -> bool:
        from alpaca.common.exceptions import APIError
        normalized = self._normalize(symbol)
        try:
            await asyncio.to_thread(self._client.close_position, symbol_or_asset_id=normalized)
            logger.info("[ALPACA] Closed position: {}", normalized)
            return True
        except APIError as e:
            logger.error("[ALPACA] close_position failed for {}: {}", symbol, e)
            return False

    async def cancel_order(self, order_id: str) -> bool:
        from alpaca.common.exceptions import APIError
        try:
            await asyncio.to_thread(self._client.cancel_order_by_id, order_id=order_id)
            return True
        except APIError as e:
            logger.error("[ALPACA] cancel_order failed for {}: {}", order_id, e)
            return False

    # ------------------------------------------------------------------
    # Account + position queries
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[dict]:
        try:
            positions = await asyncio.to_thread(self._client.get_all_positions)
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price or 0),
                    "unrealized_pl": float(p.unrealized_pl or 0),
                    "asset_class": str(p.asset_class),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("[ALPACA] get_positions failed: {}", e)
            return []

    async def get_account(self) -> dict:
        try:
            acct = await asyncio.to_thread(self._client.get_account)
            return {
                "cash": float(acct.cash),
                "portfolio_value": float(acct.portfolio_value),
                "buying_power": float(acct.buying_power),
                "pattern_day_trader": bool(acct.pattern_day_trader),
            }
        except Exception as e:
            logger.error("[ALPACA] get_account failed: {}", e)
            return {}

    async def is_market_open(self) -> bool:
        try:
            clock = await asyncio.to_thread(self._client.get_clock)
            return bool(clock.is_open)
        except Exception as e:
            logger.error("[ALPACA] is_market_open check failed: {}", e)
            return False
