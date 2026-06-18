"""utils/notifier.py — Trade and alert notifications via Telegram."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp
from loguru import logger

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """Sends trade alerts via Telegram. Silently no-ops if not configured."""

    def __init__(self, bot_token: str = "", chat_id: str = "") -> None:
        self._token   = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id   or os.getenv("TELEGRAM_CHAT_ID", "")
        self._session: aiohttp.ClientSession | None = None

    def _enabled(self) -> bool:
        return bool(self._token and self._chat_id)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def send(self, message: str) -> None:
        if not self._enabled():
            return
        url = TELEGRAM_API.format(token=self._token)
        try:
            session = await self._get_session()
            async with session.post(url, json={
                "chat_id": self._chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("[Notifier] Telegram send failed {}: {}", resp.status, body[:200])
        except Exception as e:
            logger.warning("[Notifier] send failed: {}", e)

    async def trade_alert(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        strategy: str,
        reasoning: str,
        stop: float | None = None,
        target: float | None = None,
    ) -> None:
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        lines = [
            f"{emoji} <b>{side.upper()} {symbol}</b>",
            f"Price: <b>${price:.2f}</b>  Qty: {qty}",
            f"Strategy: {strategy}",
        ]
        if stop:
            lines.append(f"Stop: ${stop:.2f}  Target: ${target:.2f}" if target else f"Stop: ${stop:.2f}")
        lines.append(f"<i>{reasoning[:200]}</i>")
        await self.send("\n".join(lines))

    async def eod_report(self, report: str) -> None:
        header = "📊 <b>Tonight's Trade Ideas</b>\n" + "─" * 30 + "\n"
        await self.send(header + report[:3500])

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
