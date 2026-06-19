"""Notification management via Telegram."""

import os
import logging
import asyncio

try:
    from telegram import Bot
except ImportError:
    Bot = None


class NotificationManager:
    """Manages notifications to users."""

    def __init__(self, logger):
        self.logger = logger
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.bot = None

        if self.telegram_token and self.telegram_chat_id and Bot:
            try:
                self.bot = Bot(token=self.telegram_token)
                self.logger.info("Telegram notifications enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Telegram: {e}")

    async def send_trade_alert(self, symbol, action, quantity, price, reason):
        """Send trade execution alert."""
        message = (
            f"🤖 *Trade Alert*\n"
            f"Action: {action.upper()}\n"
            f"Symbol: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Price: ${price:.2f}\n"
            f"Reason: {reason}"
        )
        await self.send_message(message)

    async def send_analysis(self, symbol, analysis, recommendation):
        """Send market analysis."""
        message = (
            f"📊 *Market Analysis: {symbol}*\n"
            f"Analysis: {analysis}\n"
            f"Recommendation: {recommendation}"
        )
        await self.send_message(message)

    async def send_risk_alert(self, alert_type, details):
        """Send risk management alert."""
        message = f"⚠️ *Risk Alert: {alert_type}*\n{details}"
        await self.send_message(message)

    async def send_message(self, message):
        """Send a message via Telegram."""
        if not self.bot or not self.telegram_chat_id:
            self.logger.debug(f"Notification (no Telegram): {message}")
            return

        try:
            await self.bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode="Markdown"
            )
            self.logger.debug(f"Telegram message sent")
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")

    async def send_eod_summary(self, summary):
        """Send end-of-day summary."""
        message = f"📈 *EOD Summary*\n{summary}"
        await self.send_message(message)
