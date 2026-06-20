"""Telegram notifications."""
import logging
import requests
from bot.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def send(message: str) -> bool:
    """Send a plain text message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured, skipping notification")
        return False
    try:
        resp = requests.post(
            f"{BASE_URL}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning(f"Telegram send failed: {resp.text}")
        return resp.ok
    except Exception as e:
        logger.warning(f"Telegram error: {e}")
        return False


def send_trade_alert(action: str, symbol: str, dollars: float, price: float, reasoning: str):
    emoji = "🟢 BUY" if action == "BUY" else "🔴 SELL" if action == "SELL" else "⏸ HOLD"
    msg = (
        f"{emoji} <b>{symbol}</b>\n"
        f"Price: ${price:.2f}\n"
        f"Amount: ${dollars:.2f}\n"
        f"Reason: {reasoning}"
    )
    send(msg)


def send_eod_summary(summary: str, equity: float, starting_equity: float):
    pnl = equity - starting_equity
    pnl_pct = (pnl / starting_equity * 100) if starting_equity else 0
    emoji = "📈" if pnl >= 0 else "📉"
    msg = (
        f"{emoji} <b>EOD Summary</b>\n"
        f"Equity: ${equity:.2f}\n"
        f"Day P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)\n\n"
        f"{summary}"
    )
    send(msg)


def send_error(context: str, error: str):
    send(f"⚠️ <b>Bot Error</b>\n{context}: {error}")


def send_startup(mode: str, equity: float, positions: int):
    send(
        f"🤖 <b>Trading Bot Started</b>\n"
        f"Mode: {mode}\n"
        f"Equity: ${equity:.2f}\n"
        f"Open positions: {positions}"
    )
