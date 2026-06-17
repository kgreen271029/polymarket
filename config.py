"""config.py — Load and validate environment variables into a typed config object."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key, "")
    return val


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


@dataclass
class Config:
    # Alpaca
    alpaca_api_key: str = field(default_factory=lambda: _require("ALPACA_API_KEY"))
    alpaca_api_secret: str = field(default_factory=lambda: _require("ALPACA_API_SECRET"))

    # Robinhood Agentic MCP (run get_robinhood_token.py locally to get these)
    robinhood_mcp_token: str = field(default_factory=lambda: _require("ROBINHOOD_MCP_TOKEN"))
    robinhood_refresh_token: str = field(default_factory=lambda: _require("ROBINHOOD_REFRESH_TOKEN"))
    robinhood_client_id: str = field(default_factory=lambda: _require("ROBINHOOD_CLIENT_ID"))

    # Polymarket
    polymarket_private_key: str = field(default_factory=lambda: _require("POLYMARKET_PRIVATE_KEY"))
    polymarket_api_key: str = field(default_factory=lambda: _require("POLYMARKET_API_KEY"))
    polymarket_api_secret: str = field(default_factory=lambda: _require("POLYMARKET_API_SECRET"))
    polymarket_api_passphrase: str = field(default_factory=lambda: _require("POLYMARKET_API_PASSPHRASE"))

    # AI
    anthropic_api_key: str = field(default_factory=lambda: _require("ANTHROPIC_API_KEY"))

    # News
    news_api_key: str = field(default_factory=lambda: _require("NEWS_API_KEY"))

    # Risk management
    starting_capital: float = field(default_factory=lambda: _float("STARTING_CAPITAL", 100.0))
    max_risk_per_trade: float = field(default_factory=lambda: _float("MAX_RISK_PER_TRADE", 20.0))
    daily_loss_limit_pct: float = field(default_factory=lambda: _float("DAILY_LOSS_LIMIT_PCT", 0.15))
    max_open_positions: int = field(default_factory=lambda: _int("MAX_OPEN_POSITIONS", 6))

    # System
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate(self) -> list[str]:
        """Return list of missing critical keys."""
        missing = []
        if not self.alpaca_api_key:
            missing.append("ALPACA_API_KEY")
        if not self.alpaca_api_secret:
            missing.append("ALPACA_API_SECRET")
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing

    def has_robinhood(self) -> bool:
        return bool(self.robinhood_mcp_token)

    def has_polymarket(self) -> bool:
        return bool(self.polymarket_private_key)


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
