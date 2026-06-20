import os
from dotenv import load_dotenv

load_dotenv()

ROBINHOOD_MCP_TOKEN = os.getenv("ROBINHOOD_MCP_TOKEN", "")
ROBINHOOD_REFRESH_TOKEN = os.getenv("ROBINHOOD_REFRESH_TOKEN", "")
ROBINHOOD_CLIENT_ID = os.getenv("ROBINHOOD_CLIENT_ID", "")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "92.65"))
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "20.0"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.15"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "6"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Stocks to scan each cycle
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "META", "GOOGL", "AMD", "SOFI", "PLTR",
    "MARA", "RIOT", "COIN", "HOOD", "RIVN",
    "IONQ", "RGTI", "SMCI", "MSTR", "GME",
]

# How often to scan during market hours (minutes)
SCAN_INTERVAL_MINUTES = 20

# Stop loss & take profit defaults
STOP_LOSS_PCT = 0.05      # 5% stop loss
TAKE_PROFIT_PCT = 0.12    # 12% take profit

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
