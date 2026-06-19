# Trading Bot

An automated stock trading bot that scans a universe of liquid stocks/ETFs,
scores each with a multi-signal technical strategy, and opens/closes positions
within defined risk limits.

## How it actually works

- **Market data: free, no API key.** Prices and historical OHLCV bars come from
  Yahoo Finance (`src/data_provider.py`), with caching and retry/backoff.
- **Strategy: multi-signal scoring** (`src/analyzer.py`). Each stock is scored
  across independent signals and combined into a 0–100 confidence:
  - Trend: price vs SMA20 / SMA50
  - Short-term momentum vs EMA9
  - RSI (oversold / overbought)
  - MACD crossover
  - Bollinger Band position (mean reversion)
  - Volume confirmation
- **Risk-based levels.** Stop-loss and target are derived from ATR (volatility),
  so they adapt per stock instead of using flat percentages.
- **Position management** (`src/bot.py`). Every cycle the bot re-prices open
  positions and exits on stop-loss, target, or a fresh SELL signal — then scans
  for new BUY candidates and opens the best ones within cash + position limits.
- **Risk controls** (`src/risk_manager.py`): per-trade risk cap, daily loss
  limit, and max open positions.

## Trading modes

| Mode | When | Orders go to |
|------|------|--------------|
| **Paper** (default) | No broker credentials present | Local virtual account (`paper_trading.json`) — tracks real prices, positions, and P&L |
| **Live** | Robinhood credentials in `.env` | Real Robinhood orders |

The bot runs in **paper mode** unless real broker credentials are provided.
Paper mode is a faithful simulation (real market prices, real bookkeeping) but
does **not** place real orders.

## Run it

```bash
pip install -r requirements.txt

python main.py --scan      # run one scan/trade cycle now (ignores market hours)
python main.py             # run the full session loop during market hours
python main.py --eod       # end-of-day summary
```

## Configuration (`.env`)

```
STARTING_CAPITAL=92.65
MAX_RISK_PER_TRADE=20.0
DAILY_LOSS_LIMIT_PCT=0.15
MAX_OPEN_POSITIONS=6
MIN_CONFIDENCE=50
SCAN_INTERVAL_SECONDS=300

# Optional — only needed for LIVE trading:
# ROBINHOOD_MCP_TOKEN=...
# ROBINHOOD_REFRESH_TOKEN=...
# ROBINHOOD_CLIENT_ID=...
```

## Watchlist

Edit `watchlist.txt` (one ticker per line) to change what the bot scans.
If the file is absent, a default set of ~32 liquid large-caps and ETFs is used.

## Honest limitations

- Live trading requires your own Robinhood API credentials — they cannot be
  obtained or supplied by anyone but you.
- This is not financial advice. Technical signals are not predictions. Test in
  paper mode and understand the strategy before risking real money.
