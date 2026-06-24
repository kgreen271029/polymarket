# NBA Parlay Bot

An AI-powered bot that analyzes today's NBA games and generates a 3-leg parlay with the highest statistical probability of hitting.

## How It Works

1. **Fetches** today's NBA games, live odds (The Odds API), team stats, and recent form (nba_api)
2. **Scores** every possible bet (totals, spreads, moneylines) using a weighted statistical model
3. **Selects** the top 3 picks — one per game — for maximum independence
4. **Analyzes** the picks using Claude AI (claude-sonnet-4-6) with prompt caching
5. **Prints** a clean formatted parlay with reasoning and a payout estimate

## Setup

### 1. Clone and install dependencies

```bash
cd /home/user/polymarket
pip install -r requirements.txt
```

### 2. Get API keys

| API | Where to get it | Cost |
|-----|----------------|------|
| The Odds API | https://the-odds-api.com | Free (500 req/month) |
| Anthropic | https://console.anthropic.com | Pay-per-use (~$0.003/run) |

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual keys
```

`.env` file:
```
ODDS_API_KEY=your_odds_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
RUN_TIME=10:00
```

## Usage

### Run once (get today's parlay now)

```bash
python main.py
```

### Run on a daily schedule

```bash
python main.py --schedule
```

This starts the bot and automatically generates picks every day at the time set in `RUN_TIME` (default: 10:00 AM ET). Keep the terminal open or run it in a screen/tmux session.

## Sample Output

```
╔══════════════════════════════════════════════════════════════╗
║           NBA 3-LEG PARLAY PICKS — April 21, 2026            ║
╚══════════════════════════════════════════════════════════════╝

╭────┬────────────────────────────┬──────────────────────┬───────┬───────┬────────────┬─────────╮
│  # │ Game                       │ Pick                 │  Line │  Odds │ Stat Score │ AI Conf │
├────┼────────────────────────────┼──────────────────────┼───────┼───────┼────────────┼─────────┤
│  1 │ Nuggets @ Thunder          │ UNDER 223.5          │ 223.5 │  -110 │   78/100   │ High    │
│  2 │ Lakers @ Celtics           │ Boston -6.5          │  -6.5 │  -108 │   72/100   │ High    │
│  3 │ Bucks @ Heat               │ Milwaukee ML         │   —   │  -195 │   69/100   │ Medium  │
╰────┴────────────────────────────┴──────────────────────┴───────┴───────┴────────────┴─────────╯
...
  Combined odds  : +412
  Estimated payout: $512.00
  Profit if hits : $412.00
```

## Pick Strategy

Picks are prioritized by statistical confidence:

1. **Game Totals (Over/Under)** — Most predictable. Compares combined team scoring averages to the posted total line, weighted by recent form, pace, and injury impact.
2. **Spreads** — Considers home/away ATS records, rest-day advantages, and recent margin of victory vs. the line.
3. **Moneylines** — Only heavy favorites (-180 or better). High implied probability + recent form confirmation.

**Diversity rule:** The bot never picks two legs from the same game (correlated bets reduce true parlay independence).

## Scoring Model

Each pick is scored 0–100:

| Component | Weight | Notes |
|-----------|--------|-------|
| Statistical deviation from line | 30% | How far avg pts are from the posted line |
| Recent form (last 5 games) | 25% | Weighted more than season average |
| Pace / home-away splits | 20% | High pace = Over lean; home splits for spreads |
| Line value | 15% | Bigger statistical edge = higher score |
| Injury adjustment | 10% | Key player out reduces the pick's score |

## Notes

- The bot runs in degraded mode (stats-only picks) if `ODDS_API_KEY` is missing
- Claude AI analysis falls back to raw stats if `ANTHROPIC_API_KEY` is missing
- nba_api calls respect rate limits with automatic delays
- The Odds API free tier usage is logged after each fetch

## Disclaimer

This bot is for informational and entertainment purposes only. It does not guarantee wins. Sports betting involves risk — always bet responsibly and within your means.
