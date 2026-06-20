"""LLM-powered trade analysis using Claude (Anthropic) with Groq fallback."""
import json
import logging
from typing import Optional
import anthropic
from bot.config import ANTHROPIC_API_KEY, GROQ_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_anthropic_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


SYSTEM_PROMPT = """You are an expert momentum/swing trader analyzing US equities on Robinhood.
Your job is to recommend BUY, SELL, or HOLD for a given stock based on the data provided.

Rules:
- Only recommend BUY if there is a clear momentum setup with low downside risk
- SELL if the position has hit the stop loss, take profit, or trend has reversed
- HOLD otherwise
- Keep reasoning concise (2-3 sentences max)
- Always output valid JSON

Output format (JSON only, no markdown):
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0-100,
  "reasoning": "...",
  "suggested_stop_loss_pct": 0.05,
  "suggested_take_profit_pct": 0.12
}"""


def analyze_stock(
    symbol: str,
    current_price: float,
    candles: list[dict],
    news: list[str],
    existing_position: Optional[dict] = None,
    account_equity: float = 0,
) -> dict:
    """
    Send market data to Claude and get a trade recommendation.
    Returns parsed dict with action, confidence, reasoning, stop/tp levels.
    """
    # Build technical summary from candles
    tech_summary = _build_tech_summary(candles)

    position_ctx = ""
    if existing_position:
        position_ctx = (
            f"\nEXISTING POSITION: {existing_position['qty']} shares @ "
            f"${existing_position['avg_cost']:.2f} avg cost. "
            f"P&L: {existing_position['pnl_pct']:.1f}%"
        )

    user_content = f"""Analyze {symbol} for a trade decision.

PRICE: ${current_price:.2f}
ACCOUNT EQUITY: ${account_equity:.2f}
{position_ctx}

TECHNICALS:
{tech_summary}

RECENT NEWS:
{chr(10).join(f'- {h}' for h in news) if news else '- No recent news'}

Provide your JSON recommendation."""

    try:
        client = _get_client()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text.strip()
        result = json.loads(raw)
        result["symbol"] = symbol
        logger.info(f"LLM {symbol}: {result['action']} (confidence={result['confidence']})")
        return result
    except json.JSONDecodeError:
        logger.warning(f"LLM returned non-JSON for {symbol}, defaulting to HOLD")
        return _hold_default(symbol)
    except Exception as e:
        logger.error(f"LLM analysis failed for {symbol}: {e}")
        if GROQ_API_KEY:
            return _groq_fallback(symbol, user_content)
        return _hold_default(symbol)


def analyze_eod(positions: list[dict], orders_today: list[dict], account_info: dict) -> str:
    """Generate an end-of-day summary narrative using Claude."""
    prompt = f"""Generate a brief end-of-day trading summary.

ACCOUNT EQUITY: ${account_info.get('equity', 0):.2f}
CASH: ${account_info.get('cash', 0):.2f}

OPEN POSITIONS:
{json.dumps(positions, indent=2)}

ORDERS TODAY:
{json.dumps([o for o in orders_today[:20]], indent=2)}

Write a 3-5 sentence summary: what happened today, key trades, and outlook for tomorrow.
No JSON needed — plain English."""

    try:
        client = _get_client()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.error(f"EOD analysis failed: {e}")
        return f"EOD summary unavailable. Equity: ${account_info.get('equity', 0):.2f}"


def _build_tech_summary(candles: list[dict]) -> str:
    if not candles or len(candles) < 5:
        return "Insufficient candle data."
    try:
        import pandas as pd
        import pandas_ta as ta

        df = pd.DataFrame(candles)
        df["close"] = pd.to_numeric(df["close_price"], errors="coerce")
        df["high"] = pd.to_numeric(df["high_price"], errors="coerce")
        df["low"] = pd.to_numeric(df["low_price"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df.dropna(subset=["close"], inplace=True)

        rsi = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"])
        ema9 = ta.ema(df["close"], length=9)
        ema21 = ta.ema(df["close"], length=21)
        bb = ta.bbands(df["close"], length=20)

        rsi_val = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else None
        macd_hist = float(macd["MACDh_12_26_9"].iloc[-1]) if macd is not None else None
        ema9_val = float(ema9.iloc[-1]) if ema9 is not None and not ema9.empty else None
        ema21_val = float(ema21.iloc[-1]) if ema21 is not None and not ema21.empty else None

        bb_upper = float(bb["BBU_20_2.0"].iloc[-1]) if bb is not None else None
        bb_lower = float(bb["BBL_20_2.0"].iloc[-1]) if bb is not None else None

        close = float(df["close"].iloc[-1])
        vol_avg = float(df["volume"].tail(20).mean())
        vol_now = float(df["volume"].iloc[-1])

        lines = [
            f"Close: ${close:.2f}",
            f"RSI(14): {rsi_val:.1f}" if rsi_val else "RSI: n/a",
            f"MACD Histogram: {macd_hist:.4f}" if macd_hist else "MACD: n/a",
            f"EMA9: {ema9_val:.2f} | EMA21: {ema21_val:.2f}" if ema9_val and ema21_val else "",
            f"BB Upper: {bb_upper:.2f} | BB Lower: {bb_lower:.2f}" if bb_upper and bb_lower else "",
            f"Volume: {vol_now:.0f} (20-bar avg: {vol_avg:.0f})",
        ]
        return "\n".join(l for l in lines if l)
    except Exception as e:
        logger.warning(f"Tech summary error: {e}")
        closes = [float(c.get("close_price", 0)) for c in candles[-5:]]
        return f"Last 5 closes: {closes}"


def _hold_default(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "action": "HOLD",
        "confidence": 0,
        "reasoning": "LLM unavailable, defaulting to HOLD.",
        "suggested_stop_loss_pct": 0.05,
        "suggested_take_profit_pct": 0.12,
    }


def _groq_fallback(symbol: str, user_content: str) -> dict:
    """Use Groq (llama3) as fallback if Claude is unavailable."""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)
        result["symbol"] = symbol
        result["_model"] = "groq-fallback"
        logger.info(f"Groq fallback {symbol}: {result['action']}")
        return result
    except Exception as e:
        logger.error(f"Groq fallback failed for {symbol}: {e}")
        return _hold_default(symbol)
