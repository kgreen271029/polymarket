"""
analysis/ai_analyzer.py

AI trade signal validation — uses Groq (free) with Llama 3.3 70B.
Falls back to rule-based scoring if no API key is set.

Get a free Groq key at: groq.com (no credit card, 14,400 req/day free)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Dataclasses (unchanged interface)
# ---------------------------------------------------------------------------


@dataclass
class AnalysisContext:
    """Everything the AI needs to validate a trade signal."""

    symbol: str
    asset_class: str
    strategy_name: str
    proposed_action: str
    signal_summary: dict
    news_headlines: list[str]
    available_capital: float
    open_position_count: int
    daily_pnl_pct: float


@dataclass
class AIDecision:
    """Structured trade recommendation."""

    recommendation: str          # "BUY" | "SELL" | "HOLD"
    confidence: str              # "High" | "Medium" | "Low"
    stop_price: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    risk_factors: list[str] = field(default_factory=list)
    news_relevance: str = "neutral"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert quantitative trading analyst. Validate trade signals and return structured JSON.

RULES:
1. Always respond with a single valid JSON object — no markdown, no prose outside JSON.
2. Required keys:
   {"recommendation":"BUY"|"SELL"|"HOLD","confidence":"High"|"Medium"|"Low","stop_price":<number|null>,"take_profit":<number|null>,"reasoning":"1-2 sentences","risk_factors":["..."],"news_relevance":"positive"|"negative"|"neutral"}
3. stop_price and take_profit must give at least 1.5:1 reward-to-risk ratio.
4. High = strong confluence across technicals+trend+news. Medium = mixed. Low = contradictory → HOLD.
5. RSI>75 overbought, RSI<25 oversold. Volume spike + breakout_20 = strong confirmation.
6. If daily P&L already down >3%, lean HOLD unless signals are exceptional."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class AIAnalyzer:
    """
    Validates trade signals using Groq's free Llama 3.3 70B model.
    Falls back to rule-based HOLD if no API key or on any error.

    Get a free key: groq.com → sign up → API Keys → Create
    Add to .env: GROQ_API_KEY=gsk_...
    """

    _semaphore: asyncio.Semaphore = asyncio.Semaphore(3)

    def __init__(self, api_key: str = "", groq_api_key: str = "") -> None:
        self._groq_key = groq_api_key
        self._client = None

        if self._groq_key:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self._groq_key)
                logger.info("[AI] Using Groq (free) — llama-3.3-70b-versatile")
            except ImportError:
                logger.warning("[AI] groq package not installed — run: pip install groq")
        else:
            logger.warning("[AI] No GROQ_API_KEY set — AI analysis disabled, using rule-based fallback")

    def _available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    async def analyze(self, context: AnalysisContext) -> AIDecision:
        """Validate a trade signal. Returns HOLD on any failure."""
        if not self._available():
            return self._rule_based_decision(context)

        user_msg = self._build_user_message(context)
        try:
            async with self.__class__._semaphore:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=512,
                        temperature=0.1,
                    ),
                    timeout=15.0,
                )
            raw = resp.choices[0].message.content or ""
            return self._parse_response(raw, context.symbol)
        except asyncio.TimeoutError:
            logger.warning("[AI] Timeout for {} — using rule-based fallback", context.symbol)
            return self._rule_based_decision(context)
        except Exception as e:
            logger.warning("[AI] Error for {}: {} — using rule-based fallback", context.symbol, e)
            return self._rule_based_decision(context)

    async def analyze_new_coin(self, coin_data: dict, social_data: dict) -> dict:
        """Score a new coin listing 0-10."""
        if not self._available():
            return {"score": 0, "reasoning": "AI not configured", "risks": ["no_ai"], "entry_recommendation": False}

        prompt = (
            "Score this new crypto coin 0-100 as a trading opportunity. "
            "Respond with JSON only: {\"score\":int,\"reasoning\":\"2 sentences\",\"risks\":[\"...\"],\"entry_recommendation\":bool}\n\n"
            f"Coin: {coin_data.get('name')} ({coin_data.get('symbol')})\n"
            f"Days old: {coin_data.get('days_old')}, Trending rank: {coin_data.get('trending_rank')}\n"
            f"Social velocity: {social_data.get('social_velocity')}x"
        )
        try:
            async with self.__class__._semaphore:
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0.1,
                    ),
                    timeout=15.0,
                )
            raw = resp.choices[0].message.content or ""
            cleaned = self._extract_json(raw)
            data = json.loads(cleaned)
            return {
                "score": int(data.get("score", 0)) // 10,  # convert 0-100 to 0-10
                "reasoning": str(data.get("reasoning", "")),
                "risks": list(data.get("risks", [])),
                "entry_recommendation": bool(data.get("entry_recommendation", False)),
            }
        except Exception as e:
            logger.warning("[AI] analyze_new_coin error: {}", e)
            return {"score": 0, "reasoning": str(e), "risks": ["error"], "entry_recommendation": False}

    # ------------------------------------------------------------------ #
    # Rule-based fallback (no API needed)                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rule_based_decision(ctx: AnalysisContext) -> AIDecision:
        """
        Pure technical rules when AI is unavailable.
        Conservative — only BUYs on strong multi-factor confluence.
        """
        sig = ctx.signal_summary
        rsi = sig.get("rsi", 50)
        macd = sig.get("macd_direction", "neutral")
        trend = sig.get("trend", "sideways")
        vol = sig.get("volume_ratio", 1.0)
        breakout = sig.get("breakout_20", False)
        bb_pct = sig.get("bb_pct", 0.5)
        price = sig.get("latest_close", 0)

        score = 0
        if trend == "uptrend":   score += 2
        if macd == "bullish":    score += 2
        if breakout:             score += 2
        if vol > 1.5:            score += 1
        if 40 < rsi < 65:        score += 1

        if ctx.daily_pnl_pct < -3:
            return AIDecision("HOLD", "Low", None, None,
                              "Daily loss limit caution — rule-based hold", ["daily_loss_caution"])

        if score >= 6 and ctx.proposed_action == "BUY":
            stop = round(price * 0.96, 2)
            target = round(price * 1.08, 2)
            conf = "High" if score >= 7 else "Medium"
            return AIDecision("BUY", conf, stop, target,
                              f"Rule-based: score={score}/8, trend={trend}, MACD={macd}, vol={vol:.1f}x")
        elif score <= 2:
            return AIDecision("HOLD", "Low", None, None,
                              f"Rule-based: weak signals, score={score}/8")
        else:
            return AIDecision("HOLD", "Low", None, None,
                              f"Rule-based: insufficient confluence, score={score}/8")

    # ------------------------------------------------------------------ #
    # Message builders                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_user_message(ctx: AnalysisContext) -> str:
        sig = ctx.signal_summary
        headlines = "\n".join(f"  • {h}" for h in ctx.news_headlines[:8]) or "  None"
        return (
            f"ASSET: {ctx.symbol} ({ctx.asset_class})  STRATEGY: {ctx.strategy_name}  ACTION: {ctx.proposed_action}\n\n"
            f"TECHNICALS:\n"
            f"  RSI={sig.get('rsi',50):.1f}  MACD={sig.get('macd_direction','?')}  BB%={sig.get('bb_pct',0.5):.2f}\n"
            f"  VWAP_delta={sig.get('vwap_delta_pct',0):+.2f}%  Volume={sig.get('volume_ratio',1):.1f}x  Spike={sig.get('volume_spike',False)}\n"
            f"  Trend={sig.get('trend','?')}  Breakout20={sig.get('breakout_20',False)}  Price=${sig.get('latest_close',0):.4f}\n\n"
            f"NEWS:\n{headlines}\n\n"
            f"PORTFOLIO: ${ctx.available_capital:.0f} available  {ctx.open_position_count} positions  P&L={ctx.daily_pnl_pct:+.1f}%"
        )

    # ------------------------------------------------------------------ #
    # Parsers / helpers                                                     #
    # ------------------------------------------------------------------ #

    def _parse_response(self, raw: str, symbol: str) -> AIDecision:
        try:
            data = json.loads(self._extract_json(raw))
            return AIDecision(
                recommendation=str(data.get("recommendation", "HOLD")).upper(),
                confidence=str(data.get("confidence", "Low")),
                stop_price=self._float_or_none(data.get("stop_price")),
                take_profit=self._float_or_none(data.get("take_profit")),
                reasoning=str(data.get("reasoning", "")),
                risk_factors=list(data.get("risk_factors", [])),
                news_relevance=str(data.get("news_relevance", "neutral")),
            )
        except Exception as e:
            logger.warning("[AI] Parse error for {}: {} raw={!r}", symbol, e, raw[:150])
            return self._hold_fallback("JSON parse error")

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            return text[s:e + 1]
        return text

    @staticmethod
    def _float_or_none(v) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _hold_fallback(reason: str) -> AIDecision:
        return AIDecision("HOLD", "Low", None, None, reason,
                          ["fallback: analysis unavailable"], "neutral")
