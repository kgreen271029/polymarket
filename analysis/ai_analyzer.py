"""
analysis/ai_analyzer.py

Claude API integration for trade signal validation.
Each strategy routes to the most cost-effective model capable of the task.

Strategy → Model mapping
    crypto_scalper / momentum  →  claude-haiku-4-5-20251001   (sub-100ms budget)
    news_momentum / swing /
    polymarket                 →  claude-sonnet-4-6            (richer reasoning)
    new_coin_analysis          →  claude-sonnet-4-6 + adaptive thinking
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from anthropic import AsyncAnthropic
from loguru import logger

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnalysisContext:
    """Everything Claude needs to validate a trade signal."""

    symbol: str
    asset_class: str                  # e.g. "crypto", "equity", "prediction_market"
    strategy_name: str                # must match one of the routing keys below
    proposed_action: str              # "BUY" | "SELL" | "HOLD"
    signal_summary: dict              # output of TechnicalAnalyzer.build_signal_summary()
    news_headlines: list[str]         # pre-formatted headline strings
    available_capital: float          # USD
    open_position_count: int
    daily_pnl_pct: float              # today's realised P&L as a percentage


@dataclass
class AIDecision:
    """Structured trade recommendation returned by AIAnalyzer.analyze()."""

    recommendation: str               # "BUY" | "SELL" | "HOLD"
    confidence: str                   # "High" | "Medium" | "Low"
    stop_price: Optional[float]       # None means caller should use default
    take_profit: Optional[float]
    reasoning: str
    risk_factors: list[str] = field(default_factory=list)
    news_relevance: str = "neutral"   # "positive" | "negative" | "neutral"


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

_FAST_MODEL = "claude-haiku-4-5-20251001"
_STANDARD_MODEL = "claude-sonnet-4-6"

_FAST_STRATEGIES: frozenset[str] = frozenset(
    {"crypto_scalper", "momentum", "scalper", "crypto_momentum"}
)

# ---------------------------------------------------------------------------
# System prompt (cached via prompt caching)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert quantitative trading analyst with deep experience in technical analysis, \
news-driven trading, and risk management. Your role is to validate trade signals and provide \
structured trade recommendations.

RULES:
1. Always respond with a single valid JSON object — no markdown fences, no prose outside the JSON.
2. The JSON must contain exactly these keys:
   {
     "recommendation": "BUY" | "SELL" | "HOLD",
     "confidence":     "High" | "Medium" | "Low",
     "stop_price":     <number or null>,
     "take_profit":    <number or null>,
     "reasoning":      "<1–3 sentences>",
     "risk_factors":   ["<factor>", ...],
     "news_relevance": "positive" | "negative" | "neutral"
   }
3. stop_price and take_profit must maintain a minimum 1.5 : 1 reward-to-risk ratio.
   If you cannot identify a defensible stop, set both to null.
4. Confidence mapping:
   - "High"   → you have strong confluence across technicals, trend, and news.
   - "Medium" → mixed signals; risk management is critical.
   - "Low"    → signals are contradictory or data is sparse; default to HOLD.
5. risk_factors is a list of short strings (max 5). Always include at least one.
6. Never recommend a position size — that is the risk manager's job.
7. If today's P&L is already down more than 3 %, lean toward HOLD unless signals are exceptional.
8. Treat RSI > 75 as overbought and RSI < 25 as oversold relative to the proposed action.
9. A volume_spike combined with a breakout_20 is a strong confirmation signal.
10. When news_relevance is "negative", lower confidence by one tier unless technicals are overwhelmingly positive."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class AIAnalyzer:
    """
    Validates trade signals using Claude.

    Thread-safe for concurrent async use. A class-level semaphore caps
    simultaneous in-flight requests at 3 to avoid API rate-limit errors.
    """

    _semaphore: asyncio.Semaphore = asyncio.Semaphore(3)

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

        # System prompt stored as a content block with prompt caching enabled.
        # The first request in each billing period pays for caching;
        # subsequent requests reuse the cached version at ~10 % of the token cost.
        self._system: list[dict] = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    async def analyze(self, context: AnalysisContext) -> AIDecision:
        """
        Validate a trade signal and return a structured AIDecision.

        Falls back to AIDecision(recommendation="HOLD", confidence="Low", ...)
        on API timeout (>10 s) or JSON parse failure.
        """
        model = (
            _FAST_MODEL
            if context.strategy_name in _FAST_STRATEGIES
            else _STANDARD_MODEL
        )
        user_message = self._build_user_message(context)

        try:
            async with self.__class__._semaphore:
                response = await asyncio.wait_for(
                    self._client.messages.create(
                        model=model,
                        max_tokens=512,
                        system=self._system,
                        messages=[{"role": "user", "content": user_message}],
                    ),
                    timeout=10.0,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "AIAnalyzer.analyze: timeout for {} {} — returning HOLD",
                context.symbol,
                context.strategy_name,
            )
            return self._hold_fallback("Request timed out after 10 s")
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "AIAnalyzer.analyze: unexpected error for {} — returning HOLD: {}",
                context.symbol,
                exc,
            )
            return self._hold_fallback(f"API error: {exc}")

        raw_text = response.content[0].text if response.content else ""
        return self._parse_response(raw_text, context.symbol)

    async def analyze_new_coin(
        self,
        coin_data: dict,
        social_data: dict,
    ) -> dict:
        """
        Deep-research analysis of a newly listed coin using extended (adaptive) thinking.

        Args:
            coin_data:   Keys expected: name, symbol, contract_address, launch_date,
                         market_cap_usd, liquidity_usd, holders, description.
            social_data: Keys expected: twitter_followers, telegram_members,
                         reddit_subscribers, mention_velocity_1h, avg_sentiment.

        Returns:
            {
                "score":                int (0–100),
                "reasoning":            str,
                "risks":                list[str],
                "entry_recommendation": bool,
            }
        """
        prompt = self._build_new_coin_prompt(coin_data, social_data)

        try:
            async with self.__class__._semaphore:
                response = await asyncio.wait_for(
                    self._client.messages.create(
                        model=_STANDARD_MODEL,
                        max_tokens=2048,
                        thinking={"type": "adaptive"},
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    ),
                    timeout=60.0,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "AIAnalyzer.analyze_new_coin: timeout for {} — skipping",
                coin_data.get("symbol", "UNKNOWN"),
            )
            return {
                "score": 0,
                "reasoning": "Analysis timed out.",
                "risks": ["timeout"],
                "entry_recommendation": False,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "AIAnalyzer.analyze_new_coin error for {}: {}",
                coin_data.get("symbol", "UNKNOWN"),
                exc,
            )
            return {
                "score": 0,
                "reasoning": f"API error: {exc}",
                "risks": ["api_error"],
                "entry_recommendation": False,
            }

        # The model may emit thinking blocks before the text block.
        text_block = next(
            (b for b in response.content if b.type == "text"),
            None,
        )
        if text_block is None:
            return {
                "score": 0,
                "reasoning": "No text in response.",
                "risks": ["empty_response"],
                "entry_recommendation": False,
            }

        return self._parse_new_coin_response(text_block.text, coin_data.get("symbol", ""))

    # ------------------------------------------------------------------ #
    # Message builders                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_user_message(ctx: AnalysisContext) -> str:
        sig = ctx.signal_summary

        rsi = sig.get("rsi", 50.0)
        macd_hist = sig.get("macd_hist", 0.0)
        macd_direction = sig.get("macd_direction", "neutral")
        bb_pct = sig.get("bb_pct", 0.5)
        vwap_delta_pct = sig.get("vwap_delta_pct", 0.0)
        volume_ratio = sig.get("volume_ratio", 1.0)
        volume_spike = sig.get("volume_spike", False)
        trend = sig.get("trend", "sideways")
        latest_close = sig.get("latest_close", 0.0)

        if ctx.news_headlines:
            headlines_text = "\n".join(
                f"  • {h}" for h in ctx.news_headlines[:10]
            )
        else:
            headlines_text = "  No relevant news"

        return (
            f"ASSET: {ctx.symbol} ({ctx.asset_class})\n"
            f"STRATEGY: {ctx.strategy_name}\n"
            f"PROPOSED: {ctx.proposed_action}\n"
            "\n"
            "TECHNICALS:\n"
            f"  RSI(14): {rsi:.1f}\n"
            f"  MACD histogram: {macd_hist:.4f} ({macd_direction})\n"
            f"  Bollinger %: {bb_pct:.2f} (0=lower band, 1=upper band)\n"
            f"  VWAP delta: {vwap_delta_pct:+.2f}%\n"
            f"  Volume: {volume_ratio:.1f}x average  spike={volume_spike}\n"
            f"  Trend: {trend}\n"
            f"  Price: ${latest_close:.4f}\n"
            "\n"
            "NEWS (last 30 min):\n"
            f"{headlines_text}\n"
            "\n"
            "PORTFOLIO:\n"
            f"  Available: ${ctx.available_capital:.2f}\n"
            f"  Open positions: {ctx.open_position_count}\n"
            f"  Today P&L: {ctx.daily_pnl_pct:+.1f}%"
        )

    @staticmethod
    def _build_new_coin_prompt(coin_data: dict, social_data: dict) -> str:
        return (
            "You are an expert crypto analyst specialising in newly launched tokens. "
            "Analyse the following coin and social data, then respond with a single "
            "valid JSON object (no markdown fences) containing exactly these keys:\n"
            '  "score": integer 0–100 (overall opportunity score),\n'
            '  "reasoning": string (2–4 sentences),\n'
            '  "risks": list of strings (top 3–5 risks),\n'
            '  "entry_recommendation": boolean\n\n'
            "COIN DATA:\n"
            f"  Name: {coin_data.get('name', 'N/A')}\n"
            f"  Symbol: {coin_data.get('symbol', 'N/A')}\n"
            f"  Contract: {coin_data.get('contract_address', 'N/A')}\n"
            f"  Launch date: {coin_data.get('launch_date', 'N/A')}\n"
            f"  Market cap: ${coin_data.get('market_cap_usd', 0):,.0f}\n"
            f"  Liquidity: ${coin_data.get('liquidity_usd', 0):,.0f}\n"
            f"  Holders: {coin_data.get('holders', 0):,}\n"
            f"  Description: {coin_data.get('description', 'N/A')}\n\n"
            "SOCIAL DATA:\n"
            f"  Twitter followers: {social_data.get('twitter_followers', 0):,}\n"
            f"  Telegram members: {social_data.get('telegram_members', 0):,}\n"
            f"  Reddit subscribers: {social_data.get('reddit_subscribers', 0):,}\n"
            f"  Mention velocity (1 h): {social_data.get('mention_velocity_1h', 0):.1f}x\n"
            f"  Avg sentiment: {social_data.get('avg_sentiment', 0.0):+.2f}"
        )

    # ------------------------------------------------------------------ #
    # Response parsers                                                      #
    # ------------------------------------------------------------------ #

    def _parse_response(self, raw: str, symbol: str) -> AIDecision:
        """Parse Claude's JSON reply into an AIDecision, falling back to HOLD."""
        cleaned = self._extract_json(raw)
        try:
            data = json.loads(cleaned)
            return AIDecision(
                recommendation=str(data.get("recommendation", "HOLD")).upper(),
                confidence=str(data.get("confidence", "Low")),
                stop_price=self._float_or_none(data.get("stop_price")),
                take_profit=self._float_or_none(data.get("take_profit")),
                reasoning=str(data.get("reasoning", "")),
                risk_factors=list(data.get("risk_factors", [])),
                news_relevance=str(data.get("news_relevance", "neutral")),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "AIAnalyzer: failed to parse response for {} — {}: {!r}",
                symbol,
                exc,
                raw[:200],
            )
            return self._hold_fallback("JSON parse error")

    def _parse_new_coin_response(self, raw: str, symbol: str) -> dict:
        cleaned = self._extract_json(raw)
        try:
            data = json.loads(cleaned)
            return {
                "score": int(data.get("score", 0)),
                "reasoning": str(data.get("reasoning", "")),
                "risks": list(data.get("risks", [])),
                "entry_recommendation": bool(data.get("entry_recommendation", False)),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "AIAnalyzer.analyze_new_coin: parse error for {} — {}: {!r}",
                symbol,
                exc,
                raw[:200],
            )
            return {
                "score": 0,
                "reasoning": "Failed to parse model response.",
                "risks": ["parse_error"],
                "entry_recommendation": False,
            }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences if present, then return the raw JSON."""
        text = text.strip()
        # Remove ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Attempt to isolate the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    @staticmethod
    def _float_or_none(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _hold_fallback(reason: str) -> AIDecision:
        return AIDecision(
            recommendation="HOLD",
            confidence="Low",
            stop_price=None,
            take_profit=None,
            reasoning=reason,
            risk_factors=["fallback: unable to obtain valid analysis"],
            news_relevance="neutral",
        )
