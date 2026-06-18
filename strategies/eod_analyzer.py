"""strategies/eod_analyzer.py — End-of-day research: scan setups, generate tomorrow's trade ideas."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from analysis.technical import TechnicalAnalyzer
from analysis.ai_analyzer import AIAnalyzer, AnalysisContext

WATCHLIST = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA",
    # High-volatility / momentum
    "AMD", "COIN", "MSTR", "PLTR", "ARM", "SMCI",
    # ETFs for market regime
    "SPY", "QQQ",
    # Financials + other
    "JPM", "GS", "BAC",
]

SCAN_INTERVAL = 3600 * 4  # run every 4 hours when market is closed


def _fetch_bars(symbol: str, period: str = "3mo") -> pd.DataFrame | None:
    """Download daily OHLCV bars from Yahoo Finance, normalized to TechnicalAnalyzer schema."""
    try:
        raw = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        if raw is None or len(raw) < 20:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.dropna()
        return df
    except Exception as e:
        logger.debug("[EOD] yfinance error for {}: {}", symbol, e)
        return None


def _fetch_intraday(symbol: str) -> pd.DataFrame | None:
    """5-min intraday bars for today — used for gap detection."""
    try:
        raw = yf.download(symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
        if raw is None or len(raw) < 10:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        return df.dropna()
    except Exception as e:
        logger.debug("[EOD] intraday error for {}: {}", symbol, e)
        return None


def _gap_pct(df_daily: pd.DataFrame) -> float:
    """Today's open vs yesterday's close, as a percentage."""
    if len(df_daily) < 2:
        return 0.0
    yesterday_close = float(df_daily["close"].iloc[-2])
    today_open = float(df_daily["open"].iloc[-1])
    if yesterday_close == 0:
        return 0.0
    return (today_open - yesterday_close) / yesterday_close * 100.0


def _earnings_soon(symbol: str, within_days: int = 7) -> bool:
    """Check if the stock has earnings within the next N days."""
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal is None:
            return False
        date_val = cal.get("Earnings Date") or cal.get("earningsDate")
        if date_val is None:
            return False
        if hasattr(date_val, "__iter__") and not isinstance(date_val, str):
            date_val = list(date_val)[0]
        if hasattr(date_val, "date"):
            date_val = date_val.date()
        from datetime import date
        today = date.today()
        days_out = (date_val - today).days
        return 0 <= days_out <= within_days
    except Exception:
        return False


def _score_setup(sig: dict, gap: float, earnings_soon: bool) -> int:
    """Score a stock 0-100 for trade potential based on technicals."""
    score = 0
    rsi = sig.get("rsi", 50)
    vol_ratio = sig.get("volume_ratio", 1.0)
    trend = sig.get("trend", "sideways")
    bb_pct = sig.get("bb_pct", 0.5)
    macd_dir = sig.get("macd_direction", "neutral")
    breakout = sig.get("breakout_20", False)
    price_vs_sma = sig.get("price_vs_sma20_pct", 0)

    # Momentum signals
    if trend == "uptrend":
        score += 20
    if macd_dir == "bullish":
        score += 15
    if breakout:
        score += 20
    if vol_ratio > 2.0:
        score += 15
    elif vol_ratio > 1.5:
        score += 8

    # RSI sweet spot (not overbought, trending)
    if 45 <= rsi <= 65:
        score += 10
    elif 35 <= rsi < 45:
        score += 8  # oversold recovery

    # BB position (riding upper band = momentum)
    if bb_pct > 0.8:
        score += 8
    elif bb_pct > 0.6:
        score += 4

    # Gap-up (catalyst)
    if gap > 3.0:
        score += 15
    elif gap > 1.5:
        score += 8

    # Earnings catalyst nearby (double-edged, but adds interest)
    if earnings_soon:
        score += 10

    # Price near SMA (trend following)
    if 0 < price_vs_sma < 3:
        score += 5

    return min(score, 100)


async def run_eod_scan(ai: AIAnalyzer) -> list[dict]:
    """
    Scan the watchlist, score each stock, send top candidates to Claude for analysis.
    Returns a list of trade idea dicts sorted by AI confidence.
    """
    logger.info("[EOD] Starting end-of-day scan for {} symbols...", len(WATCHLIST))
    candidates: list[dict] = []

    for symbol in WATCHLIST:
        try:
            df = _fetch_bars(symbol)
            if df is None:
                continue
            sig = TechnicalAnalyzer.build_signal_summary(df)
            if not sig:
                continue
            gap = _gap_pct(df)
            has_earnings = _earnings_soon(symbol)
            score = _score_setup(sig, gap, has_earnings)
            candidates.append({
                "symbol": symbol,
                "score": score,
                "sig": sig,
                "gap_pct": gap,
                "earnings_soon": has_earnings,
                "price": sig.get("latest_close", 0),
            })
        except Exception as e:
            logger.debug("[EOD] Error scanning {}: {}", symbol, e)

    # Sort by score, take top 6 for AI analysis
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = [c for c in candidates if c["score"] >= 30][:6]

    logger.info("[EOD] Top candidates: {}", [c["symbol"] for c in top])

    ideas: list[dict] = []
    for c in top:
        try:
            ctx = AnalysisContext(
                symbol=c["symbol"],
                asset_class="stock",
                strategy_name="EODAnalyzer",
                proposed_action="BUY",
                signal_summary=c["sig"],
                news_headlines=[],
                available_capital=92.0,
                open_position_count=0,
                daily_pnl_pct=0.0,
            )
            decision = await ai.analyze(ctx)
            if decision.recommendation == "HOLD":
                continue
            ideas.append({
                "symbol": c["symbol"],
                "price": c["price"],
                "score": c["score"],
                "gap_pct": c["gap_pct"],
                "earnings_soon": c["earnings_soon"],
                "recommendation": decision.recommendation,
                "confidence": decision.confidence,
                "stop": decision.stop_price,
                "target": decision.take_profit,
                "reasoning": decision.reasoning,
                "rsi": c["sig"].get("rsi", 0),
                "trend": c["sig"].get("trend", ""),
                "vol_ratio": c["sig"].get("volume_ratio", 1.0),
            })
        except Exception as e:
            logger.error("[EOD] AI analysis failed for {}: {}", c["symbol"], e)

    ideas.sort(key=lambda x: {"High": 3, "Medium": 2, "Low": 1}.get(x["confidence"], 0), reverse=True)
    logger.info("[EOD] Scan complete. {} trade ideas generated.", len(ideas))
    return ideas


def format_ideas_report(ideas: list[dict]) -> str:
    """Format the trade ideas list as a human-readable report."""
    if not ideas:
        return "No high-conviction setups found tonight. Market may be range-bound — waiting for clearer signals."

    lines = [f"Scan complete — {len(ideas)} trade ideas for tomorrow:\n"]
    for i, idea in enumerate(ideas, 1):
        conf_emoji = {"High": "🔥", "Medium": "⚡", "Low": "💡"}.get(idea["confidence"], "•")
        gap_str = f"  Gap: {idea['gap_pct']:+.1f}%" if abs(idea["gap_pct"]) > 0.5 else ""
        earn_str = "  ⚠️ Earnings soon" if idea["earnings_soon"] else ""
        stop_str = f"${idea['stop']:.2f}" if idea.get("stop") else "TBD"
        tgt_str  = f"${idea['target']:.2f}" if idea.get("target") else "TBD"
        lines.append(
            f"{i}. {conf_emoji} {idea['symbol']} — {idea['recommendation']} @ ${idea['price']:.2f}"
            f"\n   Confidence: {idea['confidence']} | Score: {idea['score']}/100 | RSI: {idea['rsi']:.0f} | Trend: {idea['trend']}"
            f"\n   Stop: {stop_str}  Target: {tgt_str}{gap_str}{earn_str}"
            f"\n   {idea['reasoning'][:180]}\n"
        )
    return "\n".join(lines)


async def run_once_and_print(ai: AIAnalyzer) -> str:
    ideas = await run_eod_scan(ai)
    return format_ideas_report(ideas)
