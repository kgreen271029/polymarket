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
from analysis.filters import (
    earnings_blackout,
    sector_momentum_ok,
    volatility_regime_ok,
    top_sectors,
    symbol_in_top_sectors,
    kelly_position_size,
)
from analysis.multifactor import score_symbol, chandelier_exit
from analysis.regime import detect_regime

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
    Scan the watchlist, score each stock, send top candidates to AI for analysis.
    Returns a list of trade idea dicts sorted by AI confidence.
    """
    logger.info("[EOD] Starting end-of-day scan for {} symbols...", len(WATCHLIST))

    # Market-wide filters (run once)
    regime_ok, regime_reason = volatility_regime_ok()
    if not regime_ok:
        logger.info("[EOD] Choppy regime detected ({}), lowering score threshold", regime_reason)
    market_regime = detect_regime()
    logger.info("[EOD] Market regime: {} (size x{:.1f}) — {}",
                market_regime.name, market_regime.size_mult, market_regime.detail)
    hot_sectors = top_sectors(3)
    logger.info("[EOD] Hot sectors: {}", hot_sectors)

    # SPY for relative-strength factor in multi-factor scoring
    spy_df = _fetch_bars("SPY", period="6mo")

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

            # Earnings blackout — mark but don't exclude (user may want to know)
            earn_blocked, earn_note = earnings_blackout(symbol, window_days=14)

            # Sector momentum filter
            sec_ok, sec_note = sector_momentum_ok(symbol)
            in_hot_sector = symbol_in_top_sectors(symbol, hot_sectors)

            score = _score_setup(sig, gap, has_earnings)

            # Multi-factor composite (momentum/trend/RS/VCP) — blend 50/50
            mf = {"composite": 50.0}
            if spy_df is not None and len(df) >= 60:
                try:
                    mf = score_symbol(df, spy_df)
                    score = int(score * 0.5 + mf["composite"] * 0.5)
                except Exception:
                    pass

            # Bonus for hot sector alignment
            if in_hot_sector:
                score = min(100, score + 8)
            # Penalty for trading against sector trend
            if not sec_ok:
                score = max(0, score - 15)
            # Penalty for earnings risk
            if earn_blocked:
                score = max(0, score - 20)

            candidates.append({
                "symbol":        symbol,
                "score":         score,
                "sig":           sig,
                "gap_pct":       gap,
                "earnings_soon": has_earnings,
                "earn_blocked":  earn_blocked,
                "earn_note":     earn_note,
                "sec_ok":        sec_ok,
                "sec_note":      sec_note,
                "in_hot_sector": in_hot_sector,
                "multifactor":   mf.get("composite", 0),
                "price":         sig.get("latest_close", 0),
                "regime_ok":     regime_ok,
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
            atr = c["sig"].get("atr", c["price"] * 0.02)
            ideas.append({
                "symbol": c["symbol"],
                "price": c["price"],
                "score": c["score"],
                "gap_pct": c["gap_pct"],
                "earnings_soon": c["earnings_soon"],
                "recommendation": decision.recommendation,
                "confidence": decision.confidence,
                "stop": decision.stop_price or round(c["price"] - 2.2 * atr, 2),
                "target": decision.take_profit or round(c["price"] + 3.3 * atr, 2),
                "reasoning": decision.reasoning,
                "rsi": c["sig"].get("rsi", 0),
                "trend": c["sig"].get("trend", ""),
                "vol_ratio": c["sig"].get("volume_ratio", 1.0),
                "vcp": c["multifactor"],
                "rs": round(c["multifactor"] - 50, 1),  # relative strength delta from avg
                "chandelier_stop": 0.0,  # filled below if bars available
            })
        except Exception as e:
            logger.error("[EOD] AI analysis failed for {}: {}", c["symbol"], e)

    ideas.sort(key=lambda x: {"High": 3, "Medium": 2, "Low": 1}.get(x["confidence"], 0), reverse=True)
    logger.info("[EOD] Scan complete. {} trade ideas generated.", len(ideas))
    return ideas


def _kelly_note(idea: dict, cash: float = 92.0) -> str:
    price  = idea.get("price", 0)
    stop   = idea.get("stop")
    if not price or not stop or stop >= price:
        return ""
    size = kelly_position_size(cash, price, stop, win_rate=0.50, avg_win_loss_ratio=1.5, max_pct=0.30)
    shares = size / price if price else 0
    return f"  Kelly size: ${size:.2f} ({shares:.3f} shares)"


def format_ideas_report(ideas: list[dict]) -> str:
    """Format the trade ideas list as a human-readable report."""
    if not ideas:
        return "No high-conviction setups found tonight. Market may be range-bound — waiting for clearer signals."

    lines = [f"Scan complete — {len(ideas)} trade ideas for tomorrow:\n"]
    for i, idea in enumerate(ideas, 1):
        conf_emoji = {"High": "🔥", "Medium": "⚡", "Low": "💡"}.get(idea["confidence"], "•")
        gap_str  = f"  Gap: {idea['gap_pct']:+.1f}%" if abs(idea["gap_pct"]) > 0.5 else ""
        earn_str = "  ⚠️ Earnings soon" if idea["earnings_soon"] else ""
        stop_str = f"${idea['stop']:.2f}" if idea.get("stop") else "TBD"
        tgt_str  = f"${idea['target']:.2f}" if idea.get("target") else "TBD"
        rs_val   = idea.get("rs", 0)
        rs_str   = f"  RS: {rs_val:+.1f}" if rs_val else ""
        vcp_val  = idea.get("vcp", 0)
        vcp_str  = f"  MF: {vcp_val:.0f}/100" if vcp_val else ""
        kelly_str = _kelly_note(idea)
        sec_str   = f"  Sector: {idea.get('sec_note','')}" if not idea.get("sec_ok") else ""
        # Calculate R:R ratio for display
        price = idea.get("price", 0)
        stop  = idea.get("stop", 0)
        tgt   = idea.get("target", 0)
        rr_str = ""
        if price and stop and tgt and price > stop:
            risk   = price - stop
            reward = tgt - price
            rr_str = f"  R:R {reward/risk:.1f}x" if risk > 0 else ""
        lines.append(
            f"{i}. {conf_emoji} {idea['symbol']} — {idea['recommendation']} @ ${idea['price']:.2f}"
            f"\n   Confidence: {idea['confidence']} | Score: {idea['score']}/100"
            f" | RSI: {idea['rsi']:.0f} | Trend: {idea['trend']}{vcp_str}{rs_str}"
            f"\n   Stop: {stop_str}  Target: {tgt_str}{rr_str}{gap_str}{earn_str}{sec_str}"
            f"\n{kelly_str}"
            f"\n   {idea['reasoning'][:200]}\n"
        )
    return "\n".join(lines)


async def run_once_and_print(ai: AIAnalyzer) -> str:
    ideas = await run_eod_scan(ai)
    return format_ideas_report(ideas)
