"""
Multi-signal technical analysis engine. 100% free, no API keys.

Scores each stock across several independent, well-established signals and
combines them into a single confidence score. Entry, stop, and target are
derived from ATR (volatility) so they adapt to each stock instead of using
flat percentages.
"""

import os
from src import indicators as ind


class MarketAnalyzer:
    """Generates BUY/SELL/HOLD signals from real OHLCV data."""

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("Multi-signal technical analyzer ready (free, no API key)")

    def analyze(self, symbol, bars):
        """
        bars: dict from DataProvider.get_bars (close/high/low/volume lists).
        Returns a decision dict with recommendation, confidence, levels, signals.
        """
        closes = bars.get("close", [])
        highs = bars.get("high", [])
        lows = bars.get("low", [])
        volumes = bars.get("volume", [])
        price = bars.get("price", closes[-1] if closes else 0)

        if len(closes) < 35:
            return self._hold(symbol, price, "Insufficient history for indicators")

        # --- Compute indicators ---
        sma20 = ind.sma(closes, 20)
        sma50 = ind.sma(closes, 50) if len(closes) >= 50 else ind.sma(closes, len(closes) - 1)
        ema9 = ind.ema(closes, 9)
        rsi14 = ind.rsi(closes, 14)
        macd_line, macd_signal, macd_hist = ind.macd(closes)
        bb_up, bb_mid, bb_low = ind.bollinger(closes, 20, 2)
        atr14 = ind.atr(highs, lows, closes, 14)
        mom5 = ind.pct_change(closes, 5)

        # Volume confirmation: today's volume vs 20-day average
        vol_avg = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else (volumes[-1] if volumes else 0)
        vol_ratio = (volumes[-1] / vol_avg) if vol_avg else 1.0

        signals = []
        score = 0  # positive = bullish, negative = bearish

        # 1. Trend: price above/below key moving averages
        if sma20 and sma50:
            if price > sma20 > sma50:
                score += 2; signals.append("Uptrend (price>SMA20>SMA50)")
            elif price < sma20 < sma50:
                score -= 2; signals.append("Downtrend (price<SMA20<SMA50)")

        # 2. Short-term momentum vs EMA9
        if ema9:
            if price > ema9:
                score += 1; signals.append("Above EMA9")
            else:
                score -= 1; signals.append("Below EMA9")

        # 3. RSI: avoid overbought, favor recovering oversold
        if rsi14 is not None:
            if rsi14 < 30:
                score += 2; signals.append(f"Oversold (RSI {rsi14:.0f})")
            elif rsi14 > 70:
                score -= 2; signals.append(f"Overbought (RSI {rsi14:.0f})")
            elif 40 <= rsi14 <= 60:
                score += 1; signals.append(f"Neutral RSI {rsi14:.0f}")

        # 4. MACD crossover
        if macd_hist is not None:
            if macd_hist > 0:
                score += 1; signals.append("MACD bullish")
            else:
                score -= 1; signals.append("MACD bearish")

        # 5. Bollinger position (mean reversion)
        if bb_up and bb_low:
            if price <= bb_low:
                score += 1; signals.append("At lower Bollinger band")
            elif price >= bb_up:
                score -= 1; signals.append("At upper Bollinger band")

        # 6. Volume confirmation amplifies an existing bullish bias
        if vol_ratio > 1.3 and score > 0:
            score += 1; signals.append(f"High volume ({vol_ratio:.1f}x avg)")

        # --- Convert score to decision ---
        # Max achievable bullish score is ~8. Normalize to a 0-100 confidence.
        confidence = min(100, max(0, int((score / 8) * 100)))

        atr_pct = (atr14 / price * 100) if (atr14 and price) else 2.0
        reasoning = "; ".join(signals) if signals else "No clear signals"

        if score >= 4:
            stop = price - (atr14 * 1.5 if atr14 else price * 0.05)
            target = price + (atr14 * 3 if atr14 else price * 0.10)
            return {
                "symbol": symbol, "recommendation": "BUY", "confidence": confidence,
                "entry_price": price, "stop_loss": round(stop, 2),
                "target_price": round(target, 2), "atr_pct": round(atr_pct, 2),
                "score": score, "reasoning": reasoning,
            }
        elif score <= -3:
            return {
                "symbol": symbol, "recommendation": "SELL", "confidence": confidence,
                "entry_price": price, "score": score, "reasoning": reasoning,
            }
        return self._hold(symbol, price, reasoning, score)

    def _hold(self, symbol, price, reason, score=0):
        return {
            "symbol": symbol, "recommendation": "HOLD", "confidence": 0,
            "entry_price": price, "score": score, "reasoning": reason,
        }

    # Backwards-compatible wrapper (older callers used analyze_stock)
    def analyze_stock(self, symbol, price_history, news=None, technical_indicators=None):
        bars = {"close": price_history, "high": price_history,
                "low": price_history, "volume": [0] * len(price_history),
                "price": price_history[-1] if price_history else 0}
        return self.analyze(symbol, bars)

    def get_eod_summary(self, daily_trades, portfolio_value, pnl):
        start = float(os.getenv("STARTING_CAPITAL", 92.65))
        pnl_pct = (pnl / start) * 100 if start else 0
        return (
            f"EOD Summary | Trades: {len(daily_trades)} | "
            f"Portfolio: ${portfolio_value:.2f} | "
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
            f"{'GREEN' if pnl >= 0 else 'RED'}"
        )
