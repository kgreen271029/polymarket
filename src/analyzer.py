"""Rule-based technical analysis - 100% FREE, no API keys needed!"""

import os
import logging
import json


class MarketAnalyzer:
    """Analyzes market data using simple technical rules - ZERO cost!"""

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("✅ Technical Analysis Ready (100% FREE - No API Keys!)")

    def analyze_stock(self, symbol, price_history, news, technical_indicators):
        """Analyze stock using simple technical rules - FREE!"""
        return self._technical_analysis(symbol, price_history)

    def _technical_analysis(self, symbol, price_history):
        """Simple technical analysis using momentum and volatility."""
        if not price_history or len(price_history) < 2:
            return {
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": 0,
                "reasoning": "Insufficient price data"
            }

        current_price = price_history[-1]
        prev_price = price_history[-2] if len(price_history) > 1 else current_price

        # Calculate momentum
        change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0

        # Simple strategy: momentum + mean reversion
        if change_pct > 1.5:  # Strong uptrend
            return {
                "symbol": symbol,
                "recommendation": "BUY",
                "confidence": 75,
                "reasoning": f"Momentum: +{change_pct:.2f}% - Uptrend detected",
                "stop_loss": current_price * 0.94,
                "target_price": current_price * 1.08,
                "entry_price": current_price
            }
        elif change_pct < -1.5:  # Strong downtrend
            return {
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": 60,
                "reasoning": f"Downtrend: {change_pct:.2f}% - Waiting for reversal"
            }
        else:  # Sideways movement
            return {
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": 40,
                "reasoning": f"Consolidation: {change_pct:.2f}% - No clear signal"
            }

    def get_eod_summary(self, daily_trades, portfolio_value, pnl):
        """Generate end-of-day summary - FREE!"""
        pnl_pct = (pnl / float(os.getenv("STARTING_CAPITAL", 92.65))) * 100 if pnl != 0 else 0

        summary = f"""
📊 **EOD Trading Summary**
• Trades Executed: {len(daily_trades)}
• Portfolio Value: ${portfolio_value:.2f}
• Daily P&L: ${pnl:.2f} ({pnl_pct:.2f}%)
• Status: {'✅ Green' if pnl >= 0 else '❌ Red'}

Next trading session: Monday 9:25 AM ET
"""
        return summary
