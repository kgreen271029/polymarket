"""AI-powered market analysis using Groq (FREE!)."""

import os
import logging
import json

try:
    from groq import Groq
except ImportError:
    Groq = None


class MarketAnalyzer:
    """Analyzes market data and generates trading signals using Groq."""

    def __init__(self, logger):
        self.logger = logger
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = "mixtral-8x7b-32768"

        if self.groq_key and Groq:
            try:
                self.client = Groq(api_key=self.groq_key)
                self.logger.info("✅ Groq AI loaded (FREE tier)")
            except Exception as e:
                self.logger.warning(f"Groq not available: {e}")
                self.client = None

    def analyze_stock(self, symbol, price_history, news, technical_indicators):
        """Analyze a stock using Groq (FREE AI)."""
        if not self.client:
            return self._fallback_analysis(symbol, price_history)

        try:
            context = self._build_analysis_context(
                symbol, price_history, news, technical_indicators
            )

            message = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.3,
                max_tokens=500,
            )

            response_text = message.choices[0].message.content

            # Parse the response
            analysis = self._parse_analysis(response_text)
            return analysis

        except Exception as e:
            self.logger.warning(f"Groq analysis failed for {symbol}, using fallback: {e}")
            return self._fallback_analysis(symbol, price_history)

    def _build_analysis_context(self, symbol, price_history, news, indicators):
        """Build analysis context for Claude."""
        context = f"""
Analyze {symbol} for trading opportunity.

Recent Price Data:
- Last 5 days: {price_history}
- Current Price: ${price_history[-1] if price_history else 'N/A'}

Recent News:
{self._format_news(news)}

Technical Indicators:
{json.dumps(indicators, indent=2)}

Based on this data, provide:
1. BUY, SELL, or HOLD recommendation
2. Confidence level (0-100)
3. Brief reasoning
4. Stop loss price if buying
5. Target price if buying

Respond in JSON format:
{{
    "recommendation": "BUY|SELL|HOLD",
    "confidence": 0-100,
    "reasoning": "...",
    "stop_loss": null or price,
    "target_price": null or price,
    "entry_price": null or price
}}
"""
        return context

    def _format_news(self, news_items):
        """Format news items for analysis."""
        if not news_items:
            return "No recent news available"

        formatted = []
        for i, item in enumerate(news_items[:5]):
            title = item.get("title", "")
            description = item.get("description", "")
            formatted.append(f"{i+1}. {title}\n   {description}")

        return "\n".join(formatted)

    def _parse_analysis(self, response_text):
        """Parse Claude's analysis response."""
        try:
            # Try to extract JSON from response
            import json
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except Exception as e:
            self.logger.warning(f"Failed to parse analysis JSON: {e}")

        # Fallback parsing
        rec = "HOLD"
        if "BUY" in response_text.upper():
            rec = "BUY"
        elif "SELL" in response_text.upper():
            rec = "SELL"

        return {
            "recommendation": rec,
            "confidence": 50,
            "reasoning": response_text[:200]
        }

    def get_eod_summary(self, daily_trades, portfolio_value, pnl):
        """Generate end-of-day summary."""
        if not self.client:
            return f"EOD: {len(daily_trades)} trades, Portfolio: ${portfolio_value:.2f}, P&L: ${pnl:.2f}"

        try:
            context = f"""Generate brief EOD summary:
- Trades: {len(daily_trades)}
- Portfolio: ${portfolio_value:.2f}
- Daily P&L: ${pnl:.2f}

Be concise."""

            message = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": context}],
                temperature=0.3,
                max_tokens=200,
            )

            return message.choices[0].message.content

        except Exception as e:
            self.logger.debug(f"EOD summary generation failed: {e}")
            return f"Daily P&L: ${pnl:.2f}"

    def _fallback_analysis(self, symbol, price_history):
        """Simple technical analysis fallback when AI is unavailable."""
        if not price_history or len(price_history) < 2:
            return {
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": 0,
                "reasoning": "Insufficient price data"
            }

        current_price = price_history[-1]
        prev_price = price_history[-2]

        # Simple momentum: if price going up, consider buying
        change_pct = ((current_price - prev_price) / prev_price) * 100

        if change_pct > 2:  # Up >2%, might be bullish
            return {
                "symbol": symbol,
                "recommendation": "BUY",
                "confidence": 60,
                "reasoning": f"Price up {change_pct:.2f}% - momentum signal",
                "stop_loss": current_price * 0.95,
                "target_price": current_price * 1.05,
                "entry_price": current_price
            }
        elif change_pct < -2:  # Down >2%, might be bearish
            return {
                "symbol": symbol,
                "recommendation": "SELL",
                "confidence": 50,
                "reasoning": f"Price down {change_pct:.2f}% - trend reversal signal"
            }
        else:
            return {
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": 40,
                "reasoning": f"Price change {change_pct:.2f}% - no clear signal"
            }
