"""AI-powered market analysis using Claude and Groq."""

import os
import logging
import json
from anthropic import Anthropic


class MarketAnalyzer:
    """Analyzes market data and generates trading signals."""

    def __init__(self, logger):
        self.logger = logger
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-5-sonnet-20241022"

    def analyze_stock(self, symbol, price_history, news, technical_indicators):
        """Analyze a stock using Claude."""
        try:
            context = self._build_analysis_context(
                symbol, price_history, news, technical_indicators
            )

            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            )

            response_text = message.content[0].text

            # Parse the response
            analysis = self._parse_analysis(response_text)
            return analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze {symbol}: {e}")
            return {
                "symbol": symbol,
                "recommendation": "HOLD",
                "confidence": 0.0,
                "reasoning": str(e)
            }

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
        try:
            context = f"""
Generate a brief end-of-day trading summary:
- Trades executed: {len(daily_trades)}
- Portfolio value: ${portfolio_value:.2f}
- Daily P&L: ${pnl:.2f}

Provide actionable insights for tomorrow.
"""

            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": context}]
            )

            return message.content[0].text

        except Exception as e:
            self.logger.error(f"Failed to generate EOD summary: {e}")
            return f"Daily P&L: ${pnl:.2f}"
