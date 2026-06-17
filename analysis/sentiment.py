"""analysis/sentiment.py — Fast keyword-based sentiment scoring (no API calls)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from data.market_data import HeadlineEvent

BULLISH_WORDS = [
    "surge", "soar", "rally", "breakout", "beat", "exceed", "upgrade",
    "partnership", "launch", "approval", "record", "bullish", "outperform",
    "strong", "growth", "profit", "gain", "boom", "milestone", "breakthrough",
    "positive", "optimistic", "upside", "buy", "accumulate", "moon", "pump",
    "adoption", "integration", "expand", "acquire", "merger", "deal",
]

BEARISH_WORDS = [
    "crash", "plunge", "fall", "miss", "fail", "downgrade", "sell", "dump",
    "lawsuit", "ban", "recall", "fraud", "hack", "breach", "investigation",
    "loss", "decline", "weak", "bearish", "risk", "concern", "warning",
    "halt", "suspend", "bankrupt", "default", "probe", "negative", "rug",
    "scam", "exit", "dump", "collapse", "penalty", "fine", "sec", "cftc",
]


@dataclass
class SentimentScore:
    score: float              # -1.0 to 1.0
    magnitude: float          # 0.0 to 1.0 (strength of signal)
    direction: str            # "bullish" | "bearish" | "neutral"
    bullish_hits: list[str] = field(default_factory=list)
    bearish_hits: list[str] = field(default_factory=list)


class SentimentScorer:
    def score_text(self, text: str) -> SentimentScore:
        lowered = text.lower()
        bull_hits = [w for w in BULLISH_WORDS if w in lowered]
        bear_hits = [w for w in BEARISH_WORDS if w in lowered]
        total = len(bull_hits) + len(bear_hits)
        if total == 0:
            return SentimentScore(score=0.0, magnitude=0.0, direction="neutral")

        score = (len(bull_hits) - len(bear_hits)) / total
        magnitude = min(1.0, total / 6.0)
        direction = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
        return SentimentScore(
            score=score,
            magnitude=magnitude,
            direction=direction,
            bullish_hits=bull_hits,
            bearish_hits=bear_hits,
        )

    def score_headline(self, event: HeadlineEvent) -> SentimentScore:
        combined = f"{event.title} {event.summary}"
        return self.score_text(combined)

    def aggregate_sentiment(
        self,
        scores: Sequence[tuple[SentimentScore, datetime]],
        window_minutes: int = 30,
    ) -> float:
        """Time-decayed weighted average; older events get less weight."""
        if not scores:
            return 0.0
        now = datetime.now(tz=timezone.utc)
        weighted_sum = 0.0
        weight_total = 0.0
        for score, ts in scores:
            ts_aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            age_minutes = (now - ts_aware).total_seconds() / 60.0
            age_weight = max(0.1, 1.0 - (age_minutes / window_minutes))
            w = age_weight * score.magnitude
            weighted_sum += score.score * w
            weight_total += w
        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def is_high_conviction(self, score: SentimentScore) -> bool:
        return abs(score.score) > 0.5 and score.magnitude > 0.3
