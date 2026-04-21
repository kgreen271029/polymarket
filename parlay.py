import json
import os
import sys
from datetime import date

from tabulate import tabulate

from analyzer import PickCandidate
from fetcher import FormData, TeamStats

SYSTEM_ROLE = """You are an expert NBA sports betting analyst with deep knowledge of basketball statistics, betting markets, and probability theory. You analyze statistical data to identify high-probability betting opportunities.

Your analysis framework:
1. Totals: Compare combined team scoring averages to the posted line. Weight recent form (last 5 games) more than season averages. Consider pace and defensive ratings.
2. Spreads: Look at ATS records, home/away differentials, rest advantages, and key player availability.
3. Moneylines: Confirm heavy favorites (≤-180) through recent form and situational edges.

Be critical. If a pick looks weak, say so. Focus on statistical edges, not narratives. Respond ONLY with valid JSON — no extra text."""

JSON_SCHEMA = """{
  "overall_reasoning": "string — why this parlay combination works together",
  "leg_analyses": [
    {
      "pick": "string (e.g. Celtics -5.5 or Thunder/Lakers Under 224.5)",
      "reasoning": "string — 2-3 sentences on why this pick is strong",
      "confidence": "High|Medium|Low",
      "key_factors": ["string — top 2-4 supporting factors"],
      "risks": ["string — top 1-3 risks to this pick"]
    }
  ],
  "parlay_odds_estimate": "string (e.g. +450 approx)",
  "recommendation": "string — brief final recommendation"
}"""


def select_parlay_legs(ranked_picks: list[PickCandidate]) -> list[PickCandidate]:
    """Select top 3 picks, max 1 per game."""
    seen_games: set[str] = set()
    legs: list[PickCandidate] = []

    for pick in ranked_picks:
        if pick.game_id not in seen_games:
            legs.append(pick)
            seen_games.add(pick.game_id)
        if len(legs) == 3:
            break

    if len(legs) < 3:
        print(f"[parlay] Warning: only {len(legs)} valid parlay leg(s) found.", file=sys.stderr)

    return legs


def build_stats_context(
    legs: list[PickCandidate],
    team_stats: dict[int, TeamStats],
    form: dict[int, FormData],
    injuries: dict[int, list[str]],
) -> str:
    lines = []
    for i, leg in enumerate(legs, 1):
        bet_label = _bet_label(leg)
        lines.append(f"--- LEG {i} ---")
        lines.append(f"GAME: {leg.away_team} @ {leg.home_team}")
        lines.append(f"BET: {bet_label} | LINE: {leg.line} | ODDS: {leg.price}")
        lines.append(f"STATISTICAL CONFIDENCE SCORE: {leg.confidence_score:.1f}/100")
        lines.append(f"KEY STATS: {json.dumps(leg.key_stats)}")
        lines.append(f"SCORE BREAKDOWN: {json.dumps(leg.scoring_breakdown)}")

        home_inj = injuries.get(leg.home_team, [])
        away_inj = injuries.get(leg.away_team, [])
        if home_inj:
            lines.append(f"HOME INJURIES: {', '.join(home_inj)}")
        if away_inj:
            lines.append(f"AWAY INJURIES: {', '.join(away_inj)}")
        lines.append("")

    return "\n".join(lines)


def _bet_label(pick: PickCandidate) -> str:
    bt = pick.bet_type
    if bt == "total_over":
        return f"OVER {pick.line}"
    if bt == "total_under":
        return f"UNDER {pick.line}"
    if bt == "spread_home":
        return f"{pick.home_team} {pick.line:+.1f}"
    if bt == "spread_away":
        sign = "+" if pick.line > 0 else ""
        return f"{pick.away_team} {sign}{pick.line:.1f}"
    if bt == "ml_home":
        return f"{pick.home_team} ML"
    if bt == "ml_away":
        return f"{pick.away_team} ML"
    return bt


def call_claude_for_analysis(
    legs: list[PickCandidate],
    stats_context: str,
    anthropic_client,
) -> dict:
    user_msg = (
        f"Today is {date.today().isoformat()}. "
        f"Analyze the following {len(legs)} parlay leg(s) selected by our statistical model.\n\n"
        "For each leg provide: reasoning (2-3 sentences), confidence (High/Medium/Low), "
        "key_factors (list), and risks (list). Then give overall_reasoning for the parlay combination.\n\n"
        f"Respond ONLY with valid JSON matching this schema:\n{JSON_SCHEMA}"
    )

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_ROLE,
                },
                {
                    "type": "text",
                    "text": f"STATISTICAL DATA FOR TODAY'S PARLAY:\n\n{stats_context}",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        cache_tokens = getattr(response.usage, "cache_read_input_tokens", 0)
        if cache_tokens:
            print(f"[parlay] Claude cache hit: {cache_tokens} cached tokens read.")

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"[parlay] Claude JSON parse error: {e}", file=sys.stderr)
        return _fallback_analysis(legs)
    except Exception as e:
        print(f"[parlay] Claude API error: {e}", file=sys.stderr)
        return _fallback_analysis(legs)


def _fallback_analysis(legs: list[PickCandidate]) -> dict:
    """Used when Claude is unavailable."""
    leg_analyses = []
    for leg in legs:
        leg_analyses.append({
            "pick": _bet_label(leg),
            "reasoning": f"Statistical model confidence: {leg.confidence_score:.1f}/100. "
                         f"Score breakdown: {leg.scoring_breakdown}",
            "confidence": "High" if leg.confidence_score >= 70 else "Medium" if leg.confidence_score >= 50 else "Low",
            "key_factors": [f"{k}: {v}" for k, v in leg.key_stats.items()][:4],
            "risks": ["AI analysis unavailable — review stats manually"],
        })
    return {
        "overall_reasoning": "AI analysis unavailable. Picks are based on statistical model scores.",
        "leg_analyses": leg_analyses,
        "parlay_odds_estimate": "N/A",
        "recommendation": "Review picks manually before betting.",
    }


def calculate_parlay_payout(legs: list[PickCandidate], stake: float = 100.0) -> dict:
    parlay_decimal = 1.0
    for leg in legs:
        ml = leg.price
        if ml < 0:
            decimal = (100 / abs(ml)) + 1
        else:
            decimal = (ml / 100) + 1
        parlay_decimal *= decimal

    payout = round(stake * parlay_decimal, 2)
    parlay_american = round((parlay_decimal - 1) * 100) if parlay_decimal >= 2 else round(-100 / (parlay_decimal - 1))

    return {
        "parlay_decimal": round(parlay_decimal, 3),
        "parlay_american": f"+{parlay_american}" if parlay_american >= 0 else str(parlay_american),
        "payout": payout,
        "stake": stake,
        "profit": round(payout - stake, 2),
    }


def format_parlay_output(
    legs: list[PickCandidate],
    claude_analysis: dict,
    payout_info: dict,
) -> str:
    today = date.today().strftime("%B %d, %Y")
    lines = []

    lines.append("")
    lines.append("╔" + "═" * 62 + "╗")
    lines.append(f"║{'NBA 3-LEG PARLAY PICKS — ' + today:^62}║")
    lines.append("╚" + "═" * 62 + "╝")
    lines.append("")

    # Summary table
    table_rows = []
    leg_analyses = claude_analysis.get("leg_analyses", [])
    for i, leg in enumerate(legs):
        label = _bet_label(leg)
        analysis = leg_analyses[i] if i < len(leg_analyses) else {}
        confidence = analysis.get("confidence", "?")
        game_label = f"{leg.away_team[:12]} @ {leg.home_team[:12]}"
        price_str = f"{leg.price:+d}" if leg.price != 0 else "pk"
        table_rows.append([
            i + 1,
            game_label,
            label,
            f"{leg.line:+.1f}" if leg.line != 0 else "—",
            price_str,
            f"{leg.confidence_score:.0f}/100",
            confidence,
        ])

    headers = ["#", "Game", "Pick", "Line", "Odds", "Stat Score", "AI Conf"]
    lines.append(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))
    lines.append("")

    # Per-leg detailed analysis
    for i, (leg, analysis) in enumerate(zip(legs, leg_analyses), 1):
        label = _bet_label(leg)
        lines.append(f"{'━' * 64}")
        lines.append(f"  LEG {i}: {leg.away_team} @ {leg.home_team} — {label}")
        lines.append(f"{'━' * 64}")
        lines.append(f"  {analysis.get('reasoning', '')}")
        lines.append("")

        factors = analysis.get("key_factors", [])
        if factors:
            lines.append("  Key factors:")
            for f in factors:
                lines.append(f"    ✦ {f}")

        risks = analysis.get("risks", [])
        if risks:
            lines.append("  Risks:")
            for r in risks:
                lines.append(f"    ⚠  {r}")
        lines.append("")

    # Overall reasoning
    overall = claude_analysis.get("overall_reasoning", "")
    if overall:
        lines.append("━" * 64)
        lines.append("  PARLAY OVERVIEW")
        lines.append("━" * 64)
        lines.append(f"  {overall}")
        lines.append("")

    # Payout estimate
    lines.append("━" * 64)
    lines.append(f"  PAYOUT ESTIMATE  (${payout_info['stake']:.0f} stake)")
    lines.append("━" * 64)
    lines.append(f"  Combined odds  : {payout_info['parlay_american']}")
    lines.append(f"  Estimated payout: ${payout_info['payout']:.2f}")
    lines.append(f"  Profit if hits : ${payout_info['profit']:.2f}")
    lines.append("")

    rec = claude_analysis.get("recommendation", "")
    if rec:
        lines.append(f"  Recommendation: {rec}")
        lines.append("")

    lines.append("━" * 64)
    lines.append("  ⚠  For informational purposes only. Bet responsibly.")
    lines.append("━" * 64)
    lines.append("")

    return "\n".join(lines)


def print_parlay(formatted_output: str) -> None:
    print(formatted_output)
