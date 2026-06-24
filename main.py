#!/usr/bin/env python3
"""
NBA Parlay Bot — generates a 3-leg parlay with the highest statistical chance of hitting.

Usage:
  python main.py             # Run once immediately
  python main.py --schedule  # Run daily at RUN_TIME (default 10:00 AM ET)
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def run_bot() -> None:
    odds_api_key = os.getenv("ODDS_API_KEY", "")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not odds_api_key:
        print("[main] Warning: ODDS_API_KEY not set. Odds-based picks will be skipped.")
    if not anthropic_api_key:
        print("[main] Warning: ANTHROPIC_API_KEY not set. AI analysis will use fallback mode.")

    # ── 1. Fetch today's games ──────────────────────────────────────────────
    print("[main] Fetching today's NBA games...")
    from fetcher import (
        fetch_injury_report,
        fetch_odds,
        fetch_recent_form,
        fetch_team_stats,
        get_todays_nba_games,
    )

    games = get_todays_nba_games(odds_api_key)

    if not games:
        print("\n  No NBA games found today. Check back tomorrow!\n")
        return

    print(f"[main] Found {len(games)} game(s) today.")
    for g in games:
        print(f"        {g.away_team} @ {g.home_team}")

    # ── 2. Fetch odds ───────────────────────────────────────────────────────
    print("[main] Fetching odds...")
    odds_list = fetch_odds(odds_api_key, games)

    if not odds_list and not odds_api_key:
        print("[main] No odds available — parlay will be based on stats only.")

    # ── 3. Collect all team IDs ─────────────────────────────────────────────
    all_team_ids = list({
        tid
        for g in games
        for tid in (g.home_team_id, g.away_team_id)
        if tid > 0
    })

    # ── 4. Fetch team stats and recent form ─────────────────────────────────
    print("[main] Fetching team stats...")
    team_stats = fetch_team_stats(all_team_ids)

    print("[main] Fetching recent form...")
    form = fetch_recent_form(all_team_ids)

    print("[main] Fetching injury report...")
    injuries = fetch_injury_report(all_team_ids)

    # ── 5. Score and rank all pick candidates ───────────────────────────────
    print("[main] Analyzing picks...")
    from analyzer import build_pick_candidates, rank_picks

    candidates = build_pick_candidates(games, odds_list, team_stats, form, injuries)

    if not candidates:
        print("\n  Not enough data to generate picks today. "
              "Check your API keys and try again.\n")
        return

    ranked = rank_picks(candidates)
    print(f"[main] Scored {len(candidates)} pick candidates, ranked top {len(ranked)}.")

    # ── 6. Select parlay legs ────────────────────────────────────────────────
    from parlay import (
        build_stats_context,
        calculate_parlay_payout,
        call_claude_for_analysis,
        format_parlay_output,
        print_parlay,
        select_parlay_legs,
    )

    legs = select_parlay_legs(ranked)

    if not legs:
        print("\n  Unable to select parlay legs. Insufficient data.\n")
        return

    # ── 7. Build stats context for Claude ────────────────────────────────────
    stats_context = build_stats_context(legs, team_stats, form, injuries)

    # ── 8. Claude AI analysis ────────────────────────────────────────────────
    anthropic_client = None
    if anthropic_api_key:
        try:
            import anthropic
            anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        except ImportError:
            print("[main] anthropic package not installed. Run: pip install anthropic",
                  file=sys.stderr)

    print("[main] Running AI analysis...")
    claude_analysis = call_claude_for_analysis(legs, stats_context, anthropic_client)

    # ── 9. Format and print ──────────────────────────────────────────────────
    payout_info = calculate_parlay_payout(legs)
    output = format_parlay_output(legs, claude_analysis, payout_info)
    print_parlay(output)


def main() -> None:
    if "--schedule" in sys.argv:
        from scheduler import run_scheduled
        run_scheduled(run_bot)
    else:
        run_bot()


if __name__ == "__main__":
    main()
