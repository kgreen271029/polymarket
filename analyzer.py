from dataclasses import dataclass, field

from fetcher import FormData, GameInfo, OddsData, TeamStats


@dataclass
class PickCandidate:
    game_id: str
    home_team: str
    away_team: str
    bet_type: str          # "total_over", "total_under", "spread_home", "spread_away", "ml_home", "ml_away"
    line: float
    price: int             # American odds
    confidence_score: float
    scoring_breakdown: dict = field(default_factory=dict)
    key_stats: dict = field(default_factory=dict)


def american_to_implied_prob(american_odds: int) -> float:
    if american_odds == 0:
        return 0.5
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    return 100 / (american_odds + 100)


def linear_trend(values: list[float]) -> float:
    """Returns slope of a simple linear regression."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0.0


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_totals_pick(
    odds: OddsData,
    home_stats: TeamStats,
    away_stats: TeamStats,
    home_form: FormData,
    away_form: FormData,
    injuries: dict[int, list[str]],
) -> tuple[PickCandidate, PickCandidate]:

    line = odds.total_line
    combined_avg = home_stats.pts_per_game + away_stats.pts_per_game
    deviation = (combined_avg - line) / line if line > 0 else 0.0

    # Recent scoring trend (last 5 games)
    home_last5 = home_stats.last10_pts[-5:] if home_stats.last10_pts else []
    away_last5 = away_stats.last10_pts[-5:] if away_stats.last10_pts else []
    home_trend = linear_trend(home_last5) if home_last5 else 0.0
    away_trend = linear_trend(away_last5) if away_last5 else 0.0
    avg_trend = (home_trend + away_trend) / 2

    # Pace score
    avg_pace = (home_stats.pace + away_stats.pace) / 2
    pace_score = _clip((avg_pace - 95) / (115 - 95) * 20, 0, 20)

    # Line value
    line_value = _clip(abs(deviation) * 200, 0, 15)

    # Injury penalty (2.5 pts each)
    home_inj = len(injuries.get(home_stats.team_id, []))
    away_inj = len(injuries.get(away_stats.team_id, []))
    injury_penalty = _clip((home_inj + away_inj) * 2.5, 0, 10)

    # Combined recent pts from form data
    combined_recent = home_form.avg_pts_last5 + away_form.avg_pts_last5
    recent_dev = (combined_recent - line) / line if line > 0 and combined_recent > 0 else deviation

    # Compute over score
    # deviation_score: 30pts based on how far above/below line combined avg is
    deviation_score = _clip(30 + deviation * 200, 0, 30)
    trend_score = _clip(avg_trend * 5 + 12.5, 0, 25)

    over_raw = deviation_score + trend_score + pace_score + line_value + 10 - injury_penalty
    over_score = _clip(over_raw, 0, 95)
    under_score = _clip(100 - over_raw + injury_penalty, 0, 95)

    breakdown_over = {
        "deviation_score": round(deviation_score, 1),
        "trend_score": round(trend_score, 1),
        "pace_score": round(pace_score, 1),
        "line_value": round(line_value, 1),
        "injury_penalty": round(injury_penalty, 1),
    }
    key_stats = {
        "home_ppg": home_stats.pts_per_game,
        "away_ppg": away_stats.pts_per_game,
        "combined_avg": round(combined_avg, 1),
        "line": line,
        "deviation_pct": round(deviation * 100, 1),
        "home_pace": home_stats.pace,
        "away_pace": away_stats.pace,
        "home_last5_avg": round(home_form.avg_pts_last5, 1),
        "away_last5_avg": round(away_form.avg_pts_last5, 1),
        "combined_recent_avg": round(combined_recent, 1),
    }

    over_pick = PickCandidate(
        game_id=odds.event_id,
        home_team=odds.home_team,
        away_team=odds.away_team,
        bet_type="total_over",
        line=line,
        price=odds.over_price,
        confidence_score=over_score,
        scoring_breakdown=breakdown_over,
        key_stats=key_stats,
    )
    under_pick = PickCandidate(
        game_id=odds.event_id,
        home_team=odds.home_team,
        away_team=odds.away_team,
        bet_type="total_under",
        line=line,
        price=odds.under_price,
        confidence_score=under_score,
        scoring_breakdown={k: -v if "score" in k and k != "injury_penalty" else v
                           for k, v in breakdown_over.items()},
        key_stats=key_stats,
    )
    return over_pick, under_pick


def score_spread_pick(
    odds: OddsData,
    home_stats: TeamStats,
    away_stats: TeamStats,
    home_form: FormData,
    away_form: FormData,
    injuries: dict[int, list[str]],
) -> tuple[PickCandidate, PickCandidate]:

    spread = odds.home_spread  # negative = home favored

    # W% differential (25 pts)
    w_diff = abs(home_stats.w_pct - away_stats.w_pct)
    w_pct_score = _clip(w_diff * 100 * 0.25, 0, 25)

    # Home/away split (20 pts)
    home_split_score = _clip(home_stats.home_w_pct * 20, 0, 20)

    # Last 5 avg margin vs spread (25 pts)
    home_margin = home_form.avg_pts_last5 - home_form.avg_opp_pts_last5
    vs_spread = home_margin - spread
    vs_spread_score = _clip(vs_spread * 2 + 12.5, 0, 25)

    # Rest advantage (15 pts)
    rest_diff = home_form.rest_days - away_form.rest_days
    rest_score = _clip(rest_diff * 5 + 7.5, 0, 15)

    # Injury penalty
    home_inj = len(injuries.get(home_stats.team_id, []))
    away_inj = len(injuries.get(away_stats.team_id, []))
    home_injury_penalty = _clip(home_inj * 3.0, 0, 15)
    away_injury_penalty = _clip(away_inj * 3.0, 0, 15)

    home_raw = w_pct_score + home_split_score + vs_spread_score + rest_score + 15 - home_injury_penalty
    home_score = _clip(home_raw, 0, 95)
    away_score = _clip(100 - home_raw - away_injury_penalty + home_injury_penalty, 0, 95)

    key_stats = {
        "home_w_pct": home_stats.w_pct,
        "away_w_pct": away_stats.w_pct,
        "home_home_w_pct": home_stats.home_w_pct,
        "away_away_w_pct": away_stats.away_w_pct,
        "home_last5_margin": round(home_margin, 1),
        "spread": spread,
        "margin_vs_spread": round(vs_spread, 1),
        "home_rest_days": home_form.rest_days,
        "away_rest_days": away_form.rest_days,
    }

    home_spread_pick = PickCandidate(
        game_id=odds.event_id,
        home_team=odds.home_team,
        away_team=odds.away_team,
        bet_type="spread_home",
        line=spread,
        price=odds.home_spread_price,
        confidence_score=home_score,
        scoring_breakdown={"w_pct": round(w_pct_score, 1), "home_split": round(home_split_score, 1),
                           "vs_spread": round(vs_spread_score, 1), "rest": round(rest_score, 1),
                           "injury_penalty": round(home_injury_penalty, 1)},
        key_stats=key_stats,
    )
    away_spread_pick = PickCandidate(
        game_id=odds.event_id,
        home_team=odds.home_team,
        away_team=odds.away_team,
        bet_type="spread_away",
        line=-spread,
        price=odds.away_spread_price,
        confidence_score=away_score,
        scoring_breakdown={"w_pct": round(w_pct_score, 1), "away_split": round(away_stats.away_w_pct * 20, 1),
                           "rest": round(rest_score, 1), "injury_penalty": round(away_injury_penalty, 1)},
        key_stats=key_stats,
    )
    return home_spread_pick, away_spread_pick


def score_moneyline_pick(
    odds: OddsData,
    home_stats: TeamStats,
    away_stats: TeamStats,
    home_form: FormData,
    away_form: FormData,
    injuries: dict[int, list[str]],
) -> tuple[PickCandidate | None, PickCandidate | None]:
    """Only returns candidates for heavy favorites (-180 or better)."""
    results = []

    for is_home in (True, False):
        ml = odds.home_ml if is_home else odds.away_ml
        if ml >= 0 or ml > -180:
            results.append(None)
            continue

        stats = home_stats if is_home else away_stats
        opp_stats = away_stats if is_home else home_stats
        form = home_form if is_home else away_form
        inj_count = len(injuries.get(stats.team_id, []))

        impl_prob = american_to_implied_prob(ml)
        impl_score = _clip(impl_prob * 60, 0, 40)

        season_score = _clip(stats.w_pct * 25, 0, 25)

        last5_wins = form.last5_results.count("W")
        last5_score = _clip((last5_wins / 5) * 20, 0, 20)

        rest_score = _clip(form.rest_days * 3, 0, 15)

        injury_penalty = _clip(inj_count * 4.0, 0, 20)

        raw = impl_score + season_score + last5_score + rest_score - injury_penalty
        if impl_prob < 0.60:
            raw = min(raw, 40)

        score = _clip(raw, 0, 95)

        bet_type = "ml_home" if is_home else "ml_away"
        results.append(PickCandidate(
            game_id=odds.event_id,
            home_team=odds.home_team,
            away_team=odds.away_team,
            bet_type=bet_type,
            line=0.0,
            price=ml,
            confidence_score=score,
            scoring_breakdown={
                "implied_prob_score": round(impl_score, 1),
                "season_w_pct_score": round(season_score, 1),
                "last5_score": round(last5_score, 1),
                "rest_score": round(rest_score, 1),
                "injury_penalty": round(injury_penalty, 1),
            },
            key_stats={
                "ml_odds": ml,
                "implied_prob": round(impl_prob * 100, 1),
                "season_w_pct": stats.w_pct,
                "last5_results": form.last5_results,
                "rest_days": form.rest_days,
                "opp_w_pct": opp_stats.w_pct,
            },
        ))

    home_res = results[0] if results else None
    away_res = results[1] if len(results) > 1 else None
    return home_res, away_res


def build_pick_candidates(
    games: list[GameInfo],
    odds_list: list[OddsData],
    team_stats: dict[int, TeamStats],
    form: dict[int, FormData],
    injuries: dict[int, list[str]],
) -> list[PickCandidate]:

    # Index odds by event_id and market, keeping best bookmaker (first seen)
    totals_by_event: dict[str, OddsData] = {}
    spreads_by_event: dict[str, OddsData] = {}
    ml_by_event: dict[str, OddsData] = {}

    for o in odds_list:
        if o.market == "totals" and o.event_id not in totals_by_event:
            totals_by_event[o.event_id] = o
        elif o.market == "spreads" and o.event_id not in spreads_by_event:
            spreads_by_event[o.event_id] = o
        elif o.market == "h2h" and o.event_id not in ml_by_event:
            ml_by_event[o.event_id] = o

    candidates: list[PickCandidate] = []
    default_stats = TeamStats(team_id=0, team_name="Unknown")
    default_form = FormData(team_id=0)

    for game in games:
        event_id = game.odds_event_id or game.game_id
        home_s = team_stats.get(game.home_team_id, default_stats)
        away_s = team_stats.get(game.away_team_id, default_stats)
        home_f = form.get(game.home_team_id, default_form)
        away_f = form.get(game.away_team_id, default_form)

        if event_id in totals_by_event:
            o, u = score_totals_pick(totals_by_event[event_id], home_s, away_s, home_f, away_f, injuries)
            candidates.extend([o, u])

        if event_id in spreads_by_event:
            hs, aw = score_spread_pick(spreads_by_event[event_id], home_s, away_s, home_f, away_f, injuries)
            candidates.extend([hs, aw])

        if event_id in ml_by_event:
            hm, am = score_moneyline_pick(ml_by_event[event_id], home_s, away_s, home_f, away_f, injuries)
            for pick in (hm, am):
                if pick is not None:
                    candidates.append(pick)

    return [c for c in candidates if c.confidence_score >= 30.0]


def rank_picks(candidates: list[PickCandidate]) -> list[PickCandidate]:
    """Sort by confidence_score descending, enforce max 2 picks per game."""
    sorted_picks = sorted(candidates, key=lambda c: c.confidence_score, reverse=True)
    game_count: dict[str, int] = {}
    ranked: list[PickCandidate] = []

    for pick in sorted_picks:
        count = game_count.get(pick.game_id, 0)
        if count < 2:
            ranked.append(pick)
            game_count[pick.game_id] = count + 1

    return ranked
