import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
BALLDONTLIE_BASE = "https://api.balldontlie.io/nba/v1"
NBA_API_DELAY = 1.0  # seconds between nba_api calls to avoid rate limiting


@dataclass
class GameInfo:
    game_id: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    commence_time: str
    odds_event_id: str = ""


@dataclass
class OddsData:
    event_id: str
    home_team: str
    away_team: str
    market: str
    bookmaker: str
    total_line: float = 0.0
    over_price: int = -110
    under_price: int = -110
    home_spread: float = 0.0
    home_spread_price: int = -110
    away_spread_price: int = -110
    home_ml: int = 0
    away_ml: int = 0


@dataclass
class TeamStats:
    team_id: int
    team_name: str
    pts_per_game: float = 0.0
    opp_pts_per_game: float = 0.0
    pace: float = 100.0
    w_pct: float = 0.5
    last10_pts: list = field(default_factory=list)
    last10_opp_pts: list = field(default_factory=list)
    home_w_pct: float = 0.5
    away_w_pct: float = 0.5
    ats_wins: int = 0
    ats_losses: int = 0


@dataclass
class FormData:
    team_id: int
    last5_results: list = field(default_factory=list)
    avg_pts_last5: float = 0.0
    avg_opp_pts_last5: float = 0.0
    rest_days: int = 1


def current_season() -> str:
    today = date.today()
    year = today.year
    month = today.month
    if month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    return f"{year - 1}-{str(year)[2:]}"


def _safe_get(url: str, params: dict, timeout: int = 10) -> dict | None:
    try:
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0 NBAParlaBot/1.0"})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[fetcher] GET {url} failed: {e}", file=sys.stderr)
        return None


def get_todays_schedule() -> list[GameInfo]:
    """Primary: nba_api live scoreboard. Fallback: BallDontLie."""
    games = _fetch_schedule_nba_api()
    if not games:
        games = _fetch_schedule_balldontlie()
    return games


def _fetch_schedule_nba_api() -> list[GameInfo]:
    try:
        from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
        time.sleep(NBA_API_DELAY)
        sb = live_scoreboard.ScoreBoard()
        data = sb.get_dict()
        game_list = []
        for g in data.get("scoreboard", {}).get("games", []):
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            game_list.append(GameInfo(
                game_id=str(g.get("gameId", "")),
                home_team=home.get("teamCity", "") + " " + home.get("teamName", ""),
                away_team=away.get("teamCity", "") + " " + away.get("teamName", ""),
                home_team_id=int(home.get("teamId", 0)),
                away_team_id=int(away.get("teamId", 0)),
                commence_time=g.get("gameEt", ""),
            ))
        return game_list
    except Exception as e:
        print(f"[fetcher] nba_api schedule failed: {e}", file=sys.stderr)
        return []


def _fetch_schedule_balldontlie() -> list[GameInfo]:
    today_str = date.today().isoformat()
    data = _safe_get(f"{BALLDONTLIE_BASE}/games", {"dates[]": today_str})
    if not data:
        return []
    games = []
    for g in data.get("data", []):
        home = g.get("home_team", {})
        away = g.get("visitor_team", {})
        games.append(GameInfo(
            game_id=str(g.get("id", "")),
            home_team=home.get("full_name", home.get("name", "")),
            away_team=away.get("full_name", away.get("name", "")),
            home_team_id=int(home.get("id", 0)),
            away_team_id=int(away.get("id", 0)),
            commence_time=g.get("date", today_str),
        ))
    return games


def get_todays_nba_games(odds_api_key: str) -> list[GameInfo]:
    """Fetch today's games from Odds API (most reliable for matching with odds)."""
    if not odds_api_key:
        return get_todays_schedule()

    data = _safe_get(
        f"{ODDS_API_BASE}/sports/basketball_nba/events",
        {"apiKey": odds_api_key, "dateFormat": "iso"}
    )
    if not data:
        return get_todays_schedule()

    today_str = date.today().isoformat()
    games = []
    for ev in data:
        commence = ev.get("commence_time", "")
        if today_str not in commence:
            continue
        games.append(GameInfo(
            game_id=ev.get("id", ""),
            home_team=ev.get("home_team", ""),
            away_team=ev.get("away_team", ""),
            home_team_id=0,
            away_team_id=0,
            commence_time=commence,
            odds_event_id=ev.get("id", ""),
        ))

    if games:
        _enrich_game_ids(games)
    return games


def _enrich_game_ids(games: list[GameInfo]) -> None:
    """Try to fill in nba_api team IDs for games fetched from Odds API."""
    try:
        from nba_api.stats.static import teams as nba_teams
        all_teams = nba_teams.get_teams()
        name_to_id = {}
        for t in all_teams:
            name_to_id[t["full_name"]] = t["id"]
            name_to_id[t["nickname"]] = t["id"]
            name_to_id[t["city"] + " " + t["nickname"]] = t["id"]

        for g in games:
            if g.home_team_id == 0:
                g.home_team_id = _fuzzy_team_id(g.home_team, name_to_id)
            if g.away_team_id == 0:
                g.away_team_id = _fuzzy_team_id(g.away_team, name_to_id)
    except Exception as e:
        print(f"[fetcher] team ID enrichment failed: {e}", file=sys.stderr)


def _fuzzy_team_id(name: str, name_to_id: dict) -> int:
    if name in name_to_id:
        return name_to_id[name]
    for key, tid in name_to_id.items():
        if name.lower() in key.lower() or key.lower() in name.lower():
            return tid
    return 0


def fetch_odds(odds_api_key: str, games: list[GameInfo]) -> list[OddsData]:
    if not odds_api_key:
        print("[fetcher] No ODDS_API_KEY — skipping odds fetch.", file=sys.stderr)
        return []

    data = _safe_get(
        f"{ODDS_API_BASE}/sports/basketball_nba/odds",
        {
            "apiKey": odds_api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": "draftkings,fanduel,betmgm",
        }
    )
    if data is None:
        return []

    remaining = None
    odds_list: list[OddsData] = []

    today_str = date.today().isoformat()

    for event in data:
        if today_str not in event.get("commence_time", ""):
            continue
        event_id = event.get("id", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        for bookmaker in event.get("bookmakers", []):
            bk_name = bookmaker.get("key", "")
            for market in bookmaker.get("markets", []):
                mkt = market.get("key", "")
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if mkt == "totals":
                    over = outcomes.get("Over", {})
                    under = outcomes.get("Under", {})
                    if over and under:
                        odds_list.append(OddsData(
                            event_id=event_id,
                            home_team=home_team,
                            away_team=away_team,
                            market="totals",
                            bookmaker=bk_name,
                            total_line=float(over.get("point", 220.0)),
                            over_price=int(over.get("price", -110)),
                            under_price=int(under.get("price", -110)),
                        ))

                elif mkt == "spreads":
                    home_spread_data = outcomes.get(home_team, {})
                    away_spread_data = outcomes.get(away_team, {})
                    if home_spread_data:
                        odds_list.append(OddsData(
                            event_id=event_id,
                            home_team=home_team,
                            away_team=away_team,
                            market="spreads",
                            bookmaker=bk_name,
                            home_spread=float(home_spread_data.get("point", 0.0)),
                            home_spread_price=int(home_spread_data.get("price", -110)),
                            away_spread_price=int(away_spread_data.get("price", -110) if away_spread_data else -110),
                        ))

                elif mkt == "h2h":
                    home_ml_data = outcomes.get(home_team, {})
                    away_ml_data = outcomes.get(away_team, {})
                    if home_ml_data:
                        odds_list.append(OddsData(
                            event_id=event_id,
                            home_team=home_team,
                            away_team=away_team,
                            market="h2h",
                            bookmaker=bk_name,
                            home_ml=int(home_ml_data.get("price", 0)),
                            away_ml=int(away_ml_data.get("price", 0) if away_ml_data else 0),
                        ))

    if remaining is not None:
        print(f"[fetcher] Odds API requests remaining: {remaining}")
    elif odds_list:
        print(f"[fetcher] Fetched odds for {len(set(o.event_id for o in odds_list))} games.")

    return odds_list


def fetch_team_stats(team_ids: list[int]) -> dict[int, TeamStats]:
    stats = {}
    valid_ids = [tid for tid in team_ids if tid > 0]

    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        season = current_season()
        time.sleep(NBA_API_DELAY)

        result = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_simple="PerGame",
            measure_type_detailed_defense="Base",
        )
        df = result.get_data_frames()[0]

        time.sleep(NBA_API_DELAY)
        home_result = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_simple="PerGame",
            location_nullable="Home",
        )
        home_df = home_result.get_data_frames()[0]

        time.sleep(NBA_API_DELAY)
        away_result = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_simple="PerGame",
            location_nullable="Road",
        )
        away_df = away_result.get_data_frames()[0]

        home_wpct = dict(zip(home_df["TEAM_ID"], home_df["W_PCT"]))
        away_wpct = dict(zip(away_df["TEAM_ID"], away_df["W_PCT"]))

        for _, row in df.iterrows():
            tid = int(row["TEAM_ID"])
            if valid_ids and tid not in valid_ids:
                continue
            stats[tid] = TeamStats(
                team_id=tid,
                team_name=str(row.get("TEAM_NAME", "")),
                pts_per_game=float(row.get("PTS", 0)),
                opp_pts_per_game=float(row.get("OPP_PTS", row.get("PT_DIFF", 0))),
                pace=float(row.get("PACE", 100.0)) if "PACE" in row else 100.0,
                w_pct=float(row.get("W_PCT", 0.5)),
                home_w_pct=float(home_wpct.get(tid, 0.5)),
                away_w_pct=float(away_wpct.get(tid, 0.5)),
            )

        return stats

    except Exception as e:
        print(f"[fetcher] nba_api team stats failed: {e}", file=sys.stderr)
        return _fetch_team_stats_balldontlie(valid_ids)


def _fetch_team_stats_balldontlie(team_ids: list[int]) -> dict[int, TeamStats]:
    stats = {}
    season_year = int(current_season().split("-")[0])
    data = _safe_get(
        f"{BALLDONTLIE_BASE}/season_averages",
        {"season": season_year, "team_ids[]": team_ids}
    )
    if not data:
        return {tid: TeamStats(team_id=tid, team_name="Unknown") for tid in team_ids}

    for entry in data.get("data", []):
        tid = int(entry.get("team_id", 0))
        stats[tid] = TeamStats(
            team_id=tid,
            team_name="",
            pts_per_game=float(entry.get("pts", 0)),
        )
    return stats


def fetch_recent_form(team_ids: list[int], last_n: int = 10) -> dict[int, FormData]:
    form = {}
    valid_ids = [tid for tid in team_ids if tid > 0]

    try:
        from nba_api.stats.endpoints import leaguegamelog
        season = current_season()
        time.sleep(NBA_API_DELAY)

        result = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation="T",
            sorter="DATE",
            direction="DESC",
        )
        df = result.get_data_frames()[0]

        today = date.today()

        for tid in valid_ids:
            team_df = df[df["TEAM_ID"] == tid].head(last_n)
            if team_df.empty:
                form[tid] = FormData(team_id=tid)
                continue

            last5 = team_df.head(5)
            results = []
            for _, row in last5.iterrows():
                wl = str(row.get("WL", ""))
                results.append("W" if wl.startswith("W") else "L")

            pts_col = "PTS" if "PTS" in team_df.columns else None
            opp_col = None
            for c in ["OPP_PTS", "PTS_OPP", "PT_DIFF"]:
                if c in team_df.columns:
                    opp_col = c
                    break

            avg_pts = float(last5[pts_col].mean()) if pts_col else 0.0
            avg_opp = float(last5[opp_col].mean()) if opp_col else 0.0

            last_game_date = None
            if "GAME_DATE" in team_df.columns:
                try:
                    last_game_date = datetime.strptime(
                        str(team_df.iloc[0]["GAME_DATE"]), "%Y-%m-%d"
                    ).date()
                except Exception:
                    pass

            rest_days = (today - last_game_date).days if last_game_date else 1

            form[tid] = FormData(
                team_id=tid,
                last5_results=results,
                avg_pts_last5=avg_pts,
                avg_opp_pts_last5=avg_opp,
                rest_days=max(0, rest_days),
            )

        return form

    except Exception as e:
        print(f"[fetcher] nba_api recent form failed: {e}", file=sys.stderr)
        return {tid: FormData(team_id=tid) for tid in valid_ids}


def fetch_injury_report(team_ids: list[int]) -> dict[int, list[str]]:
    data = _safe_get(f"{BALLDONTLIE_BASE}/player_injuries", {})
    if not data:
        return {}

    injuries: dict[int, list[str]] = {}
    for entry in data.get("data", []):
        player = entry.get("player", {})
        team = player.get("team", {})
        tid = int(team.get("id", 0))
        if tid in team_ids:
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
            injuries.setdefault(tid, []).append(name)

    return injuries
