#!/usr/bin/env python3
"""
MLB Game and Team Stats Updater

- Backfills and updates MLB per-player game stats and per-team game results
- Uses MLB Stats API (public) to fetch daily schedule and boxscores
- Writes to Supabase tables:
  - player_game_stats (JSON-based, robust dedupe by external_game_id)
  - team_recent_stats (typed columns; dedupe by team_id + external_game_id)

Duplicate Safety Strategy (no DB schema changes required):
- player_game_stats: pre-check existence by (player_id, stats->>game_id)
- team_recent_stats: pre-check existence by (team_id, external_game_id)

Notes:
- This script assumes MLB teams exist in `teams` table with sport_key = 'baseball_mlb'
- Players are matched by external_player_id = f"mlb_{personId}"; created if absent
- Dates are handled in local system date; you can pass --start YYYY-MM-DD and --end YYYY-MM-DD

Integrate into cron by adding to daily automation after odds ingestion.

Env required (loaded via dotenv):
- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SERVICE_KEY)

Author: Cascade
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from dateutil import tz
from dotenv import load_dotenv
from supabase import create_client, Client

MLB_SPORT_ID = 1
SPORT_NAME = "Major League Baseball"
# Teams table in DB uses sport_key = 'MLB' (not 'baseball_mlb')
TEAM_SPORT_KEY = "MLB"
PLAYER_SPORT_KEY = "baseball_mlb"
MLB_TEAMS_ENDPOINT = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
MLB_SCHEDULE_ENDPOINT = "https://statsapi.mlb.com/api/v1/schedule"
MLB_BOXSCORE_ENDPOINT = "https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"

@dataclass
class TeamInfo:
    mlb_id: int
    name: str
    abbr: Optional[str]


def load_env() -> Tuple[str, str]:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment", file=sys.stderr)
        sys.exit(1)
    return url, key


def get_client() -> Client:
    url, key = load_env()
    return create_client(url, key)


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + dt.timedelta(days=1)


def fetch_mlb_teams() -> Dict[int, TeamInfo]:
    r = requests.get(MLB_TEAMS_ENDPOINT, timeout=20)
    r.raise_for_status()
    data = r.json()
    teams = {}
    for t in data.get("teams", []):
        teams[int(t["id"])] = TeamInfo(
            mlb_id=int(t["id"]),
            name=t.get("name"),
            abbr=t.get("abbreviation") or t.get("teamCode")
        )
    return teams


def _slug_key(text: str) -> str:
    s = (text or "").lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in [' ', '-', '_']:
            out.append('-')
    key = ''.join(out)
    while '--' in key:
        key = key.replace('--', '-')
    return key.strip('-')


def get_or_create_player(sb: Client, full_name: str, mlb_person_id: int, team_abbr: Optional[str], team_id: Optional[str]) -> Optional[str]:
    external_player_id = f"mlb_{mlb_person_id}"
    # 1) Try by external_player_id
    res = sb.table("players").select("id").eq("external_player_id", external_player_id).limit(1).execute()
    if res.data:
        return res.data[0]["id"]

    # 1b) Fallback by player_name + sport (avoid duplicates)
    by_name = (
        sb.table("players").select("id")
        .eq("sport", "MLB")
        .eq("player_name", full_name)
        .limit(1)
        .execute()
    )
    if by_name.data:
        return by_name.data[0]["id"]

    # 2) Create
    player_key = f"mlb_{_slug_key(full_name)}_{mlb_person_id}"
    payload = {
        "external_player_id": external_player_id,
        "name": full_name,
        "player_name": full_name,
        "sport": "MLB",
        "sport_key": PLAYER_SPORT_KEY,
        "team": team_abbr or None,
        "team_id": team_id or None,
        "active": True,
        "status": "active",
        "player_key": player_key,
        "metadata": {"source": "mlb_stats_api"}
    }
    ins = sb.table("players").insert(payload).execute()
    if ins.data:
        return ins.data[0]["id"]
    return None


def get_team_id(sb: Client, cache: Dict[str, str], team: TeamInfo) -> Optional[str]:
    if not team:
        return None
    key = f"{TEAM_SPORT_KEY}:{team.abbr or team.name}"
    if key in cache:
        return cache[key]

    # Prefer abbreviation match
    if team.abbr:
        q = sb.table("teams").select("id").eq("sport_key", TEAM_SPORT_KEY).ilike("team_abbreviation", team.abbr).limit(1).execute()
        if q.data:
            cache[key] = q.data[0]["id"]
            return cache[key]

    # Fallback: name match
    q2 = sb.table("teams").select("id").eq("sport_key", TEAM_SPORT_KEY).ilike("team_name", team.name).limit(1).execute()
    if q2.data:
        cache[key] = q2.data[0]["id"]
        return cache[key]
    # Last resort: match by name without sport_key (legacy rows)
    q3 = sb.table("teams").select("id").ilike("team_name", team.name).limit(1).execute()
    if q3.data:
        cache[key] = q3.data[0]["id"]
        return cache[key]
    # Special-case synonyms (MLB API sometimes reports just "Athletics")
    if (team.name or "").strip().lower() == "athletics":
        q4 = sb.table("teams").select("id").eq("sport_key", TEAM_SPORT_KEY).eq("team_abbreviation", "OAK").limit(1).execute()
        if q4.data:
            cache[key] = q4.data[0]["id"]
            return cache[key]
    return None


def schedule_for_date(date: dt.date) -> List[dict]:
    params = {
        "sportId": MLB_SPORT_ID,
        "date": date.strftime("%Y-%m-%d"),
    }
    r = requests.get(MLB_SCHEDULE_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    games = []
    for d in j.get("dates", []):
        for g in d.get("games", []):
            games.append(g)
    return games


def is_final(game: dict) -> bool:
    status = (game.get("status") or {}).get("detailedState")
    return status in {"Final", "Game Over", "Completed Early", "Completed", "Postponed"} or game.get("status", {}).get("statusCode") == "F"


def fetch_boxscore(game_pk: int) -> dict:
    r = requests.get(MLB_BOXSCORE_ENDPOINT.format(gamePk=game_pk), timeout=30)
    r.raise_for_status()
    return r.json()


def parse_batting_line(b: dict) -> dict:
    return {
        "hits": b.get("hits"),
        "at_bats": b.get("atBats"),
        "home_runs": b.get("homeRuns"),
        "rbis": b.get("rbi"),
        "runs": b.get("runs"),
        "stolen_bases": b.get("stolenBases"),
        "strikeouts": b.get("strikeOuts"),
        "walks": b.get("baseOnBalls"),
        "total_bases": b.get("totalBases"),
        # Additional rate/extra-base stats used by existing flat schema
        "doubles": b.get("doubles"),
        "triples": b.get("triples"),
        "hit_by_pitch": b.get("hitByPitch"),
        "batting_average": b.get("avg"),
        "on_base_percentage": b.get("obp"),
        "slugging_percentage": b.get("slg"),
        "ops": b.get("ops"),
    }


def parse_pitching_line(p: dict) -> dict:
    return {
        "innings_pitched": p.get("inningsPitched"),
        "strikeouts_pitcher": p.get("strikeOuts"),
        "hits_allowed": p.get("hits"),
        "walks_allowed": p.get("baseOnBalls"),
        "earned_runs": p.get("earnedRuns"),
    }


def upsert_player_game_stats(
    sb: Client,
    player_id: str,
    player_name: str,
    game_date: dt.date,
    team_abbr: Optional[str],
    opp_abbr: Optional[str],
    is_home: bool,
    batting: dict,
    pitching: dict,
):
    # Determine home/away abbreviations for flat schema
    home_abbr = team_abbr if is_home else (opp_abbr or None)
    away_abbr = (opp_abbr if is_home else team_abbr) or None

    # Dedupe by (player_id, game_date, home_team, away_team)
    exists = (
        sb.table("player_game_stats")
        .select("id")
        .eq("player_id", player_id)
        .filter("stats->>game_date", "eq", game_date.isoformat())
        .filter("stats->>home_team", "eq", home_abbr or "")
        .filter("stats->>away_team", "eq", away_abbr or "")
        .limit(1)
        .execute()
    )
    if exists.data:
        return False

    # Flatten payload to match existing pybaseball_statcast-style records
    hits = int(batting.get("hits") or 0)
    doubles = int(batting.get("doubles") or 0)
    triples = int(batting.get("triples") or 0)
    home_runs = int(batting.get("home_runs") or 0)
    singles = max(0, hits - doubles - triples - home_runs)
    at_bats = int(batting.get("at_bats") or 0)
    walks = int(batting.get("walks") or 0)
    strikeouts_b = int(batting.get("strikeouts") or 0)
    total_bases = int(batting.get("total_bases") or 0)
    hbp = int(batting.get("hit_by_pitch") or 0)

    def as_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    ba = as_float(batting.get("batting_average"))
    obp = as_float(batting.get("on_base_percentage"))
    slg = as_float(batting.get("slugging_percentage"))
    ops = as_float(batting.get("ops"))

    flat_stats = {
        "data_source": "mlb_stats_api",
        "is_real_data": True,
        "player_name": player_name,
        "game_date": game_date.isoformat(),
        "home_team": home_abbr,
        "away_team": away_abbr,
        "opponent_team": opp_abbr,
        # counting stats
        "hits": hits,
        "walks": walks,
        "at_bats": at_bats,
        "doubles": doubles,
        "singles": singles,
        "triples": triples,
        "home_runs": home_runs,
        "rbis": int(batting.get("rbis") or 0),
        "runs": int(batting.get("runs") or 0),
        "stolen_bases": int(batting.get("stolen_bases") or 0),
        "strikeouts": strikeouts_b,
        "total_bases": total_bases,
        "hit_by_pitch": hbp,
        # rate stats
        "batting_average": ba,
        "on_base_percentage": obp,
        "slugging_percentage": slg,
        "ops": ops,
        # pitcher stats if present (kept flat as well)
        "innings_pitched": pitching.get("innings_pitched"),
        "strikeouts_pitcher": pitching.get("strikeouts_pitcher"),
        "hits_allowed": pitching.get("hits_allowed"),
        "walks_allowed": pitching.get("walks_allowed"),
        "earned_runs": pitching.get("earned_runs"),
    }

    payload = {
        "player_id": player_id,
        "stats": flat_stats,
    }
    sb.table("player_game_stats").insert(payload).execute()
    return True


def upsert_team_recent_stats(sb: Client, team_id: str, team_name: str, game_date: dt.date, opponent_team: str, is_home: bool, team_score: Optional[int], opponent_score: Optional[int], external_game_id: str):
    # Dedupe by team_id + external_game_id
    exists = sb.table("team_recent_stats").select("id").eq("team_id", team_id).eq("external_game_id", external_game_id).limit(1).execute()
    if exists.data:
        return False

    margin = None
    game_result = None
    if team_score is not None and opponent_score is not None:
        margin = int(team_score) - int(opponent_score)
        game_result = "W" if margin > 0 else ("L" if margin < 0 else "T")

    payload = {
        "team_id": team_id,
        "team_name": team_name,
        "sport": SPORT_NAME,
        "sport_key": TEAM_SPORT_KEY,
        "game_date": game_date.isoformat(),
        "opponent_team": opponent_team,
        "is_home": is_home,
        "team_score": team_score,
        "opponent_score": opponent_score,
        "game_result": game_result,
        "margin": margin,
        "external_game_id": external_game_id,
    }
    sb.table("team_recent_stats").insert(payload).execute()
    return True


def latest_pgs_mlb_date(sb: Client) -> Optional[dt.date]:
    """Determine the latest MLB game_date present in player_game_stats.

    We detect MLB-like rows by either:
    - presence of flat stat keys (batting_average/OBP/SLG/OPS), or
    - data_source in ['pybaseball_statcast', 'mlb_stats_api']
    """
    res = sb.table("player_game_stats").select("created_at, stats").order("created_at", desc=True).limit(2000).execute()
    max_date = None
    for row in res.data or []:
        s = row.get("stats") or {}
        if not s:
            continue
        is_mlb_like = (
            ("batting_average" in s or "on_base_percentage" in s or "slugging_percentage" in s or "ops" in s)
            or (s.get("data_source") in ("pybaseball_statcast", "mlb_stats_api"))
        )
        if not is_mlb_like:
            continue
        gd = s.get("game_date")
        if not gd:
            continue
        try:
            d = dt.date.fromisoformat(gd[:10])
            if not max_date or d > max_date:
                max_date = d
        except Exception:
            continue
    return max_date


def main():
    parser = argparse.ArgumentParser(description="Update MLB player and team stats into Supabase")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (inclusive). Defaults to today.")
    parser.add_argument("--sleep", type=float, default=0.6, help="Sleep between network requests (sec)")
    args = parser.parse_args()

    sb = get_client()

    today = dt.date.today()
    start_date: Optional[dt.date] = dt.date.fromisoformat(args.start) if args.start else None
    end_date: dt.date = dt.date.fromisoformat(args.end) if args.end else today

    if not start_date:
        last = latest_pgs_mlb_date(sb)
        if last:
            start_date = last + dt.timedelta(days=1)
        else:
            # conservative default window
            start_date = today - dt.timedelta(days=7)
    if start_date > end_date:
        print(f"Up-to-date. Nothing to do (start {start_date} > end {end_date})")
        return

    print(f"Updating MLB stats from {start_date} to {end_date} (inclusive)")

    mlb_teams = fetch_mlb_teams()
    team_cache: Dict[str, str] = {}

    for day in daterange(start_date, end_date):
        try:
            games = schedule_for_date(day)
        except Exception as e:
            print(f"Schedule fetch failed for {day}: {e}", file=sys.stderr)
            continue

        print(f"{day}: {len(games)} scheduled games")
        for g in games:
            game_pk = g.get("gamePk")
            if not game_pk:
                continue
            game_pk = int(game_pk)
            external_game_id = f"mlb_{game_pk}"
            status = (g.get("status") or {}).get("detailedState")
            if not is_final(g):
                # Only ingest final/complete games
                continue

            # Team info and score from schedule (already final)
            home_info = g.get("teams", {}).get("home", {})
            away_info = g.get("teams", {}).get("away", {})
            home_team_name = ((home_info.get("team") or {}).get("name"))
            away_team_name = ((away_info.get("team") or {}).get("name"))
            home_score = home_info.get("score")
            away_score = away_info.get("score")

            # Resolve MLB team IDs via mapping
            home_mlb_id = ((home_info.get("team") or {}).get("id"))
            away_mlb_id = ((away_info.get("team") or {}).get("id"))
            home_team = mlb_teams.get(int(home_mlb_id)) if home_mlb_id else None
            away_team = mlb_teams.get(int(away_mlb_id)) if away_mlb_id else None

            # Map to Supabase teams
            home_team_id = get_team_id(sb, team_cache, home_team) if home_team else None
            away_team_id = get_team_id(sb, team_cache, away_team) if away_team else None

            # Insert team_recent_stats (home and away)
            try:
                if home_team_id:
                    upsert_team_recent_stats(
                        sb,
                        home_team_id,
                        home_team.name if home_team else (home_team_name or ""),
                        day,
                        away_team.name if away_team else (away_team_name or ""),
                        True,
                        home_score,
                        away_score,
                        external_game_id,
                    )
                else:
                    print(f"WARN: Could not resolve home team id for {home_team_name}")
                if away_team_id:
                    upsert_team_recent_stats(
                        sb,
                        away_team_id,
                        away_team.name if away_team else (away_team_name or ""),
                        day,
                        home_team.name if home_team else (home_team_name or ""),
                        False,
                        away_score,
                        home_score,
                        external_game_id,
                    )
                else:
                    print(f"WARN: Could not resolve away team id for {away_team_name}")
            except Exception as e:
                print(f"ERROR inserting team_recent_stats for game {external_game_id}: {e}", file=sys.stderr)

            # Fetch boxscore and insert per-player stats
            try:
                box = fetch_boxscore(game_pk)
                # Teams payload includes players dict keyed by 'IDxxxx'
                for side in ("home", "away"):
                    tblock = (box.get("teams") or {}).get(side) or {}
                    team_obj = tblock.get("team") or {}
                    parent_id = team_obj.get("id")
                    team_info = mlb_teams.get(int(parent_id)) if parent_id else None
                    opp_info = home_team if side == "away" else away_team
                    is_home = (side == "home")
                    # resolve supabase team_id for this side
                    side_team_id = get_team_id(sb, team_cache, team_info) if team_info else None

                    players = tblock.get("players") or {}
                    for pid, pdata in players.items():
                        person = pdata.get("person") or {}
                        full_name = person.get("fullName")
                        person_id = person.get("id")
                        if not full_name or not person_id:
                            continue
                        splits = (pdata.get("stats") or {}).get("batting") or {}
                        psplits = (pdata.get("stats") or {}).get("pitching") or {}

                        batting = parse_batting_line(splits)
                        pitching = parse_pitching_line(psplits)

                        # ensure player exists
                        player_id = get_or_create_player(
                            sb,
                            full_name,
                            int(person_id),
                            (team_info.abbr if team_info else None),
                            side_team_id,
                        )
                        if not player_id:
                            print(f"WARN: Could not create/find player {full_name} ({person_id})")
                            continue

                        # Insert player_game_stats
                        try:
                            upsert_player_game_stats(
                                sb,
                                player_id,
                                full_name,
                                day,
                                team_info.abbr if team_info else None,
                                opp_info.abbr if opp_info else None,
                                is_home,
                                batting,
                                pitching,
                            )
                        except Exception as e:
                            print(f"ERROR inserting player_game_stats {full_name} game {external_game_id}: {e}", file=sys.stderr)
                time.sleep(args.sleep)
            except Exception as e:
                print(f"ERROR boxscore fetch/parse for game {external_game_id}: {e}", file=sys.stderr)
                continue

    print("Done.")


if __name__ == "__main__":
    main()
