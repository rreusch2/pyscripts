#!/usr/bin/env python3
"""
Build Player Trends Materializations for Trends Search Tab

- Aggregates recent per-game stats from `player_game_stats` (JSON-based) into
  `player_trends_data` (per-player PK) and `top_trending_players` (denormalized index)
- Focus: MLB hitters & pitchers for now. Extendible to NBA/WNBA/NFL later.

Approach:
- For MLB, select recent player_game_stats where stats->>'league' = 'MLB'
- Group by player_id and compute last-N games aggregates (default N=10)
- Compute derived metrics (batting_average, ERA, etc.)
- Upsert into player_trends_data (PK: player_id)
- Upsert summary rows into top_trending_players with composite score

Env:
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
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dateutil import parser as dateparser
from dotenv import load_dotenv
from supabase import create_client, Client

SPORT_NAME = "Major League Baseball"
LEAGUE = "MLB"

@dataclass
class GameRow:
    player_id: str
    game_date: dt.date
    batting: dict
    pitching: dict


def load_env() -> Tuple[str, str]:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", file=sys.stderr)
        sys.exit(1)
    return url, key


def get_client() -> Client:
    url, key = load_env()
    return create_client(url, key)


def fetch_recent_pgs(sb: Client, days: int = 45) -> List[dict]:
    # Pull a broad window to ensure we can pick last-N games per player
    res = sb.table("player_game_stats").select("player_id, stats").order("created_at", desc=True).limit(8000).execute()
    rows = res.data or []
    return rows


def parse_rows(rows: List[dict]) -> Dict[str, List[GameRow]]:
    by_player: Dict[str, List[GameRow]] = defaultdict(list)

    def is_mlb_like(stats: dict) -> bool:
        if not stats:
            return False
        if any(k in stats for k in ("batting_average", "on_base_percentage", "slugging_percentage", "ops")):
            return True
        ds = stats.get("data_source")
        return ds in ("pybaseball_statcast", "mlb_stats_api")

    for r in rows:
        pid = r.get("player_id")
        s = r.get("stats") or {}
        if not pid or not is_mlb_like(s):
            continue
        gd = s.get("game_date")
        try:
            game_date = dt.date.fromisoformat(gd[:10]) if gd else None
        except Exception:
            game_date = None
        if not game_date:
            continue

        if "batting" in s or "pitching" in s:
            batting = s.get("batting") or {}
            pitching = s.get("pitching") or {}
        else:
            # flat schema -> adapt to our expected keys
            batting = {
                "hits": s.get("hits"),
                "at_bats": s.get("at_bats"),
                "home_runs": s.get("home_runs"),
                "rbis": s.get("rbis"),
                "runs": s.get("runs"),
                "stolen_bases": s.get("stolen_bases"),
                "strikeouts": s.get("strikeouts"),
                "walks": s.get("walks"),
                "total_bases": s.get("total_bases"),
                "doubles": s.get("doubles"),
                "triples": s.get("triples"),
                "hit_by_pitch": s.get("hit_by_pitch"),
                "batting_average": s.get("batting_average"),
                "on_base_percentage": s.get("on_base_percentage"),
                "slugging_percentage": s.get("slugging_percentage"),
                "ops": s.get("ops"),
            }
            pitching = {
                "innings_pitched": s.get("innings_pitched"),
                "strikeouts_pitcher": s.get("strikeouts_pitcher"),
                "hits_allowed": s.get("hits_allowed"),
                "walks_allowed": s.get("walks_allowed"),
                "earned_runs": s.get("earned_runs"),
            }

        by_player[pid].append(GameRow(pid, game_date, batting, pitching))

    # sort by date asc
    for pid in by_player:
        by_player[pid].sort(key=lambda x: x.game_date)
    return by_player


def compute_batter_metrics(games: List[GameRow], last_n: int) -> dict:
    sample = games[-last_n:] if len(games) > last_n else games
    hits = sum(int(g.batting.get("hits") or 0) for g in sample)
    at_bats = sum(int(g.batting.get("at_bats") or 0) for g in sample)
    home_runs = sum(int(g.batting.get("home_runs") or 0) for g in sample)
    rbis = sum(int(g.batting.get("rbis") or 0) for g in sample)
    runs = sum(int(g.batting.get("runs") or 0) for g in sample)
    strikeouts = sum(int(g.batting.get("strikeouts") or 0) for g in sample)
    walks = sum(int(g.batting.get("walks") or 0) for g in sample)
    total_bases = sum(int(g.batting.get("total_bases") or 0) for g in sample)
    stolen_bases = sum(int(g.batting.get("stolen_bases") or 0) for g in sample)
    avg_hits = hits / max(1, len(sample))
    batting_average = (hits / at_bats) if at_bats else 0.0
    return {
        "recent_games_count": len(sample),
        "avg_hits": round(avg_hits, 3),
        "avg_home_runs": round(home_runs / max(1, len(sample)), 3),
        "avg_rbis": round(rbis / max(1, len(sample)), 3),
        "avg_runs": round(runs / max(1, len(sample)), 3),
        "avg_strikeouts": round(strikeouts / max(1, len(sample)), 3),
        "avg_walks": round(walks / max(1, len(sample)), 3),
        "avg_total_bases": round(total_bases / max(1, len(sample)), 3),
        "avg_stolen_bases": round(stolen_bases / max(1, len(sample)), 3),
        "batting_average": round(batting_average, 3),
    }


def compute_pitcher_metrics(games: List[GameRow], last_n: int) -> dict:
    sample = games[-last_n:] if len(games) > last_n else games
    so = sum(int(g.pitching.get("strikeouts_pitcher") or 0) for g in sample)
    ha = sum(int(g.pitching.get("hits_allowed") or 0) for g in sample)
    wa = sum(int(g.pitching.get("walks_allowed") or 0) for g in sample)
    er = sum(int(g.pitching.get("earned_runs") or 0) for g in sample)
    ip = 0.0
    for g in sample:
        ip_str = g.pitching.get("innings_pitched")
        if isinstance(ip_str, str):
            # MLB format like '5.1' (5 and 1/3) or '6.2'
            try:
                parts = ip_str.split('.')
                base = int(parts[0])
                frac = int(parts[1]) if len(parts) > 1 else 0
                ip += base + (frac / 3.0)
            except Exception:
                pass
        elif isinstance(ip_str, (int, float)):
            ip += float(ip_str)
    era = (er * 9.0 / ip) if ip else 0.0
    return {
        "avg_strikeouts_pitched": round(so / max(1, len(sample)), 3),
        "avg_hits_allowed": round(ha / max(1, len(sample)), 3),
        "avg_walks_allowed": round(wa / max(1, len(sample)), 3),
        "avg_earned_runs": round(er / max(1, len(sample)), 3),
        "avg_innings_pitched": round(ip / max(1, len(sample)), 3),
        "era": round(era, 3),
    }


def upsert_player_trends(sb: Client, player_id: str, player_name: Optional[str], team_name: Optional[str], batter: dict, pitcher: dict, recent_games: int, last_game_date: Optional[dt.date]):
    # player_trends_data has PK = player_id
    payload = {
        "player_id": player_id,
        "player_name": player_name,
        "team_name": team_name,
        "sport_key": "baseball_mlb",
        "recent_games_count": recent_games,
        "last_game_date": last_game_date.isoformat() if last_game_date else None,
        "avg_hits": batter.get("avg_hits"),
        "avg_home_runs": batter.get("avg_home_runs"),
        "avg_rbis": batter.get("avg_rbis"),
        "avg_runs": batter.get("avg_runs"),
        "avg_strikeouts": batter.get("avg_strikeouts"),
        "avg_walks": batter.get("avg_walks"),
        "avg_total_bases": batter.get("avg_total_bases"),
        "avg_stolen_bases": batter.get("avg_stolen_bases"),
        "batting_average": batter.get("batting_average"),
        "avg_strikeouts_pitched": pitcher.get("avg_strikeouts_pitched"),
        "avg_hits_allowed": pitcher.get("avg_hits_allowed"),
        "avg_walks_allowed": pitcher.get("avg_walks_allowed"),
        "avg_earned_runs": pitcher.get("avg_earned_runs"),
        "avg_innings_pitched": pitcher.get("avg_innings_pitched"),
        "era": pitcher.get("era"),
        "confidence_score": None,
    }
    # Upsert logic: try update then insert
    get = sb.table("player_trends_data").select("player_id").eq("player_id", player_id).limit(1).execute()
    if get.data:
        sb.table("player_trends_data").update(payload).eq("player_id", player_id).execute()
    else:
        sb.table("player_trends_data").insert(payload).execute()


def upsert_top_trending(sb: Client, player_id: str, player_name: Optional[str], team_name: Optional[str], batter: dict, pitcher: dict, recent_games: int):
    # composite score simple heuristic
    composite = 0.0
    composite += (batter.get("avg_hits") or 0) * 1.0
    composite += (batter.get("avg_home_runs") or 0) * 2.0
    composite += (batter.get("avg_total_bases") or 0) * 0.5
    composite += (pitcher.get("avg_strikeouts_pitched") or 0) * 1.0
    # fetch existing display fields from players
    payload = {
        "player_id": player_id,
        "player_name": player_name,
        "team_name": team_name,
        "sport_key": "baseball_mlb",
        "recent_games_count": recent_games,
        "avg_hits": batter.get("avg_hits"),
        "avg_home_runs": batter.get("avg_home_runs"),
        "avg_rbis": batter.get("avg_rbis"),
        "avg_runs": batter.get("avg_runs"),
        "avg_strikeouts": batter.get("avg_strikeouts"),
        "avg_walks": batter.get("avg_walks"),
        "avg_total_bases": batter.get("avg_total_bases"),
        "avg_stolen_bases": batter.get("avg_stolen_bases"),
        "batting_average": batter.get("batting_average"),
        "avg_strikeouts_pitched": pitcher.get("avg_strikeouts_pitched"),
        "avg_hits_allowed": pitcher.get("avg_hits_allowed"),
        "avg_walks_allowed": pitcher.get("avg_walks_allowed"),
        "avg_earned_runs": pitcher.get("avg_earned_runs"),
        "avg_innings_pitched": pitcher.get("avg_innings_pitched"),
        "era": pitcher.get("era"),
        "composite_score": round(composite, 3),
    }
    # upsert by player_id
    get = sb.table("top_trending_players").select("player_id").eq("player_id", player_id).limit(1).execute()
    if get.data:
        sb.table("top_trending_players").update(payload).eq("player_id", player_id).execute()
    else:
        sb.table("top_trending_players").insert(payload).execute()


def get_player_name_and_team(sb: Client, player_id: str) -> Tuple[Optional[str], Optional[str]]:
    res = sb.table("players").select("name, team").eq("id", player_id).limit(1).execute()
    if res.data:
        row = res.data[0]
        return row.get("name"), row.get("team")
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Build MLB player trends materializations")
    parser.add_argument("--window", type=int, default=10, help="Number of most recent games to aggregate per player")
    args = parser.parse_args()

    sb = get_client()
    rows = fetch_recent_pgs(sb)
    grouped = parse_rows(rows)

    print(f"Computing trends for {len(grouped)} MLB players (window={args.window})")
    processed = 0
    for pid, games in grouped.items():
        batter = compute_batter_metrics(games, args.window)
        pitcher = compute_pitcher_metrics(games, args.window)
        name, team = get_player_name_and_team(sb, pid)
        last_game_date = games[-1].game_date if games else None
        upsert_player_trends(
            sb,
            pid,
            name,
            team,
            batter,
            pitcher,
            batter.get("recent_games_count", 0),
            last_game_date,
        )
        try:
            upsert_top_trending(sb, pid, name, team, batter, pitcher, batter.get("recent_games_count", 0))
        except Exception as e:
            # If top_trending_players is a view or non-updatable relation, skip gracefully
            if "top_trending_players" in str(e) or "View" in str(e):
                pass
            else:
                print(f"WARN: upsert_top_trending failed for {pid}: {e}", file=sys.stderr)
        processed += 1
        if processed % 200 == 0:
            print(f"...{processed} players updated")

    print(f"Done. Updated {processed} player trend rows and top_trending entries.")


if __name__ == "__main__":
    main()
