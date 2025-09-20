#!/usr/bin/env python3
"""
Daily AI Report Generator
Intelligently analyzes sports data, trends, and betting patterns to generate insightful reports
Uses advanced prompt engineering with xAI Grok for autonomous analysis
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Set
from dotenv import load_dotenv
import httpx
import re
from supabase import create_client, Client
from collections import defaultdict

# Load environment variables
load_dotenv()

# Mapping of NFL abbreviations to full team names for display matching
NFL_ABBR_TO_FULL = {
    'LAC': 'Los Angeles Chargers',
    'LV': 'Las Vegas Raiders',
    'CHI': 'Chicago Bears',
    'TB': 'Tampa Bay Buccaneers',
    'HOU': 'Houston Texans',
    'KC': 'Kansas City Chiefs',
    'SEA': 'Seattle Seahawks',
    'NYJ': 'New York Jets',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'GB': 'Green Bay Packers',
    'MIN': 'Minnesota Vikings',
    'DET': 'Detroit Lions',
    'NO': 'New Orleans Saints',
    'ATL': 'Atlanta Falcons',
    'NE': 'New England Patriots',
    'BUF': 'Buffalo Bills',
    'MIA': 'Miami Dolphins',
    'NYG': 'New York Giants',
    'PHI': 'Philadelphia Eagles',
    'WAS': 'Washington Commanders',
    'BAL': 'Baltimore Ravens',
    'PIT': 'Pittsburgh Steelers',
    'CIN': 'Cincinnati Bengals',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'TEN': 'Tennessee Titans',
    'ARI': 'Arizona Cardinals',
    'LAR': 'Los Angeles Rams',
    'SF': 'San Francisco 49ers'
}

def infer_team_display(player_team: Optional[str], sport: Optional[str], ev: Dict[str, Any]) -> Optional[str]:
    """
    Return the event team name to display for a player prop, or None if uncertain.
    We never guess outside the event's teams; if player's team doesn't match either team (after simple NFL abbr mapping), we omit.
    """
    if not ev:
        return None
    home = (ev.get('home_team') or '').strip()
    away = (ev.get('away_team') or '').strip()
    pt = (player_team or '').strip()
    sp = (sport or '').strip()
    # Exact match
    if pt and (pt == home or pt == away):
        return pt
    # NFL simple abbr->full comparison
    if sp in ('NFL', 'National Football League', 'americanfootball_nfl') and pt in NFL_ABBR_TO_FULL:
        full = NFL_ABBR_TO_FULL.get(pt)
        if full == home or full == away:
            return full
    # Unknown/conflict -> None to avoid mislabeling
    return None

class StatMuseClient:
    """Minimal client for the ParleyApp StatMuse API service.

    Endpoints supported by backend proxy or direct service:
    - GET /health
    - POST /query
    - POST /head-to-head
    - POST /team-record
    - POST /player-stats
    """

    def __init__(self, base_url: Optional[str] = None):
        # Prefer env, then argument, then local default
        self.base_url = base_url or os.getenv("STATMUSE_API_URL") or "http://127.0.0.1:5001"
        self._client = httpx.AsyncClient(timeout=15.0)

    async def get(self, path: str) -> Dict[str, Any]:
        try:
            resp = await self._client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = await self._client.post(
                f"{self.base_url}{path}", json=body, headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def health(self) -> Dict[str, Any]:
        return await self.get("/health")

    async def query(self, q: str) -> Dict[str, Any]:
        return await self.post("/query", {"query": q})

    async def head_to_head(self, team_a: str, team_b: str, sport: Optional[str] = None) -> Dict[str, Any]:
        payload = {"team_a": team_a, "team_b": team_b}
        if sport:
            payload["sport"] = sport
        return await self.post("/head-to-head", payload)

    async def team_record(self, team: str, sport: Optional[str] = None) -> Dict[str, Any]:
        payload = {"team": team}
        if sport:
            payload["sport"] = sport
        return await self.post("/team-record", payload)

    async def player_stats(self, player: str, sport: Optional[str] = None) -> Dict[str, Any]:
        payload = {"player": player}
        if sport:
            payload["sport"] = sport
        return await self.post("/player-stats", payload)

    async def close(self):
        try:
            await self._client.aclose()
        except Exception:
            pass


class DailyAIReportGenerator:
    def __init__(self):
        """Initialize the AI Report Generator with database and API connections"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        # Prefer service role key for unrestricted, accurate reads; fall back to anon
        self.supabase_service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        self.supabase_key = self.supabase_service_role_key or os.getenv('SUPABASE_ANON_KEY')
        self.xai_api_key = os.getenv('XAI_API_KEY')

        if not all([self.supabase_url, self.supabase_key, self.xai_api_key]):
            raise ValueError("Missing required environment variables")

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.xai_client = httpx.AsyncClient(timeout=120.0)  # Increased timeout for Grok-4 reasoning
        # Optional StatMuse client for ground-truth checks
        self.statmuse = StatMuseClient(os.getenv('STATMUSE_API_URL'))
        
    @staticmethod
    def _extract_capitalized_phrases(text: str) -> List[str]:
        """Extract simple capitalized multi-word phrases to heuristically detect entity mentions.
        Strips markdown headings and ignores common report structure phrases to reduce false positives.
        """
        # Remove markdown headings and bullet markers
        lines = []
        for line in text.splitlines():
            if line.lstrip().startswith('#'):
                continue  # ignore headings
            if line.lstrip().startswith(('-', '*')):
                line = line.lstrip('-* ').strip()
            lines.append(line)
        cleaned_text = '\n'.join(lines)

        # Two+ capitalized words in a row (e.g., 'Los Angeles', 'Keenan Allen')
        candidates = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", cleaned_text)
        # Common non-entity phrases to ignore
        stop = {
            'Executive Summary', 'Hot Trends', 'Statistical Edges', 'High Confidence Plays',
            'Risk Alerts', 'Expert Insights', 'Odds Movements', 'No Data', 'No data available',
            'Daily Sports', 'Sports Analytics', 'Mobile Display',
            'Generated At', 'Generated at', 'Games Today', 'Games Tomorrow',
            'Quick Look', 'Upcoming Games', 'Quick Look: Upcoming Games',
            'Daily Sports Analytics Report',
            # odds field labels that are not entities
            'Over Odds', 'Under Odds'
        }

        # Helper: minimal NFL abbreviation to full-name map to align with sports_events
        nfl_abbr_to_full = {
            'LAC': 'Los Angeles Chargers',
            'LV': 'Las Vegas Raiders',
            'CHI': 'Chicago Bears',
            'TB': 'Tampa Bay Buccaneers',
            'HOU': 'Houston Texans',
            'KC': 'Kansas City Chiefs',
            'SEA': 'Seattle Seahawks',
            'NYJ': 'New York Jets',
            'CLE': 'Cleveland Browns',
            'DAL': 'Dallas Cowboys',
            'GB': 'Green Bay Packers',
            'MIN': 'Minnesota Vikings',
            'DET': 'Detroit Lions',
            'NO': 'New Orleans Saints',
            'ATL': 'Atlanta Falcons',
            'NE': 'New England Patriots',
            'BUF': 'Buffalo Bills',
            'MIA': 'Miami Dolphins',
            'NYG': 'New York Giants',
            'PHI': 'Philadelphia Eagles',
            'WAS': 'Washington Commanders',
            'BAL': 'Baltimore Ravens',
            'PIT': 'Pittsburgh Steelers',
            'CIN': 'Cincinnati Bengals',
            'IND': 'Indianapolis Colts',
            'JAX': 'Jacksonville Jaguars',
            'TEN': 'Tennessee Titans',
            'ARI': 'Arizona Cardinals',
            'LAR': 'Los Angeles Rams',
            'SF': 'San Francisco 49ers'
        }

        def infer_team_display(player_team: Optional[str], sport: Optional[str], ev: Dict[str, Any]) -> Optional[str]:
            """Return the event team name to display for a player prop, or None if uncertain.
            We never guess outside the event's teams; if player's team doesn't match either team (after simple NFL abbr mapping), we omit.
            """
            if not ev:
                return None
            home = (ev.get('home_team') or '').strip()
            away = (ev.get('away_team') or '').strip()
            pt = (player_team or '').strip()
            sp = (sport or '').strip()
            # Exact match
            if pt and (pt == home or pt == away):
                return pt
            # NFL simple abbr->full comparison
            if sp in ('NFL', 'National Football League', 'americanfootball_nfl') and pt in nfl_abbr_to_full:
                full = nfl_abbr_to_full.get(pt)
                if full == home or full == away:
                    return full
            # Unknown/conflict -> None to avoid mislabeling
            return None
        cleaned = []
        for c in candidates:
            c_strip = c.strip()
            if c_strip in stop:
                continue
            cleaned.append(c_strip)
        return list(dict.fromkeys(cleaned))  # de-duplicate, preserve order

    @staticmethod
    def _validate_entities(text: str, allowed_players: List[str], allowed_teams: List[str], allowed_props: List[str]) -> Dict[str, Any]:
        """Return a validation report with any unknown phrases that look like entities."""
        phrases = DailyAIReportGenerator._extract_capitalized_phrases(text)
        allowed = set([*allowed_players, *allowed_teams, *allowed_props])
        allowed_lower = {a.lower() for a in allowed}
        generic_stop = {
            'Team Context', 'Limited Prop Data', 'Start Time',
            'Executive Summary', 'Hot Trends', 'Statistical Edges', 'High Confidence Plays',
            'Risk Alerts', 'Expert Insights', 'Odds Movements', 'No Data', 'No data available',
        }
        # Allow partial matches where phrase is substring of an allowed entity (e.g., 'Dodgers' within 'Los Angeles Dodgers')
        def normalize(p: str) -> str:
            q = p.strip()
            # Remove trailing qualifiers like Over/Under/Total
            q = re.sub(r"\s+(Over|Under|Total)$", "", q, flags=re.IGNORECASE)
            return q
        def is_allowed(p: str) -> bool:
            if p in generic_stop:
                return True
            q = normalize(p)
            ql = q.lower()
            if ql in allowed_lower:
                return True
            for a in allowed:
                al = a.lower()
                if ql in al or al in ql:
                    return True
            return False
        unknown = [p for p in phrases if not is_allowed(p)]
        return {
            'phrases_found': phrases,
            'unknown_entities': unknown,
            'unknown_count': len(unknown)
        }
        
    async def get_current_date_context(self) -> Dict[str, Any]:
        """Get current date and determine which sports are in season"""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = now + timedelta(days=1)
        tomorrow_start = today_start + timedelta(days=1)

        # Check for games today and tomorrow
        today_games = (
            self.supabase
            .table('sports_events')
            .select('id,sport,league,home_team,away_team,start_time')
            .gte('start_time', today_start.isoformat())
            .lt('start_time', tomorrow_start.isoformat())
            .execute()
        )

        tomorrow_games = (
            self.supabase
            .table('sports_events')
            .select('id,sport,league,home_team,away_team,start_time')
            .gte('start_time', tomorrow_start.isoformat())
            .lt('start_time', (tomorrow_start + timedelta(days=1)).isoformat())
            .execute()
        )
        
        # Get unique sports
        active_sports = set()
        if today_games.data:
            active_sports.update([game['sport'] for game in today_games.data])
        if tomorrow_games.data:
            active_sports.update([game['sport'] for game in tomorrow_games.data])
        
        return {
            'current_date': now.strftime('%Y-%m-%d'),
            'current_time': now.strftime('%H:%M:%S'),
            'tomorrow_date': tomorrow.strftime('%Y-%m-%d'),
            'active_sports': list(active_sports),
            'today_game_count': len(today_games.data) if today_games.data else 0,
            'tomorrow_game_count': len(tomorrow_games.data) if tomorrow_games.data else 0
        }
    
    async def fetch_trending_data(self, sports: List[str]) -> Dict[str, Any]:
        """Fetch ground-truth data only: upcoming games, props (joined with players and prop types),
        recent predictions, player trends, and recent player stats summaries. Absolutely no fabrication."""
        data: Dict[str, Any] = {
            'upcoming_games': [],
            'player_props_with_names': [],
            'player_trends': [],
            'recent_predictions': [],
            'odds_movements': [],
            'recent_player_stats_summary': [],
            'entities': {
                'teams': [],
                'players': [],
                'prop_types': [],
                'sports': []
            }
        }

        now = datetime.now(timezone.utc)
        window_end = now + timedelta(days=2)

        # 1) Upcoming games (next 48 hours)
        games_resp = (
            self.supabase
            .table('sports_events')
            .select('id,sport,league,home_team,away_team,start_time,venue')
            .gte('start_time', now.isoformat())
            .lt('start_time', window_end.isoformat())
            .order('start_time')
            .limit(50)
            .execute()
        )
        events = games_resp.data or []
        data['upcoming_games'] = events

        event_ids: List[str] = [g['id'] for g in events]
        teams_set: Set[str] = set()
        sports_set: Set[str] = set()
        for g in events:
            if g.get('home_team'):
                teams_set.add(g['home_team'])
            if g.get('away_team'):
                teams_set.add(g['away_team'])
            if g.get('sport'):
                sports_set.add(g['sport'])
            # Also allow league names as sports display labels (e.g., Major League Baseball, Ultimate Fighting Championship)
            if g.get('league'):
                sports_set.add(g['league'])

        # 2) Recent AI predictions (for context)
        recent_preds = (
            self.supabase
            .table('ai_predictions')
            .select('match_teams,pick,odds,confidence,sport,reasoning,created_at,status,bet_type')
            .order('created_at', desc=True)
            .limit(20)
            .execute()
        )
        data['recent_predictions'] = recent_preds.data or []
        for p in data['recent_predictions']:
            # extract teams from match_teams field if present
            mt = p.get('match_teams') or ''
            parts = [x.strip() for x in mt.split('vs') if x.strip()] if 'vs' in mt else []
            for t in parts:
                teams_set.add(t)
            if p.get('sport'):
                sports_set.add(p['sport'])

        # Expand allowed sports with common league/synonym names for stricter-but-practical validation
        expanded_sports: Set[str] = set(sports_set)
        lower_sports = {s.lower() for s in sports_set}
        # MLB
        if any(s in lower_sports for s in ['mlb', 'baseball_mlb', 'major league baseball']):
            expanded_sports.update({'MLB', 'Major League Baseball'})
        # WNBA (add formal long name and guard against NBA phrasing slip-ups)
        if any(s in lower_sports for s in ['wnba', "women's national basketball association", 'basketball_wnba']):
            expanded_sports.update({'WNBA', "Women's National Basketball Association", 'National Basketball Association'})
        # NFL
        if any(s in lower_sports for s in ['nfl', 'americanfootball_nfl', 'national football league']):
            expanded_sports.update({'NFL', 'National Football League'})
        # MMA / UFC
        if any(s in lower_sports for s in ['mma', 'ultimate fighting championship', 'mma_mixed_martial_arts']):
            expanded_sports.update({'MMA', 'Ultimate Fighting Championship', 'UFC'})
        # Replace with expanded set
        sports_set = expanded_sports

        # 3) Player props for those upcoming events, joined with players and prop types
        props_with_names: List[Dict[str, Any]] = []
        players_map: Dict[str, Dict[str, Any]] = {}
        prop_types_map: Dict[str, Dict[str, Any]] = {}

        if event_ids:
            props_resp = (
                self.supabase
                .table('player_props_odds')
                .select('id,event_id,player_id,prop_type_id,line,over_odds,under_odds,created_at')
                .in_('event_id', event_ids)
                .order('created_at', desc=True)
                .limit(200)
                .execute()
            )
            props_rows = props_resp.data or []

            player_ids = sorted({r['player_id'] for r in props_rows if r.get('player_id')})
            prop_type_ids = sorted({r['prop_type_id'] for r in props_rows if r.get('prop_type_id')})

            if player_ids:
                players_resp = (
                    self.supabase
                    .table('players')
                    .select('id,name,player_name,team,sport,sport_key,position')
                    .in_('id', player_ids)
                    .limit(1000)
                    .execute()
                )
                for row in players_resp.data or []:
                    nm = row.get('name') or row.get('player_name') or ''
                    players_map[row['id']] = {
                        'name': nm,
                        'team': row.get('team'),
                        'sport': row.get('sport') or row.get('sport_key')
                    }
                    if nm:
                        data['entities']['players'].append(nm)

            if prop_type_ids:
                types_resp = (
                    self.supabase
                    .table('player_prop_types')
                    .select('id,prop_key,prop_name,sport_key')
                    .in_('id', prop_type_ids)
                    .limit(1000)
                    .execute()
                )
                for row in types_resp.data or []:
                    prop_types_map[row['id']] = row
                    if row.get('prop_name'):
                        data['entities']['prop_types'].append(row['prop_name'])

            events_by_id = {e['id']: e for e in events}
            for r in props_rows:
                player = players_map.get(r.get('player_id'))
                ptype = prop_types_map.get(r.get('prop_type_id'))
                ev = events_by_id.get(r.get('event_id'))
                if not (player and ptype and ev):
                    continue
                team_display = infer_team_display(player.get('team'), player.get('sport'), ev)
                item = {
                    'player_name': player['name'],
                    'player_team_from_db': player.get('team'),
                    'team_display': team_display,  # Use this for UI/report; if null, omit team tag
                    'sport': player.get('sport'),
                    'prop_type': ptype.get('prop_name') or ptype.get('prop_key'),
                    'line': r.get('line'),
                    'over_odds': r.get('over_odds'),
                    'under_odds': r.get('under_odds'),
                    'event': {
                        'home_team': ev.get('home_team'),
                        'away_team': ev.get('away_team'),
                        'start_time': ev.get('start_time')
                    },
                    'created_at': r.get('created_at')
                }
                props_with_names.append(item)

        data['player_props_with_names'] = props_with_names

        # 4) Player trends - prefer for players we actually have props on
        if data['entities']['players']:
            # We don't have player_id mapping here easily; use names filter if available in this table
            # Fall back to latest trends if name filter is not possible from client.
            trends_resp = (
                self.supabase
                .table('player_trends_data')
                .select('player_name, team_name, sport_key, avg_hits, avg_home_runs, avg_strikeouts, batting_average, form_trend, confidence_score, last_updated')
                .order('last_updated', desc=True)
                .limit(50)
                .execute()
            )
        else:
            trends_resp = (
                self.supabase
                .table('player_trends_data')
                .select('player_name, team_name, sport_key, avg_hits, avg_home_runs, avg_strikeouts, batting_average, form_trend, confidence_score, last_updated')
                .order('last_updated', desc=True)
                .limit(20)
                .execute()
            )
        data['player_trends'] = trends_resp.data or []
        for t in data['player_trends']:
            if t.get('player_name'):
                data['entities']['players'].append(t['player_name'])
            if t.get('team_name'):
                teams_set.add(t['team_name'])

        # 5) Recent player stats summary (last 5 games) for players with props
        recent_stats_summary: Dict[str, Dict[str, Any]] = {}
        if players_map:
            pid_list = list(players_map.keys())
            # Supabase 'in' filter supports up to a certain size; handle moderately sized lists
            chunk_size = 100
            rows: List[Dict[str, Any]] = []
            for i in range(0, len(pid_list), chunk_size):
                chunk = pid_list[i:i+chunk_size]
                rs_resp = (
                    self.supabase
                    .table('player_recent_stats')
                    .select('player_id,player_name,game_date,hits,home_runs,strikeouts,walks,at_bats,total_bases,points,rebounds,assists,receptions,passing_yards,rushing_yards,receiving_yards')
                    .in_('player_id', chunk)
                    .order('game_date', desc=True)
                    .limit(500)
                    .execute()
                )
                rows.extend(rs_resp.data or [])

            # group by player and compute last-5 averages of common stats
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in rows:
                grouped[r['player_id']].append(r)

            for pid, games in grouped.items():
                games_sorted = sorted(games, key=lambda x: x.get('game_date', ''), reverse=True)[:5]
                def avg(key: str) -> Optional[float]:
                    vals = [g.get(key) for g in games_sorted if isinstance(g.get(key), (int, float))]
                    return round(sum(vals)/len(vals), 2) if vals else None

                recent_stats_summary[pid] = {
                    'player_name': players_map.get(pid, {}).get('name'),
                    'last5_hits': avg('hits'),
                    'last5_total_bases': avg('total_bases'),
                    'last5_home_runs': avg('home_runs'),
                    'last5_strikeouts': avg('strikeouts'),
                    'last5_walks': avg('walks'),
                    'last5_at_bats': avg('at_bats'),
                    'last5_receptions': avg('receptions'),
                    'last5_passing_yards': avg('passing_yards'),
                    'last5_rushing_yards': avg('rushing_yards'),
                    'last5_receiving_yards': avg('receiving_yards'),
                }

        data['recent_player_stats_summary'] = [
            {'player_id': pid, **summary} for pid, summary in recent_stats_summary.items()
        ]

        # 6) Odds movements (recent)
        odds = (
            self.supabase
            .table('odds_data')
            .select('outcome_name,outcome_price,outcome_point,created_at')
            .order('created_at', desc=True)
            .limit(20)
            .execute()
        )
        data['odds_movements'] = odds.data or []

        # Build entity lists
        data['entities']['teams'] = sorted({t for t in teams_set if t})
        data['entities']['players'] = sorted({p for p in data['entities']['players'] if p})
        data['entities']['prop_types'] = sorted({p for p in data['entities']['prop_types'] if p})
        data['entities']['sports'] = sorted({s for s in sports_set if s})

        return data
    
    async def analyze_with_ai(self, context: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Use xAI Grok to analyze ONLY the provided data and generate a report.

        Absolute rules to prevent hallucinations are embedded in the system prompt.
        """

        # Ultra-strict anti-hallucination system prompt
        system_prompt = (
            "You are an elite sports analytics AI. You MUST ONLY use facts present in the provided data.\n"
            "Absolutely DO NOT invent players, teams, games, lines, odds, or statistics.\n"
            "If a detail is missing, explicitly write: 'No data available'.\n"
            "Mention ONLY entities that appear in the Allowed Entities lists.\n"
            "Do NOT generalize from historical priors unless that data is present here.\n"
            "Your tone is analytical and concise, formatted in clean mobile-friendly Markdown."
        )

        # Build allowed entities snapshot to constrain references
        allowed = data.get('entities', {})
        allowed_players = allowed.get('players', [])
        allowed_teams = allowed.get('teams', [])
        allowed_props = allowed.get('prop_types', [])
        allowed_sports = allowed.get('sports', [])

        # Provide compact data snapshots (truncate to respect token limits)
        snapshot = {
            'upcoming_games_sample': (data.get('upcoming_games') or [])[:20],
            'props_sample': (data.get('player_props_with_names') or [])[:50],
            'player_trends_sample': (data.get('player_trends') or [])[:20],
            'recent_predictions_sample': (data.get('recent_predictions') or [])[:20],
            'odds_movements_sample': (data.get('odds_movements') or [])[:20],
            'recent_player_stats_summary': (data.get('recent_player_stats_summary') or [])[:50],
        }

        analysis_prompt = f"""
Current Context:
- Date: {context['current_date']} at {context['current_time']}
- Active Sports: {', '.join(context['active_sports']) if context.get('active_sports') else 'None'}
- Games Today: {context['today_game_count']}
- Games Tomorrow: {context['tomorrow_game_count']}

ALLOWED ENTITIES (do not mention anything outside these lists):
- Teams: {', '.join(allowed_teams) if allowed_teams else 'None'}
- Players: {', '.join(allowed_players[:60]) + (' ...' if len(allowed_players) > 60 else '') if allowed_players else 'None'}
- Prop Types: {', '.join(allowed_props) if allowed_props else 'None'}
- Sports: {', '.join(allowed_sports) if allowed_sports else 'None'}

DATA SNAPSHOTS (ground truth only):
{json.dumps(snapshot, indent=2, default=str)[:9000]}

TASK: Generate a comprehensive daily report using ONLY the above data, with this exact section order and formatting:
1) Title line: "# 🏟️ Daily Sports Analytics Report - {context['current_date']}"
2) Subtitle line: "Generated at {context['current_time']} | Games Today: {context['today_game_count']} | Games Tomorrow: {context['tomorrow_game_count']}"
3) "## 📋 Executive Summary" — 2-4 concise bullets summarizing the strongest edges. If no data, write "No data available".
4) "## 🔥 Hot Trends" — Bulleted real trends from snapshots. If none, write "No data available".
5) "## 📊 Statistical Edges" — Bulleted, data-backed edges. Use only lines/odds present in props_sample. If none, write "No data available".
6) "## 🎯 High Confidence Plays" — 1-3 items, ONLY if props_sample provides exact lines/odds. Include confidence as an integer percent derived from available stats; if insufficient data, omit this section.
7) "## ⚠️ Risk Alerts" — Specific, data-supported cautions. If none, write "No data available".
8) "## 💡 Expert Insights" — Cross-reference multiple provided data points. If none, write "No data available".
9) "## 📅 Quick Look: Upcoming Games" — List a few upcoming games (from upcoming_games_sample) with start times; no embellishment.

Hard constraints:
- If a section lacks data, write exactly: "No data available".
- Use only the provided props/odds for any recommendations. Do not invent lines.
- Mention only Allowed Entities (teams/players/prop types/sports). If unsure, omit.
- Keep bullets specific, quantifiable, and sourced from the snapshots.
- When referencing a player's team for a prop, use item.team_display if present; if null, OMIT the team label. Never use item.player_team_from_db in the narrative.
"""

        try:
            response = await self.xai_client.post(
                'https://api.x.ai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.xai_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'grok-3-latest',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': analysis_prompt}
                    ],
                    'max_tokens': 3000
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                error_text = response.text if hasattr(response, 'text') else 'Unknown error'
                return f"Error generating report: {response.status_code} - {error_text}"
                
        except Exception as e:
            print(f"Full exception: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    async def generate_report(self) -> Dict[str, Any]:
        """Main method to generate the complete AI report"""
        try:
            # Get current context
            context = await self.get_current_date_context()
            
            if not context['active_sports']:
                return {
                    'success': False,
                    'error': 'No active sports found for today or tomorrow',
                    'generated_at': datetime.now().isoformat()
                }
            
            # Fetch all relevant data (ground-truth only)
            data = await self.fetch_trending_data(context['active_sports'])

            # Generate AI analysis
            report_content = await self.analyze_with_ai(context, data)

            # Post-generation validation: block saving if unknown entities are mentioned
            allowed = data.get('entities', {})
            validation = self._validate_entities(
                report_content,
                allowed_players=allowed.get('players', []),
                allowed_teams=allowed.get('teams', []),
                allowed_props=allowed.get('prop_types', []),
            )
            if validation.get('unknown_count', 0) > 0:
                return {
                    'success': False,
                    'error': 'validation_failed_unknown_entities',
                    'details': validation,
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }

            # Store report in database
            report_record = {
                'report_type': 'daily_ai_analysis',
                'content': report_content,
                'metadata': {
                    'active_sports': context['active_sports'],
                    'data_points_analyzed': {
                        'player_props': len(data['player_props_with_names']),
                        'player_trends': len(data['player_trends']),
                        'predictions': len(data['recent_predictions']),
                        'odds_movements': len(data['odds_movements'])
                    },
                    'entities': data.get('entities', {}),
                    'sources': ['supabase:sports_events', 'supabase:player_props_odds', 'supabase:players', 'supabase:player_prop_types', 'supabase:player_trends_data', 'supabase:ai_predictions', 'supabase:odds_data', 'supabase:player_recent_stats', 'statmuse:(optional)'],
                    'validation': validation
                },
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Store in a new ai_reports table (you may need to create this)
            stored = self.supabase.table('ai_reports').insert(report_record).execute()
            
            return {
                'success': True,
                'report': report_content,
                'metadata': report_record['metadata'],
                'generated_at': report_record['generated_at'],
                'report_id': stored.data[0]['id'] if stored.data else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    async def close(self):
        """Clean up connections"""
        await self.xai_client.aclose()
        await self.statmuse.close()

async def main():
    """Main execution function"""
    generator = DailyAIReportGenerator()
    
    try:
        print("🚀 Generating Daily AI Report...")
        result = await generator.generate_report()
        
        if result['success']:
            print("✅ Report generated successfully!")
            print("\n📊 Report Content:")
            print(result['report'])
            
            # Output JSON for backend consumption
            output = {
                'success': True,
                'report': result['report'],
                'metadata': result['metadata'],
                'generated_at': result['generated_at']
            }
            print("\n📄 JSON Output:")
            print(json.dumps(output, indent=2))
        else:
            print(f"❌ Error: {result['error']}")
            if isinstance(result, dict) and result.get('details'):
                print("\n🔎 Validation details:")
                print(json.dumps(result['details'], indent=2))
            sys.exit(1)
            
    finally:
        await generator.close()

if __name__ == "__main__":
    asyncio.run(main())
