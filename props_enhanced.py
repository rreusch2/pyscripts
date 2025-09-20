import os
import json
import logging
import asyncio
import requests
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import httpx
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from dotenv import load_dotenv
import time
import re # Added for JSON fixing

# Load environment variables
load_dotenv("backend/.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class PlayerProp:
    player_name: str
    prop_type: str
    line: float
    over_odds: Optional[int]
    under_odds: Optional[int]
    event_id: str
    team: str
    bookmaker: str

@dataclass
class ResearchInsight:
    source: str
    query: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime

class StatMuseClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def query(self, question: str) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/query",
                json={"query": question},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"StatMuse query failed: {e}")
            return {"error": str(e)}
    
    def player_stats(self, player_name: str, stat_type: str = "recent") -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/player-stats",
                json={"player": player_name, "stat_type": stat_type},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"StatMuse player stats failed: {e}")
            return {"error": str(e)}

class WebSearchClient:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.google_search_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.google_api_key or not self.search_engine_id:
            logger.warning("Google Search API credentials not found. Web search will use fallback.")
    
    def search(self, query: str) -> Dict[str, Any]:
        logger.info(f"🌐 Web search: {query}")
        
        try:
            # Try Google Custom Search first
            if self.google_api_key and self.search_engine_id:
                return self._google_search(query)
            else:
                logger.warning("Google Search API not configured, using fallback")
                return self._fallback_search(query)
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return self._fallback_search(query)
    
    def _google_search(self, query: str) -> Dict[str, Any]:
        """Perform real Google Custom Search"""
        try:
            params = {
                "q": query,
                "key": self.google_api_key,
                "cx": self.search_engine_id,
                "num": 5  # Limit to 5 results
            }
            
            response = requests.get(self.google_search_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            items = data.get("items", [])
            
            results = []
            for item in items:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "source": "Google Search"
                })
            
            # Create summary from top results
            summary_parts = []
            for result in results[:3]:  # Use top 3 results for summary
                if result["snippet"]:
                    summary_parts.append(f"{result['title']}: {result['snippet']}")
            
            summary = " | ".join(summary_parts) if summary_parts else "No relevant information found."
            
            web_result = {
                "query": query,
                "results": results,
                "summary": summary[:800] + "..." if len(summary) > 800 else summary
            }
            
            logger.info(f"🌐 Google search returned {len(results)} results for: {query}")
            return web_result
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> Dict[str, Any]:
        """Fallback when Google Search is unavailable"""
        logger.warning(f"Using fallback search for: {query}")
        return {
            "query": query,
            "results": [{
                "title": "Search Unavailable",
                "snippet": "Real-time web search is currently unavailable. Using cached data where possible.",
                "url": "N/A",
                "source": "Fallback"
            }],
            "summary": f"Web search unavailable for query: {query}. Using available data sources."
        }

class DatabaseClient:
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required")
            
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def get_games_for_date(self, target_date: datetime.date) -> List[Dict[str, Any]]:
        try:
            # IMPORTANT: Use the same approach as setupOddsIntegration.ts and teams_enhanced.py
            # For current day: use current time to end of day
            # For tomorrow: use start of tomorrow to end of tomorrow
            
            now = datetime.now()
            current_date = now.date()
            
            if target_date == current_date:
                # Today - get ALL games for today (including games that already happened)
                start_time_local = datetime.combine(current_date, datetime.min.time())
                start_time = start_time_local - timedelta(hours=8)  # Pad for timezone differences
                # End of day in EST, converted to UTC (EST games can run until ~3 AM UTC next day)
                end_time_local = datetime.combine(current_date, datetime.min.time().replace(hour=23, minute=59, second=59))
                end_time = end_time_local + timedelta(hours=8)  # EST to UTC conversion (worst case)
            else:
                # Tomorrow or specified date - use full day with timezone padding
                # Start of day in EST converted to UTC (EST midnight = 5 AM UTC typically)
                start_time_local = datetime.combine(target_date, datetime.min.time())
                start_time = start_time_local - timedelta(hours=8)  # Pad for timezone differences
                
                # End of day in EST converted to UTC (EST 11:59 PM can be up to 8 AM UTC next day)
                end_time_local = datetime.combine(target_date, datetime.min.time().replace(hour=23, minute=59, second=59))
                end_time = end_time_local + timedelta(hours=8)  # Pad for timezone differences
            
            start_iso = start_time.isoformat()
            end_iso = end_time.isoformat()
            
            logger.info(f"Fetching games from UTC range ({start_iso}) to ({end_iso}) and filtering for local date {target_date}")
            
            # Fetch games from selected sports with player props support
            all_games = []
            # Determine sport filter if provided
            if hasattr(self, 'sport_filter') and getattr(self, 'sport_filter'):
                sports = list(getattr(self, 'sport_filter'))
                logger.info(f"🎯 Sport filter active (props): {sports}")
            elif hasattr(self, 'nfl_only_mode') and getattr(self, 'nfl_only_mode'):
                sports = ["National Football League"]
                logger.info("🏈 NFL-only mode (props): Fetching NFL games only")
            else:
                sports = [
                    "Major League Baseball",
                    "Women's National Basketball Association",
                    "National Football League",
                    "College Football"
                ]
            
            for sport in sports:
                response = self.supabase.table("sports_events").select(
                    "id, home_team, away_team, start_time, sport, metadata"
                ).gte("start_time", start_iso).lte("start_time", end_iso).eq("sport", sport).order("start_time").execute()
                
                if response.data:
                    # Filter games to only include those that happen on the target local date
                    filtered_games = []
                    for game in response.data:
                        # Parse the UTC timestamp and convert to EST to check local date
                        game_utc = datetime.fromisoformat(game['start_time'].replace('Z', '+00:00'))
                        # Convert to EST (UTC-5, but we'll use a simple approximation)
                        game_est = game_utc - timedelta(hours=5)  # EST offset
                        game_local_date = game_est.date()
                        
                        # Only include if it falls on our target date in local time
                        if game_local_date == target_date:
                            filtered_games.append(game)
                    
                    logger.info(f"Found {len(filtered_games)} {sport} games for local date {target_date}")
                    all_games.extend(filtered_games)
            
            # Sort all games by start time
            all_games.sort(key=lambda x: x['start_time'])
            logger.info(f"Total games found for time window: {len(all_games)}")
            return all_games
        except Exception as e:
            logger.error(f"Failed to fetch games for {target_date}: {e}")
            return []

    def get_upcoming_games(self, target_date: Optional[datetime.date] = None) -> List[Dict[str, Any]]:
        """Fetch games for the specified date (defaults to today)"""
        if target_date is None:
            target_date = datetime.now().date()
        
        return self.get_games_for_date(target_date)
    
    def _safe_int_convert(self, value) -> Optional[int]:
        """Safely convert a value to int, handling strings and None"""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert odds value to int: {value}")
            return None
    
    def get_tomorrow_games(self) -> List[Dict[str, Any]]:
        """Fetch games for tomorrow"""
        tomorrow_date = datetime.now().date() + timedelta(days=1)
        return self.get_games_for_date(tomorrow_date)
    
    def get_player_props_for_games(self, game_ids: List[str]) -> List[PlayerProp]:
        if not game_ids:
            return []
        
        try:
            response = self.supabase.table("player_props_odds").select(
                "line, over_odds, under_odds, event_id, "
                "players(name, player_name, team), "
                "player_prop_types(prop_name)"
            ).in_("event_id", game_ids).execute()
            
            props = []
            for row in response.data:
                if row.get("players") and row.get("player_prop_types") and row["player_prop_types"].get("prop_name"):
                    # Prefer full name; fallback to abbreviated player_name
                    player_full = row["players"].get("name") or row["players"].get("player_name")
                    if not player_full:
                        continue
                    props.append(PlayerProp(
                        player_name=player_full,
                        prop_type=row["player_prop_types"]["prop_name"],
                        line=float(row["line"]),
                        over_odds=self._safe_int_convert(row["over_odds"]),
                        under_odds=self._safe_int_convert(row["under_odds"]),
                        event_id=row["event_id"],
                        team=row["players"].get("team") or "Unknown",
                        bookmaker="fanduel"
                    ))
            
            return props
        except Exception as e:
            logger.error(f"Failed to fetch player props: {e}")
            return []
    
    def store_ai_predictions(self, predictions: List[Dict[str, Any]]):
        try:
            # Sort predictions to control UI display order (UI shows newest first):
            # Save order (oldest -> newest): WNBA -> MLB -> CFB -> NFL
            # This makes NFL saved last, so it appears FIRST in UI lists.
            def sport_priority(pred):
                sport = pred.get("sport", "MLB")
                if sport == "WNBA":
                    return 1  # earliest
                if sport == "MLB":
                    return 2
                if sport in ("CFB", "College Football"):
                    return 3
                if sport == "NFL" or sport == "National Football League":
                    return 4  # latest (top in UI)
                return 5
            
            sorted_predictions = sorted(predictions, key=sport_priority)
            logger.info(f"📊 Saving predictions in UI order: WNBA → MLB → CFB → NFL (NFL will display first)")
            
            for pred in sorted_predictions:
                reasoning = pred.get("reasoning", "")
                if not reasoning and pred.get("metadata"):
                    reasoning = pred["metadata"].get("reasoning", "")
                
                metadata = pred.get("metadata", {})
                roi_estimate_str = metadata.get("roi_estimate", "0%")
                value_percentage_str = metadata.get("value_percentage", "0%")
                implied_probability_str = metadata.get("implied_probability", "50%")
                
                try:
                    roi_estimate = float(roi_estimate_str.replace("%", "")) if roi_estimate_str else 0.0
                    value_percentage = float(value_percentage_str.replace("%", "")) if value_percentage_str else 0.0
                    implied_probability = float(implied_probability_str.replace("%", "")) if implied_probability_str else 50.0
                except (ValueError, AttributeError):
                    roi_estimate = 0.0
                    value_percentage = 0.0
                    implied_probability = 50.0
                
                # Calculate Kelly stake for props
                try:
                    odds_value = float(str(pred.get("odds", 100)).replace("+", "").replace("-", ""))
                    confidence = pred.get("confidence", 75)
                    kelly_stake = max(0, min(10, (confidence/100 - 0.5) * 10))
                except:
                    kelly_stake = 2.5
                
                # Calculate expected value
                try:
                    confidence = pred.get("confidence", 75)
                    expected_value = (confidence - 50) * 0.2
                except:
                    expected_value = 5.0
                
                # Determine risk level
                confidence = pred.get("confidence", 75)
                if confidence >= 80:
                    risk_level = "Low"
                elif confidence >= 65:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                prediction_data = {
                    "user_id": "c19a5e12-4297-4b0f-8d21-39d2bb1a2c08",
                    "confidence": pred.get("confidence", 0),
                    "pick": pred.get("pick", ""),
                    "odds": str(pred.get("odds", 0)),
                    "sport": pred.get("sport", "MLB"),
                    "event_time": pred.get("event_time"),
                    "bet_type": pred.get("bet_type", "player_prop"),
                    "game_id": str(pred.get("event_id", "")),
                    "match_teams": pred.get("match_teams", ""),
                    "reasoning": reasoning,
                    "line_value": pred.get("line_value") or pred.get("line", 0),
                    "prediction_value": pred.get("prediction_value"),
                    "prop_market_type": pred.get("prop_market_type") or pred.get("prop_type", ""),
                    "roi_estimate": roi_estimate,
                    "value_percentage": value_percentage,
                    "kelly_stake": kelly_stake,
                    "expected_value": expected_value,
                    "risk_level": risk_level,
                    "implied_probability": implied_probability,
                    "fair_odds": metadata.get("fair_odds", pred.get("odds", 0)),
                    "key_factors": metadata.get("key_factors", []),
                    "status": "pending",
                    "metadata": metadata
                }
                
                # Remove None values to avoid database errors
                prediction_data = {k: v for k, v in prediction_data.items() if v is not None}
                
                self.supabase.table("ai_predictions").insert(prediction_data).execute()
                
            logger.info(f"Successfully stored {len(predictions)} AI predictions")
            
        except Exception as e:
            logger.error(f"Failed to store AI predictions: {e}")

class IntelligentPlayerPropsAgent:
    def __init__(self):
        self.db = DatabaseClient()
        self.statmuse = StatMuseClient()
        self.web_search = WebSearchClient()
        self.grok_client = AsyncOpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        # Add session for StatMuse context scraping
        self.session = requests.Session()
        self.statmuse_base_url = "http://localhost:5001"
        # WNBA-only mode flag
        self.wnba_only_mode = False
        # NFL week mode flag (disabled by default; can be enabled via CLI)
        # We are moving to AI-driven distribution instead of hard-coded NFL-day overrides.
        self.nfl_week_mode = False
        # NFL-only mode flag
        self.nfl_only_mode = False
    
    async def fetch_upcoming_games(self) -> List[Dict[str, Any]]:
        if self.nfl_week_mode:
            return self.get_nfl_week_games()
        return self.db.get_upcoming_games()  # Get today's games
    
    def get_nfl_week_games(self) -> List[Dict[str, Any]]:
        """Get all NFL games for the current week (Thu-Sun)"""
        try:
            current_date = datetime.now().date()
            
            # Get games for the next 7 days to capture the full NFL week
            all_nfl_games = []
            for days_ahead in range(8):  # Today + next 7 days
                target_date = current_date + timedelta(days=days_ahead)
                games = self.db.get_games_for_date(target_date)
                # Filter for NFL only
                nfl_games = [g for g in games if g.get('sport') == 'National Football League']
                all_nfl_games.extend(nfl_games)
            
            logger.info(f"🏈 Found {len(all_nfl_games)} NFL games for the week ahead")
            return all_nfl_games
        except Exception as e:
            logger.error(f"Failed to fetch NFL week games: {e}")
            return []
    
    async def fetch_player_props(self) -> List[PlayerProp]:
        if self.nfl_week_mode:
            games = self.get_nfl_week_games()
        else:
            games = self.db.get_upcoming_games()  # Only today's games
        
        if not games:
            return []
        game_ids = [game["id"] for game in games]
        return self.db.get_player_props_for_games(game_ids)
        
    def _distribute_props_by_sport(self, games: List[Dict], target_picks: int = 25) -> Dict[str, int]:
        """Generate abundant props across all available sports for frontend filtering"""
        sport_counts = {"MLB": 0, "WNBA": 0, "NFL": 0, "CFB": 0}
        
        # Count available games by sport (map full names to abbreviations)
        for game in games:
            sport = game.get("sport", "MLB")
            if sport == "Major League Baseball":
                sport_counts["MLB"] += 1
            elif sport == "Women's National Basketball Association":
                sport_counts["WNBA"] += 1
            elif sport == "National Football League":
                sport_counts["NFL"] += 1
            elif sport == "College Football":
                sport_counts["CFB"] += 1
        
        logger.info(f"Available games by sport for props: {sport_counts}")
        
        # WNBA-ONLY MODE: Override all logic
        if self.wnba_only_mode:
            wnba_games = sport_counts.get("WNBA", 0)
            if wnba_games > 0:
                logger.info(f"🏀 WNBA-only mode: targeting {target_picks} WNBA picks from {wnba_games} games")
                return {"WNBA": target_picks, "MLB": 0, "NFL": 0, "CFB": 0}
            else:
                logger.warning("🏀 WNBA-only mode requested but no WNBA games found!")
                return {"WNBA": 0, "MLB": 0, "NFL": 0, "CFB": 0}

        # NFL-ONLY MODE: Override all logic
        if self.nfl_only_mode:
            nfl_games = sport_counts.get("NFL", 0)
            if nfl_games > 0:
                logger.info(f"🏈 NFL-only mode: targeting {target_picks} NFL props from {nfl_games} games")
                return {"NFL": target_picks, "MLB": 0, "WNBA": 0, "CFB": 0}
            else:
                logger.warning("🏈 NFL-only mode requested but no NFL games found!")
                return {"NFL": 0, "MLB": 0, "WNBA": 0, "CFB": 0}
        
        # NFL SUNDAY PRIORITY DISTRIBUTION
        if self.nfl_week_mode and sport_counts["NFL"] > 0:
            logger.info("🏈 NFL Sunday detected - prioritizing NFL player props")
            # NFL Sunday gets majority of props picks
            distribution = {
                "NFL": min(sport_counts["NFL"] * 6, 20),  # 6 props per NFL game, max 20
                "MLB": min(sport_counts["MLB"] * 2, 8) if sport_counts["MLB"] > 0 else 0,
                "CFB": min(sport_counts["CFB"] * 2, 5) if sport_counts["CFB"] > 0 else 0,
                "WNBA": min(sport_counts["WNBA"] * 1, 2) if sport_counts["WNBA"] > 0 else 0
            }
        else:
            # REGULAR DISTRIBUTION - PRIORITIZE FOOTBALL (NFL/CFB) DURING SEASON, THEN MLB, THEN WNBA
            distribution = {}
            
            mlb_games = sport_counts.get("MLB", 0)
            wnba_games = sport_counts.get("WNBA", 0)
            nfl_games = sport_counts.get("NFL", 0)
            cfb_games = sport_counts.get("CFB", 0)
            
            total_football_games = nfl_games + cfb_games
            total_sports_with_games = sum(1 for count in [mlb_games, wnba_games, total_football_games] if count > 0)
            
            # Football season priority (NFL + CFB combined get highest priority)
            if total_football_games > 0 and mlb_games > 0 and wnba_games > 0:
                # All sports available - Football priority (60%), MLB (25%), WNBA (15%)
                football_allocation = int(target_picks * 0.60)
                distribution["NFL"] = min(15, max(0, int(football_allocation * (nfl_games / max(total_football_games, 1)))))
                distribution["CFB"] = min(15, max(0, football_allocation - distribution["NFL"]))
                distribution["MLB"] = min(12, max(5, int(target_picks * 0.25)))
                distribution["WNBA"] = min(8, max(3, int(target_picks * 0.15)))
            elif total_football_games > 0 and mlb_games > 0:
                # Football + MLB - Football priority (70%), MLB (30%)
                football_allocation = int(target_picks * 0.70)
                distribution["NFL"] = min(20, max(0, int(football_allocation * (nfl_games / max(total_football_games, 1)))))
                distribution["CFB"] = min(20, max(0, football_allocation - distribution["NFL"]))
                distribution["MLB"] = min(15, max(8, int(target_picks * 0.30)))
                distribution["WNBA"] = 0
            elif total_football_games > 0 and wnba_games > 0:
                # Football + WNBA - Football priority (80%), WNBA (20%)
                football_allocation = int(target_picks * 0.80)
                distribution["NFL"] = min(22, max(0, int(football_allocation * (nfl_games / max(total_football_games, 1)))))
                distribution["CFB"] = min(22, max(0, football_allocation - distribution["NFL"]))
                distribution["WNBA"] = min(10, max(5, int(target_picks * 0.20)))
                distribution["MLB"] = 0
            elif mlb_games > 0 and wnba_games > 0:
                # MLB + WNBA (no football) - heavily favor MLB (80%), cap WNBA
                distribution["MLB"] = min(32, max(24, int(target_picks * 0.80)))
                distribution["WNBA"] = min(8, max(4, int(target_picks * 0.20)))
                distribution["NFL"] = 0
                distribution["CFB"] = 0
            elif total_football_games > 0:
                # Only football available - split between NFL/CFB based on games available
                distribution["NFL"] = min(25, max(0, int(target_picks * (nfl_games / max(total_football_games, 1)))))
                distribution["CFB"] = min(25, max(0, target_picks - distribution["NFL"]))
                distribution["MLB"] = 0
                distribution["WNBA"] = 0
            elif mlb_games > 0:
                # Only MLB available
                distribution["MLB"] = min(30, max(20, target_picks))
                distribution["WNBA"] = 0
                distribution["NFL"] = 0
                distribution["CFB"] = 0
            elif wnba_games > 0:
                # Only WNBA available
                distribution["WNBA"] = min(20, max(15, target_picks))
                distribution["MLB"] = 0
                distribution["NFL"] = 0
                distribution["CFB"] = 0
            else:
                distribution["MLB"] = 0
                distribution["WNBA"] = 0
                distribution["NFL"] = 0
                distribution["CFB"] = 0
        
        logger.info(f"Generous props distribution for frontend filtering: {distribution}")
        return distribution
    
    def _format_sport_distribution_requirements(self, sport_distribution: Dict[str, int], target_picks: int) -> str:
        """Generate dynamic prop distribution requirements based on available sports and games"""
        if not sport_distribution:
            return f"- Generate EXACTLY {target_picks} total props across all available sports"
        
        # Filter out sports with 0 props
        active_sports = {sport: props for sport, props in sport_distribution.items() if props > 0}
        
        if not active_sports:
            return f"- Generate EXACTLY {target_picks} total props across all available sports"
        
        requirements = []
        total_expected = sum(active_sports.values())
        
        # Generate requirements for each sport  
        sport_order = ["MLB", "WNBA"]  # Prioritize MLB in instructions
        for sport in sport_order:
            if sport in active_sports:
                props_count = active_sports[sport]
                requirements.append(f"- Generate EXACTLY {props_count} {sport} player prop picks")
        
        # Add any sports not in the preferred order
        for sport, props_count in active_sports.items():
            if sport not in sport_order:
                requirements.append(f"- Generate EXACTLY {props_count} {sport} player prop picks")
        
        requirements.append(f"- TOTAL: Generate EXACTLY {total_expected} player props across all sports")
        requirements.append("- Focus on generating the FULL amount for each sport to maximize frontend filtering options")
        requirements.append("- DIVERSIFY prop types within each sport (hits, home runs, RBIs for MLB; points, rebounds, assists for WNBA)")
        
        return "\n".join(requirements)
    
    async def generate_daily_picks(self, target_date: Optional[datetime.date] = None, target_picks: int = 15) -> List[Dict[str, Any]]:
        if target_date is None:
            target_date = datetime.now().date()
            
        logger.info(f"🚀 Starting intelligent multi-sport player props analysis for {target_date}...")
        
        games = self.db.get_upcoming_games(target_date)
        logger.info(f"📅 Found {len(games)} games for {target_date} across MLB, WNBA, NFL, and CFB")
        
        if not games:
            logger.warning(f"No games found for {target_date}")
            return []
        
        # Decide pick distribution using AI based on actual available props (fallbacks to heuristic if needed)
        game_ids = [game["id"] for game in games]
        available_props = self.db.get_player_props_for_games(game_ids)
        logger.info(f"🎯 Found {len(available_props)} available player props across all sports")
        if not available_props:
            logger.warning("No player props found")
            return []

        sport_distribution = await self.decide_pick_distribution_ai(available_props, games, target_picks)
        if not sport_distribution:
            logger.warning("AI distribution failed, falling back to heuristic distribution")
            sport_distribution = self._distribute_props_by_sport(games, target_picks)
        
        research_plan = await self.create_research_plan(available_props, games, sport_distribution)
        statmuse_count = len(research_plan.get("statmuse_queries", []))
        web_search_count = len(research_plan.get("web_searches", []))
        total_queries = statmuse_count + web_search_count
        logger.info(f"📋 Created research plan with {statmuse_count} StatMuse + {web_search_count} web queries = {total_queries} total")
        
        insights = await self.execute_research_plan(research_plan, available_props)
        logger.info(f"🔍 Gathered {len(insights)} research insights across all stages")
        
        picks = await self.generate_picks_with_reasoning(insights, available_props, games, target_picks, sport_distribution)
        logger.info(f"🎲 Generated {len(picks)} intelligent picks")
        
        if picks:
            self.db.store_ai_predictions(picks)
            logger.info(f"💾 Stored {len(picks)} picks in database")
        
        return picks
    
    def scrape_statmuse_context(self) -> Dict[str, Any]:
        """Scrape StatMuse main pages for current context and insights"""
        try:
            logger.info("🔍 Scraping StatMuse main pages for current context...")
            response = self.session.get(
                f"{self.statmuse_base_url}/scrape-context",
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                logger.info("✅ StatMuse context scraping successful")
                return result.get('context', {})
            else:
                logger.warning(f"⚠️ StatMuse context scraping failed: {result.get('error')}")
                return {}
        except Exception as e:
            logger.error(f"❌ StatMuse context scraping error: {e}")
            return {}
    
    async def create_research_plan(self, props: List[PlayerProp], games: List[Dict], desired_distribution: Dict[str, int] = None) -> Dict[str, Any]:
        """Create intelligent research plan based on actual available props data, desired pick distribution, and current StatMuse context"""
        
        # STEP 1: Scrape StatMuse main pages for current context
        statmuse_context = self.scrape_statmuse_context()
        
        # STEP 2: Separate props by detected sport using enhanced detection  
        mlb_props = [p for p in props if self._get_prop_sport(p, games) == 'MLB']
        wnba_props = [p for p in props if self._get_prop_sport(p, games) == 'WNBA']
        nfl_props = [p for p in props if self._get_prop_sport(p, games) == 'NFL']
        cfb_props = [p for p in props if self._get_prop_sport(p, games) == 'CFB']
        unknown_props = [p for p in props if self._get_prop_sport(p, games) == 'Unknown']
        
        logger.info(f"🔍 Enhanced prop detection results:")
        logger.info(f"  MLB: {len(mlb_props)} props")
        logger.info(f"  WNBA: {len(wnba_props)} props")
        logger.info(f"  NFL: {len(nfl_props)} props") 
        logger.info(f"  CFB: {len(cfb_props)} props")
        logger.info(f"  Unknown: {len(unknown_props)} props")
        
        if unknown_props:
            logger.info(f"  Unknown teams: {list(set(p.team for p in unknown_props[:5]))}")
        
        # If a desired distribution is provided, allocate research based on it; otherwise use legacy allocation (with optional NFL week mode)
        if desired_distribution and any(v > 0 for v in desired_distribution.values()):
            total_desired = max(1, sum(v for v in desired_distribution.values() if v))
            # Establish a reasonable total research query budget based on props volume
            base_queries = min(28, max(12, int(len(props) * 0.25)))
            def alloc_for(sport_key: str, available_count: int) -> int:
                desired = max(0, desired_distribution.get(sport_key, 0))
                if desired == 0 or available_count == 0:
                    return 0
                share = desired / total_desired
                # Weight by both desired share and availability
                weighted = share * base_queries
                # Cap to available_count to avoid wasted queries; ensure minimum of 2 for selected sports
                return max(2, min(15, int(round(min(weighted, available_count * 0.8)))))

            target_nfl_queries = alloc_for('NFL', len(nfl_props))
            target_cfb_queries = alloc_for('CFB', len(cfb_props))
            target_mlb_queries = alloc_for('MLB', len(mlb_props))
            target_wnba_queries = alloc_for('WNBA', len(wnba_props))
        # NFL SUNDAY PRIORITY: Override all research allocation for NFL Sunday
        elif self.nfl_week_mode and len(nfl_props) > 0:
            logger.info(f"🏈 NFL Sunday mode: prioritizing research on {len(nfl_props)} NFL props")
            target_nfl_queries = min(15, max(8, int(len(nfl_props) * 0.6)))
            target_mlb_queries = min(8, max(3, int(len(mlb_props) * 0.4))) if len(mlb_props) > 0 else 0
            target_wnba_queries = min(4, max(2, int(len(wnba_props) * 0.3))) if len(wnba_props) > 0 else 0
            target_cfb_queries = min(6, max(2, int(len(cfb_props) * 0.4))) if len(cfb_props) > 0 else 0
        elif self.wnba_only_mode:
            if len(wnba_props) > 0:
                # WNBA-only mode: max research focus on WNBA
                target_mlb_queries = 0
                target_wnba_queries = min(20, max(15, len(wnba_props)))
                target_nfl_queries = 0
                target_cfb_queries = 0
                logger.info(f"🏀 WNBA-only mode: focusing ALL research on {len(wnba_props)} WNBA props")
            else:
                logger.warning("🏀 WNBA-only mode but no WNBA props found!")
                target_mlb_queries = 0
                target_wnba_queries = 0
                target_nfl_queries = 0
                target_cfb_queries = 0
        else:
            # NORMAL MODE: Multi-sport allocation
            total_football_props = len(nfl_props) + len(cfb_props)
            
            if total_football_props > 0 and len(mlb_props) > 0 and len(wnba_props) > 0:
                # All sports available - Football priority (50%), MLB (30%), WNBA (20%)
                target_nfl_queries = min(12, max(0, int(len(nfl_props) * 0.6)))
                target_cfb_queries = min(8, max(0, int(len(cfb_props) * 0.6)))
                target_mlb_queries = min(10, max(5, int(len(mlb_props) * 0.6)))
                target_wnba_queries = min(6, max(2, int(len(wnba_props) * 0.4)))
            elif total_football_props > 0 and len(mlb_props) > 0:
                # Football + MLB - Football priority (60%), MLB (40%)
                target_nfl_queries = min(15, max(0, int(len(nfl_props) * 0.7)))
                target_cfb_queries = min(10, max(0, int(len(cfb_props) * 0.7)))
                target_mlb_queries = min(12, max(6, int(len(mlb_props) * 0.7)))
                target_wnba_queries = 0
            elif len(mlb_props) > 0 and len(wnba_props) > 0:
                # MLB + WNBA - heavily favor MLB research (75% MLB, 25% WNBA)  
                target_nfl_queries = 0
                target_cfb_queries = 0
                target_mlb_queries = min(18, max(10, int(len(mlb_props) * 0.8)))
                target_wnba_queries = min(8, max(3, int(len(wnba_props) * 0.6)))
            elif len(mlb_props) > 0:
                # Only MLB available
                target_nfl_queries = 0
                target_cfb_queries = 0
                target_mlb_queries = min(20, max(12, len(mlb_props)))
                target_wnba_queries = 0
            elif len(wnba_props) > 0:
                # Only WNBA available  
                target_nfl_queries = 0
                target_cfb_queries = 0
                target_mlb_queries = 0
                target_wnba_queries = min(15, max(8, len(wnba_props)))
            elif total_football_props > 0:
                # Only Football available
                target_nfl_queries = min(15, max(0, int(len(nfl_props) * 0.7)))
                target_cfb_queries = min(12, max(0, int(len(cfb_props) * 0.7)))
                target_mlb_queries = 0
                target_wnba_queries = 0
            else:
                # No props available
                target_nfl_queries = 0
                target_cfb_queries = 0
                target_mlb_queries = 0
                target_wnba_queries = 0
        
        total_queries = target_mlb_queries + target_wnba_queries + target_nfl_queries + target_cfb_queries
        logger.info(f"🎯 Dynamic research allocation: NFL={target_nfl_queries}, MLB={target_mlb_queries}, CFB={target_cfb_queries}, WNBA={target_wnba_queries}, Total={total_queries}")
        
        # STEP 3: Create dynamic analysis using Grok AI (like teams_enhanced.py)
        mlb_sample = [{"player": p.player_name, "prop": p.prop_type, "line": p.line, "team": p.team} for p in mlb_props[:20]]
        wnba_sample = [{"player": p.player_name, "prop": p.prop_type, "line": p.line, "team": p.team} for p in wnba_props[:15]]
        nfl_sample = [{"player": p.player_name, "prop": p.prop_type, "line": p.line, "team": p.team} for p in nfl_props[:15]]
        cfb_sample = [{"player": p.player_name, "prop": p.prop_type, "line": p.line, "team": p.team} for p in cfb_props[:10]]
        
        # NFL SUNDAY PRIORITY CHECK
        is_nfl_sunday = target_nfl_queries > target_mlb_queries and len(nfl_props) > 0
        
        prompt = f"""You are an elite sports betting analyst. {"🏈 IT'S NFL SUNDAY - PRIORITIZE NFL RESEARCH!" if is_nfl_sunday else "Analyze the available player props and create an INTELLIGENT, DYNAMIC research strategy."}

# {"🚨 NFL SUNDAY CRITICAL PRIORITY 🚨" if is_nfl_sunday else "CRITICAL ANALYSIS TASK:"}
{"TODAY IS NFL SUNDAY - The allocation shows NFL gets MORE research than MLB. You MUST focus on NFL players first!" if is_nfl_sunday else "Analyze the actual props data below and create DIVERSE, VALUE-FOCUSED research queries."}

## AVAILABLE PROPS DATA:

**NFL PROPS ({len(nfl_props)} total):** {"⭐ PRIMARY FOCUS - NFL SUNDAY ⭐" if is_nfl_sunday else ""}
{json.dumps(nfl_sample, indent=2)}

**CFB PROPS ({len(cfb_props)} total):**
{json.dumps(cfb_sample, indent=2)}

**MLB PROPS ({len(mlb_props)} total):** {"⚠️ SECONDARY PRIORITY ⚠️" if is_nfl_sunday else ""}
{json.dumps(mlb_sample, indent=2)}

**WNBA PROPS ({len(wnba_props)} total):**  
{json.dumps(wnba_sample, indent=2)}

**CURRENT GAMES TODAY:**
{json.dumps([{"sport": g.get("sport"), "home": g.get("home_team"), "away": g.get("away_team")} for g in games[:10]], indent=2)}

# MANDATORY RESEARCH ALLOCATION:
- **NFL Queries**: {target_nfl_queries} {"🏈 MUST PRIORITIZE NFL PLAYERS FIRST!" if is_nfl_sunday else ""}
- **CFB Queries**: {target_cfb_queries}
- **MLB Queries**: {target_mlb_queries} {"(Secondary to NFL today)" if is_nfl_sunday else ""}
- **WNBA Queries**: {target_wnba_queries}
- **Web Searches**: 5-8 total (injury/lineup news)

# {"🏈 NFL SUNDAY REQUIREMENTS:" if is_nfl_sunday else "YOUR INTELLIGENCE TASK:"}
{"1. **NFL FIRST**: Research NFL players BEFORE any MLB players - this is Sunday!" if is_nfl_sunday else "1. **ANALYZE THE ACTUAL PROPS**: What players have props? What prop types? What lines look interesting?"}
{"2. **NFL DIVERSITY**: Cover QBs, RBs, WRs, TEs from different NFL games" if is_nfl_sunday else "2. **IDENTIFY VALUE OPPORTUNITIES**: Which players/props might be mispriced based on recent performance?"}
{"3. **NFL FOCUS**: Use most of your research allocation on NFL props analysis" if is_nfl_sunday else "3. **CREATE DIVERSE RESEARCH**: Don't repeat the same players/queries every time - be INTELLIGENT and ADAPTIVE"}
{"4. **THEN MLB**: Only research MLB after covering NFL thoroughly" if is_nfl_sunday else "4. **FOCUS ON ACTIONABLE DATA**: Research recent form, matchups, injuries that could affect these specific props"}

## {"NFL SUNDAY" if is_nfl_sunday else "RESEARCH"} STRATEGY REQUIREMENTS:
- **{"🏈 NFL ABSOLUTE PRIORITY" if is_nfl_sunday else "NFL PRIORITY"}**: Research diverse NFL players from the props list (QBs, RBs, WRs, TEs)
- **CFB ANALYSIS**: College football key players and prop opportunities  
- **{"MLB SECONDARY" if is_nfl_sunday else "MLB FOCUS"}**: Research MLB players {"ONLY AFTER NFL research is covered" if is_nfl_sunday else "from the props list (batters AND pitchers)"}
- **VARIED PROP TYPES**: Research different prop types (passing yards, rushing yards, hits, HRs, etc.)
- **DIFFERENT TEAMS**: Spread research across multiple teams and games
- **AVOID REPETITION**: Don't use the same players/queries every single time
- **VALUE HUNTING**: Look for mispriced lines based on recent trends and matchups

**StatMuse Works Best For:**
- "[Player Name] passing yards this season" (NFL)
- "[Player Name] rushing yards last 5 games" (NFL/CFB) 
- "[Player Name] receiving yards this season" (NFL)
- "[Player Name] touchdowns this season" (NFL/CFB)
- "[Player Name] hits this season" (MLB)
- "[Player Name] home runs last 10 games" (MLB)
- "[Player Name] strikeouts this season" (MLB)
- "[Player Name] points this season" (WNBA)
- "[Player Name] rebounds last 5 games" (WNBA)

**Web Search For:**
- "[Player Name] injury status lineup news"
- "[Team Name] starting lineup injury report"
- "[NFL Team] Week [X] injury report"

Generate intelligent research plan as JSON:
{{
    "analysis_summary": "Brief analysis of the props data and research strategy",
    "statmuse_queries": [
        {{
            "query": "Specific player stat query based on available props",
            "priority": "high/medium/low", 
            "sport": "NFL/CFB/MLB/WNBA",
            "reasoning": "Why this player/stat is worth researching"
        }}
    ],
    "web_searches": [
        {{
            "query": "Injury/lineup search query",
            "priority": "high/medium/low",
            "sport": "NFL/CFB/MLB/WNBA"
        }}
    ]
}}

**BE INTELLIGENT**: Look at the ACTUAL props data and create research that will help evaluate those SPECIFIC props!"""
        
        try:
            response = await self.grok_client.chat.completions.create(
                model="grok-3-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            plan_text = response.choices[0].message.content
            start_idx = plan_text.find("{")
            end_idx = plan_text.rfind("}") + 1
            plan_json = json.loads(plan_text[start_idx:end_idx])
            
            return plan_json
            
        except Exception as e:
            logger.error(f"Failed to create research plan: {e}")
            return self._create_fallback_research_plan(props)

    async def decide_pick_distribution_ai(self, props: List[PlayerProp], games: List[Dict], target_picks: int) -> Dict[str, int]:
        """Use Grok to intelligently decide pick distribution across sports, given available props and target count."""
        try:
            # Count available props by sport using reliable mapping
            counts = {"MLB": 0, "WNBA": 0, "NFL": 0, "CFB": 0}
            samples = {"MLB": [], "WNBA": [], "NFL": [], "CFB": []}
            for p in props:
                sp = self._get_prop_sport(p, games)
                if sp in counts:
                    counts[sp] += 1
                    if len(samples[sp]) < 8:
                        samples[sp].append({
                            "player": p.player_name,
                            "prop_type": p.prop_type,
                            "line": p.line,
                            "team": p.team
                        })

            # Remove zero-count sports from consideration
            active_counts = {k: v for k, v in counts.items() if v > 0}
            if not active_counts:
                return {}

            prompt = f"""
You are a world-class betting strategist. Decide how to allocate exactly {target_picks} player prop picks across the available sports based on supply and slate importance.

Rules:
- Only include sports that have available props.
- Output MUST be a single JSON object with keys from ["MLB","WNBA","NFL","CFB"].
- Values are non-negative integers that sum to exactly {target_picks}.
- Favor sports with higher availability and slate importance (e.g., NFL Sundays), but be reasonable and diversified.
- If one sport overwhelmingly dominates (e.g., NFL Sunday), it can take most or all of the picks.

Available props by sport: {json.dumps(active_counts)}

Sample props (first few) by sport to understand the slate quality:
MLB: {json.dumps(samples['MLB'], indent=2)}
WNBA: {json.dumps(samples['WNBA'], indent=2)}
NFL: {json.dumps(samples['NFL'], indent=2)}
CFB: {json.dumps(samples['CFB'], indent=2)}

Return ONLY compact JSON like:
{{"NFL":10, "MLB":3, "WNBA":2}}
"""

            response = await self.grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON braces
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end <= start:
                logger.warning("AI distribution response did not contain JSON; falling back")
                return {}
            json_str = text[start:end]
            try:
                dist = json.loads(json_str)
            except Exception as pe:
                logger.warning(f"Failed to parse AI distribution JSON: {pe}")
                return {}

            # Validate keys and sum
            valid_keys = {"MLB", "WNBA", "NFL", "CFB"}
            dist = {k: int(v) for k, v in dist.items() if k in valid_keys and isinstance(v, (int, float))}
            total = sum(dist.values())
            if total != target_picks:
                logger.warning(f"AI distribution does not sum to target ({total}!={target_picks}); normalizing")
                if total > 0:
                    # Normalize proportionally and fix rounding
                    scaled = {k: max(0, int(round(v * target_picks / total))) for k, v in dist.items()}
                    # Fix rounding drift
                    drift = target_picks - sum(scaled.values())
                    for k in sorted(scaled.keys(), key=lambda x: -scaled[x]):
                        if drift == 0:
                            break
                        scaled[k] += 1 if drift > 0 else -1
                        drift = target_picks - sum(scaled.values())
                    dist = scaled
                else:
                    return {}

            # Remove sports with no available props
            dist = {k: v for k, v in dist.items() if active_counts.get(k, 0) > 0 and v > 0}
            logger.info(f"🧠 AI pick distribution decided: {dist}")
            return dist
        except Exception as e:
            logger.error(f"Failed to decide AI distribution: {e}")
            return {}
    
    def _analyze_available_props(self, props: List[PlayerProp], games: List[Dict]) -> Dict[str, Any]:
        """Analyze the actual props data to understand what's available"""
        
        # Group props by sport
        props_by_sport = {}
        players_by_sport = {}
        prop_types_by_sport = {}
        
        # Create MLB team abbreviation mapping
        mlb_team_mapping = {
            'ari': ['arizona diamondbacks', 'diamondbacks'],
            'atl': ['atlanta braves', 'braves'], 
            'bal': ['baltimore orioles', 'orioles'],
            'bos': ['boston red sox', 'red sox'],
            'chc': ['chicago cubs', 'cubs'],
            'cws': ['chicago white sox', 'white sox'],
            'cin': ['cincinnati reds', 'reds'],
            'cle': ['cleveland guardians', 'guardians'],
            'col': ['colorado rockies', 'rockies'],
            'det': ['detroit tigers', 'tigers'], 
            'hou': ['houston astros', 'astros'],
            'kc': ['kansas city royals', 'royals'],
            'laa': ['los angeles angels', 'angels'],
            'lad': ['los angeles dodgers', 'dodgers'],
            'mia': ['miami marlins', 'marlins'],
            'mil': ['milwaukee brewers', 'brewers'],
            'min': ['minnesota twins', 'twins'],
            'nym': ['new york mets', 'mets'],
            'nyy': ['new york yankees', 'yankees'],
            'oak': ['oakland athletics', 'athletics'],
            'phi': ['philadelphia phillies', 'phillies'],
            'pit': ['pittsburgh pirates', 'pirates'],
            'sd': ['san diego padres', 'padres'],
            'sf': ['san francisco giants', 'giants'],
            'sea': ['seattle mariners', 'mariners'],
            'stl': ['st. louis cardinals', 'cardinals'],
            'tb': ['tampa bay rays', 'rays'],
            'tex': ['texas rangers', 'rangers'],
            'tor': ['toronto blue jays', 'blue jays'],
            'wsh': ['washington nationals', 'nationals']
        }
        
        # WNBA team mapping
        wnba_team_mapping = {
            'liberty': ['new york liberty'],
            'wings': ['dallas wings'],
            'storm': ['seattle storm'],
            'aces': ['las vegas aces'],
            'mystics': ['washington mystics'],
            'sun': ['connecticut sun'],
            'fever': ['indiana fever'],
            'sky': ['chicago sky'],
            'dream': ['atlanta dream'],
            'lynx': ['minnesota lynx'],
            'mercury': ['phoenix mercury'],
            'sparks': ['los angeles sparks']
        }
        
        # Debug: Log all game teams for reference
        logger.info(f"🔍 Available games and teams:")
        for game in games:
            logger.info(f"  {game.get('sport', 'Unknown')}: {game.get('home_team')} vs {game.get('away_team')}")
        
        for prop in props:
            # Use the new enhanced sport detection
            sport = self._get_prop_sport(prop, games)
            
            if sport not in props_by_sport:
                props_by_sport[sport] = []
                players_by_sport[sport] = set()
                prop_types_by_sport[sport] = set()
            
            props_by_sport[sport].append({
                "player": prop.player_name,
                "prop_type": prop.prop_type,
                "line": prop.line,
                "over_odds": prop.over_odds,
                "under_odds": prop.under_odds,
                "team": prop.team
            })
            
            players_by_sport[sport].add(prop.player_name)
            prop_types_by_sport[sport].add(prop.prop_type)
        
        # Debug logging for sport detection
        logger.info(f"🔍 Sport detection results:")
        for sport, sport_props in props_by_sport.items():
            logger.info(f"  {sport}: {len(sport_props)} props")
            if sport_props:
                sample_players = list(set(prop['player'] for prop in sport_props[:5]))
                logger.info(f"    Sample players: {sample_players}")
        
        # Create analysis summary
        analysis = {
            "total_props": len(props),
            "sports_breakdown": {},
            "sport_distribution": {},  # Add this for research allocation
            "top_players_by_sport": {},
            "prop_types_by_sport": {},
            "sample_props_by_sport": {}
        }
        
        for sport, sport_props in props_by_sport.items():
            analysis["sports_breakdown"][sport] = len(sport_props)
            analysis["sport_distribution"][sport] = len(sport_props)  # Same as sports_breakdown for now
            analysis["top_players_by_sport"][sport] = list(players_by_sport[sport])[:15]
            analysis["prop_types_by_sport"][sport] = list(prop_types_by_sport[sport])
            analysis["sample_props_by_sport"][sport] = sport_props[:20]  # Sample for analysis
        
        return analysis
    
    def _get_prop_sport(self, prop: PlayerProp, games: List[Dict]) -> str:
        """Determine sport for a prop using reliable event_id mapping, with fallbacks."""
        # 1) Most reliable: map by event_id to the game's sport
        for game in games:
            if str(prop.event_id) == str(game.get('id')):
                sport_full = game.get('sport', 'Unknown')
                if sport_full == "Women's National Basketball Association":
                    return "WNBA"
                elif sport_full == "Major League Baseball":
                    return "MLB"
                elif sport_full == "National Football League":
                    return "NFL"
                elif sport_full == "College Football":
                    return "CFB"
                elif sport_full == "Ultimate Fighting Championship":
                    return "MMA"
                else:
                    return "Unknown"

        # 2) Fallback: try team name matching against game teams
        for game in games:
            home_team = game.get('home_team', '').lower()
            away_team = game.get('away_team', '').lower()
            prop_team = (prop.team or "").lower()

            if self._teams_match_enhanced(prop_team, home_team, away_team):
                sport_full = game.get('sport', 'Unknown')
                if sport_full == "Women's National Basketball Association":
                    return "WNBA"
                elif sport_full == "Major League Baseball":
                    return "MLB"
                elif sport_full == "National Football League":
                    return "NFL"
                elif sport_full == "College Football":
                    return "CFB"
                elif sport_full == "Ultimate Fighting Championship":
                    return "MMA"
                else:
                    return "MLB"  # Default to MLB
        
        # If no match found, try to infer from team name patterns
        prop_team_lower = (prop.team or "").lower()
        
        # Common NFL abbreviations (from database analysis)
        nfl_teams = ['cin', 'buf', 'tb', 'mia', 'ind', 'hou', 'jax', 'ari', 'car', 'gb', 'ne', 
                     'det', 'no', 'sea', 'bal', 'den', 'cle', 'ten', 'lar', 'nyj', 'atl', 'sf',
                     'lac', 'lv', 'dal', 'kc', 'nyg', 'phi', 'pit', 'was', 'chi', 'min']
        
        # Common CFB abbreviations (basic set)
        cfb_teams = ['duke', 'illinois', 'indiana', 'kennesaw', 'iowa', 'penn', 'pitt', 'smu',
                     'baylor', 'syracuse', 'uconn', 'texas', 'clemson', 'oregon', 'alabama']
        
        # Common MLB abbreviations
        mlb_teams = ['hou', 'mia', 'bos', 'nyy', 'nym', 'lad', 'sf', 'phi', 'atl', 'det', 'min',
                     'pit', 'cle', 'bal', 'wsh', 'oak', 'kc', 'tex', 'tb', 'tor', 'cws',
                     'chc', 'mil', 'stl', 'cin', 'col', 'ari', 'sd', 'sea', 'laa']
        
        # Common WNBA teams  
        wnba_teams = ['liberty', 'wings', 'storm', 'aces', 'mystics', 'sun', 'fever', 
                     'sky', 'dream', 'lynx', 'mercury', 'sparks']
        
        if prop_team_lower in nfl_teams:
            return "NFL"
        elif prop_team_lower in cfb_teams:
            return "CFB"
        elif prop_team_lower in mlb_teams:
            return "MLB"
        elif prop_team_lower in wnba_teams:
            return "WNBA"
        elif prop_team_lower in nfl_teams:
            return "NFL"
        elif prop_team_lower in cfb_teams:
            return "CFB"

        return "Unknown"
    
    def _teams_match_enhanced(self, prop_team: str, home_team: str, away_team: str) -> bool:
        """Enhanced team matching using multiple strategies"""
        # Direct matches
        if prop_team == home_team or prop_team == away_team:
            return True
            
        # Partial matches
        if prop_team in home_team or home_team in prop_team:
            return True
        if prop_team in away_team or away_team in prop_team:
            return True
            
        # Word matching for significant words
        prop_words = prop_team.split()
        home_words = home_team.split()
        away_words = away_team.split()
        
        for prop_word in prop_words:
            if len(prop_word) > 3:
                for home_word in home_words:
                    if len(home_word) > 3 and prop_word == home_word:
                        return True
                for away_word in away_words:
                    if len(away_word) > 3 and prop_word == away_word:
                        return True
        
        return False
    
    def _create_fallback_research_plan(self, props: List[PlayerProp]) -> Dict[str, Any]:
        """Create a basic research plan if AI planning fails"""
        
        # Get diverse set of players and prop types
        unique_players = list(set(prop.player_name for prop in props))[:15]
        unique_prop_types = list(set(prop.prop_type for prop in props))
        
        return {
            "analysis_summary": "Fallback research plan based on available props",
            "statmuse_queries": [{
                "query": f"{player} {prop_type.replace('Batter ', '').replace('Player ', '').lower()} this season",
                "priority": "medium"
            } for player, prop_type in zip(unique_players[:8], unique_prop_types[:8])],
            "web_searches": [{
                "query": f"{player} injury status and recent news",
                "priority": "medium"
            } for player in unique_players[:3]]
        }
    
    async def execute_research_plan(self, plan: Dict[str, Any], props: List[PlayerProp]) -> List[ResearchInsight]:
        all_insights = []
        
        logger.info("🔬 STAGE 1: Initial Research")
        stage1_insights = await self._execute_initial_research(plan)
        all_insights.extend(stage1_insights)
        
        logger.info("🧠 STAGE 2: Analyzing findings and generating follow-up research")
        stage2_insights = await self._execute_adaptive_followup(stage1_insights, props)
        all_insights.extend(stage2_insights)
        
        logger.info("🎯 STAGE 3: Final Targeted Research")
        stage3_insights = await self._execute_final_research(all_insights, props)
        all_insights.extend(stage3_insights)
        
        logger.info(f"🔍 Total research insights gathered: {len(all_insights)}")
        return all_insights
    
    async def _execute_initial_research(self, plan: Dict[str, Any]) -> List[ResearchInsight]:
        insights = []
        
        statmuse_queries = plan.get("statmuse_queries", [])[:8]
        web_searches = plan.get("web_searches", [])[:3]
        
        # BALANCED LIMITS: More MLB research since 7 picks needed vs 3 WNBA picks
        max_statmuse = min(18, len(statmuse_queries))  # Reasonable limit for both sports
        max_web = min(12, len(web_searches))  # Focused web searches
        
        for query_obj in statmuse_queries[:max_statmuse]:
            try:
                query_text = query_obj.get("query", query_obj) if isinstance(query_obj, dict) else query_obj
                priority = query_obj.get("priority", "medium") if isinstance(query_obj, dict) else "medium"
                
                logger.info(f"🔍 StatMuse query ({priority}): {query_text}")
                result = self.statmuse.query(query_text)
                
                if result and "error" not in result:
                    result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                    logger.info(f"📊 StatMuse result: {result_preview}")
                    
                    confidence = 0.9 if priority == "high" else 0.7 if priority == "medium" else 0.5
                    insights.append(ResearchInsight(
                        source="statmuse",
                        query=query_text,
                        data=result,
                        confidence=confidence,
                        timestamp=datetime.now()
                    ))
                else:
                    logger.warning(f"❌ StatMuse query failed: {result}")
                
                await asyncio.sleep(1.5)
                
            except Exception as e:
                logger.error(f"❌ StatMuse query failed for \'{query_text}\': {e}")
        
        web_searches = plan.get("web_searches", [])[:3]
        for search_obj in web_searches:
            try:
                search_query = search_obj.get("query", search_obj) if isinstance(search_obj, dict) else search_obj
                priority = search_obj.get("priority", "medium") if isinstance(search_obj, dict) else "medium"
                
                logger.info(f"🌐 Web search ({priority}): {search_query}")
                result = self.web_search.search(search_query)
                
                confidence = 0.8 if priority == "high" else 0.6 if priority == "medium" else 0.4
                insights.append(ResearchInsight(
                    source="web_search",
                    query=search_query,
                    data=result,
                    confidence=confidence,
                    timestamp=datetime.now()
                ))
                
            except Exception as e:
                logger.error(f"❌ Initial web search failed for \'{search_query}\': {e}")
        
        return insights
    
    async def _execute_adaptive_followup(self, initial_insights: List[ResearchInsight], props: List[PlayerProp]) -> List[ResearchInsight]:
        insights_summary = []
        for insight in initial_insights:
            insights_summary.append({
                "source": insight.source,
                "query": insight.query,
                "data": str(insight.data)[:600],
                "confidence": insight.confidence
            })
        
        top_props = [{
            "player": prop.player_name,
            "prop_type": prop.prop_type,
            "line": prop.line,
            "over_odds": prop.over_odds,
            "under_odds": prop.under_odds
        } for prop in props[:30]]
        
        prompt = f"""
You are analyzing initial research findings to identify gaps and generate intelligent follow-up queries.

INITIAL RESEARCH FINDINGS:
{json.dumps(insights_summary, indent=2)}

AVAILABLE PROPS TO ANALYZE:
{json.dumps(top_props, indent=2)}

Based on these findings, identify:
1. **KNOWLEDGE GAPS**: What key information is missing?
2. **SURPRISING FINDINGS**: Any results that suggest new research directions?
3. **PROP MISMATCHES**: Props that need more specific research?

Generate ADAPTIVE follow-up queries that will fill these gaps.

Return JSON with this structure:
{{
    "analysis": "Brief analysis of findings and gaps identified",
    "followup_statmuse_queries": [
        {{
            "query": "Specific StatMuse question",
            "reasoning": "Why this query is needed based on initial findings",
            "priority": "high/medium/low"
        }}
    ],
    "followup_web_searches": [
        {{
            "query": "Web search query",
            "reasoning": "Why this search is needed",
            "priority": "high/medium/low"
        }}
    ]
}}

Generate 3-6 high-value follow-up queries that will maximize our edge.
"""
        
        try:
            response = await self.grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            followup_text = response.choices[0].message.content
            start_idx = followup_text.find("{")
            end_idx = followup_text.rfind("}") + 1
            followup_plan = json.loads(followup_text[start_idx:end_idx])
            
            # Execute the follow-up queries
            followup_insights = []
            
            # Execute StatMuse follow-up queries
            for query_obj in followup_plan.get("followup_statmuse_queries", [])[:5]:
                try:
                    query_text = query_obj.get("query", query_obj) if isinstance(query_obj, dict) else query_obj
                    priority = query_obj.get("priority", "medium") if isinstance(query_obj, dict) else "medium"
                    
                    logger.info(f"🔍 Follow-up StatMuse ({priority}): {query_text}")
                    result = self.statmuse.query(query_text)
                    
                    if result and "error" not in result:
                        confidence = 0.9 if priority == "high" else 0.7 if priority == "medium" else 0.5
                        followup_insights.append(ResearchInsight(
                            source="statmuse_followup",
                            query=query_text,
                            data=result,
                            confidence=confidence,
                            timestamp=datetime.now()
                        ))
                    
                    await asyncio.sleep(1.5)
                    
                except Exception as e:
                    logger.error(f"❌ Follow-up StatMuse query failed: {e}")
            
            # Execute web follow-up searches
            for search_obj in followup_plan.get("followup_web_searches", [])[:3]:
                try:
                    search_query = search_obj.get("query", search_obj) if isinstance(search_obj, dict) else search_obj
                    priority = search_obj.get("priority", "medium") if isinstance(search_obj, dict) else "medium"
                    
                    logger.info(f"🌐 Follow-up web search ({priority}): {search_query}")
                    result = self.web_search.search(search_query)
                    
                    if result:
                        confidence = 0.8 if priority == "high" else 0.6 if priority == "medium" else 0.4
                        followup_insights.append(ResearchInsight(
                            source="web_followup",
                            query=search_query,
                            data=result,
                            confidence=confidence,
                            timestamp=datetime.now()
                        ))
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Follow-up web search failed: {e}")
            
            return followup_insights
            
        except Exception as e:
            logger.error(f"Failed to generate adaptive follow-up: {e}")
            return []
    
    async def _execute_final_research(self, all_insights: List[ResearchInsight], props: List[PlayerProp]) -> List[ResearchInsight]:
        final_insights = []
        
        statmuse_count = len([i for i in all_insights if "statmuse" in i.source])
        web_count = len([i for i in all_insights if "web" in i.source])
        
        logger.info(f"📊 Research Summary: {statmuse_count} StatMuse + {web_count} Web insights")
        
        if len(all_insights) < 8:
            logger.info("🎯 Adding final broad research queries")
            # Could add more research here if needed
        
        return final_insights
    
    def _distribute_props_by_game_and_sport(self, props: List[PlayerProp], games: List[Dict], sport_distribution: Dict[str, int]) -> Dict[str, List[PlayerProp]]:
        """Create game-aware distribution to ensure picks spread across multiple games"""
        logger.info(f"🎯 Creating game-aware prop distribution for optimal spread")
        
        # Group props by game_id first
        props_by_game = {}
        for prop in props:
            game_id = prop.event_id
            if game_id not in props_by_game:
                props_by_game[game_id] = []
            props_by_game[game_id].append(prop)
        
        # Get game info for each game with props
        game_info = {}
        for game in games:
            game_id = game.get('id', '')
            if game_id in props_by_game:
                game_info[game_id] = {
                    'home_team': game.get('home_team', ''),
                    'away_team': game.get('away_team', ''),
                    'sport': game.get('sport', ''),
                    'prop_count': len(props_by_game[game_id])
                }
        
        logger.info(f"📊 Props distribution by game:")
        for game_id, props_list in props_by_game.items():
            game = game_info.get(game_id, {})
            teams = f"{game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}"
            sport = game.get('sport', 'Unknown')
            logger.info(f"  {teams} ({sport}): {len(props_list)} props")
        
        # Calculate target picks per game based on available props and sport priority
        target_games = min(8, len(props_by_game))  # Target 8 games max for good distribution
        picks_per_game = max(2, 10 // target_games)  # At least 2 picks per game
        
        logger.info(f"🎯 Target distribution: {picks_per_game} picks per game across {target_games} games")
        
        # Select top games by prop count and sport priority for diverse coverage
        games_ranked = []
        for game_id, props_list in props_by_game.items():
            game = game_info.get(game_id, {})
            sport = game.get('sport', '')
            prop_count = len(props_list)
            
            # Priority scoring: MLB=3, NFL=2, WNBA=2, CFB=1, plus prop count
            sport_priority = 3 if 'Baseball' in sport else (2 if 'Football' in sport or 'Basketball' in sport else 1)
            score = (sport_priority * 10) + min(prop_count, 20)  # Cap prop count influence
            
            games_ranked.append((score, game_id, props_list, game))
        
        # Sort by score (highest first) and select top games
        games_ranked.sort(reverse=True)
        selected_games = games_ranked[:target_games]
        
        # Create the final distribution
        game_distribution = {}
        for score, game_id, props_list, game_info in selected_games:
            teams = f"{game_info.get('away_team', 'Unknown')} @ {game_info.get('home_team', 'Unknown')}"
            sport = game_info.get('sport', 'Unknown')
            game_distribution[f"{teams} ({sport})"] = props_list[:picks_per_game * 2]  # 2x for AI selection flexibility
            logger.info(f"✅ Selected: {teams} ({sport}) - {len(props_list)} props available")
        
        return game_distribution
    
    def _format_game_diversity_requirements(self, game_prop_distribution: Dict[str, List[PlayerProp]]) -> str:
        """Format game diversity requirements for AI prompt"""
        if not game_prop_distribution:
            return "- Distribute picks across all available games evenly"
        
        requirements = [
            f"🎯 MANDATORY GAME DISTRIBUTION - You MUST select from {len(game_prop_distribution)} different games:",
            ""
        ]
        
        picks_per_game = max(1, 10 // len(game_prop_distribution))  # Distribute 10 picks across games
        
        for i, (game_desc, props_list) in enumerate(game_prop_distribution.items(), 1):
            prop_types = list(set(p.prop_type for p in props_list))[:5]  # Show first 5 prop types
            requirements.append(f"  {i}. {game_desc}")
            requirements.append(f"     - Select {picks_per_game}-{picks_per_game+1} picks from this game")
            requirements.append(f"     - Available prop types: {', '.join(prop_types)}")
            requirements.append("")
        
        requirements.extend([
            "🚨 CRITICAL DISTRIBUTION RULES:",
            f"- You MUST select picks from AT LEAST {max(3, len(game_prop_distribution)-1)} different games",
            "- NO MORE than 3 picks from any single game",
            "- Prioritize games with more diverse prop types available",
            "- Ensure good mix of over/under recommendations across all games"
        ])
        
        return "\n".join(requirements)
    
    def _format_game_specific_props(self, game_prop_distribution: Dict[str, List[PlayerProp]]) -> str:
        """Format game-specific prop data for AI analysis"""
        if not game_prop_distribution:
            return "No game-specific distribution available"
        
        formatted_sections = []
        
        for game_desc, props_list in game_prop_distribution.items():
            formatted_sections.append(f"\n📊 {game_desc}:")
            formatted_sections.append(f"Available Props ({len(props_list)} total):")
            
            # Group by player for better organization
            players_props = {}
            for prop in props_list[:20]:  # Limit to 20 props per game for prompt size
                if prop.player_name not in players_props:
                    players_props[prop.player_name] = []
                players_props[prop.player_name].append(prop)
            
            for player, player_props in list(players_props.items())[:8]:  # Max 8 players per game
                prop_details = []
                for prop in player_props[:3]:  # Max 3 props per player
                    over_odds = f"+{prop.over_odds}" if prop.over_odds and prop.over_odds > 0 else prop.over_odds
                    under_odds = f"+{prop.under_odds}" if prop.under_odds and prop.under_odds > 0 else prop.under_odds
                    prop_details.append(f"{prop.prop_type} {prop.line} (O:{over_odds}/U:{under_odds})")
                
                formatted_sections.append(f"  • {player}: {', '.join(prop_details)}")
        
        return "\n".join(formatted_sections)

    async def generate_picks_with_reasoning(
        self, 
        insights: List[ResearchInsight], 
        props: List[PlayerProp], 
        games: List[Dict],
        target_picks: int,
        sport_distribution: Dict[str, int] = None
    ) -> List[Dict[str, Any]]:
        try:
            insights_summary = []
            for insight in insights[:40]:
                insights_summary.append({
                    "source": insight.source,
                    "query": insight.query,
                    "data": str(insight.data)[:800],
                    "confidence": insight.confidence,
                    "timestamp": insight.timestamp.isoformat()
                })
            
            # GAME-AWARE DISTRIBUTION: Create game distribution for picks spread
            game_prop_distribution = self._distribute_props_by_game_and_sport(props, games, sport_distribution or {})
            logger.info(f"🎯 Game-aware distribution created with {len(game_prop_distribution)} games")
            
            # Generate game diversity requirements for AI prompt
            game_diversity_requirements = self._format_game_diversity_requirements(game_prop_distribution)
            game_specific_props = self._format_game_specific_props(game_prop_distribution)
            
            MAX_ODDS = 450
            
            filtered_props = []
            long_shot_count = 0
            
            for prop in props:
                # Allow props with either over OR under odds (not requiring both)
                has_over = prop.over_odds is not None
                has_under = prop.under_odds is not None
                
                # Must have at least one side with reasonable odds
                if has_over or has_under:
                    over_reasonable = not has_over or abs(prop.over_odds) <= MAX_ODDS
                    under_reasonable = not has_under or abs(prop.under_odds) <= MAX_ODDS
                
                    if over_reasonable and under_reasonable:
                        filtered_props.append(prop)
                    else:
                        long_shot_count += 1
                        over_str = f"+{prop.over_odds}" if prop.over_odds and prop.over_odds > 0 else str(prop.over_odds) if prop.over_odds else "N/A"
                        under_str = f"+{prop.under_odds}" if prop.under_odds and prop.under_odds > 0 else str(prop.under_odds) if prop.under_odds else "N/A"
                        logger.info(f"🚫 Filtered long shot: {prop.player_name} {prop.prop_type} (Over: {over_str}, Under: {under_str})")
                else:
                    long_shot_count += 1
                    logger.info(f"🚫 Filtered no odds: {prop.player_name} {prop.prop_type} (no odds available)")
            
            logger.info(f"🎯 Filtered props: {len(props)} → {len(filtered_props)} (removed {long_shot_count} long shots with odds > +{MAX_ODDS})")
            
            # Analyze filtered props by sport to understand supply before AI generation
            try:
                event_sport_map = {}
                for g in games:
                    g_s = g.get('sport')
                    sp = 'Other'
                    if g_s == "Women's National Basketball Association":
                        sp = "WNBA"
                    elif g_s == "Major League Baseball":
                        sp = "MLB"
                    elif g_s == "Ultimate Fighting Championship":
                        sp = "UFC"
                    elif g_s == "National Football League":
                        sp = "NFL"
                    event_sport_map[str(g.get('id'))] = sp
                filtered_counts = {}
                for p in filtered_props:
                    sp = event_sport_map.get(str(p.event_id), "Unknown")
                    filtered_counts[sp] = filtered_counts.get(sp, 0) + 1
                logger.info(f"📊 Filtered props by sport: {filtered_counts}")
            except Exception as e:
                logger.warning(f"Failed to summarize filtered props by sport: {e}")
            
            props_data = []
            for prop in filtered_props:
                props_data.append({
                    "player": prop.player_name,
                    "prop_type": prop.prop_type,
                    "line": prop.line,
                    "over_odds": prop.over_odds,
                    "under_odds": prop.under_odds,
                    "team": prop.team,
                    "event_id": prop.event_id,
                    "bookmaker": prop.bookmaker
                })
            
            games_info = json.dumps(games[:10], indent=2, default=str)
            props_info = json.dumps(props_data, indent=2)
            research_summary = json.dumps(insights_summary, indent=2)
            
            props = filtered_props
            
            # Determine active sports from filtered props for dynamic prompt
            active_sports = set()
            for prop in filtered_props[:50]:
                sport = self._get_prop_sport(prop, games)
                if sport != "Unknown":
                    active_sports.add(sport)
            
            sports_list = ", ".join(sorted(active_sports)) if active_sports else "MLB, WNBA, NFL"
            
            prompt = f"""
You are a professional sports betting analyst with 15+ years experience handicapping multi-sport player props ({sports_list}).
Your job is to find PROFITABLE betting opportunities across ALL available sports, not just predict outcomes.

🏆 **SPORT EXPERTISE:**
- **NFL**: Quarterback efficiency, rushing matchups, receiving targets, weather/field conditions, injury reports
- **MLB**: Batter performance trends, pitcher matchups, weather impacts, ballpark factors
- **WNBA**: Player usage rates, pace of play, defensive matchups, rest/travel factors
- **CFB**: College player development, team systems, conference strength

TODAY\'S DATA:

🏟️ UPCOMING GAMES ({len(games)}):
{games_info}

🎯 AVAILABLE PLAYER PROPS ({len(filtered_props)}) - **ONLY PICK FROM THESE FILTERED PROPS**:
{props_info}

💡 **SMART FILTERING**: Long shot props (odds > +{MAX_ODDS}) have been removed to focus on PROFITABLE opportunities.

⚠️  **CRITICAL**: You MUST pick from the exact player names and prop types listed above. 
Available prop types in this data: {set(prop.prop_type for prop in filtered_props[:50])}
Available players in this data: {list(set(prop.player_name for prop in filtered_props[:30]))[:20]}

🔍 RESEARCH INSIGHTS ({len(insights_summary)}):

**STATMUSE DATA FINDINGS:**
{self._format_statmuse_insights(insights_summary)}

**WEB SEARCH INTEL:**
{self._format_web_insights(insights_summary)}

**RAW RESEARCH DATA:**
{research_summary}

TASK: Generate exactly {target_picks} strategic player prop picks that maximize expected value and long-term profit.

🚨 **MANDATORY SPORT DISTRIBUTION:**
{self._format_sport_distribution_requirements(sport_distribution, target_picks)}

🎯 **GAME DIVERSITY REQUIREMENTS:**
{game_diversity_requirements}

🏟️ **GAME-SPECIFIC PROP ANALYSIS:**
{game_specific_props}

🔍 **COMPREHENSIVE ANALYSIS REQUIRED:**
- You have access to {len(filtered_props)} total player props across all games
- DO NOT just pick the same star players repeatedly (Wilson, Stewart, etc.)
- ANALYZE THE ENTIRE POOL of available props before making selections
- Research data covers ALL players with props - use this broad analysis
- Look for VALUE in lesser-known players, not just popular names
- DIVERSIFY prop types: points, rebounds, assists, hits, home runs, RBIs, etc.
- Select the BEST {target_picks} picks from your comprehensive analysis of ALL options

🚨 **BETTING DISCIPLINE REQUIREMENTS:**
1. **MANDATORY ODDS CHECK**: Before picking, check the over_odds and under_odds in the data
2. **ODDS PRESENCE FOR CHOSEN SIDE**: Only recommend a pick if odds for your chosen side (over or under) are present; skip if the chosen side's odds are missing
3. **NO HIGH-ODDS PICKS**: Never pick sides with odds higher than +{MAX_ODDS} (even if available)
4. **AVOID LONG SHOTS**: Props with +450, +500, +950, +1300 odds are SUCKER BETS - ignore them!
5. **FOCUS ON VALUE RANGE**: Target odds between -250 and +250 for best long-term profit
6. **DIVERSIFY PROP TYPES**: 
   - **MLB**: Hits, Home Runs, RBIs, Runs Scored, Stolen Bases, Total Bases
   - **WNBA**: Points, Rebounds, Assists (see available props below)
7. **MIX OVER/UNDER**: Don't just pick all overs - find spots where under has value
8. **REALISTIC CONFIDENCE**: Most picks should be 55-65% confidence (sharp betting range)
9. **VALUE HUNTING**: Focus on lines that seem mispriced based on data

PROFITABLE BETTING STRATEGY:
- **Focus on -200 to +200 odds**: This is the profitable betting sweet spot
- **0.5 Hit props**: Look for struggling hitters vs tough pitchers (UNDER value)
- **1.5 Hit props**: Target hot hitters vs weak pitching (OVER value)  
- **1.5 Total Base props**: Consider park factors, weather, matchup history
- **Fade public favorites**: Elite players often have inflated lines
- **Target situational spots**: Day games, travel, pitcher handedness
- **Avoid "lottery tickets"**: High-odds props (+500+) are designed to lose money

CONFIDENCE SCALE (BE REALISTIC):
- 52-55%: Marginal edge, small value (only if great odds)
- 56-60%: Solid spot, good value (most picks should be here)
- 61-65%: Strong conviction, clear edge
- 66-70%: Exceptional opportunity (very rare)

💰 **REMEMBER**: Professional bettors win by finding small edges consistently, NOT by chasing big payouts!
- 71%+: Only for obvious mispricing

🚨 CRITICAL RESPONSE FORMAT REQUIREMENTS:
- MUST respond with ONLY a valid JSON array starting with [ and ending with ]
- NO explanatory text before or after the JSON
- NO markdown code blocks or formatting
- NO comments or additional text
- Just pure JSON array format

FORMAT RESPONSE AS JSON ARRAY (ONLY JSON, NO OTHER TEXT):
[
  {{
    "player_name": "Full Player Name",
    "prop_type": "Hits", "Home Runs", "RBIs", "Runs Scored", "Stolen Bases", "Hits Allowed", "Innings Pitched", "Strikeouts (Pitcher)", "Walks Allowed",
    "recommendation": "over" or "under",
    "line": line_value,
    "odds": american_odds_value,
    "confidence": confidence_percentage,
    "reasoning": "4-6 sentence comprehensive analysis. Start with the key edge or advantage identified, explain the supporting data or trends that led to this conclusion, mention any relevant player/team factors, and conclude with why this represents value. Be specific about numbers, trends, or situational factors that support the pick.",
    "key_factors": ["factor_1", "factor_2", "factor_3"],
    "roi_estimate": "percentage like 8.5% or 12.3%",
    "value_percentage": "percentage like 15.2% or 22.8%",
    "implied_probability": "percentage like 45.5% or 62.1%",
    "fair_odds": "what the odds should be like -140 or +165"
  }}
]

IMPORTANT: Your response must start with [ and end with ]. No other text allowed.

🧮 **CALCULATION REQUIREMENTS:**

**ROI Estimate:** (Expected Win Amount / Risk Amount) - 1
- Example: If you bet $100 at +150 odds with 55% win rate: ROI = (55% × $150 - 45% × $100) / $100 = 37.5%
- Target range: 5-25% for sustainable profit

**Value Percentage:** (Your Win Probability - Implied Probability) × 100
- Example: You think 60% chance, odds imply 52% = 8% value
- Positive value = good bet, negative value = bad bet

**Implied Probability:** Convert American odds to probability
- Positive odds: 100 / (odds + 100)
- Negative odds: |odds| / (|odds| + 100)

**Fair Odds:** What odds should be based on your confidence
- If you think 60% chance: Fair odds = +67 (100/40 - 1)
- If you think 45% chance: Fair odds = +122 (100/45 - 1)

THINK LIKE A SHARP: Find spots where the oddsmakers may have made mistakes or where public perception differs from reality.

REMEMBER:
- **DIVERSIFY ACROSS ALL PROP TYPES**: Use Hits, Home Runs, RBIs, Runs Scored, Stolen Bases, and Pitcher props
- Mix overs and unders based on VALUE, not bias  
- Keep confidence realistic (most picks 55-65%)
- Focus on profitable opportunities, not just likely outcomes
- Each pick should be one you\'d bet your own money on
- **Available Batter Props**: Hits, Home Runs, RBIs, Runs Scored, Stolen Bases
- **Available Pitcher Props**: Hits Allowed, Innings Pitched, Strikeouts, Walks Allowed

**CRITICAL ODDS RULES:**
- **NEVER recommend "UNDER 0.5 Home Runs"** - impossible (can't get negative home runs)
- **NEVER recommend "UNDER 0.5 Stolen Bases"** - impossible (can't get negative steals)
- **Only recommend props where both over/under make logical sense**
- **Home Runs 0.5**: Only bet OVER (under is impossible)
- **Stolen Bases 0.5**: Only bet OVER (under is impossible)
"""
            
            try:
                response = await self.grok_client.chat.completions.create(
                    model="grok-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                picks_text = response.choices[0].message.content.strip()
                logger.info(f"🧠 Grok raw response: {picks_text[:500]}...")
                
                # DEBUG: Log the full response to understand the format
                logger.info(f"🔍 FULL Grok response for debugging:\n{picks_text}")
                
                # Try to remove markdown code blocks if present
                if "```json" in picks_text.lower():
                    logger.info("Detected markdown JSON code block, extracting...")
                    start_marker = picks_text.lower().find("```json") + 7
                    end_marker = picks_text.find("```", start_marker)
                    if end_marker != -1:
                        picks_text = picks_text[start_marker:end_marker].strip()
                        logger.info(f"Extracted JSON from markdown: {picks_text[:200]}...")
                elif "```" in picks_text:
                    logger.info("Detected markdown code block, extracting...")
                    start_marker = picks_text.find("```") + 3
                    end_marker = picks_text.find("```", start_marker)
                    if end_marker != -1:
                        picks_text = picks_text[start_marker:end_marker].strip()
                        logger.info(f"Extracted content from markdown: {picks_text[:200]}...")
                
                # Find and extract the JSON array
                start_idx = picks_text.find("[")
                end_idx = picks_text.rfind("]") + 1
                
                logger.info(f"🔍 JSON search results: start_idx={start_idx}, end_idx={end_idx}")
                
                if start_idx == -1 or end_idx == 0:
                    logger.error("No JSON array found in Grok response")
                    logger.error(f"Response length: {len(picks_text)}")
                    logger.error(f"First 1000 chars: {picks_text[:1000]}")
                    logger.error(f"Last 500 chars: {picks_text[-500:]}")
                    
                    # Try to find if there's JSON object without array brackets
                    if "{" in picks_text and "}" in picks_text:
                        logger.info("Found JSON object syntax, attempting to wrap in array...")
                        # Try to extract objects and wrap them in an array
                        objects = []
                        # Find all JSON objects
                        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                        matches = re.findall(pattern, picks_text, re.DOTALL)
                        for match in matches:
                            try:
                                obj = json.loads(match)
                                objects.append(obj)
                            except:
                                continue
                        if objects:
                            logger.info(f"Extracted {len(objects)} JSON objects, proceeding...")
                            ai_picks = objects
                        else:
                            return []
                    else:
                        return []
                else:
                    json_str = picks_text[start_idx:end_idx]
                    
                    # Enhanced error handling for JSON parsing
                    try:
                        ai_picks = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error: {e}")
                        
                        # Attempt to fix common JSON issues
                        fixed_json_str = self._fix_json_string(json_str)
                        logger.info("Attempting to parse with fixed JSON")
                        
                        try:
                            ai_picks = json.loads(fixed_json_str)
                            logger.info("Successfully parsed JSON after fixes")
                        except json.JSONDecodeError as e2:
                            logger.error(f"Still failed to parse JSON after fixes: {e2}")
                            
                            # Last resort: manual object extraction
                            ai_picks = self._manual_json_parser(json_str)
                            if not ai_picks:
                                logger.error("Manual parsing failed, no valid picks found")
                                return []
            
            except Exception as e:
                logger.error(f"Failed to process AI response: {e}")
                return []
                
            formatted_picks = []
            for pick in ai_picks:
                matching_prop = self._find_matching_prop(pick, props)
                
                if matching_prop:
                    # CRITICAL: Validate that the pick has valid odds for the recommendation
                    recommendation = pick.get("recommendation", "").lower()
                    prop_type = pick.get("prop_type", "").lower()
                    line = float(pick.get("line", 0))
                    
                    # Check for impossible props that should be skipped
                    is_impossible = False
                    
                    # Home runs under 0.5 is impossible (can't get negative home runs)
                    if "home run" in prop_type and recommendation == "under" and line <= 0.5:
                        logger.warning(f"🚫 Skipping impossible prop: {pick['player_name']} {prop_type} UNDER {line} (impossible)")
                        is_impossible = True
                    
                    # Stolen bases under 0.5 is impossible
                    if "stolen base" in prop_type and recommendation == "under" and line <= 0.5:
                        logger.warning(f"🚫 Skipping impossible prop: {pick['player_name']} {prop_type} UNDER {line} (impossible)")
                        is_impossible = True
                    
                    # Check if the recommendation has valid odds
                    if not is_impossible:
                        if recommendation == "over" and matching_prop.over_odds is None:
                            logger.warning(f"🚫 Skipping pick with missing over odds: {pick['player_name']} {prop_type} OVER {line}")
                            is_impossible = True
                        elif recommendation == "under" and matching_prop.under_odds is None:
                            logger.warning(f"🚫 Skipping pick with missing under odds: {pick['player_name']} {prop_type} UNDER {line}")
                            is_impossible = True
                    
                    if is_impossible:
                        continue
                    
                    game = next((g for g in games if str(g.get("id")) == str(matching_prop.event_id)), None)
                    
                    # Determine sport from game data, not hardcoded
                    sport = "MLB"  # default
                    if game:
                        game_sport = game.get('sport', 'MLB')
                        if game_sport == "Women's National Basketball Association":
                            sport = "WNBA"
                        elif game_sport == "Major League Baseball":
                            sport = "MLB"
                        elif game_sport == "Ultimate Fighting Championship":
                            sport = "UFC"
                        elif game_sport == "National Football League":
                            sport = "NFL"
                        elif game_sport == "College Football":
                            sport = "CFB"
                        
                        # Create proper game info with team matchup
                        home_team = game.get('home_team', 'Unknown')
                        away_team = game.get('away_team', 'Unknown')
                        game_info = f"{away_team} @ {home_team}"
                    else:
                        # Fallback: try to determine sport from player name patterns
                        player_name = pick.get('player_name', '').lower()
                        wnba_players = ['paige bueckers', 'arike ogunbowale', 'skylar diggins-smith', 
                                       'nneka ogwumike', 'gabby williams', 'li yueru', 'erica wheeler']
                        if any(wnba_player in player_name for wnba_player in wnba_players):
                            sport = "WNBA"
                        game_info = f"{matching_prop.team} game"
                    
                    formatted_picks.append({
                        "match_teams": game_info,
                        "pick": self._format_pick_string(pick, matching_prop),
                        "odds": pick.get("odds") or (
                            matching_prop.over_odds if pick["recommendation"] == "over" and matching_prop.over_odds is not None
                            else matching_prop.under_odds if pick["recommendation"] == "under" and matching_prop.under_odds is not None
                            else None  # Don't use wrong odds as fallback
                        ),
                        "confidence": pick.get("confidence", 75),
                        "sport": sport,
                        "event_time": game.get("start_time") if game else None,
                        "bet_type": "player_prop",
                        "bookmaker": matching_prop.bookmaker,
                        "event_id": matching_prop.event_id,
                        "team": matching_prop.team,
                        "metadata": {
                            "player_name": pick["player_name"],
                            "prop_type": pick["prop_type"],
                            "line": pick["line"],
                            "recommendation": pick["recommendation"],
                            "reasoning": pick.get("reasoning", "AI-generated pick"),
                            "roi_estimate": pick.get("roi_estimate", "0%"),
                            "value_percentage": pick.get("value_percentage", "0%"),
                            "implied_probability": pick.get("implied_probability", "50%"),
                            "fair_odds": pick.get("fair_odds", pick.get("odds", 0)),
                            "key_factors": pick.get("key_factors", []),
                            "risk_level": pick.get("risk_level", "medium"),
                            "expected_value": pick.get("expected_value", "Positive EV expected"),
                            "research_support": pick.get("research_support", "Based on comprehensive analysis"),
                            "ai_generated": True,
                            "research_insights_count": len(insights),
                            "model_used": "grok-4"
                        }
                    })
                else:
                    logger.warning(f"No matching prop found for {pick.get('player_name')} {pick.get('prop_type')}")
            
            # Sort all picks by confidence for best-first presentation
            formatted_picks.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Log AI picks by sport BEFORE enforcement
            pre_counts = {}
            for p in formatted_picks:
                sp = p.get("sport", "Unknown")
                pre_counts[sp] = pre_counts.get(sp, 0) + 1
            pre_summary = " + ".join([f"{count} {sport}" for sport, count in pre_counts.items()])
            logger.info(f"🧮 AI picks before enforcement: {pre_summary} = {len(formatted_picks)} total picks")
            
            # Enforce sport quotas programmatically to prioritize MLB and cap WNBA
            final_picks = formatted_picks
            if sport_distribution:
                desired = {s: c for s, c in sport_distribution.items() if c and c > 0}
                # Hard cap WNBA at 8
                if "WNBA" in desired:
                    desired["WNBA"] = min(desired["WNBA"], 8)
                
                # Normalize desired quotas to sum exactly to target_picks
                total_desired = sum(desired.values()) if desired else 0
                if total_desired > 0 and total_desired != target_picks:
                    logger.info(f"📐 Normalizing quotas from {desired} (sum {total_desired}) to target {target_picks}")
                    # Proportional scaling with largest remainder distribution
                    ratio = target_picks / max(1, total_desired)
                    scaled = {sp: int(desired[sp] * ratio) for sp in desired}
                    # Compute remainders for fair rounding
                    remainders = {sp: (desired[sp] * ratio) - scaled[sp] for sp in desired}
                    # Ensure at least 1 for any sport that originally had quota > 0 when target allows
                    nonzero_sports = [sp for sp, c in desired.items() if c > 0]
                    # Distribute remaining slots by largest remainders
                    current_sum = sum(scaled.values())
                    remaining_slots = max(0, target_picks - current_sum)
                    if remaining_slots > 0:
                        for sp, _ in sorted(remainders.items(), key=lambda kv: kv[1], reverse=True):
                            if remaining_slots <= 0:
                                break
                            scaled[sp] = scaled.get(sp, 0) + 1
                            remaining_slots -= 1
                    desired = scaled
                logger.info(f"📐 Desired quotas by sport (normalized): {desired}")

                # Organize picks by sport (already confidence-sorted)
                name_map = {
                    "Major League Baseball": "MLB",
                    "Women's National Basketball Association": "WNBA",
                    "National Football League": "NFL",
                    "Ultimate Fighting Championship": "UFC",
                    "College Football": "CFB",
                    "MLB": "MLB",
                    "WNBA": "WNBA",
                    "NFL": "NFL",
                    "UFC": "UFC",
                    "CFB": "CFB",
                }
                by_sport = {}
                for p in formatted_picks:
                    sp_raw = p.get("sport", "Other")
                    sp = name_map.get(sp_raw, sp_raw)
                    by_sport.setdefault(sp, []).append(p)
                
                # Build dynamic allocation order by desired quotas (highest first)
                allocation_order = [sp for sp, _ in sorted(desired.items(), key=lambda kv: kv[1], reverse=True)]
                # Fallback order for any sports not in desired
                fallback_order = allocation_order + [sp for sp in by_sport.keys() if sp not in allocation_order] + ["Other"]
                
                selected = []
                used_ids = set()
                
                # Helper to add picks without duplicates
                def add_from_bucket(bucket, limit):
                    added = 0
                    for item in bucket:
                        uid = f"{item.get('event_id')}|{item['metadata'].get('player_name')}|{item['metadata'].get('prop_type')}|{item['metadata'].get('recommendation')}"
                        if uid in used_ids:
                            continue
                        selected.append(item)
                        used_ids.add(uid)
                        added += 1
                        if added >= limit:
                            break
                    return added
                
                # Allocate quotas strictly by desired distribution
                for sp in allocation_order:
                    take_quota = min(desired.get(sp, 0), len(by_sport.get(sp, [])))
                    if take_quota > 0:
                        add_from_bucket(by_sport.get(sp, []), take_quota)
                
                # Fill remaining up to target with extras, starting from sports with remaining depth
                remaining = max(0, target_picks - len(selected))
                if remaining > 0:
                    # Compute extras per sport (beyond quota)
                    for sp in allocation_order:
                        if remaining <= 0:
                            break
                        start_idx = desired.get(sp, 0)
                        extras = by_sport.get(sp, [])[start_idx:]
                        if extras:
                            remaining -= add_from_bucket(extras, remaining)
                
                # Still remaining? Use any other sports (including CFB) not in desired
                if remaining > 0:
                    for sp in fallback_order:
                        if remaining <= 0:
                            break
                        # Skip already exhausted slices
                        start_idx = desired.get(sp, 0)
                        extras = by_sport.get(sp, [])[start_idx:]
                        if extras:
                            remaining -= add_from_bucket(extras, remaining)

                final_picks = selected if selected else formatted_picks
                # If we overshot, trim to target_picks while preserving priority order
                if len(final_picks) > target_picks:
                    final_picks = final_picks[:target_picks]
            
            # Log sport distribution AFTER enforcement
            sport_counts = {}
            for pick in final_picks:
                sport = pick.get("sport", "Unknown")
                sport_counts[sport] = sport_counts.get(sport, 0) + 1
            sport_summary = " + ".join([f"{count} {sport}" for sport, count in sport_counts.items()])
            logger.info(f"🎯 Final selection (enforced): {sport_summary} = {len(final_picks)} total picks")
            
            if final_picks:
                prop_types = {}
                recommendations = {"over": 0, "under": 0}
                confidence_ranges = {"50-60": 0, "61-70": 0, "71+": 0}
                
                for pick in final_picks:
                    prop_type = pick["metadata"]["prop_type"]
                    prop_types[prop_type] = prop_types.get(prop_type, 0) + 1
                    
                    rec = pick["metadata"]["recommendation"]
                    recommendations[rec] += 1
                    
                    conf = pick["confidence"]
                    if conf <= 60:
                        confidence_ranges["50-60"] += 1
                    elif conf <= 70:
                        confidence_ranges["61-70"] += 1
                    else:
                        confidence_ranges["71+"] += 1
                
                logger.info(f"📊 Pick Diversity Analysis:")
                logger.info(f"  Prop Types: {dict(prop_types)}")
                logger.info(f"  Over/Under: {dict(recommendations)}")
                logger.info(f"  Confidence Ranges: {dict(confidence_ranges)}")
                
                logger.info(f"📝 Generated {len(final_picks)} diverse picks:")
                for i, pick in enumerate(final_picks, 1):
                    meta = pick["metadata"]
                    logger.info(f"  {i}. {meta['player_name']} {meta['prop_type']} {meta['recommendation'].upper()} {meta['line']} ({pick['confidence']}% conf)")
            
            return final_picks
            
        except Exception as e:
            logger.error(f"Failed to generate picks: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _fix_json_string(self, json_str: str) -> str:
        """Attempt to fix common JSON format issues"""
        try:
            # First, try direct parsing
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            logger.info(f"Fixing JSON error: {str(e)}")
            
            # Get error details
            error_msg = str(e)
            
            # Handle specific error: Missing comma
            if "Expecting ',' delimiter" in error_msg:
                # Extract line and column information
                match = re.search(r"line (\d+) column (\d+)", error_msg)
                if match:
                    line_num = int(match.group(1))
                    col_num = int(match.group(2))
                    
                    # Split JSON by lines
                    lines = json_str.split('\n')
                    
                    # If the line exists
                    if 0 <= line_num - 1 < len(lines):
                        # Insert comma at the right position
                        problematic_line = lines[line_num - 1]
                        if col_num <= len(problematic_line):
                            fixed_line = problematic_line[:col_num] + ',' + problematic_line[col_num:]
                            lines[line_num - 1] = fixed_line
                            json_str = '\n'.join(lines)
                            logger.info(f"Added missing comma at line {line_num}, column {col_num}")
            
            # Apply all other fixes
            
            # Remove trailing commas before closing braces/brackets
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Add missing commas between objects in arrays
            json_str = re.sub(r'}\s*{', '}, {', json_str)
            
            # Replace Python quotes with JSON quotes
            json_str = re.sub(r'None', 'null', json_str)
            json_str = re.sub(r'True', 'true', json_str)
            json_str = re.sub(r'False', 'false', json_str)
            
            # Add missing quotes around keys
            json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', json_str)
            
            # Try to handle missing quotes around strings
            json_str = re.sub(r':\s*([^"{}\[\],\d][^,}\]]*?)([,}\]])', r': "\1"\2', json_str)
            
            # Remove any illegal control characters
            json_str = re.sub(r'[\x00-\x09\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
            
            # Try to balance brackets/braces
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            # Add missing closing brackets/braces
            json_str = json_str.rstrip()
            if open_brackets > close_brackets:
                json_str += ']' * (open_brackets - close_brackets)
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            
            # Last resort: use regex to extract what looks like a valid JSON array
            if '[' in json_str and ']' in json_str:
                match = re.search(r'\[.*?\]', json_str, re.DOTALL)
                if match:
                    extracted = match.group(0)
                    logger.info(f"Extracted JSON array: {extracted[:100]}...")
                    return extracted
                
            return json_str
    
    def _manual_json_parser(self, json_str: str) -> List[Dict]:
        """Manual fallback parser for when automatic JSON fixing fails"""
        logger.info("Attempting manual JSON array parsing")
        picks = []
        
        # Extract JSON objects within the array
        object_pattern = re.compile(r'{(.*?)}', re.DOTALL)
        matches = object_pattern.finditer(json_str)
        
        for match in matches:
            obj_str = '{' + match.group(1) + '}'
            try:
                # Try to parse each object individually
                fixed_obj = self._fix_json_string(obj_str)
                pick_obj = json.loads(fixed_obj)
                
                # Validate required fields
                required_fields = ["player_name", "prop_type", "recommendation", "line"]
                if all(field in pick_obj for field in required_fields):
                    picks.append(pick_obj)
                    logger.info(f"Successfully parsed pick for {pick_obj.get('player_name')}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse individual JSON object")
                continue
        
        logger.info(f"Manually extracted {len(picks)} valid picks")
        return picks

    def _format_statmuse_insights(self, insights_summary: List[Dict]) -> str:
        statmuse_insights = [i for i in insights_summary if i.get("source") == "statmuse"]
        if not statmuse_insights:
            return "No StatMuse data available"
        
        formatted = []
        for insight in statmuse_insights[:10]:
            query = insight.get("query", "")
            data = insight.get("data", "")
            confidence = insight.get("confidence", 0.5)
            
            data_clean = str(data).replace("{", "").replace("}", "").replace("\"", "")
            if len(data_clean) > 300:
                data_clean = data_clean[:300] + "..."
            
            formatted.append(f"• Q: {query}\n  A: {data_clean} (confidence: {confidence:.1f})")
        
        return "\n\n".join(formatted)
    
    def _format_web_insights(self, insights_summary: List[Dict]) -> str:
        web_insights = [i for i in insights_summary if i.get("source") == "web_search"]
        if not web_insights:
            return "No web search data available"
        
        formatted = []
        for insight in web_insights[:5]:
            query = insight.get("query", "")
            data = insight.get("data", "")
            
            data_clean = str(data).replace("{", "").replace("}", "").replace("\"", "")
            if len(data_clean) > 200:
                data_clean = data_clean[:200] + "..."
            
            formatted.append(f"• Search: {query}\n  Result: {data_clean}")
        
        return "\n\n".join(formatted)
    
    def _find_matching_prop(self, pick: Dict, props: List[PlayerProp]) -> PlayerProp:
        player_name = pick.get("player_name", "")
        prop_type = pick.get("prop_type", "")
        
        exact_match = next(
            (p for p in props 
             if p.player_name == player_name and p.prop_type == prop_type),
            None
        )
        if exact_match:
            return exact_match
        
        name_variations = [
            player_name,
            player_name.replace(" Jr.", ""),
            player_name.replace(" Sr.", ""),
            player_name.replace(".", ""),
        ]
        
        for name_var in name_variations:
            fuzzy_match = next(
                (p for p in props 
                 if name_var.lower() in p.player_name.lower() 
                 and p.prop_type == prop_type),
                None
            )
            if fuzzy_match:
                logger.info(f"✅ Fuzzy matched \'{player_name}\' to \'{fuzzy_match.player_name}\'")
                return fuzzy_match
        
        prop_type_mappings = {
            "Batter Hits O/U": ["batter_hits", "hits"],
            "Batter Total Bases O/U": ["batter_total_bases", "total_bases"],
            "Batter Home Runs O/U": ["batter_home_runs", "home_runs"],
            "Batter RBIs O/U": ["batter_rbis", "rbis"]
        }
        
        for mapped_type, variations in prop_type_mappings.items():
            if prop_type == mapped_type:
                for var in variations:
                    prop_var_match = next(
                        (p for p in props 
                         if p.player_name == player_name and var in p.prop_type.lower()),
                        None
                    )
                    if prop_var_match:
                        logger.info(f"✅ Prop type matched \'{prop_type}\' to \'{prop_var_match.prop_type}\'")
                        return prop_var_match
        
        available_for_player = [p.prop_type for p in props if p.player_name == player_name]
        logger.warning(f"❌ No match for {player_name} {prop_type}. Available for this player: {available_for_player[:5]}")
        
        return None

    def _format_pick_string(self, pick: Dict, matching_prop: PlayerProp) -> str:
        """Formats the pick string for clarity."""
        player_name = pick.get("player_name", "")
        prop_type = pick.get("prop_type", "")
        recommendation = pick.get("recommendation", "").lower()
        line = pick.get("line")

        if prop_type in ["Hits", "Home Runs", "RBIs", "Runs Scored", "Stolen Bases"]:
            return f"{player_name} {prop_type} {recommendation.capitalize()} {line}"
        elif prop_type in ["Hits Allowed", "Innings Pitched", "Strikeouts (Pitcher)", "Walks Allowed"]:
            return f"{player_name} {prop_type} {recommendation.capitalize()} {line}"
        return f"{player_name} {prop_type} {recommendation} {line}" # Fallback

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate AI player prop betting picks')
    parser.add_argument('--tomorrow', action='store_true', 
                      help='Generate picks for tomorrow instead of today')
    parser.add_argument('--date', type=str, 
                      help='Specific date to generate picks for (YYYY-MM-DD)')
    parser.add_argument('--picks', type=int, default=15,
                      help='Target number of total props to generate (default: 15)')
    parser.add_argument('--wnba', action='store_true',
                      help='Generate 5 best WNBA picks only (overrides --picks)')
    parser.add_argument('--nfl-week', action='store_true',
                      help='Generate 5 best NFL picks for the entire week ahead (Thu-Sun)')
    parser.add_argument('--nfl-only', action='store_true',
                      help='Generate picks for NFL games only (ignore other sports)')
    parser.add_argument('--sport', type=str, choices=['NFL', 'MLB', 'WNBA', 'CFB'],
                      help='Limit props to a single sport (overrides multi-sport distribution)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            return
    elif args.tomorrow:
        target_date = datetime.now().date() + timedelta(days=1)
    else:
        target_date = datetime.now().date()
    
    logger.info(f"🤖 Starting Intelligent Player Props Agent for {target_date}")
    target_picks = args.picks
    
    agent = IntelligentPlayerPropsAgent()
    # Apply sport filters to agent and DB
    try:
        if hasattr(agent, 'db'):
            # NFL-only flag
            setattr(agent, 'nfl_only_mode', bool(args.nfl_only))
            setattr(agent.db, 'nfl_only_mode', bool(args.nfl_only))
            # Generic sport filter via --sport
            if args.sport:
                sport_map = {
                    'NFL': 'National Football League',
                    'MLB': 'Major League Baseball',
                    'WNBA': "Women's National Basketball Association",
                    'CFB': 'College Football'
                }
                full = sport_map.get(args.sport.upper())
                if full:
                    setattr(agent.db, 'sport_filter', [full])
                    logger.info(f"🎯 Sport filter enabled (props): only '{full}' games will be used")
                if args.sport.upper() == 'NFL':
                    setattr(agent, 'nfl_only_mode', True)
                    setattr(agent.db, 'nfl_only_mode', True)
    except Exception as e:
        logger.warning(f"Could not apply sport filters to props DB client: {e}")
    
    # EXACT pick target: honor requested --picks without escalation
    target_picks = args.picks
    # Set NFL week mode if flag is provided
    agent.nfl_week_mode = args.nfl_week
    
    # Keep the requested target_picks exact; distribution will be decided intelligently later
    # (Legacy dynamic escalation removed to honor explicit --picks value.)
    picks = await agent.generate_daily_picks(
        target_date=target_date,
        target_picks=target_picks
    )
    
    if picks:
        logger.info(f"✅ Successfully generated {len(picks)} intelligent picks for {target_date}!")
        
        # Group picks by sport for summary
        picks_by_sport = {}
        for pick in picks:
            sport = pick.get("sport", "Unknown")
            if sport not in picks_by_sport:
                picks_by_sport[sport] = []
            picks_by_sport[sport].append(pick)
        
        for sport, sport_picks in picks_by_sport.items():
            logger.info(f"📊 {sport}: {len(sport_picks)} picks")
            for i, pick in enumerate(sport_picks[:3], 1):  # Show first 3 picks per sport
                logger.info(f"  {i}. {pick['pick']} (Confidence: {pick['confidence']}%)")
            if len(sport_picks) > 3:
                logger.info(f"  ... and {len(sport_picks) - 3} more {sport} picks")
    else:
        logger.warning(f"❌ No picks generated for {target_date}")

if __name__ == "__main__":
    asyncio.run(main())