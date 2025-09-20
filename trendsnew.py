#!/usr/bin/env python3
"""
Intelligent Trends Generator
Uses AI to determine which players/teams to analyze based on upcoming games and available props,
then scrapes Baseball Reference and uses StatMuse strategically to generate targeted trends.

Workflow:
1. Analyze upcoming games and available props
2. AI selects most valuable players/teams to research  
3. Scrape Baseball Reference for selected players
4. Run targeted StatMuse queries
5. AI synthesizes data into 9 player prop + 6 team trends
"""

import os
import sys
import requests
import json
import asyncio
import httpx
import re
from datetime import datetime, timedelta, date
from supabase import create_client, Client
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
import time
from typing import Dict, List, Optional, Tuple, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentTrendsGenerator:
    def __init__(self):
        # Initialize Supabase
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize OpenAI (Grok)
        self.grok_client = AsyncOpenAI(
            api_key=os.getenv('XAI_API_KEY'),
            base_url="https://api.x.ai/v1"
        )
        
        # Initialize Apify client
        self.apify_api_token = os.getenv('APIFY_API_TOKEN')
        if not self.apify_api_token:
            logger.warning("APIFY_API_TOKEN not found in environment variables")
        self.statmuse_api_url = os.getenv('STATMUSE_API_URL', 'https://web-production-f090e.up.railway.app')
        
        logger.info(f"Connecting to Supabase at: {self.supabase_url[:50]}...")
        
        # Load player mappings
        self.player_mappings = self.load_player_mappings()
        
        # Simple PlayerProp class for compatibility
        class PlayerProp:
            def __init__(self, player_name, prop_type, line, over_odds, under_odds, event_id, team, bookmaker):
                self.player_name = player_name
                self.prop_type = prop_type
                self.line = line
                self.over_odds = over_odds
                self.under_odds = under_odds
                self.event_id = event_id
                self.team = team
                self.bookmaker = bookmaker
        
        self.PlayerProp = PlayerProp
        
    def load_player_mappings(self) -> Dict[str, Dict]:
        """Load player mappings from database for name resolution"""
        try:
            response = self.supabase.table('players')\
                .select('id, name, player_name, team')\
                .eq('sport', 'MLB')\
                .execute()
            
            mappings = {}
            for player in response.data:
                # Create multiple mapping keys for flexible matching
                names_to_try = []
                if player.get('name'):
                    names_to_try.append(player['name'])
                if player.get('player_name'):
                    names_to_try.append(player['player_name'])
                
                for name in names_to_try:
                    if name:
                        # Full name
                        mappings[name.lower()] = player
                        # Create abbreviated version (F. Freeman from Freddie Freeman)
                        parts = name.split()
                        if len(parts) >= 2:
                            abbreviated = f"{parts[0][0]}. {parts[-1]}"
                            mappings[abbreviated.lower()] = player
            
            logger.info(f"Loaded {len(mappings)} player name mappings")
            return mappings
        except Exception as e:
            logger.error(f"Error loading player mappings: {e}")
            return {}

    async def analyze_upcoming_games_and_props(self) -> Dict:
        """Step 1: Analyze upcoming games and available props using working patterns from props_enhanced.py"""
        try:
            # Use working database patterns from props_enhanced.py
            target_date = datetime.now().date()
            
            # Get games using the working method from props_enhanced.py
            games_data = self.get_games_for_date(target_date)
            
            # Get recent AI predictions for context 
            predictions_response = self.supabase.table('ai_predictions')\
                .select('*')\
                .eq('sport', 'MLB')\
                .gte('created_at', (datetime.now() - timedelta(days=7)).isoformat())\
                .limit(50)\
                .execute()
            
            # Get props using working method
            props_data = []
            if games_data:
                game_ids = [game["id"] for game in games_data]
                props_raw = self.get_player_props_for_games(game_ids)
                # Convert PlayerProp objects to dictionaries for JSON serialization
                props_data = []
                for prop in props_raw:
                    props_data.append({
                        'player_name': prop.player_name,
                        'prop_type': prop.prop_type,
                        'line': prop.line,
                        'over_odds': prop.over_odds,
                        'under_odds': prop.under_odds,
                        'event_id': prop.event_id,
                        'team': prop.team,
                        'bookmaker': prop.bookmaker
                    })
            
            analysis_data = {
                'upcoming_games': games_data,
                'recent_predictions': predictions_response.data,
                'available_props': props_data,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Found {len(games_data)} upcoming games, {len(predictions_response.data)} recent predictions, {len(props_data)} prop bets")
            
            # If no upcoming games (offseason), add some context
            if len(games_data) == 0:
                analysis_data['offseason_mode'] = True
                logger.info("No upcoming games found - running in offseason mode")
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error analyzing games and props: {e}")
            return {'upcoming_games': [], 'recent_predictions': [], 'available_props': []}

    def get_games_for_date(self, target_date: datetime.date) -> List[Dict[str, Any]]:
        """Get games for date using working patterns from props_enhanced.py"""
        try:
            # Use the exact same approach as props_enhanced.py
            now = datetime.now()
            current_date = now.date()
            
            if target_date == current_date:
                # Today - start from now
                start_time = now
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
            
            # Fetch games from MLB only - as per requirements (focus on MLB)
            all_games = []
            sports = ["Major League Baseball"]  # Use correct full sport name
            
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

    def _safe_int_convert(self, value) -> Optional[int]:
        """Safely convert a value to int, handling strings and None"""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert odds value to int: {value}")
            return None
    
    def get_player_props_for_games(self, game_ids: List[str]) -> List:
        """Get player props using working patterns from props_enhanced.py"""
        if not game_ids:
            return []
        
        try:
            response = self.supabase.table("player_props_odds").select(
                "line, over_odds, under_odds, event_id, "
                "players(name, team), "
                "player_prop_types(prop_name)"
            ).in_("event_id", game_ids).execute()
            
            props = []
            for row in response.data:
                if (row.get("players") and 
                    row.get("player_prop_types") and 
                    row["players"].get("name") and 
                    row["player_prop_types"].get("prop_name")):
                    
                    props.append(self.PlayerProp(
                        player_name=row["players"]["name"],
                        prop_type=row["player_prop_types"]["prop_name"],
                        line=float(row["line"]),
                        over_odds=self._safe_int_convert(row["over_odds"]),
                        under_odds=self._safe_int_convert(row["under_odds"]),
                        event_id=row["event_id"],
                        team=row["players"]["team"] if row["players"]["team"] else "Unknown",
                        bookmaker="fanduel"
                    ))
            
            logger.info(f"Found {len(props)} player props from {len(game_ids)} games")
            return props
        except Exception as e:
            logger.error(f"Failed to fetch player props: {e}")
            return []

    async def ai_select_focus_players_and_queries(self, analysis_data: Dict) -> Dict:
        """Step 2: Use AI to intelligently select which players to scrape and StatMuse queries to run"""
        
        offseason_mode = analysis_data.get('offseason_mode', False)
        
        if offseason_mode:
            prompt = f"""
You are an expert sports betting analyst. Since we're in the MLB offseason with no upcoming games, select the most popular and valuable MLB star players to analyze for historical prop betting trends that will be useful when the season resumes.

RECENT PREDICTIONS: {json.dumps(analysis_data['recent_predictions'][:5], indent=2)}

Focus on:
1. TOP 10 STAR PLAYERS to scrape from Baseball Reference (focus on popular betting targets, consistent performers, major market players)
2. TOP 8 STATMUSE QUERIES for general team and league trends

Select players like:
- Top sluggers (Judge, Ohtani, Freeman, etc.)
- Popular prop betting targets
- Consistent performers across seasons
- Players from major market teams

For each selected player, provide:
- Full name (e.g., "Aaron Judge") 
- Team abbreviation (e.g., "NYY")
- Why they're valuable (star power, popular props, consistent performance)
- Key props to focus on (RBIs, Hits, Total Bases, Home Runs, etc.)

For StatMuse queries, focus on:
- Historical team performance patterns
- Player consistency trends
- League-wide statistical trends
- Seasonal performance patterns

Return JSON format:
{{
  "selected_players": [
    {{
      "name": "Aaron Judge",
      "team": "NYY", 
      "reason": "Star slugger, popular HR and RBI props, consistent performer",
      "focus_props": ["Home Runs", "RBIs", "Total Bases", "Hits"]
    }}
  ],
  "statmuse_queries": [
    {{
      "query": "Which MLB teams have the most consistent offensive production across seasons?",
      "purpose": "Historical team trend analysis"  
    }}
  ]
}}
"""
        else:
            prompt = f"""
You are an expert sports betting analyst. Based on the upcoming MLB games and available data, select the most valuable players to analyze for prop betting trends.

UPCOMING GAMES: {json.dumps(analysis_data['upcoming_games'][:10], indent=2)}
RECENT PREDICTIONS: {json.dumps(analysis_data['recent_predictions'][:10], indent=2)}  
AVAILABLE PROPS: {json.dumps(analysis_data['available_props'][:10], indent=2)}

Your task is to identify:
1. TOP 15 PLAYERS to scrape from Baseball Reference (focus on upcoming games, popular props, star players)
2. TOP 10 STATMUSE QUERIES for team trends and additional context

For each selected player, provide:
- Full name (e.g., "Freddie Freeman") 
- Team
- Why they're valuable to analyze (upcoming games, prop availability, recent performance)
- Key props to focus on (RBIs, Hits, Total Bases, etc.)

For StatMuse queries, focus on:
- Team performance trends
- Pitching matchup insights  
- Recent team form
- Weather/venue impacts

Return JSON format:
{{
  "selected_players": [
    {{
      "name": "Freddie Freeman",
      "team": "LAD", 
      "reason": "Playing tomorrow, popular RBI props, strong recent form",
      "focus_props": ["RBIs", "Hits", "Total Bases"]
    }}
  ],
  "statmuse_queries": [
    {{
      "query": "How have teams performed in their last 10 games at home vs road?",
      "purpose": "Team trend analysis"  
    }}
  ]
}}
"""

        try:
            # Add randomization to avoid deterministic behavior
            import random
            temp = random.uniform(0.4, 0.8)  # More variation in AI responses
            
            response = await self.grok_client.chat.completions.create(
                model="grok-2-1212",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=3000  # Ensure enough tokens for full response
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"AI response received ({len(content)} chars), attempting to parse...")
            
            # More robust JSON extraction
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                result = json.loads(json_content)
                
                # Validate the result has required fields
                if (result.get('selected_players') and 
                    result.get('statmuse_queries') and 
                    len(result['selected_players']) >= 5):
                    
                    logger.info(f"✅ AI successfully selected {len(result['selected_players'])} players and {len(result['statmuse_queries'])} queries")
                    return result
                else:
                    logger.warning("AI response missing required fields or insufficient players, using fallback")
                    raise ValueError("Invalid AI response structure")
            else:
                logger.warning("Could not extract JSON from AI response, using fallback")
                raise ValueError("No JSON found in response")
            
        except Exception as e:
            logger.error(f"💥 AI SELECTION COMPLETELY FAILED: {e}")
            logger.error("🚫 NO FALLBACK - SYSTEM REQUIRES AI TO WORK!")
            
            # NO FALLBACK! If AI fails, the whole system fails.
            # This forces us to fix AI issues rather than relying on hardcoded shit.
            raise Exception(f"AI player selection failed and no fallback available: {e}")

    def get_baseball_reference_id(self, player_name: str) -> str:
        """Generate Baseball Reference player ID using the correct format with known mappings"""
        try:
            # Known player ID mappings for common players
            known_ids = {
                "pete alonso": "alonspe01",
                "aaron judge": "judgeaa01", 
                "shohei ohtani": "ohtansh01",
                "mookie betts": "bettsmo01",
                "freddie freeman": "freemfr01",
                "ronald acuna jr.": "acunaro01",
                "ronald acuña jr.": "acunaro01",  # Handle accent
                "juan soto": "sotoju01",
                "jose altuve": "altuvjo01",
                "vladimir guerrero jr.": "guerrvl02",
                "trea turner": "turnetr01",
                "bryce harper": "harpebr03",
                "manny machado": "machama01",
                "francisco lindor": "lindofr01",
                "fernando tatis jr.": "tatisfe02",
                "kyle tucker": "tuckeke01",
                "yordan alvarez": "alvaryo01",
                "jose ramirez": "ramirjo01",
                "bo bichette": "bichebo01",
                "george springer": "springg01",
                "marcus semien": "semiema01",
                "corey seager": "seageco01",
                "max muncy": "muncyma01",
                "christian walker": "walkech01",
                "gleyber torres": "torregl01",
                "yordan alvarez": "alvaryo01",
                "kyle schwarber": "schwaky01",
                "nick castellanos": "casteni02",
                "j.t. realmuto": "realmj.01",
                "alec bohm": "bohmal01",
                "brandon marsh": "marshbr01",
                "trea turner": "turnetr01"
            }
            
            # Normalize the name for matching
            normalized_name = player_name.lower().strip()
            # Remove periods and handle common abbreviations
            normalized_name = normalized_name.replace(".", "")
            
            # Debug logging
            logger.info(f"Looking up Baseball Reference ID for: '{player_name}' (normalized: '{normalized_name}')")
            
            # Check for known mapping first
            if normalized_name in known_ids:
                logger.info(f"Found known ID for {player_name}: {known_ids[normalized_name]}")
                return known_ids[normalized_name]
            
            # Fallback to standard generation logic
            parts = normalized_name.split()
            if len(parts) < 2:
                logger.warning(f"Cannot generate ID for single name: {player_name}")
                return None
                
            last_name = parts[-1]
            first_name = parts[0]
            
            # Handle common name variations
            if "jr" in normalized_name:
                # Remove Jr from parts to get clean last name
                parts = [p for p in parts if p not in ["jr", "jr."]]
                if len(parts) >= 2:
                    last_name = parts[-1]
                    first_name = parts[0]
            
            # Standard format: first 5 chars of last_name + first 2 chars of first_name + 01
            last_name_part = last_name[:5] if len(last_name) > 5 else last_name
            first_name_part = first_name[:2]
            
            player_id = f"{last_name_part}{first_name_part}01"
            
            logger.info(f"Generated ID for {player_name}: {player_id}")
            return player_id
            
        except Exception as e:
            logger.error(f"Error generating Baseball Reference ID for {player_name}: {e}")
            return None

    async def scrape_baseball_reference_player(self, player_info: Dict) -> Dict:
        """Scrape Baseball Reference using Apify API with multiple scraper fallbacks"""
        player_name = player_info['name']
        player_id = self.get_baseball_reference_id(player_name)
        
        if not player_id:
            logger.warning(f"Could not generate Baseball Reference ID for {player_name}")
            return {'success': False, 'player': player_name, 'error': 'Invalid player ID'}
            
        if not self.apify_api_token:
            logger.error("No Apify API token available")
            return {'success': False, 'player': player_name, 'error': 'No Apify token'}
            
        # Try multiple URL variations for better success rate
        urls_to_try = [
            f"https://www.baseball-reference.com/players/{player_id[0]}/{player_id}.shtml",
        ]
        
        # Add alternative URL formats for common issues
        parts = player_name.lower().split()
        if len(parts) >= 2:
            # Try alternative numbering (02, 03) for common names
            base_id = player_id[:-2]  # Remove the "01"
            urls_to_try.extend([
                f"https://www.baseball-reference.com/players/{player_id[0]}/{base_id}02.shtml",
                f"https://www.baseball-reference.com/players/{player_id[0]}/{base_id}03.shtml",
            ])
        
        # Try different scrapers in order of preference
        scrapers_to_try = [
            ("cheerio", "apify~cheerio-scraper"),
            ("puppeteer", "apify~web-scraper"),
            ("content", "apify~website-content-crawler")
        ]
        
        last_error = None
        for attempt, url in enumerate(urls_to_try):
            for scraper_name, scraper_id in scrapers_to_try:
                try:
                    logger.info(f"Scraping {player_name} via {scraper_name} (attempt {attempt+1}/{len(urls_to_try)}): {url}")
                    result = await self._try_apify_scrape_with_method(url, player_name, player_info, scraper_id, scraper_name)
                    
                    if result.get('success'):
                        logger.info(f"✅ Successfully scraped {player_name} using {scraper_name}")
                        return result
                    else:
                        last_error = result.get('error', 'Unknown error')
                        logger.warning(f"❌ {scraper_name} failed for {player_name}: {last_error}")
                        
                        # If we get a 404-like error, try next URL (but continue with other scrapers first)
                        if '404' in str(last_error).lower() or 'not found' in str(last_error).lower():
                            break  # Try next URL
                            
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"❌ {scraper_name} exception for {player_name}: {e}")
                    continue
        
        # If all attempts failed
        logger.error(f"All scraping attempts failed for {player_name}. Last error: {last_error}")
        return {'success': False, 'player': player_name, 'error': f'All URLs and scrapers failed: {last_error}'}

    async def _try_apify_scrape_with_method(self, url: str, player_name: str, player_info: Dict, scraper_id: str, scraper_name: str) -> Dict:
        """Try scraping a single URL via Apify with specific scraper method"""
        try:
            apify_url = f"https://api.apify.com/v2/acts/{scraper_id}/runs"
            
            # Configure payload based on scraper type
            if scraper_name == "cheerio":
                payload = {
                    "startUrls": [{"url": url}],
                    "maxRequestRetries": 3,
                    "maxConcurrency": 1,
                    "pageFunction": """
                        async function pageFunction(context) {
                            const { request, log, $ } = context;
                            
                            // Get all text content from the page
                            const text = $('body').text();
                            
                            return {
                                url: request.url,
                                text: text,
                                html: $('body').html()
                            };
                        }
                    """,
                    "additionalMimeTypes": ["text/html"],
                    "ignoreSslErrors": True,
                    "requestTimeoutSecs": 60
                }
            elif scraper_name == "puppeteer":
                payload = {
                    "startUrls": [{"url": url}],
                    "maxRequestRetries": 3,
                    "maxConcurrency": 1,
                    "pageFunction": """
                        async function pageFunction(context) {
                            const { page, request, log } = context;
                            
                            // Wait for content to load
                            await page.waitForSelector('body', { timeout: 10000 });
                            
                            // Get all text content from the page
                            const text = await page.evaluate(() => document.body.innerText);
                            
                            return {
                                url: request.url,
                                text: text
                            };
                        }
                    """,
                    "requestTimeoutSecs": 90,
                    "maxRequestRetries": 2
                }
            else:  # content crawler
                payload = {
                    "startUrls": [{"url": url}],
                    "maxRequestsPerCrawl": 1,
                    "maxRequestRetries": 3,
                    "htmlTransformer": "readableText",
                    "readableTextCharThreshold": 100
                }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.apify_api_token}"
            }
            
            # Start the crawl
            async with httpx.AsyncClient() as client:
                response = await client.post(apify_url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                
                run_data = response.json()
                run_id = run_data["data"]["id"]
                
                logger.info(f"Started {scraper_name} crawl {run_id} for {player_name}")
                
                # Poll for completion - use the correct status URL based on scraper
                status_url = f"https://api.apify.com/v2/acts/{scraper_id}/runs/{run_id}"
                
                for attempt in range(30):  # Wait up to 5 minutes
                    await asyncio.sleep(10)
                    
                    status_response = await client.get(status_url, headers=headers)
                    status_data = status_response.json()
                    status = status_data["data"]["status"]
                    
                    if status == "SUCCEEDED":
                        break
                    elif status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                        logger.error(f"{scraper_name} crawl failed for {player_name}: {status}")
                        return {'success': False, 'player': player_name, 'error': f'{scraper_name} crawl {status}'}
                        
                    logger.info(f"Waiting for {player_name} {scraper_name} crawl to complete... ({status}) - attempt {attempt+1}/30")
                
                if status != "SUCCEEDED":
                    logger.error(f"{scraper_name} crawl timed out for {player_name}")
                    return {'success': False, 'player': player_name, 'error': f'{scraper_name} crawl timeout'}
                
                # Get the results
                dataset_id = run_data['data']['defaultDatasetId']
                results_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
                results_response = await client.get(results_url, headers=headers)
                results_data = results_response.json()
                
                if not results_data:
                    logger.warning(f"No results from {scraper_name} for {player_name}")
                    return {'success': False, 'player': player_name, 'error': f'No {scraper_name} results'}
                
                # Parse the scraped content
                scraped_text = results_data[0].get("text", "")
                
                # Log more info about what we got
                logger.info(f"Got {len(scraped_text)} characters from {scraper_name} for {player_name}")
                
                # Check for 404 or other error pages
                if ("page not found" in scraped_text.lower() or "404 error" in scraped_text.lower() or
                    "we apologize, but we could not find the page" in scraped_text.lower()):
                    logger.warning(f"404 error detected for {player_name} at {url} using {scraper_name}")
                    return {'success': False, 'player': player_name, 'error': '404 page not found'}
                
                # Check if we got the full detailed page (look for key indicators)
                has_game_logs = any(indicator in scraped_text for indicator in [
                    "Last 5 Games", "Date Tm Opp Result", "Standard Batting", "Game Logs"
                ])
                
                if not has_game_logs and any(phrase in scraped_text for phrase in ["How old is", "was born", "Stats, Height, Weight"]):
                    logger.warning(f"Got basic player info page instead of detailed stats for {player_name} using {scraper_name}")
                    return {'success': False, 'player': player_name, 'error': 'basic info page instead of game logs'}
                
                # NEW APPROACH: Skip complex parsing, let AI analyze raw scraped content directly
                logger.info(f"🤖 Sending {len(scraped_text)} characters to AI for direct analysis...")
                
                ai_analysis = await self.analyze_scraped_content_with_ai(scraped_text, player_name, player_info)
                
                if not ai_analysis:
                    logger.error(f"AI analysis failed for {player_name} using {scraper_name}")
                    return {'success': False, 'player': player_name, 'error': 'AI analysis failed'}
                
                # Extract player_id from URL for result
                player_id_from_url = url.split('/')[-1].replace('.shtml', '')
                
                result = {
                    'success': True,
                    'player': player_name,
                    'team': player_info.get('team', ''),
                    'player_id': player_id_from_url,
                    'ai_analysis': ai_analysis,
                    'scrape_timestamp': datetime.now().isoformat(),
                    'source': f'apify-{scraper_name}',
                    'url_used': url,
                    'text_length': len(scraped_text)
                }
                
                logger.info(f"✅ Successfully analyzed {player_name} via {scraper_name} using AI")
                return result
                
        except Exception as e:
            logger.error(f"Error scraping {player_name} with {scraper_name}: {e}")
            return {'success': False, 'player': player_name, 'error': str(e)}

    async def analyze_scraped_content_with_ai(self, scraped_text: str, player_name: str, player_info: Dict) -> Dict:
        """Let AI analyze raw scraped content directly to extract trends and insights"""
        try:
            # Create a focused analysis prompt for the AI
            analysis_prompt = f"""
You are analyzing scraped Baseball Reference data for {player_name}. 

SCRAPED CONTENT:
{scraped_text[:8000]}  # Truncate to avoid token limits

TASK: Analyze this player's recent performance and extract key insights for sports betting trends.

IMPORTANT: Extract data from the "Last 5 Games" table at the top of the page. Look for the table with columns like Date, Tm, Opp, Result, Pos, AB, R, H, 2B, 3B, HR, RBI, BB, SO.

Please provide a JSON response with:
1. last_5_games: Array of exactly 5 games with stats: [{{"date": "2025-08-03", "opponent": "SFG", "hits": 1, "home_runs": 0, "rbis": 0, "at_bats": 4}}]
2. performance_trends: Key performance patterns (hot/cold streaks, matchup preferences)
3. prop_insights: Insights relevant to betting props (hits, HRs, RBIs, total bases)
4. situational_analysis: How player performs in different situations
5. betting_recommendations: Specific prop bet recommendations with confidence levels

Focus on actionable betting insights. Be concise but thorough.
"""

            logger.info(f"🤖 Sending AI analysis request for {player_name}...")
            
            response = await self.grok_client.chat.completions.create(
                model="grok-2-1212",
                messages=[
                    {"role": "system", "content": "You are an expert sports analyst specializing in baseball player performance analysis for betting insights. Provide detailed, actionable analysis in JSON format."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"🤖 AI analysis completed for {player_name}")
            
            # Try to extract JSON from the response
            try:
                import json
                # Look for JSON content in the response
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = ai_response[json_start:json_end]
                    parsed_analysis = json.loads(json_content)
                    return parsed_analysis
                else:
                    # Fallback: Return the raw analysis if JSON parsing fails
                    return {
                        'raw_analysis': ai_response,
                        'player': player_name,
                        'analysis_type': 'text_analysis'
                    }
                    
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from AI response for {player_name}, using raw text")
                return {
                    'raw_analysis': ai_response,
                    'player': player_name,
                    'analysis_type': 'text_analysis'
                }
                
        except Exception as e:
            logger.error(f"Error in AI analysis for {player_name}: {e}")
            return None

    def parse_baseball_reference_text(self, scraped_text: str, player_name: str) -> List[Dict]:
        """Parse Baseball Reference scraped text to extract game data"""
        games = []
        
        try:
            # Debug: Save the scraped text to see what we're working with
            logger.info(f"Raw scraped text sample for {player_name}: {scraped_text[:500]}...")
            
            # Save full text to file for debugging
            with open(f"debug_{player_name.replace(' ', '_')}_scraped.txt", "w") as f:
                f.write(scraped_text)
            logger.info(f"Saved full scraped text to debug_{player_name.replace(' ', '_')}_scraped.txt")
            
            # Check if this looks like a player stats page (has game logs)
            if ("Last 5 Games" in scraped_text or "Date Tm Opp Result" in scraped_text or 
                "Last 5 Games Table" in scraped_text):
                
                logger.info(f"Found game log section for {player_name}")
                
                # Split into lines for easier parsing
                lines = scraped_text.split('\n')
                
                # Look for game data with more flexible date patterns: Aug 3, Jul 30, etc.
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Skip obvious header/navigation lines
                    if any(header in line for header in ["Date Tm Opp", "Last 5 Games", "POWERED BY", "Share & Export", 
                                                        "Sports Reference", "Baseball", "Players", "Teams", "Welcome"]):
                        continue
                    
                    # Look for lines that start with YYYY-MM-DD date format (Baseball Reference format)
                    date_pattern = r'^(20\d{2}-\d{2}-\d{2})'
                    date_match = re.match(date_pattern, line)
                    
                    if date_match:
                        logger.info(f"Processing game line: {line[:100]}...")
                        
                        try:
                            # Parse the specific Baseball Reference format:
                            # 2025-08-03NYMSFGL, 4-12 1B40000000100010004-0.0480.39-0.04%0.60-1.15850
                            
                            # Extract date
                            game_date = date_match.group(1)
                            
                            # Find position (1B, OF, etc.) to locate stats start
                            pos_match = re.search(r'(1B|2B|3B|SS|C|OF|DH|LF|CF|RF|P)', line)
                            if pos_match:
                                # Stats start right after the position
                                pos_end = pos_match.end()
                                stats_part = line[pos_end:]
                                
                                # Look for the concatenated stats sequence (like "40000000100010004")
                                stat_sequence_match = re.search(r'^(\d{10,})', stats_part)
                                if stat_sequence_match:
                                    stat_sequence = stat_sequence_match.group(1)
                                    
                                    # Extract individual digits as stats: AB, R, H, 2B, 3B, HR, RBI, BB, SO, etc.
                                    if len(stat_sequence) >= 9:  # Need at least 9 stats
                                        digits = [int(d) for d in stat_sequence]
                                        
                                        at_bats = digits[0]
                                        runs = digits[1] 
                                        hits = digits[2]
                                        doubles = digits[3]
                                        triples = digits[4]
                                        home_runs = digits[5]
                                        rbis = digits[6]
                                        walks = digits[7]
                                        strikeouts = digits[8]
                                        
                                        # Basic sanity checks
                                        if (1 <= at_bats <= 6 and 
                                            0 <= hits <= at_bats and 
                                            0 <= home_runs <= 4 and
                                            0 <= rbis <= 9):
                                            
                                            game_data = {
                                                'date': game_date,
                                                'at_bats': at_bats,
                                                'runs': runs,
                                                'hits': hits,
                                                'doubles': doubles,
                                                'triples': triples,
                                                'home_runs': home_runs,
                                                'rbis': rbis,
                                                'walks': walks,
                                                'strikeouts': strikeouts,
                                                'avg': round(hits / at_bats, 3) if at_bats > 0 else 0.0
                                            }
                                            
                                            games.append(game_data)
                                            logger.info(f"✅ Parsed game: {game_date} - {hits}H/{at_bats}AB, {home_runs}HR, {rbis}RBI")
                                            
                                            if len(games) >= 5:  # Limit to 5 games
                                                break
                        
                        except Exception as e:
                            logger.warning(f"Error parsing game line: {e}")
                            continue
                    
                    # Limit to 5 most recent games
                    if len(games) >= 5:
                        break
                
                logger.info(f"Successfully parsed {len(games)} games for {player_name}")
                
            else:
                # Check if this looks like a basic player info page instead of game logs
                if any(phrase in scraped_text for phrase in ["How old is", "was born", "Stats, Height, Weight"]):
                    logger.warning(f"Got player info page instead of game logs for {player_name}")
                    logger.warning("This suggests the URL might be pointing to a summary page rather than the main player page")
                else:
                    logger.warning(f"No game log section found for {player_name}")
            
            # Final validation
            if not games:
                logger.error(f"FAILED TO PARSE ANY GAMES FOR {player_name}")
                logger.error(f"Text length: {len(scraped_text)} characters")
                logger.error("Check the debug file to see what Apify actually returned")
            
            return games[:10]  # Return last 10 games
            
        except Exception as e:
            logger.error(f"Error parsing Baseball Reference text for {player_name}: {e}")
            return []

    def calculate_prop_performance(self, games: List[Dict], focus_props: List[str]) -> Dict:
        """Calculate prop betting performance from game logs"""
        if not games:
            return {}
        
        prop_thresholds = {
            'RBIs': [0.5, 1.5],
            'Hits': [0.5, 1.5, 2.5], 
            'Runs': [0.5, 1.5],
            'Total Bases': [0.5, 1.5, 2.5],
            'Home Runs': [0.5],
            'Walks': [0.5, 1.5],
            'Strikeouts': [0.5, 1.5]
        }
        
        performance = {}
        
        for prop in focus_props:
            if prop not in prop_thresholds:
                continue
                
            prop_key = prop.lower().replace(' ', '_')
            performance[prop] = {}
            
            for threshold in prop_thresholds[prop]:
                successes = 0
                for game in games:
                    if game.get(prop_key, 0) > threshold:
                        successes += 1
                
                success_rate = (successes / len(games)) * 100 if games else 0
                performance[prop][f"over_{threshold}"] = {
                    'success_count': successes,
                    'total_games': len(games),
                    'success_rate': round(success_rate, 1),
                    'trend': f"{successes}/{len(games)} games"
                }
        
        return performance

    async def run_statmuse_queries(self, queries: List[Dict]) -> List[Dict]:
        """Execute StatMuse queries for team trends and additional insights"""
        results = []
        for query_info in queries:
            try:
                response = requests.post(
                    f"{self.statmuse_api_url}/query",
                    json={'query': query_info['query']},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        results.append(result)
                    else:
                        logger.warning(f"StatMuse query failed: {query_info['query']} - {result.get('error')}")
                else:
                    logger.warning(f"StatMuse API returned {response.status_code} for query: {query_info['query']}")
            except Exception as e:
                logger.error(f"Error executing StatMuse query: {e}")
                continue
        
        return results

    async def ai_generate_final_trends(self, scraped_data: List[Dict], statmuse_data: List[Dict]) -> Dict:
        """Step 5: AI synthesizes all data to generate final 9 player prop + 6 team trends with enhanced fields"""
        
        prompt = f"""
You are an expert sports betting analyst. Based on the scraped Baseball Reference data and StatMuse insights, generate exactly 9 player prop trends and 6 team trends for MLB betting.

SCRAPED PLAYER DATA:
{json.dumps(scraped_data, indent=2, default=str)}

STATMUSE INSIGHTS:
{json.dumps(statmuse_data, indent=2, default=str)}

CRITICAL REQUIREMENTS:
1. ALL charts must be BAR CHARTS (chart_type: "bar")
2. Use exactly 5 games in chart_data.recent_games (from Last 5 Games table)
3. Generate sensible Y-axis intervals (no weird duplicates like 0,0,1,1,1)
4. Create meaningful key_stats (no nonsensical phrases like "Ba Vs Rhp - higher")
5. For team trends, provide simplified data or skip charts if data quality is poor

Generate trends that are:
1. Actionable for bettors
2. Based on strong statistical evidence  
3. Focused on upcoming games
4. Clear and specific

For each trend, provide:
- A SHORT catchy headline (5-8 words max) for the trend card display
- Chart data with EXACTLY 5 games from Baseball Reference Last 5 Games table
- Meaningful key statistics (avoid nonsensical abbreviations)
- Proper Y-axis values with clear intervals

Return JSON format:
{{
  "player_prop_trends": [
    {{
      "title": "Freddie Freeman RBI Hot Streak",
      "headline": "Freeman RBI Surge",
      "description": "Freeman has recorded RBIs in 4 of his last 5 games with strong recent production.",
      "insight": "Consider Over 0.5 RBIs props for Freeman",
      "supporting_data": "4/5 games over 0.5 RBIs in last 5 games, facing weak bullpens",
      "confidence": 85,
      "player_name": "Freddie Freeman",
      "prop_type": "RBIs",
      "trend_type": "player_prop",
      "trend_category": "streak",
      "key_stats": {{
        "Last 5 Games RBI Rate": "4 of 5 games",
        "Total RBIs Last 5": 7,
        "Average RBIs per Game": 1.4,
        "Current Streak": "3 games with RBI"
      }},
      "chart_data": {{
        "recent_games": [
          {{"date": "Aug 3", "rbis": 3, "game_number": 1}},
          {{"date": "Aug 2", "rbis": 2, "game_number": 2}},
          {{"date": "Aug 1", "rbis": 0, "game_number": 3}},
          {{"date": "Jul 30", "rbis": 1, "game_number": 4}},
          {{"date": "Jul 29", "rbis": 1, "game_number": 5}}
        ],
        "y_axis_max": 4,
        "y_axis_intervals": [0, 1, 2, 3, 4],
        "trend_direction": "up",
        "success_rate": 80
      }},
      "visual_data": {{
        "chart_type": "bar",
        "x_axis": "Last 5 Games",
        "y_axis": "RBIs",
        "trend_color": "#3b82f6",
        "bar_color": "#1e40af"
      }}
    }}
  ],
  "team_trends": [
    {{
      "title": "Dodgers Recent Offensive Surge",
      "headline": "LAD Offense Hot",
      "description": "Dodgers have scored 6+ runs in 4 of last 5 games, showing strong offensive consistency",
      "insight": "Team totals and run lines trending over",
      "supporting_data": "Averaging 7.2 runs per game over last 5, well above season average",
      "confidence": 78,
      "team": "LAD",
      "trend_type": "team",
      "trend_category": "form",
      "key_stats": {{
        "Recent Runs Average": "7.2 per game",
        "Season Runs Average": "5.1 per game",
        "Games with 6+ Runs": "4 of 5",
        "Offensive Improvement": "+41% vs season avg"
      }},
      "chart_data": null,
      "visual_data": null
    }}
  ]
}}

NOTE: For team trends, if you don't have reliable game-by-game team data, set chart_data and visual_data to null to avoid displaying poor quality charts.
"""

        try:
            response = await self.grok_client.chat.completions.create(
                model="grok-2-1212",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            # Clean the response content to extract JSON
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Clean up any extra text before/after JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                content = content[start:end]
            
            result = json.loads(content)
            
            # Validate we have the right number of trends
            player_trends = result.get('player_prop_trends', [])
            team_trends = result.get('team_trends', [])
            
            logger.info(f"AI generated {len(player_trends)} player prop trends and {len(team_trends)} team trends")
            return result
            
        except Exception as e:
            logger.error(f"Error generating final trends: {e}")
            return {'player_prop_trends': [], 'team_trends': []}

    async def store_trends_in_database(self, trends_data: Dict, scraped_data: List[Dict]) -> bool:
        """Store generated trends in ai_trends table with enhanced metadata"""
        try:
            # Clear existing global trends
            self.supabase.table('ai_trends').delete().eq('is_global', True).execute()
            logger.info("Cleared existing global trends")
            
            trends_to_store = []
            
            # Store player prop trends
            for trend in trends_data.get('player_prop_trends', []):
                # Find matching scraped data for this player
                player_data = None
                for data in scraped_data:
                    if data.get('success') and data.get('player', '').lower() == trend.get('player_name', '').lower():
                        player_data = data
                        break
                
                # Try to resolve player from database
                player_id = None
                full_name = trend.get('player_name', '')
                if full_name.lower() in self.player_mappings:
                    player_info = self.player_mappings[full_name.lower()]
                    player_id = player_info.get('id')
                    full_name = player_info.get('name', full_name)
                
                trend_entry = {
                    'user_id': "00000000-0000-0000-0000-000000000000",  # System user for global trends
                    'trend_type': 'player_prop',
                    'title': trend.get('title', ''),
                    'description': trend.get('description', ''),
                    'insight': trend.get('insight', ''),
                    'supporting_data': trend.get('supporting_data', ''),
                    'confidence_score': trend.get('confidence', 50),
                    'trend_text': trend.get('description', ''),  # Use description as trend_text
                    'headline': trend.get('headline', ''),  # New: Short catchy headline
                    'chart_data': trend.get('chart_data', {}),  # New: Chart visualization data
                    'trend_category': trend.get('trend_category', 'general'),  # New: Trend category
                    'key_stats': trend.get('key_stats', {}),  # New: Key statistics
                    'visual_data': trend.get('visual_data', {}),  # New: Visual elements for display
                    'sport': 'MLB',
                    'is_global': True,
                    'player_id': player_id,
                    'full_player_name': full_name,
                    'scraped_prop_data': player_data.get('ai_analysis', {}) if player_data else {},
                    'prop_performance_stats': player_data.get('ai_analysis', {}).get('last_5_games', []) if player_data and player_data.get('ai_analysis') else [],
                    'data_sources': ['baseball_reference', 'ai_analysis'],
                    'metadata': {
                        'prop_type': trend.get('prop_type', ''),
                        'games_analyzed': 5,  # Always 5 games from Last 5 Games table
                        'scrape_timestamp': player_data.get('scrape_timestamp') if player_data else None,
                        'chart_type': 'bar'  # Always bar charts
                    }
                }
                trends_to_store.append(trend_entry)
            
            # Store team trends
            for trend in trends_data.get('team_trends', []):
                trend_entry = {
                    'user_id': "00000000-0000-0000-0000-000000000000",  # System user for global trends
                    'trend_type': 'team',
                    'title': trend.get('title', ''),
                    'description': trend.get('description', ''),
                    'insight': trend.get('insight', ''),
                    'supporting_data': trend.get('supporting_data', ''),
                    'confidence_score': trend.get('confidence', 50),
                    'trend_text': trend.get('description', ''),  # Use description as trend_text
                    'headline': trend.get('headline', ''),  # New: Short catchy headline
                    'chart_data': trend.get('chart_data'),  # Allow null for teams with poor data
                    'trend_category': trend.get('trend_category', 'general'),  # New: Trend category
                    'key_stats': trend.get('key_stats', {}),  # New: Key statistics
                    'visual_data': trend.get('visual_data'),  # Allow null for teams with poor data
                    'sport': 'MLB',
                    'is_global': True,
                    'data_sources': ['statmuse', 'ai_analysis'],
                    'metadata': {
                        'team': trend.get('team', ''),
                        'analysis_type': 'team_performance',
                        'has_chart': trend.get('chart_data') is not None
                    }
                }
                trends_to_store.append(trend_entry)
            
            # Batch insert trends
            if trends_to_store:
                for trend in trends_to_store:
                    self.supabase.table('ai_trends').insert(trend).execute()
                    await asyncio.sleep(0.1)  # Small delay between inserts
                
                logger.info(f"Successfully stored {len(trends_to_store)} trends in database")
                return True
            else:
                logger.warning("No trends to store")
                return False
                
        except Exception as e:
            logger.error(f"Error storing trends in database: {e}")
            return False

    async def run_intelligent_analysis(self):
        """Main workflow: Intelligent analysis and trend generation"""
        try:
            logger.info("Starting intelligent trends analysis...")
            
            # Step 1: Analyze upcoming games and props
            logger.info("Step 1: Analyzing upcoming games and available props...")
            analysis_data = await self.analyze_upcoming_games_and_props()
            
            # Step 2: AI selects focus players and queries
            logger.info("Step 2: AI selecting focus players and StatMuse queries...")
            selection_data = await self.ai_select_focus_players_and_queries(analysis_data)
            
            # Step 3: Scrape Baseball Reference using Apify
            logger.info("Step 3: Scraping Baseball Reference for selected players using Apify...")
            successful_scrapes = []
            
            if selection_data.get('selected_players'):
                for player_info in selection_data['selected_players'][:12]:  # Analyze 12 players for comprehensive trends
                    scrape_result = await self.scrape_baseball_reference_player(player_info)
                    if scrape_result.get('success'):
                        successful_scrapes.append(scrape_result)
                        logger.info(f"✅ Successfully analyzed {scrape_result['player']} with AI")
                    else:
                        logger.warning(f"❌ Failed to analyze {scrape_result['player']}: {scrape_result.get('error')}")
            
            logger.info(f"Successfully scraped {len(successful_scrapes)} of {len(selection_data.get('selected_players', []))} players via Apify")
            
            # Step 4: Run StatMuse queries
            logger.info("Step 4: Running StatMuse queries...")
            statmuse_data = await self.run_statmuse_queries(selection_data.get('statmuse_queries', []))
            
            # Step 5: AI generates final trends
            logger.info("Step 5: AI generating final trends...")
            final_trends = await self.ai_generate_final_trends(successful_scrapes, statmuse_data)
            
            # Step 6: Store trends in database
            logger.info("Step 6: Storing trends in database...")
            success = await self.store_trends_in_database(final_trends, successful_scrapes)
            
            if success:
                logger.info("✅ Intelligent trends analysis completed successfully!")
                return {
                    'success': True,
                    'players_analyzed': len(successful_scrapes),
                    'statmuse_queries': len(statmuse_data),
                    'player_prop_trends': len(final_trends.get('player_prop_trends', [])),
                    'team_trends': len(final_trends.get('team_trends', []))
                }
            else:
                logger.error("❌ Failed to store trends in database")
                return {'success': False, 'error': 'Database storage failed'}
                
        except Exception as e:
            logger.error(f"Error in intelligent analysis: {e}")
            return {'success': False, 'error': str(e)}


async def main():
    """Main entry point"""
    generator = IntelligentTrendsGenerator()
    result = await generator.run_intelligent_analysis()
    
    if result.get('success'):
        print(f"\n✅ Analysis Complete!")
        print(f"Players Analyzed: {result.get('players_analyzed', 0)}")
        print(f"StatMuse Queries: {result.get('statmuse_queries', 0)}")
        print(f"Player Prop Trends: {result.get('player_prop_trends', 0)}")
        print(f"Team Trends: {result.get('team_trends', 0)}")
    else:
        print(f"\n❌ Analysis Failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())