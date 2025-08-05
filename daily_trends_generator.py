#!/usr/bin/env python3
"""
Daily Trends Generator
Fetches upcoming games & odds for next 48h, builds StatMuse query list,
calls StatMuse API with caching, parses answers to structured trend objects,
and stores 15 trends into ai_trends table (is_global=true).

Pattern after daily_insights_generator.py
"""

import os
import sys
import requests
import json
import argparse
import random
import asyncio
from datetime import datetime, timedelta, date
from supabase import create_client, Client
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyTrendsGenerator:
    def __init__(self, sport_filter=None, dry_run=False):
        self.sport_filter = sport_filter
        self.dry_run = dry_run
        # Initialize Supabase client with service key for admin operations
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_ANON_KEY')

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")

        logger.info(f"Connecting to Supabase at: {self.supabase_url[:50]}...")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # StatMuse API server URL
        self.statmuse_api_url = os.getenv('STATMUSE_API_URL', 'http://localhost:5001')
        
        # Initialize Grok client properly like your other scripts
        self.grok_client = AsyncOpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        
        if not os.getenv("XAI_API_KEY"):
            raise ValueError("Please set XAI_API_KEY environment variable for Grok-3")

    def fetch_upcoming_games_and_odds(self):
        """Fetch upcoming games and odds for the next 48 hours"""
        try:
            now = datetime.now()
            two_days_later = now + timedelta(hours=48)
            
            # Fetch games
            games_query = self.supabase.table('sports_events').select('*')
            if self.sport_filter:
                games_query = games_query.eq('sport', self.sport_filter)
            games_query = games_query.gte('start_time', now.isoformat()).lte('start_time', two_days_later.isoformat())
            games = games_query.execute().data

            if not games:
                logger.warning("No games found for the next 48 hours")
                return [], []

            # Fetch player props with actual player names
            player_props_query = self.supabase.table('player_props_odds').select(
                '*', 
                'players(name, team)'
            ).gte('created_at', (now - timedelta(days=1)).isoformat())
            player_props = player_props_query.execute().data

            logger.info(f"Fetched {len(games)} games and {len(player_props)} player props")
            return games, player_props

        except Exception as e:
            logger.error(f"Error fetching games and odds: {e}")
            return [], []

    async def grok_select_best_queries(self, games, player_props):
        """Use Grok-3 to intelligently select the most betting-relevant StatMuse queries"""
        try:
            # Prepare context for Grok
            games_context = []
            for game in games[:20]:
                games_context.append({
                    'home_team': game.get('home_team', ''),
                    'away_team': game.get('away_team', ''),
                    'start_time': game.get('start_time', ''),
                    'sport': game.get('sport', 'MLB')
                })
            
            props_context = []
            for prop in player_props[:30]:
                if prop.get('players'):
                    props_context.append({
                        'player': prop['players'].get('name', ''),
                        'team': prop['players'].get('team', ''),
                        'prop_type': prop.get('prop_type', ''),
                        'line': prop.get('line', '')
                    })
            
            grok_prompt = f"""You are Grok, an expert sports betting analyst. Generate 50-60 StatMuse queries that will produce betting insights. You MUST generate BOTH individual player queries AND team queries.

UPCOMING GAMES (next 48 hours):
{json.dumps(games_context, indent=2)}

PLAYER PROPS WITH LINES:
{json.dumps(props_context, indent=2)}

CRITICAL: Generate EXACTLY these types of queries:

**30-35 INDIVIDUAL PLAYER QUERIES (for player props):**
- "[Player Name] batting average in last 10 games"
- "[Player Name] hits in last 15 games" 
- "[Player Name] RBIs in last 7 games"
- "[Player Name] home runs in last 20 games"
- "[Player Name] strikeouts in last 10 games"
- "[Player Name] on-base percentage in last 15 games"

**20-25 TEAM QUERIES (for team trends):**
- "[Team Name] record in last 10 games"
- "[Team Name] runs scored in last 7 games"
- "[Team Name] batting average in last 15 games at home"
- "[Team Name] bullpen ERA in last 10 games"

Use the ACTUAL player names from the props data above. Use the ACTUAL team names from the games data above.

Return ONLY a JSON array of queries:
["query1", "query2", ...]

MUST include individual player names for player prop betting insights."""

            response = await self.grok_client.chat.completions.create(
                model="grok-2-1212",
                messages=[{"role": "user", "content": grok_prompt}],
                temperature=0.7
            )
            
            grok_response = response.choices[0].message.content.strip()
            
            # Parse JSON response from Grok
            if grok_response.startswith('['):
                queries = json.loads(grok_response)
                logger.info(f"Grok selected {len(queries)} betting-focused queries")
                return queries
            else:
                logger.warning("Grok didn't return proper JSON array")
                return self.fallback_queries(games, player_props)
                    
        except Exception as e:
            logger.error(f"Error in Grok query selection: {e}")
            return self.fallback_queries(games, player_props)
    
    def fallback_queries(self, games, player_props):
        """Balanced player and team betting queries"""
        queries = []
        
        # HIGH-VALUE INDIVIDUAL PLAYER queries (prioritize these)
        for prop in player_props[:20]:
            if prop.get('players') and prop.get('players', {}).get('name'):
                player = prop['players']['name']
                queries.extend([
                    f"{player} batting average in last 10 games",
                    f"{player} hits in last 15 games",
                    f"{player} RBIs in last 10 games",
                    f"{player} home runs in last 20 games",
                    f"{player} strikeouts in last 10 games",
                    f"{player} on-base percentage in last 15 games"
                ])
        
        # Team betting queries (fewer than before)
        for game in games[:6]:
            home = game.get('home_team', '')
            away = game.get('away_team', '')
            if home and away:
                queries.extend([
                    f"{home} record in last 10 games",
                    f"{away} road record in last 10 games", 
                    f"{home} runs scored per game last 10 games",
                    f"{away} bullpen ERA last 15 games"
                ])
        
        # Remove duplicates and return
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen and len(query.strip()) > 10:
                seen.add(query)
                unique_queries.append(query)
        
        logger.info(f"Generated {len(unique_queries)} balanced betting queries (prioritizing player props)")
        return unique_queries[:50]

    def call_statmuse_api(self, queries):
        """Call the StatMuse API server with caching"""
        try:
            results = []
            for query in queries:
                try:
                    response = requests.post(
                        f"{self.statmuse_api_url}/query",
                        json={'query': query},
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            results.append(result)
                        else:
                            logger.warning(f"StatMuse query failed: {query} - {result.get('error')}")
                    else:
                        logger.warning(f"StatMuse API returned {response.status_code} for query: {query}")
                        
                except Exception as query_error:
                    logger.warning(f"Error with individual query '{query}': {query_error}")
                    continue
                    
            logger.info(f"Received {len(results)} successful results from StatMuse API")
            return results
            
        except Exception as e:
            logger.error(f"Error calling StatMuse API: {e}")
            return []

    async def grok_select_best_trends(self, responses):
        """Use Grok-3 to intelligently select the 15+ BEST betting trends and fix grammar"""
        try:
            # Prepare all StatMuse results for Grok analysis
            statmuse_results = []
            for response in responses:
                if response.get('success') and response.get('answer'):
                    statmuse_results.append({
                        'query': response.get('query', ''),
                        'answer': response.get('answer', ''),
                        'source': 'StatMuse'
                    })
            
            if len(statmuse_results) < 5:
                logger.warning("Not enough StatMuse results to analyze")
                return []
            
            grok_prompt = f"""You are Grok, a sharp sports betting analyst. Analyze all these StatMuse results and select the 15 MOST VALUABLE betting insights for sharp bettors.

STATMUSE RESULTS TO ANALYZE:
{json.dumps(statmuse_results, indent=2)}

CRITICAL REQUIREMENT: You MUST generate EXACTLY this mix:
- EXACTLY 8 trends about INDIVIDUAL PLAYERS (classify as "player_prop") 
- EXACTLY 7 trends about TEAMS (classify as "team")

CLASSIFICATION RULES - FOLLOW THESE EXACTLY:
- "player_prop": When trend is about a SPECIFIC PLAYER's individual performance
  Examples: "Aaron Judge has 15 hits in his last 10 games", "Mookie Betts is batting .350 in his last 15 games"
- "team": When trend is about a TEAM's collective performance  
  Examples: "The Yankees have won 8 of their last 10 games", "The Dodgers have scored 5+ runs in 7 straight games"

PRIORITIZE HIGH-VALUE BETTING INSIGHTS:
- Recent hot/cold streaks (last 7-15 games)
- Home/away performance splits
- Head-to-head matchup advantages
- Situational stats that impact prop lines
- Form changes that create betting opportunities

AVOID LOW-VALUE TRENDS:
- Zero stats ("Player has 0 steals")
- Mediocre records (".500 teams")
- Season averages without context

Return EXACTLY this JSON format with 15 trends (8 player_prop + 7 team):
{{
  "trends": [
    {{
      "trend_text": "Clean, grammatically perfect insight here",
      "trend_type": "player_prop",
      "confidence_score": 0.8,
      "betting_value": "High"
    }},
    {{
      "trend_text": "Another perfectly clean team insight", 
      "trend_type": "team",
      "confidence_score": 0.7,
      "betting_value": "Medium"
    }}
  ]
}}

MANDATORY: Must return EXACTLY 8 player_prop trends and EXACTLY 7 team trends for a total of 15."""

            response = await self.grok_client.chat.completions.create(
                model="grok-2-1212",
                messages=[{"role": "user", "content": grok_prompt}],
                temperature=0.3
            )
            
            grok_response = response.choices[0].message.content.strip()
            
            # Parse JSON response from Grok
            if '{' in grok_response and 'trends' in grok_response:
                result = json.loads(grok_response)
                trends = result.get('trends', [])
                logger.info(f"Grok selected {len(trends)} high-value betting trends")
                
                # Ensure we have at least 15 trends for Elite users
                if len(trends) < 15:
                    logger.warning(f"Grok only returned {len(trends)} trends, padding with fallback")
                    fallback_trends = self.fallback_trend_parsing(responses)
                    trends.extend(fallback_trends[:15-len(trends)])
                
                return trends[:15]  # Return exactly 15
            else:
                logger.warning("Grok didn't return proper JSON format")
                return self.fallback_trend_parsing(responses)
                    
        except Exception as e:
            logger.error(f"Error in Grok trend analysis: {e}")
            return self.fallback_trend_parsing(responses)
    
    def fallback_trend_parsing(self, responses):
        """Simplified fallback that trusts data more and filters less"""
        trends = []
        
        # Collect all valid responses with minimal filtering
        all_responses = []
        for response in responses:
            if response.get('success') and response.get('answer'):
                trend_text = response['answer'].strip()
                
                # Only skip truly empty or error responses
                if (len(trend_text) < 10 or 
                    'no data' in trend_text.lower() or 
                    'not available' in trend_text.lower()):
                    continue
                
                all_responses.append({
                    'trend_text': trend_text,
                    'response': response
                })
        
        # Simple classification based on content - less mechanical
        player_trends = []
        team_trends = []
        
        for item in all_responses:
            trend_text = item['trend_text']
            
            # Simple classification: if it mentions a person name (no "The" prefix), it's likely a player
            # If it starts with "The" or mentions team words, it's likely a team
            is_player_trend = (
                not trend_text.lower().startswith('the ') and
                any(indicator in trend_text.lower() for indicator in ['has ', 'is ', 'batting', 'hitting'])
            )
            
            is_team_trend = (
                trend_text.lower().startswith('the ') or
                any(team_word in trend_text.lower() for team_word in ['team', 'record', 'have a', 'have won'])
            )
            
            # Prefer player trends to balance the mix
            if is_player_trend and len(player_trends) < 8:
                player_trends.append({
                    'trend_text': trend_text,
                    'trend_type': 'player_prop',
                    'confidence_score': 0.7,
                    'betting_value': 'Medium'
                })
            elif len(team_trends) < 7:
                team_trends.append({
                    'trend_text': trend_text,
                    'trend_type': 'team',
                    'confidence_score': 0.7,
                    'betting_value': 'Medium'
                })
            
            if len(player_trends) >= 8 and len(team_trends) >= 7:
                break
        
        # Fill remaining slots with whatever we have
        final_trends = player_trends + team_trends
        remaining_needed = 15 - len(final_trends)
        
        if remaining_needed > 0:
            for item in all_responses[len(final_trends):]:
                if remaining_needed <= 0:
                    break
                final_trends.append({
                    'trend_text': item['trend_text'],
                    'trend_type': 'team',  # Default to team if unsure
                    'confidence_score': 0.6,
                    'betting_value': 'Medium'
                })
                remaining_needed -= 1
        
        logger.info(f"Fallback generated {len(final_trends)} trends ({len(player_trends)} player_prop, {len(team_trends)} team)")
        return final_trends[:15]


    
    def fix_grammar_and_spelling(self, text):
        """Fix common grammar and spelling issues in trend text"""
        import re
        
        fixed_text = text
        
        # VERY SPECIFIC fixes for known bad patterns only
        specific_fixes = {
            'David Fryhas': 'David Fry has',
            'José Ramírezhas': 'José Ramírez has', 
            'Angelsput': 'Angels put',
            'Soxyesterday': 'Sox yesterday',
            'White Soxyesterday': 'White Sox yesterday',
            'Dodgershave': 'Dodgers have',
            'Giantshit': 'Giants hit',
            'Metsare': 'Mets are',
            'Yankeesscored': 'Yankees scored'
        }
        
        # Apply specific fixes
        for wrong, correct in specific_fixes.items():
            fixed_text = fixed_text.replace(wrong, correct)
        
        # Only fix obvious concatenation issues (name + verb without space)
        # Be very conservative - only fix when we're 100% sure
        common_name_endings = ['ez', 'son', 'ez', 'er', 'man', 'ski', 'ton', 'ham']
        verbs = ['has', 'have', 'put', 'hit', 'scored']
        
        for verb in verbs:
            # Look for patterns where a likely name (ending with common suffixes) is directly followed by a verb
            pattern = r'\b([A-Z][a-z]+(?:' + '|'.join(common_name_endings) + r'))(' + verb + r')\b'
            fixed_text = re.sub(pattern, r'\1 \2', fixed_text)
        
        # Fix team names + yesterday/today only
        team_words = ['Sox', 'Angels', 'Dodgers', 'Giants', 'Mets', 'Yankees', 'Cubs', 'Reds']
        time_words = ['yesterday', 'today', 'tomorrow']
        
        for team in team_words:
            for time_word in time_words:
                pattern = team + time_word
                replacement = team + ' ' + time_word
                fixed_text = fixed_text.replace(pattern, replacement)
        
        # Clean up multiple spaces
        fixed_text = re.sub(r'\s+', ' ', fixed_text)
        
        return fixed_text.strip()
    

    
    def store_trends(self, trends):
        """Store the trends into the ai_trends table with global scope"""
        try:
            # Clear existing global trends for today
            today = date.today().isoformat()
            
            # Delete existing global trends (is_global=true)
            try:
                self.supabase.table('ai_trends').delete().eq('is_global', True).execute()
                logger.info("Cleared existing global trends")
            except Exception as delete_error:
                logger.warning(f"Could not clear existing trends: {delete_error}")
            
            # Use a global admin user ID or make user_id nullable
            global_user_id = '00000000-0000-0000-0000-000000000000'  # Global trends user
            
            stored_count = 0
            for i, trend in enumerate(trends):
                try:
                    record = {
                        'user_id': global_user_id,
                        'trend_text': trend['trend_text'][:500],  # Truncate if too long
                        'trend_type': trend.get('trend_type', 'general'),
                        'sport': self.sport_filter or 'MLB',
                        'confidence_score': float(trend['confidence_score']),
                        'data': {
                            'query': trend.get('query', ''),
                            'generated_at': datetime.now().isoformat(),
                            'order': i + 1
                        },
                        'is_global': True,
                        'expires_at': (datetime.now() + timedelta(days=2)).isoformat()
                    }
                    
                    result = self.supabase.table('ai_trends').insert(record).execute()
                    stored_count += 1
                    
                except Exception as insert_error:
                    logger.warning(f"Failed to store trend {i+1}: {insert_error}")
                    continue

            logger.info(f"Successfully stored {stored_count} out of {len(trends)} trends")
    
        except Exception as e:
            logger.error(f"Error storing trends: {e}")

async def main():
    parser = argparse.ArgumentParser(description='Generate daily sports trends with Grok-3')
    parser.add_argument('--sport', type=str, help='Filter by sport')
    parser.add_argument('--dry-run', action='store_true', help='Execute script without storing results')
    args = parser.parse_args()

    generator = DailyTrendsGenerator(sport_filter=args.sport, dry_run=args.dry_run)
    games, player_props = generator.fetch_upcoming_games_and_odds()
    queries = await generator.grok_select_best_queries(games, player_props)
    responses = generator.call_statmuse_api(queries)
    trends = await generator.grok_select_best_trends(responses)
    if not generator.dry_run:
        generator.store_trends(trends)

if __name__ == "__main__":
    asyncio.run(main())

