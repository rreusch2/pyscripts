import os
import json
import logging
import asyncio
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import httpx
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv("backend/.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class ResearchInsight:
    source: str
    query: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime

class StatMuseClient:
    def __init__(self, base_url: str = None):
        # Get the StatMuse API URL from environment variable, with fallback
        self.base_url = base_url or os.getenv("STATMUSE_API_URL", "http://127.0.0.1:5001")
        self.session = requests.Session()
        logger.info(f"Using StatMuse API URL: {self.base_url}")
        
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
            return {"error": str(e), "data": None}

class WebSearchClient:
    def __init__(self):
        # Use backend AI API for web search
        self.backend_url = os.getenv("BACKEND_URL", "https://zooming-rebirth-production-a305.up.railway.app")
        
    def search(self, query: str) -> Dict[str, Any]:
        """Perform web search using backend AI API"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/ai/chat",
                json={
                    "message": f"Search the web for: {query}",
                    "userId": "insights-agent",
                    "useWebSearch": True
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "query": query,
                "results": result.get("response", "No results found"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return {
                "query": query,
                "results": f"Search failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

class DatabaseClient:
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required")
            
        self.supabase: Client = create_client(supabase_url, supabase_key)

    def clear_daily_insights(self) -> bool:
        """Clear the daily_professor_insights_dev table before populating new insights"""
        try:
            # Delete all existing insights
            response = self.supabase.table("daily_professor_insights_dev").delete().neq("id", "").execute()
            logger.info("✅ Cleared daily_professor_insights_dev table")
            return True
        except Exception as e:
            logger.error(f"Error clearing daily insights: {e}")
            return False

    def get_upcoming_games(self, hours_ahead: int = 48) -> List[Dict[str, Any]]:
        """Fetch upcoming games from ALL sports"""
        try:
            now = datetime.now()
            end_time = now + timedelta(hours=hours_ahead)
            
            response = self.supabase.table("sports_events").select(
                "id, sport, home_team, away_team, start_time, metadata"
            ).gte("start_time", now.isoformat()).lte("start_time", end_time.isoformat()).execute()
            
            games = response.data
            logger.info(f"Found {len(games)} upcoming games across all sports")
            
            # Group by sport for logging
            sport_counts = {}
            for game in games:
                sport = game.get('sport', 'Unknown')
                sport_counts[sport] = sport_counts.get(sport, 0) + 1
            
            for sport, count in sport_counts.items():
                logger.info(f"  {sport}: {count} games")
            
            return games
        except Exception as e:
            logger.error(f"Error fetching upcoming games: {e}")
            return []

    def store_daily_insights(self, insights: List[Dict[str, Any]]) -> bool:
        """Store daily insights with sport metadata for filtering in DEV table"""
        try:
            # Add created_at timestamp and ensure sport field exists
            for insight in insights:
                insight['created_at'] = datetime.now().isoformat()
                if 'sport' not in insight:
                    insight['sport'] = 'Multi-Sport'  # Default for cross-sport insights
                    
            response = self.supabase.table("daily_professor_insights_dev").insert(insights).execute()
            logger.info(f"Successfully stored {len(insights)} daily insights in DEV table")
            return True
        except Exception as e:
            logger.error(f"Error storing daily insights: {e}")
            return False

class PersonalizedInsightsAgent:
    def __init__(self):
        self.db = DatabaseClient()
        self.statmuse = StatMuseClient()
        self.web_search = WebSearchClient()
        self.grok_client = AsyncOpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )

    async def generate_personalized_insights_by_sport(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate 7-8 insights per sport that has active games"""
        logger.info("🧠 Starting personalized multi-sport insights generation...")
        
        # Clear existing insights first
        self.db.clear_daily_insights()
        
        # Fetch games
        games = self.db.get_upcoming_games()
        if not games:
            logger.warning("No upcoming games found")
            return {}
        
        # Group games by sport
        games_by_sport = {}
        for game in games:
            sport = game['sport']
            if sport not in games_by_sport:
                games_by_sport[sport] = []
            games_by_sport[sport].append(game)
        
        # Generate insights for each sport
        all_insights_by_sport = {}
        
        for sport, sport_games in games_by_sport.items():
            if len(sport_games) < 2:  # Need minimum games for meaningful insights
                logger.info(f"Skipping {sport} - insufficient games ({len(sport_games)})")
                continue
                
            logger.info(f"📊 Generating 12 insights for {sport} ({len(sport_games)} games available)")
            
            insights = await self._generate_insights_for_sport(sport, sport_games, target_insights=12)
            
            if insights:
                all_insights_by_sport[sport] = insights
                logger.info(f"✅ Generated {len(insights)} insights for {sport}")
        
        return all_insights_by_sport

    async def _generate_insights_for_sport(self, sport: str, games: List[Dict], target_insights: int = 8) -> List[Dict[str, Any]]:
        """Generate insights for a specific sport"""
        try:
            # Create research plan
            research_plan = await self.create_research_plan(sport, games)
            
            # Execute research
            research_insights = await self.execute_research_plan(research_plan, games)
            
            # Generate final insights with reasoning
            insights = await self.generate_insights_with_reasoning(sport, research_insights, games, target_insights)
            
            return insights
        except Exception as e:
            logger.error(f"Error generating insights for {sport}: {e}")
            return []

    async def create_research_plan(self, sport: str, games: List[Dict]) -> Dict[str, Any]:
        """Create intelligent research plan for specific sport insights"""
        try:
            # Analyze available games
            game_analysis = self._analyze_available_games(games)
            
            # Create sport-specific research prompt
            prompt = f"""You are an expert {sport} analyst creating a research strategy for generating daily betting insights.

AVAILABLE DATA:
- {len(games)} {sport} games today
- Key matchups: {', '.join([f"{g.get('away_team', 'TBD')} @ {g.get('home_team', 'TBD')}" for g in games[:8]])}

RESEARCH OBJECTIVES:
1. Identify key trends and patterns across {sport} slate
2. Find situational advantages and betting angles
3. Analyze weather, injuries, and lineup impacts (if applicable)
4. Discover cross-game correlations and betting strategies
5. Uncover market inefficiencies and value opportunities

Create a strategic research plan with:
- 8-10 StatMuse queries focused on {sport} trends, matchups, and situational factors
- 6-8 web search queries for {sport} breaking news, weather, injuries, lineup changes
- Priority order for research execution
- Expected insights from each query

Focus on generating insights that will help bettors across the ENTIRE {sport} slate, not just individual games.

Return a JSON research plan with 'statmuse_queries' and 'web_queries' arrays."""

            response = await self.grok_client.chat.completions.create(
                model="grok-3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            
            plan_text = response.choices[0].message.content
            
            # Try to parse JSON from response
            try:
                if '```json' in plan_text:
                    json_start = plan_text.find('```json') + 7
                    json_end = plan_text.find('```', json_start)
                    plan_text = plan_text[json_start:json_end]
                
                plan = json.loads(plan_text)
                logger.info(f"Created research plan for {sport} with {len(plan.get('statmuse_queries', []))} StatMuse queries and {len(plan.get('web_queries', []))} web queries")
                return plan
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse research plan JSON for {sport}, using fallback")
                return self._create_fallback_research_plan(sport, games)
                
        except Exception as e:
            logger.error(f"Error creating research plan for {sport}: {e}")
            return self._create_fallback_research_plan(sport, games)

    def _create_fallback_research_plan(self, sport: str, games: List[Dict]) -> Dict[str, Any]:
        """Create a basic research plan if AI planning fails"""
        top_teams = []
        for game in games[:5]:
            top_teams.extend([game.get('home_team', ''), game.get('away_team', '')])
        top_teams = [t for t in top_teams if t][:6]
        
        return {
            "statmuse_queries": [
                f"{sport} trends this season",
                f"Best {sport} betting trends today",
                f"{sport} home field advantage statistics",
                f"{sport} weather impact on games",
                f"{sport} injury report impact on betting",
                f"Most profitable {sport} betting strategies",
                f"{sport} line movement patterns",
                f"{sport} public betting vs sharp money"
            ],
            "web_queries": [
                f"{sport} injury report today",
                f"{sport} weather conditions today",
                f"{sport} lineup changes today",
                f"Best {sport} betting picks today",
                f"{sport} breaking news today",
                f"{sport} line movements today"
            ]
        }

    def _analyze_available_games(self, games: List[Dict]) -> Dict[str, Any]:
        """Analyze the available games"""
        teams = set()
        for game in games:
            teams.add(game.get('home_team', ''))
            teams.add(game.get('away_team', ''))
        
        return {
            "total_games": len(games),
            "teams": list(teams),
            "matchups": [f"{g.get('away_team', 'TBD')} @ {g.get('home_team', 'TBD')}" for g in games]
        }

    async def execute_research_plan(self, plan: Dict[str, Any], games: List[Dict]) -> List[ResearchInsight]:
        """Execute the research plan and gather insights"""
        insights = []
        
        # Execute StatMuse queries
        statmuse_queries = plan.get('statmuse_queries', [])[:8]  # Limit to 8 queries
        for query in statmuse_queries:
            try:
                result = self.statmuse.query(query)
                if result and not result.get('error'):
                    insight = ResearchInsight(
                        source="StatMuse",
                        query=query,
                        data=result,
                        confidence=0.8,
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
                    logger.info(f"✅ StatMuse query completed: {query[:50]}...")
                else:
                    logger.warning(f"❌ StatMuse query failed: {query}")
            except Exception as e:
                logger.error(f"StatMuse query error: {e}")
        
        # Execute web search queries
        web_queries = plan.get('web_queries', [])[:6]  # Limit to 6 queries
        for query in web_queries:
            try:
                result = self.web_search.search(query)
                insight = ResearchInsight(
                    source="WebSearch",
                    query=query,
                    data=result,
                    confidence=0.6,
                    timestamp=datetime.now()
                )
                insights.append(insight)
                logger.info(f"✅ Web search completed: {query[:50]}...")
            except Exception as e:
                logger.error(f"Web search error: {e}")
        
        logger.info(f"Research complete: {len(insights)} insights gathered")
        return insights

    async def generate_insights_with_reasoning(
        self, 
        sport: str,
        research_insights: List[ResearchInsight], 
        games: List[Dict],
        target_insights: int
    ) -> List[Dict[str, Any]]:
        """Generate final insights with detailed reasoning for specific sport"""
        try:
            # Format research insights for AI
            statmuse_insights = [i for i in research_insights if i.source == "StatMuse"]
            web_insights = [i for i in research_insights if i.source == "WebSearch"]
            
            insights_summary = []
            
            # Format StatMuse insights
            for insight in statmuse_insights:
                insights_summary.append({
                    "source": "StatMuse",
                    "query": insight.query,
                    "data": insight.data.get('data', 'No data available'),
                    "confidence": insight.confidence
                })
            
            # Format web insights
            for insight in web_insights:
                insights_summary.append({
                    "source": "Web Search",
                    "query": insight.query,
                    "results": insight.data.get('results', 'No results available'),
                    "confidence": insight.confidence
                })
            
            # Create comprehensive prompt
            prompt = f"""You are Professor Lock, a sharp {sport} betting analyst generating {target_insights} intelligent daily insights for serious bettors.

RESEARCH DATA:
{json.dumps(insights_summary, indent=2)}

TODAY'S {sport.upper()} SLATE:
{self._format_games_for_ai(games)}

REQUIREMENTS:
1. Generate EXACTLY {target_insights} insights
2. Focus on {sport} betting opportunities and angles
3. Each insight must include:
   - title (catchy, specific to the insight)
   - content (2-3 sentences with actionable intelligence)
   - category (trend/injury/weather/matchup/research/value/situational)
   - confidence (70-95 range)
   - sport (set to "{sport}")

4. Cover different aspects of {sport} betting:
   - Cross-game trends and patterns
   - Situational advantages
   - Weather/injury impacts (if applicable)
   - Market inefficiencies
   - Sharp vs public betting angles
   - Historical matchup data

5. Make insights actionable for bettors across the entire slate
6. Use research data to support each insight
7. Write in Professor Lock's sharp, confident style

Return a JSON array of insights. Each should provide real betting value."""

            response = await self.grok_client.chat.completions.create(
                model="grok-3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=3000
            )
            
            insights_text = response.choices[0].message.content
            
            # Parse insights
            try:
                if '```json' in insights_text:
                    json_start = insights_text.find('```json') + 7
                    json_end = insights_text.find('```', json_start)
                    insights_text = insights_text[json_start:json_end]
                
                insights_data = json.loads(insights_text)
                
                # Process and validate insights
                processed_insights = []
                for insight in insights_data:
                    processed_insight = {
                        'title': insight.get('title', f'{sport} Betting Insight'),
                        'content': insight.get('content', 'Market analysis reveals betting opportunities.'),
                        'category': insight.get('category', 'trend'),
                        'confidence': min(95, max(70, int(insight.get('confidence', 80)))),
                        'sport': sport,
                        'created_at': datetime.now().isoformat()
                    }
                    processed_insights.append(processed_insight)
                
                logger.info(f"Generated {len(processed_insights)} valid {sport} insights from {len(insights_data)} AI suggestions")
                return processed_insights[:target_insights]  # Ensure we don't exceed target
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse insights JSON for {sport}: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating insights for {sport}: {e}")
            return []

    def _format_games_for_ai(self, games: List[Dict]) -> str:
        """Format games data for AI consumption"""
        formatted = []
        for game in games:
            start_time = game.get('start_time', 'TBD')
            if start_time != 'TBD':
                try:
                    dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    time_str = dt.strftime('%I:%M %p ET')
                except:
                    time_str = 'TBD'
            else:
                time_str = 'TBD'
            
            formatted.append(f"• {game.get('away_team', 'TBD')} @ {game.get('home_team', 'TBD')} ({time_str})")
        
        return '\n'.join(formatted)

    async def store_all_insights(self, insights_by_sport: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Store all insights from all sports"""
        all_insights = []
        for sport, insights in insights_by_sport.items():
            all_insights.extend(insights)
        
        if all_insights:
            success = self.db.store_daily_insights(all_insights)
            logger.info(f"Stored {len(all_insights)} total insights across {len(insights_by_sport)} sports")
            return success
        return False

async def main():
    """Main execution function"""
    agent = PersonalizedInsightsAgent()
    
    # Generate insights for all sports
    insights_by_sport = await agent.generate_personalized_insights_by_sport()
    
    if insights_by_sport:
        # Store all insights
        await agent.store_all_insights(insights_by_sport)
        
        # Log summary
        for sport, insights in insights_by_sport.items():
            logger.info(f"✅ {sport}: {len(insights)} insights generated")
    else:
        logger.warning("No insights generated for any sport")

if __name__ == "__main__":
    asyncio.run(main())
