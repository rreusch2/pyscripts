import asyncio
from playwright.async_api import async_playwright
import json
import sys


async def fetch_revenuecat_metrics(email, password):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # Navigate to login page
            await page.goto('https://app.revenuecat.com/login')
            
            # Fill in credentials and submit
            await page.fill('input[name="email"]', email)
            await page.fill('input[name="password"]', password)
            await page.click('button[type="submit"]')
            
            # Wait for navigation to the dashboard
            await page.wait_for_url('https://app.revenuecat.com/projects/**')
            
            # Extract metrics from the dashboard
            # Note: These selectors may need to be updated if RevenueCat changes its dashboard structure.
            active_trials = await page.locator('div:has-text("Active Trials") >> nth=1').inner_text()
            active_subscriptions = await page.locator('div:has-text("Active Subscriptions") >> nth=1').inner_text()
            mrr = await page.locator('div:has-text("MRR") >> nth=1').inner_text()
            revenue = await page.locator('div:has-text("Revenue") >> nth=1').inner_text()
            
            metrics = {
                'active_trials': active_trials.split('\n')[0],
                'active_subscriptions': active_subscriptions.split('\n')[0],
                'mrr': mrr.split('\n')[0],
                'revenue_last_28_days': revenue.split('\n')[0],
            }
            
            return metrics
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        finally:
            await browser.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python revenuecat_scraper.py <email> <password>")
        sys.exit(1)
        
    email = sys.argv[1]
    password = sys.argv[2]
    
    metrics = asyncio.run(fetch_revenuecat_metrics(email, password))
    
    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        print(json.dumps({"error": "Failed to fetch metrics."}))
        sys.exit(1)
