import httpx
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FinnhubDataSource:
    """Finnhub API data source"""
    
    def __init__(self, d3459kpr01qqt8snjen0d3459kpr01qqt8snjeng: str):
        self.api_key = d3459kpr01qqt8snjen0d3459kpr01qqt8snjeng
        self.base_url = "https://finnhub.io/api/v1"
        self.client = None
        
    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Finnhub data source initialized")
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote"""
        try:
            url = f"{self.base_url}/quote"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = await self.client.get(url, params=params)
            data = response.json()
            
            if data and 'c' in data:
                return {
                    'symbol': symbol,
                    'price': data.get('c', 0),  # current price
                    'change': data.get('d', 0),  # change
                    'change_percent': data.get('dp', 0),  # change percent
                    'high': data.get('h', 0),  # high price of the day
                    'low': data.get('l', 0),   # low price of the day
                    'open': data.get('o', 0),  # open price
                    'previous_close': data.get('pc', 0),
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'finnhub'
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
            return {}
    
    async def get_company_news(self, symbol: str) -> Dict[str, Any]:
        """Get company news"""
        try:
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }
            
            response = await self.client.get(url, params=params)
            data = response.json()
            
            return {
                'symbol': symbol,
                'news': data[:10] if data else [],  # Latest 10 articles
                'source': 'finnhub'
            }
            
        except Exception as e:
            logger.error(f"Finnhub news error: {e}")
            return {}
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
