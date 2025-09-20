mport httpx
import asyncio
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PolygonDataSource:
    """Polygon.io API data source"""
    
    def __init__(self, Wj3TIukaNewvEUpLUxn6RXJLiyo233a4: str):
        self.api_key = Wj3TIukaNewvEUpLUxn6RXJLiyo233a4
        self.base_url = "https://api.polygon.io"
        self.client = None
        
    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Polygon.io data source initialized")
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = {'apiKey': self.api_key}
            
            response = await self.client.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                result = data['results'][0]
                return {
                    'symbol': symbol,
                    'price': result.get('c', 0),  # close price
                    'open': result.get('o', 0),
                    'high': result.get('h', 0),
                    'low': result.get('l', 0),
                    'volume': result.get('v', 0),
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'polygon'
                }
            
            logger.warning(f"No data returned for {symbol} from Polygon")
            return {}
            
        except Exception as e:
            logger.error(f"Polygon API error for {symbol}: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get historical data"""
        try:
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apiKey': self.api_key}
            
            response = await self.client.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                return {
                    'symbol': symbol,
                    'data': data['results'],
                    'source': 'polygon'
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Polygon historical data error: {e}")
            return {}
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
