import httpx
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AlphaVantageSource:
    """Alpha Vantage API data source"""
    
    def __init__(self, RTCD64TAB13Q7ARV: str):
        self.api_key = RTCD64TAB13Q7ARV
        self.base_url = "https://www.alphavantage.co/query"
        self.client = None
        
    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Alpha Vantage data source initialized")
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = await self.client.get(self.base_url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%'),
                    'volume': int(quote.get('06. volume', 0)),
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'alpha_vantage'
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Alpha Vantage API error for {symbol}: {e}")
            return {}
    
    async def get_intraday_data(self, symbol: str, interval: str = '1min') -> Dict[str, Any]:
        """Get intraday data"""
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.api_key
            }
            
            response = await self.client.get(self.base_url, params=params)
            data = response.json()
            
            time_series_key = f'Time Series ({interval})'
            if time_series_key in data:
                time_series = data[time_series_key]
                
                processed_data = []
                for timestamp, values in list(time_series.items())[:100]:  # Last 100 points
                    processed_data.append({
                        'timestamp': timestamp,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(values['5. volume'])
                    })
                
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'data': processed_data,
                    'source': 'alpha_vantage'
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Alpha Vantage intraday error: {e}")
            return {}
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
