import asyncio
import aioredis
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Dict, Any, List
import json
import logging
from datetime import datetime, timedelta
import httpx

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.mongo_client = None
        self.db = None
        self.data_sources = {}
        
    async def initialize(self):
        """Initialize data pipeline connections"""
        # Redis connection
        self.redis_client = aioredis.from_url(
            self.config['redis_url'],
            decode_responses=True
        )
        
        # MongoDB connection
        self.mongo_client = AsyncIOMotorClient(self.config['mongodb_uri'])
        self.db = self.mongo_client.get_database()
        
        # Initialize data sources
        await self._initialize_data_sources()
        
    async def _initialize_data_sources(self):
        """Initialize various data sources"""
        self.data_sources = {
            'alpha_vantage': AlphaVantageSource(self.config['alpha_vantage_api_key']),
            'news': NewsDataSource(self.config['news_api_key']),
            'sentiment': SentimentDataSource(self.config['twitter_bearer_token'])
        }
        
        for source in self.data_sources.values():
            await source.initialize()
    
    async def start_real_time_feed(self):
        """Start real-time data feed"""
        tasks = [
            self._market_data_loop(),
            self._news_data_loop(),
            self._sentiment_data_loop()
        ]
        
        await asyncio.gather(*tasks)
    
    async def _market_data_loop(self):
        """Real-time market data collection loop"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        while True:
            try:
                for symbol in symbols:
                    data = await self.data_sources['alpha_vantage'].get_real_time_data(symbol)
                    
                    if data:
                        # Store in Redis for real-time access
                        await self.redis_client.setex(
                            f"market_data:{symbol}",
                            300,  # 5 minute expiry
                            json.dumps(data)
                        )
                        
                        # Store in MongoDB for historical data
                        await self.db.market_data.insert_one({
                            'symbol': symbol,
                            'data': data,
                            'timestamp': datetime.utcnow()
                        })
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Market data loop error: {e}")
                await asyncio.sleep(30)
    
    async def _news_data_loop(self):
        """News data collection loop"""
        while True:
            try:
                news_data = await self.data_sources['news'].get_latest_news()
                
                if news_data:
                    await self.redis_client.setex(
                        "news_data",
                        1800,  # 30 minute expiry
                        json.dumps(news_data)
                    )
                    
                    # Store in MongoDB
                    await self.db.news_data.insert_many(news_data)
                
                await asyncio.sleep(900)  # Update every 15 minutes
                
            except Exception as e:
                logger.error(f"News data loop error: {e}")
                await asyncio.sleep(300)
    
    async def _sentiment_data_loop(self):
        """Sentiment data collection loop"""
        while True:
            try:
                sentiment_data = await self.data_sources['sentiment'].get_market_sentiment()
                
                if sentiment_data:
                    await self.redis_client.setex(
                        "sentiment_data",
                        600,  # 10 minute expiry
                        json.dumps(sentiment_data)
                    )
                    
                    # Store in MongoDB
                    await self.db.sentiment_data.insert_one({
                        'data': sentiment_data,
                        'timestamp': datetime.utcnow()
                    })
                
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logger.error(f"Sentiment data loop error: {e}")
                await asyncio.sleep(180)
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get latest market data for symbol"""
        try:
            cached_data = await self.redis_client.get(f"market_data:{symbol}")
            if cached_data:
                return json.loads(cached_data)
            
            # Fallback to database
            result = await self.db.market_data.find_one(
                {'symbol': symbol},
                sort=[('timestamp', -1)]
            )
            
            return result.get('data', {}) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical data for symbol"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            cursor = self.db.market_data.find({
                'symbol': symbol,
                'timestamp': {'$gte': start_date}
            }).sort('timestamp', 1)
            
            return await cursor.to_list(length=None)
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
