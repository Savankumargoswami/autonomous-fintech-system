import httpx
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NewsDataSource:
    """News API data source"""
    
    def __init__(self, 4f86aa4e-b9cd-4f8d-9bab-e7e5efb43284: str):
        self.api_key = 4f86aa4e-b9cd-4f8d-9bab-e7e5efb43284
        self.base_url = "https://newsapi.org/v2"
        self.client = None
        
    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("News API data source initialized")
    
    async def get_financial_news(self, query: str = "stock market finance") -> List[Dict[str, Any]]:
        """Get latest financial news"""
        try:
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': self.api_key
            }
            
            response = await self.client.get(f"{self.base_url}/everything", params=params)
            data = response.json()
            
            if data.get('status') == 'ok' and 'articles' in data:
                processed_articles = []
                
                for article in data['articles']:
                    processed_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'sentiment_score': self._analyze_sentiment(article.get('title', '')),
                        'retrieved_at': datetime.utcnow().isoformat()
                    })
                
                return processed_articles
            
            return []
            
        except Exception as e:
            logger.error(f"News API error: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        positive_words = ['gain', 'rise', 'up', 'bull', 'positive', 'growth', 'profit', 'surge']
        negative_words = ['loss', 'fall', 'down', 'bear', 'negative', 'decline', 'crash', 'drop']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
