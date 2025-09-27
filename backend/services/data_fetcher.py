"""
Data Fetcher - Real-time and historical market data fetcher
"""
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import json
import time

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Service for fetching market data from various sources
    """
    
    def __init__(self):
        self.api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'polygon': os.getenv('POLYGON_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'news_api': os.getenv('NEWS_API_KEY')
        }
        self.base_urls = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'polygon': 'https://api.polygon.io',
            'finnhub': 'https://finnhub.io/api/v1',
            'news_api': 'https://newsapi.org/v2'
        }
        self.cache = {}
        self.cache_duration = 60  # seconds
        
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a symbol
        """
        try:
            # Check cache
            cache_key = f"quote_{symbol}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Try multiple sources
            quote = self._get_quote_polygon(symbol)
            if not quote:
                quote = self._get_quote_finnhub(symbol)
            if not quote:
                quote = self._get_quote_alpha_vantage(symbol)
            
            # If still no quote, generate simulated data for paper trading
            if not quote:
                quote = self._generate_simulated_quote(symbol)
            
            # Cache the result
            self._cache_data(cache_key, quote)
            
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return self._generate_simulated_quote(symbol)
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical price data
        """
        try:
            # Check cache
            cache_key = f"historical_{symbol}_{days}"
            if self._is_cache_valid(cache_key, duration=3600):  # 1 hour cache
                return self.cache[cache_key]['data']
            
            # Try to fetch from API
            data = self._get_historical_polygon(symbol, days)
            if data is None or data.empty:
                data = self._get_historical_alpha_vantage(symbol, days)
            
            # If still no data, generate simulated data
            if data is None or data.empty:
                data = self._generate_simulated_historical(symbol, days)
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache the result
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return self._generate_simulated_historical(symbol, days)
    
    def get_market_news(self, symbol: str = None) -> List[Dict]:
        """
        Get market news
        """
        try:
            # Check cache
            cache_key = f"news_{symbol if symbol else 'market'}"
            if self._is_cache_valid(cache_key, duration=1800):  # 30 minutes cache
                return self.cache[cache_key]['data']
            
            news = []
            
            # Try NewsAPI
            if self.api_keys['news_api']:
                news.extend(self._get_news_newsapi(symbol))
            
            # Try Finnhub
            if self.api_keys['finnhub']:
                news.extend(self._get_news_finnhub(symbol))
            
            # If no news, generate sample news for demo
            if not news:
                news = self._generate_sample_news(symbol)
            
            # Cache the result
            self._cache_data(cache_key, news)
            
            return news[:10]  # Return top 10 news items
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return self._generate_sample_news(symbol)
    
    def _get_quote_polygon(self, symbol: str) -> Optional[Dict]:
        """Fetch quote from Polygon.io"""
        if not self.api_keys['polygon']:
            return None
        
        try:
            url = f"{self.base_urls['polygon']}/v2/last/trade/{symbol}"
            params = {'apiKey': self.api_keys['polygon']}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK':
                    result = data.get('results', {})
                    return {
                        'symbol': symbol,
                        'price': result.get('p', 0),
                        'volume': result.get('s', 0),
                        'timestamp': datetime.fromtimestamp(result.get('t', 0) / 1000).isoformat(),
                        'source': 'polygon'
                    }
        except Exception as e:
            logger.debug(f"Polygon quote fetch failed: {e}")
        
        return None
    
    def _get_quote_finnhub(self, symbol: str) -> Optional[Dict]:
        """Fetch quote from Finnhub"""
        if not self.api_keys['finnhub']:
            return None
        
        try:
            url = f"{self.base_urls['finnhub']}/quote"
            params = {
                'symbol': symbol,
                'token': self.api_keys['finnhub']
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'price': data.get('c', 0),
                    'open': data.get('o', 0),
                    'high': data.get('h', 0),
                    'low': data.get('l', 0),
                    'prev_close': data.get('pc', 0),
                    'change': data.get('c', 0) - data.get('pc', 0),
                    'change_percent': ((data.get('c', 0) - data.get('pc', 0)) / data.get('pc', 1)) * 100,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'finnhub'
                }
        except Exception as e:
            logger.debug(f"Finnhub quote fetch failed: {e}")
        
        return None
    
    def _get_quote_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """Fetch quote from Alpha Vantage"""
        if not self.api_keys['alpha_vantage']:
            return None
        
        try:
            url = self.base_urls['alpha_vantage']
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_keys['alpha_vantage']
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                quote = data.get('Global Quote', {})
                
                if quote:
                    return {
                        'symbol': symbol,
                        'price': float(quote.get('05. price', 0)),
                        'open': float(quote.get('02. open', 0)),
                        'high': float(quote.get('03. high', 0)),
                        'low': float(quote.get('04. low', 0)),
                        'volume': int(quote.get('06. volume', 0)),
                        'prev_close': float(quote.get('08. previous close', 0)),
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': quote.get('10. change percent', '0%').replace('%', ''),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'alpha_vantage'
                    }
        except Exception as e:
            logger.debug(f"Alpha Vantage quote fetch failed: {e}")
        
        return None
    
    def _get_historical_polygon(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from Polygon.io"""
        if not self.api_keys['polygon']:
            return None
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_urls['polygon']}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apiKey': self.api_keys['polygon']}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK' and data.get('results'):
                    df = pd.DataFrame(data['results'])
                    df.columns = ['volume', 'vwap', 'open', 'close', 'high', 'low', 'timestamp', 'transactions']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.debug(f"Polygon historical fetch failed: {e}")
        
        return None
    
    def _get_historical_alpha_vantage(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from Alpha Vantage"""
        if not self.api_keys['alpha_vantage']:
            return None
        
        try:
            url = self.base_urls['alpha_vantage']
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'compact' if days <= 100 else 'full',
                'apikey': self.api_keys['alpha_vantage']
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                time_series = data.get('Time Series (Daily)', {})
                
                if time_series:
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    df = df.astype(float)
                    df = df.sort_index()
                    return df.tail(days)
        except Exception as e:
            logger.debug(f"Alpha Vantage historical fetch failed: {e}")
        
        return None
    
    def _generate_simulated_quote(self, symbol: str) -> Dict:
        """Generate simulated quote for paper trading"""
        base_price = 100.0
        volatility = 0.02
        
        # Generate random price movement
        change_pct = np.random.normal(0, volatility)
        price = base_price * (1 + change_pct)
        
        return {
            'symbol': symbol,
            'price': round(price, 2),
            'open': round(base_price, 2),
            'high': round(price * 1.01, 2),
            'low': round(price * 0.99, 2),
            'volume': np.random.randint(1000000, 10000000),
            'prev_close': base_price,
            'change': round(price - base_price, 2),
            'change_percent': round(change_pct * 100, 2),
            'timestamp': datetime.now().isoformat(),
            'source': 'simulated'
        }
    
    def _generate_simulated_historical(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate simulated historical data for paper trading"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate price data with realistic patterns
        base_price = 100.0
        prices = [base_price]
        
        for i in range(1, days):
            # Add trend and random walk
            trend = 0.0001 * i  # Slight upward trend
            random_walk = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + trend + random_walk)
            prices.append(new_price)
        
        # Create DataFrame
        data = []
        for i, date in enumerate(dates):
            price = prices[i]
            daily_volatility = np.random.uniform(0.005, 0.02)
            
            high = price * (1 + daily_volatility)
            low = price * (1 - daily_volatility)
            open_price = price * (1 + np.random.uniform(-daily_volatility/2, daily_volatility/2))
            close = price
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        if df.empty:
            return df
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        
        # Moving averages
        df['sma_20'] = sma_20
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = ema_12
        df['ema_26'] = ema_26
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _get_news_newsapi(self, symbol: str) -> List[Dict]:
        """Get news from NewsAPI"""
        if not self.api_keys['news_api']:
            return []
        
        try:
            url = f"{self.base_urls['news_api']}/everything"
            params = {
                'q': symbol,
                'apiKey': self.api_keys['news_api'],
                'sortBy': 'relevancy',
                'pageSize': 5
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                news = []
                for article in articles:
                    news.append({
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'url': article.get('url'),
                        'published': article.get('publishedAt'),
                        'source': article.get('source', {}).get('name')
                    })
                
                return news
        except Exception as e:
            logger.debug(f"NewsAPI fetch failed: {e}")
        
        return []
    
    def _get_news_finnhub(self, symbol: str) -> List[Dict]:
        """Get news from Finnhub"""
        if not self.api_keys['finnhub']:
            return []
        
        try:
            url = f"{self.base_urls['finnhub']}/company-news"
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            
            params = {
                'symbol': symbol,
                'from': week_ago.strftime('%Y-%m-%d'),
                'to': today.strftime('%Y-%m-%d'),
                'token': self.api_keys['finnhub']
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                articles = response.json()
                
                news = []
                for article in articles[:5]:
                    news.append({
                        'title': article.get('headline'),
                        'description': article.get('summary'),
                        'url': article.get('url'),
                        'published': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                        'source': article.get('source')
                    })
                
                return news
        except Exception as e:
            logger.debug(f"Finnhub news fetch failed: {e}")
        
        return []
    
    def _generate_sample_news(self, symbol: str) -> List[Dict]:
        """Generate sample news for demo"""
        templates = [
            f"{symbol} shows strong momentum amid market rally",
            f"Analysts upgrade {symbol} target price to new highs",
            f"{symbol} announces breakthrough in Q4 earnings",
            f"Market watch: {symbol} among top performers today",
            f"Technical analysis suggests bullish trend for {symbol}"
        ]
        
        news = []
        for i, title in enumerate(templates[:3]):
            news.append({
                'title': title,
                'description': f"Lorem ipsum analysis for {symbol} showing positive trends...",
                'url': f"https://example.com/news/{i}",
                'published': (datetime.now() - timedelta(hours=i)).isoformat(),
                'source': 'Demo News'
            })
        
        return news
    
    def _is_cache_valid(self, key: str, duration: int = None) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        cache_duration = duration if duration else self.cache_duration
        cache_time = self.cache[key].get('timestamp', 0)
        
        return (time.time() - cache_time) < cache_duration
    
    def _cache_data(self, key: str, data):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
