"""
Sentiment Agent - Market sentiment analysis from news and social media
"""
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import requests
import json

logger = logging.getLogger(__name__)

class SentimentAgent:
    """
    AI agent for sentiment analysis from multiple sources
    """
    
    def __init__(self):
        self.sentiment_sources = ['news', 'social', 'analyst', 'technical']
        self.sentiment_weights = {
            'news': 0.35,
            'social': 0.25,
            'analyst': 0.30,
            'technical': 0.10
        }
        self.api_keys = {
            'news_api': None,
            'twitter': None
        }
        
    def analyze_sentiment(self, symbol: str) -> Dict:
        """
        Analyze market sentiment for a symbol from multiple sources
        """
        try:
            # Analyze different sources
            news_sentiment = self._analyze_news_sentiment(symbol)
            social_sentiment = self._analyze_social_sentiment(symbol)
            analyst_sentiment = self._analyze_analyst_sentiment(symbol)
            technical_sentiment = self._analyze_technical_sentiment(symbol)
            
            # Calculate weighted sentiment score
            weighted_score = (
                news_sentiment['score'] * self.sentiment_weights['news'] +
                social_sentiment['score'] * self.sentiment_weights['social'] +
                analyst_sentiment['score'] * self.sentiment_weights['analyst'] +
                technical_sentiment['score'] * self.sentiment_weights['technical']
            )
            
            # Determine sentiment level
            if weighted_score > 0.3:
                sentiment_level = 'bullish'
            elif weighted_score < -0.3:
                sentiment_level = 'bearish'
            else:
                sentiment_level = 'neutral'
            
            # Calculate confidence based on consistency
            confidence = self._calculate_confidence(
                news_sentiment, 
                social_sentiment, 
                analyst_sentiment,
                technical_sentiment
            )
            
            return {
                'symbol': symbol,
                'score': round(weighted_score, 3),
                'level': sentiment_level,
                'confidence': round(confidence, 2),
                'sources': {
                    'news': news_sentiment,
                    'social': social_sentiment,
                    'analyst': analyst_sentiment,
                    'technical': technical_sentiment
                },
                'signals': self._generate_signals(weighted_score, sentiment_level),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'score': 0,
                'level': 'neutral',
                'confidence': 0,
                'error': str(e)
            }
    
    def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment"""
        try:
            # In production, this would fetch real news
            # For demo, using simulated sentiment with realistic patterns
            
            # Simulated news analysis
            positive_words = ['growth', 'profit', 'beat', 'upgrade', 'strong', 'record', 'surge']
            negative_words = ['loss', 'decline', 'miss', 'downgrade', 'weak', 'concern', 'fall']
            
            # Generate realistic sentiment score
            base_score = np.random.uniform(-0.5, 0.5)
            
            # Add some market correlation
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                base_score += 0.1  # Tech bias
            
            article_count = np.random.randint(5, 25)
            positive_count = int(max(0, (base_score + 0.5) * article_count * 0.7))
            negative_count = int(max(0, (0.5 - base_score) * article_count * 0.7))
            
            return {
                'score': round(base_score, 3),
                'article_count': article_count,
                'positive_mentions': positive_count,
                'negative_mentions': negative_count,
                'trending': abs(base_score) > 0.3,
                'latest_headline': self._generate_headline(symbol, base_score),
                'momentum': 'increasing' if base_score > 0.2 else 'decreasing' if base_score < -0.2 else 'stable'
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis error: {e}")
            return {'score': 0, 'article_count': 0, 'error': str(e)}
    
    def _analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze social media sentiment"""
        try:
            # Simulated social media sentiment
            base_score = np.random.uniform(-0.5, 0.5)
            
            # Add volatility for social media
            volatility_factor = np.random.uniform(0.8, 1.2)
            adjusted_score = base_score * volatility_factor
            
            mention_count = np.random.randint(100, 5000)
            sentiment_ratio = 0.5 + (adjusted_score * 0.5)  # Convert to 0-1 range
            
            # Calculate engagement metrics
            likes = int(mention_count * sentiment_ratio * np.random.uniform(2, 5))
            shares = int(mention_count * sentiment_ratio * np.random.uniform(0.5, 1.5))
            
            return {
                'score': round(adjusted_score, 3),
                'mention_count': mention_count,
                'sentiment_ratio': round(sentiment_ratio, 2),
                'volume_change': round(np.random.uniform(-50, 100), 1),
                'engagement': {
                    'likes': likes,
                    'shares': shares,
                    'comments': np.random.randint(50, 500)
                },
                'influencer_sentiment': 'positive' if adjusted_score > 0.1 else 'negative' if adjusted_score < -0.1 else 'neutral',
                'viral_score': min(100, int(abs(adjusted_score) * 200))
            }
            
        except Exception as e:
            logger.error(f"Social sentiment analysis error: {e}")
            return {'score': 0, 'mention_count': 0, 'error': str(e)}
    
    def _analyze_analyst_sentiment(self, symbol: str) -> Dict:
        """Analyze analyst ratings and recommendations"""
        try:
            # Simulated analyst sentiment (more conservative)
            base_score = np.random.uniform(-0.3, 0.3)
            
            # Analysts are typically more bullish on large caps
            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                base_score += 0.15
            
            ratings = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
            
            # Generate realistic distribution
            if base_score > 0.2:
                weights = [0.35, 0.30, 0.25, 0.07, 0.03]
            elif base_score > 0:
                weights = [0.20, 0.30, 0.35, 0.10, 0.05]
            elif base_score > -0.2:
                weights = [0.10, 0.20, 0.40, 0.20, 0.10]
            else:
                weights = [0.05, 0.10, 0.35, 0.30, 0.20]
            
            consensus = np.random.choice(ratings, p=weights)
            
            # Calculate target price
            base_price = 100  # Simplified base price
            target_price = base_price * (1 + base_score * 0.3)
            
            analyst_count = np.random.randint(5, 20)
            
            # Generate rating distribution
            rating_distribution = {
                rating: int(analyst_count * weight) 
                for rating, weight in zip(ratings, weights)
            }
            
            return {
                'score': round(base_score, 3),
                'consensus': consensus,
                'target_price': round(target_price, 2),
                'analyst_count': analyst_count,
                'rating_distribution': rating_distribution,
                'avg_rating': round(2.5 - base_score * 2, 1),  # 1-5 scale
                'recent_changes': {
                    'upgrades': max(0, int(base_score * 5)),
                    'downgrades': max(0, int(-base_score * 5)),
                    'initiated': np.random.randint(0, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Analyst sentiment analysis error: {e}")
            return {'score': 0, 'consensus': 'Hold', 'error': str(e)}
    
    def _analyze_technical_sentiment(self, symbol: str) -> Dict:
        """Analyze technical indicators sentiment"""
        try:
            # Simulated technical sentiment
            rsi = np.random.uniform(20, 80)
            macd_signal = np.random.uniform(-1, 1)
            
            # Calculate technical score
            rsi_score = 0
            if rsi < 30:
                rsi_score = 0.5  # Oversold - bullish
            elif rsi > 70:
                rsi_score = -0.5  # Overbought - bearish
            
            technical_score = (rsi_score + macd_signal) / 2
            
            return {
                'score': round(technical_score, 3),
                'indicators': {
                    'rsi': round(rsi, 2),
                    'macd_signal': 'bullish' if macd_signal > 0 else 'bearish',
                    'moving_average': 'above' if technical_score > 0 else 'below',
                    'volume_trend': np.random.choice(['increasing', 'decreasing', 'stable'])
                },
                'pattern': np.random.choice([
                    'ascending_triangle',
                    'descending_triangle', 
                    'head_shoulders',
                    'double_bottom',
                    'flag',
                    'none'
                ]),
                'support_resistance': {
                    'near_support': rsi < 40,
                    'near_resistance': rsi > 60
                }
            }
            
        except Exception as e:
            logger.error(f"Technical sentiment analysis error: {e}")
            return {'score': 0, 'indicators': {}, 'error': str(e)}
    
    def _calculate_confidence(self, news: Dict, social: Dict, 
                            analyst: Dict, technical: Dict) -> float:
        """Calculate confidence in sentiment analysis"""
        confidence = 0.5  # Base confidence
        
        # Check data availability
        if news.get('article_count', 0) > 10:
            confidence += 0.1
        if social.get('mention_count', 0) > 500:
            confidence += 0.1
        if analyst.get('analyst_count', 0) > 10:
            confidence += 0.1
        
        # Check consistency across sources
        scores = [
            news.get('score', 0),
            social.get('score', 0),
            analyst.get('score', 0),
            technical.get('score', 0)
        ]
        
        # All positive or all negative = high confidence
        if all(s > 0 for s in scores) or all(s < 0 for s in scores):
            confidence += 0.25
        # Mixed signals = lower confidence
        elif len([s for s in scores if s > 0]) == 2:
            confidence -= 0.1
        
        # Check signal strength
        avg_abs_score = np.mean([abs(s) for s in scores])
        if avg_abs_score > 0.3:
            confidence += 0.15
        
        return min(1.0, max(0.1, confidence))
    
    def _generate_signals(self, score: float, level: str) -> List[Dict]:
        """Generate trading signals based on sentiment"""
        signals = []
        
        if score > 0.4:
            signals.append({
                'type': 'strong_buy',
                'strength': 'high',
                'description': 'Strong positive sentiment across multiple sources'
            })
        elif score > 0.2:
            signals.append({
                'type': 'buy',
                'strength': 'medium',
                'description': 'Moderate positive sentiment'
            })
        elif score < -0.4:
            signals.append({
                'type': 'strong_sell',
                'strength': 'high',
                'description': 'Strong negative sentiment across multiple sources'
            })
        elif score < -0.2:
            signals.append({
                'type': 'sell',
                'strength': 'medium',
                'description': 'Moderate negative sentiment'
            })
        else:
            signals.append({
                'type': 'hold',
                'strength': 'low',
                'description': 'Mixed or neutral sentiment'
            })
        
        return signals
    
    def _generate_headline(self, symbol: str, score: float) -> str:
        """Generate a realistic news headline based on sentiment"""
        positive_templates = [
            f"{symbol} beats earnings expectations, stock surges",
            f"Analysts upgrade {symbol} on strong growth prospects",
            f"{symbol} announces record quarterly revenue",
            f"Institutional investors increase {symbol} holdings"
        ]
        
        negative_templates = [
            f"{symbol} misses revenue targets amid challenges",
            f"Concerns grow over {symbol} market position",
            f"{symbol} faces headwinds in competitive market",
            f"Analysts downgrade {symbol} on valuation concerns"
        ]
        
        neutral_templates = [
            f"{symbol} trading steady as investors await earnings",
            f"Market watches {symbol} for breakout signals",
            f"{symbol} consolidates near support levels",
            f"Mixed signals for {symbol} as market digests news"
        ]
        
        if score > 0.2:
            return np.random.choice(positive_templates)
        elif score < -0.2:
            return np.random.choice(negative_templates)
        else:
            return np.random.choice(neutral_templates)
    
    def get_market_mood(self, symbols: List[str]) -> Dict:
        """Get overall market mood from multiple symbols"""
        try:
            sentiments = []
            for symbol in symbols[:10]:  # Limit to 10 symbols
                sentiment = self.analyze_sentiment(symbol)
                sentiments.append(sentiment['score'])
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            if avg_sentiment > 0.2:
                mood = 'risk-on'
            elif avg_sentiment < -0.2:
                mood = 'risk-off'
            else:
                mood = 'neutral'
            
            return {
                'market_mood': mood,
                'average_sentiment': round(avg_sentiment, 3),
                'bullish_count': len([s for s in sentiments if s > 0.2]),
                'bearish_count': len([s for s in sentiments if s < -0.2]),
                'neutral_count': len([s for s in sentiments if -0.2 <= s <= 0.2]),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market mood analysis error: {e}")
            return {'market_mood': 'neutral', 'error': str(e)}
