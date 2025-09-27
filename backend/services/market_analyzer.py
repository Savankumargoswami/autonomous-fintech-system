"""
Market Analyzer - Real-time market analysis and insights
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """
    Market analysis service for technical and fundamental analysis
    """
    
    def __init__(self):
        self.indicators = {
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger': self._calculate_bollinger_bands,
            'stochastic': self._calculate_stochastic,
            'atr': self._calculate_atr,
            'adx': self._calculate_adx,
            'obv': self._calculate_obv,
            'vwap': self._calculate_vwap
        }
        
    def analyze_market(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive market analysis
        """
        try:
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'technical_indicators': {},
                'patterns': [],
                'support_resistance': {},
                'trend': {},
                'volume_analysis': {},
                'price_action': {},
                'market_structure': {}
            }
            
            # Calculate technical indicators
            for name, calc_func in self.indicators.items():
                try:
                    analysis['technical_indicators'][name] = calc_func(data)
                except Exception as e:
                    logger.warning(f"Failed to calculate {name}: {e}")
            
            # Identify patterns
            analysis['patterns'] = self._identify_patterns(data)
            
            # Find support and resistance levels
            analysis['support_resistance'] = self._find_support_resistance(data)
            
            # Analyze trend
            analysis['trend'] = self._analyze_trend(data)
            
            # Volume analysis
            analysis['volume_analysis'] = self._analyze_volume(data)
            
            # Price action analysis
            analysis['price_action'] = self._analyze_price_action(data)
            
            # Market structure
            analysis['market_structure'] = self._analyze_market_structure(data)
            
            # Generate trading signals
            analysis['signals'] = self._generate_signals(analysis)
            
            # Overall market score
            analysis['market_score'] = self._calculate_market_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {'error': str(e)}
    
    def get_recommendation(self, strategy: Dict, risk: Dict, sentiment: Dict) -> Dict:
        """
        Generate trading recommendation based on multiple factors
        """
        try:
            # Combine scores
            strategy_score = strategy.get('score', 0) if isinstance(strategy.get('score'), (int, float)) else 0
            risk_score = 100 - risk.get('risk_score', 50)
            sentiment_score = sentiment.get('score', 0) if isinstance(sentiment.get('score'), (int, float)) else 0
            
            # Weighted average
            weights = {'strategy': 0.4, 'risk': 0.3, 'sentiment': 0.3}
            combined_score = (
                strategy_score * weights['strategy'] +
                (risk_score / 100) * weights['risk'] +
                sentiment_score * weights['sentiment']
            )
            
            # Determine action
            if combined_score > 0.3:
                action = 'BUY'
                confidence = min(combined_score, 1.0)
            elif combined_score < -0.3:
                action = 'SELL'
                confidence = min(abs(combined_score), 1.0)
            else:
                action = 'HOLD'
                confidence = 1 - abs(combined_score)
            
            return {
                'action': action,
                'confidence': round(confidence * 100, 2),
                'combined_score': round(combined_score, 3),
                'factors': {
                    'strategy': round(strategy_score, 3),
                    'risk': round(risk_score, 2),
                    'sentiment': round(sentiment_score, 3)
                },
                'reasoning': self._generate_reasoning(action, strategy, risk, sentiment)
            }
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'error': str(e)
            }
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate RSI indicator"""
        closes = data['close']
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        return {
            'value': round(current_rsi, 2) if not pd.isna(current_rsi) else 50,
            'signal': 'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral',
            'trend': 'bullish' if current_rsi > 50 else 'bearish'
        }
    
    def _calculate_macd(self, data: pd.DataFrame) -> Dict:
        """Calculate MACD indicator"""
        closes = data['close']
        
        ema_12 = closes.ewm(span=12).mean()
        ema_26 = closes.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': round(macd_line.iloc[-1], 4) if not pd.isna(macd_line.iloc[-1]) else 0,
            'signal': round(signal_line.iloc[-1], 4) if not pd.isna(signal_line.iloc[-1]) else 0,
            'histogram': round(histogram.iloc[-1], 4) if not pd.isna(histogram.iloc[-1]) else 0,
            'crossover': 'bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'bearish'
        }
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        closes = data['close']
        
        sma = closes.rolling(window=period).mean()
        std = closes.rolling(window=period).std()
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        current_price = closes.iloc[-1]
        
        return {
            'upper': round(upper.iloc[-1], 2) if not pd.isna(upper.iloc[-1]) else current_price,
            'middle': round(sma.iloc[-1], 2) if not pd.isna(sma.iloc[-1]) else current_price,
            'lower': round(lower.iloc[-1], 2) if not pd.isna(lower.iloc[-1]) else current_price,
            'price': round(current_price, 2),
            'position': 'above' if current_price > upper.iloc[-1] else 'below' if current_price < lower.iloc[-1] else 'inside'
        }
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return {
            'k': round(k_percent.iloc[-1], 2) if not pd.isna(k_percent.iloc[-1]) else 50,
            'd': round(d_percent.iloc[-1], 2) if not pd.isna(d_percent.iloc[-1]) else 50,
            'signal': 'oversold' if k_percent.iloc[-1] < 20 else 'overbought' if k_percent.iloc[-1] > 80 else 'neutral'
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]
        
        return {
            'value': round(current_atr, 2) if not pd.isna(current_atr) else 0,
            'percentage': round((current_atr / current_price) * 100, 2) if current_price > 0 else 0,
            'volatility': 'high' if (current_atr / current_price) > 0.03 else 'low'
        }
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Average Directional Index"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        atr = self._calculate_atr(data, period)['value']
        
        if atr > 0:
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (abs(minus_dm.rolling(period).mean()) / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return {
                'value': round(adx.iloc[-1], 2) if not pd.isna(adx.iloc[-1]) else 25,
                'plus_di': round(plus_di.iloc[-1], 2) if not pd.isna(plus_di.iloc[-1]) else 0,
                'minus_di': round(minus_di.iloc[-1], 2) if not pd.isna(minus_di.iloc[-1]) else 0,
                'trend_strength': 'strong' if adx.iloc[-1] > 25 else 'weak'
            }
        
        return {'value': 25, 'plus_di': 0, 'minus_di': 0, 'trend_strength': 'weak'}
    
    def _calculate_obv(self, data: pd.DataFrame) -> Dict:
        """Calculate On-Balance Volume"""
        close = data['close']
        volume = data['volume']
        
        obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
        obv_ma = obv.rolling(window=20).mean()
        
        return {
            'value': int(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else 0,
            'ma': int(obv_ma.iloc[-1]) if not pd.isna(obv_ma.iloc[-1]) else 0,
            'trend': 'bullish' if obv.iloc[-1] > obv_ma.iloc[-1] else 'bearish'
        }
    
    def _calculate_vwap(self, data: pd.DataFrame) -> Dict:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        current_price = data['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        return {
            'value': round(current_vwap, 2) if not pd.isna(current_vwap) else current_price,
            'price': round(current_price, 2),
            'position': 'above' if current_price > current_vwap else 'below'
        }
    
    def _identify_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Identify chart patterns"""
        patterns = []
        
        # Check for common patterns
        if self._is_double_top(data):
            patterns.append({'name': 'Double Top', 'type': 'bearish', 'confidence': 75})
        
        if self._is_double_bottom(data):
            patterns.append({'name': 'Double Bottom', 'type': 'bullish', 'confidence': 75})
        
        if self._is_head_shoulders(data):
            patterns.append({'name': 'Head and Shoulders', 'type': 'bearish', 'confidence': 70})
        
        if self._is_ascending_triangle(data):
            patterns.append({'name': 'Ascending Triangle', 'type': 'bullish', 'confidence': 65})
        
        return patterns
    
    def _is_double_top(self, data: pd.DataFrame) -> bool:
        """Check for double top pattern"""
        if len(data) < 50:
            return False
        
        highs = data['high'].iloc[-50:]
        peaks = highs.nlargest(2)
        
        if len(peaks) == 2:
            diff_pct = abs((peaks.iloc[0] - peaks.iloc[1]) / peaks.iloc[0]) * 100
            return diff_pct < 3  # Within 3% of each other
        
        return False
    
    def _is_double_bottom(self, data: pd.DataFrame) -> bool:
        """Check for double bottom pattern"""
        if len(data) < 50:
            return False
        
        lows = data['low'].iloc[-50:]
        troughs = lows.nsmallest(2)
        
        if len(troughs) == 2:
            diff_pct = abs((troughs.iloc[0] - troughs.iloc[1]) / troughs.iloc[0]) * 100
            return diff_pct < 3  # Within 3% of each other
        
        return False
    
    def _is_head_shoulders(self, data: pd.DataFrame) -> bool:
        """Check for head and shoulders pattern"""
        # Simplified pattern detection
        if len(data) < 60:
            return False
        
        # This is a placeholder for more complex pattern recognition
        return False
    
    def _is_ascending_triangle(self, data: pd.DataFrame) -> bool:
        """Check for ascending triangle pattern"""
        if len(data) < 40:
            return False
        
        highs = data['high'].iloc[-40:]
        lows = data['low'].iloc[-40:]
        
        # Check if highs are relatively flat and lows are rising
        high_slope = (highs.iloc[-1] - highs.iloc[0]) / len(highs)
        low_slope = (lows.iloc[-1] - lows.iloc[0]) / len(lows)
        
        return high_slope < 0.01 and low_slope > 0.01
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        if len(data) < 50:
            return {'support': [], 'resistance': []}
        
        highs = data['high'].iloc[-100:] if len(data) > 100 else data['high']
        lows = data['low'].iloc[-100:] if len(data) > 100 else data['low']
        
        # Find key levels
        resistance_levels = highs.nlargest(3).tolist()
        support_levels = lows.nsmallest(3).tolist()
        
        current_price = data['close'].iloc[-1]
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels, reverse=True),
            'nearest_support': min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else 0,
            'nearest_resistance': min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else 0
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze price trend"""
        closes = data['close']
        
        # Calculate moving averages
        sma_20 = closes.rolling(window=20).mean()
        sma_50 = closes.rolling(window=50).mean() if len(closes) >= 50 else sma_20
        sma_200 = closes.rolling(window=200).mean() if len(closes) >= 200 else sma_50
        
        current_price = closes.iloc[-1]
        
        # Determine trend
        short_trend = 'bullish' if current_price > sma_20.iloc[-1] else 'bearish'
        medium_trend = 'bullish' if current_price > sma_50.iloc[-1] else 'bearish'
        long_trend = 'bullish' if len(closes) >= 200 and current_price > sma_200.iloc[-1] else medium_trend
        
        # Calculate trend strength
        price_change_20d = (current_price - closes.iloc[-20]) / closes.iloc[-20] * 100 if len(closes) >= 20 else 0
        
        return {
            'short_term': short_trend,
            'medium_term': medium_trend,
            'long_term': long_trend,
            'strength': abs(price_change_20d),
            'direction': 'up' if price_change_20d > 0 else 'down',
            'sma_20': round(sma_20.iloc[-1], 2) if not pd.isna(sma_20.iloc[-1]) else current_price,
            'sma_50': round(sma_50.iloc[-1], 2) if not pd.isna(sma_50.iloc[-1]) else current_price,
            'sma_200': round(sma_200.iloc[-1], 2) if len(closes) >= 200 and not pd.isna(sma_200.iloc[-1]) else current_price
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        volume = data['volume']
        
        avg_volume = volume.rolling(window=20).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
        
        # Volume trend
        volume_trend = 'increasing' if volume.iloc[-5:].mean() > avg_volume.iloc[-1] else 'decreasing'
        
        return {
            'current': int(current_volume),
            'average': int(avg_volume.iloc[-1]) if not pd.isna(avg_volume.iloc[-1]) else 0,
            'ratio': round(volume_ratio, 2),
            'trend': volume_trend,
            'signal': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
        }
    
    def _analyze_price_action(self, data: pd.DataFrame) -> Dict:
        """Analyze price action"""
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else current
        
        # Calculate candle properties
        body = abs(current['close'] - current['open'])
        upper_shadow = current['high'] - max(current['close'], current['open'])
        lower_shadow = min(current['close'], current['open']) - current['low']
        
        # Determine candle type
        if current['close'] > current['open']:
            candle_type = 'bullish'
        elif current['close'] < current['open']:
            candle_type = 'bearish'
        else:
            candle_type = 'doji'
        
        # Check for specific patterns
        is_hammer = lower_shadow > body * 2 and upper_shadow < body * 0.5
        is_shooting_star = upper_shadow > body * 2 and lower_shadow < body * 0.5
        
        return {
            'candle_type': candle_type,
            'body_size': round(body, 2),
            'upper_shadow': round(upper_shadow, 2),
            'lower_shadow': round(lower_shadow, 2),
            'patterns': {
                'hammer': is_hammer,
                'shooting_star': is_shooting_star,
                'doji': candle_type == 'doji'
            }
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure"""
        closes = data['close']
        highs = data['high']
        lows = data['low']
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(data) - 2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
               highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                swing_highs.append(highs.iloc[i])
            
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
               lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                swing_lows.append(lows.iloc[i])
        
        # Determine market structure
        if swing_highs and swing_lows:
            higher_highs = all(swing_highs[i] > swing_highs[i-1] for i in range(1, len(swing_highs)))
            higher_lows = all(swing_lows[i] > swing_lows[i-1] for i in range(1, len(swing_lows)))
            
            if higher_highs and higher_lows:
                structure = 'uptrend'
            elif not higher_highs and not higher_lows:
                structure = 'downtrend'
            else:
                structure = 'ranging'
        else:
            structure = 'undefined'
        
        return {
            'type': structure,
            'swing_highs': len(swing_highs),
            'swing_lows': len(swing_lows),
            'volatility': 'high' if len(swing_highs) + len(swing_lows) > 10 else 'low'
        }
    
    def _generate_signals(self, analysis: Dict) -> List[Dict]:
        """Generate trading signals from analysis"""
        signals = []
        
        # RSI signals
        rsi = analysis['technical_indicators'].get('rsi', {})
        if rsi.get('signal') == 'oversold':
            signals.append({'indicator': 'RSI', 'signal': 'BUY', 'strength': 'medium'})
        elif rsi.get('signal') == 'overbought':
            signals.append({'indicator': 'RSI', 'signal': 'SELL', 'strength': 'medium'})
        
        # MACD signals
        macd = analysis['technical_indicators'].get('macd', {})
        if macd.get('crossover') == 'bullish':
            signals.append({'indicator': 'MACD', 'signal': 'BUY', 'strength': 'high'})
        elif macd.get('crossover') == 'bearish':
            signals.append({'indicator': 'MACD', 'signal': 'SELL', 'strength': 'high'})
        
        # Bollinger Bands signals
        bb = analysis['technical_indicators'].get('bollinger', {})
        if bb.get('position') == 'below':
            signals.append({'indicator': 'Bollinger', 'signal': 'BUY', 'strength': 'medium'})
        elif bb.get('position') == 'above':
            signals.append({'indicator': 'Bollinger', 'signal': 'SELL', 'strength': 'medium'})
        
        return signals
    
    def _calculate_market_score(self, analysis: Dict) -> float:
        """Calculate overall market score"""
        score = 50  # Neutral starting point
        
        # Adjust based on signals
        for signal in analysis.get('signals', []):
            if signal['signal'] == 'BUY':
                score += 10 if signal['strength'] == 'high' else 5
            elif signal['signal'] == 'SELL':
                score -= 10 if signal['strength'] == 'high' else 5
        
        # Adjust based on trend
        trend = analysis.get('trend', {})
        if trend.get('short_term') == 'bullish':
            score += 5
        else:
            score -= 5
        
        # Cap score between 0 and 100
        return max(0, min(100, score))
    
    def _generate_reasoning(self, action: str, strategy: Dict, risk: Dict, sentiment: Dict) -> List[str]:
        """Generate reasoning for recommendation"""
        reasons = []
        
        if action == 'BUY':
            reasons.append("Positive market conditions detected")
            if strategy.get('score', 0) > 0:
                reasons.append("Strong technical indicators support buying")
            if sentiment.get('score', 0) > 0:
                reasons.append("Positive market sentiment")
        elif action == 'SELL':
            reasons.append("Negative market conditions detected")
            if strategy.get('score', 0) < 0:
                reasons.append("Technical indicators suggest selling")
            if risk.get('risk_score', 0) > 70:
                reasons.append("High risk levels detected")
        else:
            reasons.append("Market conditions are neutral")
            reasons.append("Waiting for clearer signals")
        
        return reasons
