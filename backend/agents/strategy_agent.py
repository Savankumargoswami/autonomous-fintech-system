"""
Strategy Agent - AI-driven trading strategy selection and optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

# Machine learning imports
try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from stable_baselines3 import PPO, SAC, TD3
    import gym
    from gym import spaces
except ImportError as e:
    logging.warning(f"ML libraries not fully installed: {e}")

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.current_balance = initial_balance
        self.shares_held = 0
        self.trades = []
        
        # Action space: [hold, buy, sell] x [0-100% of balance]
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([2, 1]),
            dtype=np.float32
        )
        
        # Observation space: price data + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # 20 features
            dtype=np.float32
        )
    
    def reset(self):
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.shares_held = 0
        self.trades = []
        return self._get_observation()
    
    def step(self, action):
        # Execute action
        action_type = int(action[0])
        amount = action[1]
        
        current_price = self.data.iloc[self.current_step]['close']
        
        reward = 0
        if action_type == 1:  # Buy
            shares_to_buy = int((self.current_balance * amount) / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.current_balance -= cost
                self.shares_held += shares_to_buy
                self.trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'step': self.current_step
                })
        elif action_type == 2:  # Sell
            shares_to_sell = int(self.shares_held * amount)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                self.current_balance += revenue
                self.shares_held -= shares_to_sell
                self.trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'shares': shares_to_sell,
                    'step': self.current_step
                })
                # Calculate reward based on profit
                if len(self.trades) > 1:
                    last_buy_price = next(
                        (t['price'] for t in reversed(self.trades) if t['type'] == 'buy'),
                        current_price
                    )
                    reward = (current_price - last_buy_price) / last_buy_price
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate total portfolio value
        total_value = self.current_balance + (self.shares_held * current_price)
        
        # Additional reward based on portfolio performance
        portfolio_return = (total_value - self.initial_balance) / self.initial_balance
        reward += portfolio_return * 0.01  # Small continuous reward
        
        return self._get_observation(), reward, done, {'total_value': total_value}
    
    def _get_observation(self):
        """Get current market observation"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape[0])
        
        row = self.data.iloc[self.current_step]
        
        # Basic price features
        features = [
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume'],
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('signal', 0),
            row.get('bb_upper', row['close']),
            row.get('bb_lower', row['close']),
            row.get('sma_20', row['close']),
            row.get('sma_50', row['close']),
            row.get('ema_12', row['close']),
            row.get('ema_26', row['close']),
            self.current_balance / self.initial_balance,
            self.shares_held,
            row['close'] if self.shares_held > 0 else 0,
            len(self.trades),
            self.current_step / len(self.data),
            (self.current_balance + self.shares_held * row['close']) / self.initial_balance
        ]
        
        return np.array(features, dtype=np.float32)

class StrategyAgent:
    """
    AI agent for trading strategy selection and optimization
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.models = {}
        self.strategies = {
            'momentum': self._momentum_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'breakout': self._breakout_strategy,
            'trend_following': self._trend_following_strategy,
            'ml_ensemble': self._ml_ensemble_strategy
        }
        self.current_strategy = 'momentum'
        self.performance_history = []
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize RL agents
            if self.model_path:
                # Load pre-trained models if available
                self._load_models()
            else:
                # Initialize with default parameters
                logger.info("Initializing strategy models with default parameters")
                self.scaler = StandardScaler()
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def analyze(self, market_data: Dict) -> Dict:
        """
        Analyze market data and recommend trading strategy
        
        Args:
            market_data: Dictionary containing price data and indicators
            
        Returns:
            Strategy recommendation with confidence scores
        """
        try:
            # Convert market data to DataFrame
            df = self._prepare_data(market_data)
            
            # Evaluate each strategy
            strategy_scores = {}
            for name, strategy_func in self.strategies.items():
                score, signals = strategy_func(df)
                strategy_scores[name] = {
                    'score': score,
                    'signals': signals,
                    'confidence': self._calculate_confidence(score, df)
                }
            
            # Select best strategy
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1]['score'])
            self.current_strategy = best_strategy[0]
            
            # Generate detailed recommendation
            recommendation = self._generate_recommendation(
                best_strategy[0],
                best_strategy[1],
                df
            )
            
            return {
                'selected_strategy': best_strategy[0],
                'all_strategies': strategy_scores,
                'recommendation': recommendation,
                'market_regime': self._detect_market_regime(df),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Strategy analysis error: {e}")
            return {
                'selected_strategy': 'hold',
                'recommendation': {
                    'action': 'hold',
                    'confidence': 0.0,
                    'reason': 'Analysis error'
                }
            }
    
    def _prepare_data(self, market_data: Dict) -> pd.DataFrame:
        """Prepare and validate market data"""
        if isinstance(market_data, pd.DataFrame):
            return market_data
        
        # Convert to DataFrame if dictionary
        df = pd.DataFrame(market_data)
        
        # Add technical indicators if not present
        if 'rsi' not in df.columns:
            df['rsi'] = self._calculate_rsi(df['close'])
        if 'macd' not in df.columns:
            df['macd'], df['signal'] = self._calculate_macd(df['close'])
        if 'bb_upper' not in df.columns:
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        return df
    
    def _momentum_strategy(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Momentum-based trading strategy"""
        signals = {'buy': [], 'sell': []}
        score = 0.0
        
        # Calculate momentum indicators
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
        
        # Price momentum
        returns = df['close'].pct_change()
        momentum = returns.rolling(window=20).mean().iloc[-1]
        
        # Generate signals
        if rsi < 30 and momentum > 0:
            signals['buy'].append('RSI oversold with positive momentum')
            score += 0.3
        elif rsi > 70 and momentum < 0:
            signals['sell'].append('RSI overbought with negative momentum')
            score -= 0.3
        
        if macd > 0:
            signals['buy'].append('MACD positive')
            score += 0.2
        else:
            signals['sell'].append('MACD negative')
            score -= 0.2
        
        # Volume confirmation
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        if volume_ratio > 1.5:
            score *= 1.2  # Amplify signal with high volume
        
        return score, signals
    
    def _mean_reversion_strategy(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Mean reversion trading strategy"""
        signals = {'buy': [], 'sell': []}
        score = 0.0
        
        # Calculate deviation from mean
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        
        current_price = df['close'].iloc[-1]
        mean_price = sma_20.iloc[-1]
        
        z_score = (current_price - mean_price) / std_20.iloc[-1] if std_20.iloc[-1] > 0 else 0
        
        # Generate signals
        if z_score < -2:
            signals['buy'].append('Price significantly below mean')
            score += 0.4
        elif z_score > 2:
            signals['sell'].append('Price significantly above mean')
            score -= 0.4
        
        # Check for support/resistance levels
        support = df['low'].rolling(window=50).min().iloc[-1]
        resistance = df['high'].rolling(window=50).max().iloc[-1]
        
        if current_price <= support * 1.02:
            signals['buy'].append('Near support level')
            score += 0.2
        elif current_price >= resistance * 0.98:
            signals['sell'].append('Near resistance level')
            score -= 0.2
        
        return score, signals
    
    def _breakout_strategy(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Breakout trading strategy"""
        signals = {'buy': [], 'sell': []}
        score = 0.0
        
        # Identify consolidation and breakout
        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        
        current_price = df['close'].iloc[-1]
        prev_high = high_20.iloc[-2] if len(high_20) > 1 else current_price
        prev_low = low_20.iloc[-2] if len(low_20) > 1 else current_price
        
        # Check for breakout
        if current_price > prev_high:
            signals['buy'].append('Bullish breakout detected')
            score += 0.5
        elif current_price < prev_low:
            signals['sell'].append('Bearish breakdown detected')
            score -= 0.5
        
        # Volume confirmation for breakout
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        if current_volume > avg_volume * 1.5 and abs(score) > 0:
            if score > 0:
                signals['buy'].append('Volume confirms breakout')
            else:
                signals['sell'].append('Volume confirms breakdown')
            score *= 1.3
        
        return score, signals
    
    def _trend_following_strategy(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Trend following strategy using moving averages"""
        signals = {'buy': [], 'sell': []}
        score = 0.0
        
        # Calculate moving averages
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean() if len(df) >= 200 else sma_50
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        
        current_price = df['close'].iloc[-1]
        
        # Golden cross / Death cross
        if len(df) >= 200:
            if sma_50.iloc[-1] > sma_200.iloc[-1] and sma_50.iloc[-2] <= sma_200.iloc[-2]:
                signals['buy'].append('Golden cross detected')
                score += 0.6
            elif sma_50.iloc[-1] < sma_200.iloc[-1] and sma_50.iloc[-2] >= sma_200.iloc[-2]:
                signals['sell'].append('Death cross detected')
                score -= 0.6
        
        # EMA crossover
        if ema_12.iloc[-1] > ema_26.iloc[-1]:
            signals['buy'].append('EMA bullish crossover')
            score += 0.2
        else:
            signals['sell'].append('EMA bearish crossover')
            score -= 0.2
        
        # Trend strength
        adx = self._calculate_adx(df) if len(df) > 14 else 25
        if adx > 25:
            score *= 1.2  # Strong trend
        
        return score, signals
    
    def _ml_ensemble_strategy(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Machine learning ensemble strategy"""
        signals = {'buy': [], 'sell': []}
        score = 0.0
        
        try:
            # Combine multiple strategy signals
            strategies_results = []
            for name, strategy_func in self.strategies.items():
                if name != 'ml_ensemble':  # Avoid recursion
                    s, _ = strategy_func(df)
                    strategies_results.append(s)
            
            # Weighted ensemble
            ensemble_score = np.mean(strategies_results)
            
            if ensemble_score > 0.3:
                signals['buy'].append(f'Ensemble score: {ensemble_score:.2f}')
                score = ensemble_score
            elif ensemble_score < -0.3:
                signals['sell'].append(f'Ensemble score: {ensemble_score:.2f}')
                score = ensemble_score
            else:
                signals['hold'] = [f'Neutral ensemble score: {ensemble_score:.2f}']
            
        except Exception as e:
            logger.error(f"ML ensemble error: {e}")
            score = 0.0
        
        return score, signals
    
    def _calculate_confidence(self, score: float, df: pd.DataFrame) -> float:
        """Calculate confidence level for strategy score"""
        # Base confidence on score magnitude
        confidence = min(abs(score), 1.0)
        
        # Adjust based on market volatility
        volatility = df['close'].pct_change().std()
        if volatility > 0.03:  # High volatility
            confidence *= 0.8
        
        # Adjust based on data quality
        if len(df) < 50:
            confidence *= 0.7
        
        return round(confidence, 2)
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        returns = df['close'].pct_change()
        
        # Calculate regime indicators
        volatility = returns.std()
        trend = returns.rolling(window=20).mean().iloc[-1] if len(returns) > 20 else 0
        
        if volatility > 0.03:
            return 'high_volatility'
        elif trend > 0.01:
            return 'bullish_trend'
        elif trend < -0.01:
            return 'bearish_trend'
        else:
            return 'ranging'
    
    def _generate_recommendation(self, strategy: str, strategy_data: Dict, df: pd.DataFrame) -> Dict:
        """Generate detailed trading recommendation"""
        score = strategy_data['score']
        confidence = strategy_data['confidence']
        
        # Determine action
        if score > 0.3:
            action = 'buy'
            size = min(score * 0.5, 0.3)  # Max 30% position size
        elif score < -0.3:
            action = 'sell'
            size = min(abs(score) * 0.5, 0.3)
        else:
            action = 'hold'
            size = 0.0
        
        # Risk management
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)
        
        stop_loss = current_price - (2 * atr) if action == 'buy' else current_price + (2 * atr)
        take_profit = current_price + (3 * atr) if action == 'buy' else current_price - (3 * atr)
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': round(size, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'strategy': strategy,
            'reasons': strategy_data['signals'],
            'risk_reward_ratio': 1.5
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return upper, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else (high.iloc[-1] - low.iloc[-1])
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = self._calculate_atr(df, 1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (abs(minus_dm.rolling(period).mean()) / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean().iloc[-1]
        
        return adx if not pd.isna(adx) else 25
    
    def train(self, historical_data: pd.DataFrame, epochs: int = 100):
        """Train the strategy agent on historical data"""
        try:
            logger.info(f"Training strategy agent with {len(historical_data)} data points")
            
            # Create trading environment
            env = TradingEnvironment(historical_data)
            
            # Train RL agent
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=epochs * len(historical_data))
            
            # Save model
            if self.model_path:
                model.save(f"{self.model_path}/strategy_agent_ppo")
                logger.info("Strategy model saved successfully")
            
            # Store trained model
            self.models['ppo'] = model
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load saved models if they exist
            pass
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
