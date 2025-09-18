import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .base_agent import BaseAgent
import torch
import torch.nn as nn

class StrategyAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.model = None
        self.portfolio_weights = {}
        self.signals = {}
        
    async def initialize(self):
        """Initialize strategy models"""
        self.model = self._build_strategy_model()
        await self._load_historical_data()
    
    def _build_strategy_model(self):
        """Build neural network for strategy decisions"""
        class StrategyNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, output_size),
                    nn.Tanh()  # Output between -1 and 1
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return StrategyNet(input_size=50, hidden_size=256, output_size=10)
    
    async def _load_historical_data(self):
        """Load historical market data for training"""
        # Implement data loading logic
        pass
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate signals"""
        try:
            market_data = data.get('market_data', {})
            features = self._extract_features(market_data)
            
            if len(features) > 0:
                signals = await self._generate_signals(features)
                self.signals = signals
                
            return {
                'agent_id': self.agent_id,
                'signals': self.signals,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Strategy agent error: {e}")
            return {}
    
    def _extract_features(self, market_data: Dict) -> np.ndarray:
        """Extract features from market data"""
        features = []
        
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'close' in data:
                # Technical indicators
                close_prices = np.array(data.get('close', []))
                if len(close_prices) > 20:
                    # Moving averages
                    ma_5 = np.mean(close_prices[-5:])
                    ma_20 = np.mean(close_prices[-20:])
                    
                    # RSI
                    rsi = self._calculate_rsi(close_prices)
                    
                    # Volatility
                    volatility = np.std(close_prices[-20:])
                    
                    features.extend([ma_5, ma_20, rsi, volatility])
        
        return np.array(features[:50])  # Limit to 50 features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    async def _generate_signals(self, features: np.ndarray) -> Dict[str, float]:
        """Generate trading signals using the model"""
        if self.model is None or len(features) == 0:
            return {}
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            output = self.model(input_tensor)
            signals_array = output.squeeze().numpy()
        
        # Map signals to symbols (simplified)
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'CRM']
        signals = {}
        
        for i, symbol in enumerate(symbols):
            if i < len(signals_array):
                signals[symbol] = float(signals_array[i])
        
        return signals
    
    async def make_decision(self) -> Dict[str, Any]:
        """Make portfolio allocation decisions"""
        if not self.signals:
            return {'action': 'hold', 'allocations': {}}
        
        # Simple allocation based on signals
        allocations = {}
        total_signal = sum(abs(signal) for signal in self.signals.values())
        
        if total_signal > 0:
            for symbol, signal in self.signals.items():
                if abs(signal) > 0.1:  # Minimum signal threshold
                    allocation = (signal / total_signal) * 0.8  # Max 80% allocation
                    allocations[symbol] = max(min(allocation, 0.2), -0.2)  # Cap at 20% per position
        
        return {
            'action': 'rebalance' if allocations else 'hold',
            'allocations': allocations,
            'confidence': min(total_signal, 1.0)
        }
