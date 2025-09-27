"""
Risk Agent - AI-driven risk assessment and management
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class RiskAgent:
    """
    AI agent for risk assessment and management
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'var_95': 0.05,  # 5% VaR threshold
            'max_drawdown': 0.20,  # 20% max drawdown
            'sharpe_min': 0.5,  # Minimum Sharpe ratio
            'correlation_max': 0.8  # Maximum correlation
        }
        
    def assess_risk(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """
        Assess risk for a symbol
        """
        try:
            risk_assessment = {
                'symbol': symbol,
                'risk_score': 50,  # Base score
                'risk_level': 'medium',
                'factors': {},
                'warnings': [],
                'recommendations': []
            }
            
            # Calculate volatility risk
            volatility = self._calculate_volatility(market_data)
            risk_assessment['factors']['volatility'] = round(volatility * 100, 2)
            
            if volatility > 0.03:  # 3% daily volatility
                risk_assessment['risk_score'] += 20
                risk_assessment['warnings'].append('High volatility detected')
            
            # Calculate drawdown risk
            max_drawdown = self._calculate_max_drawdown(market_data)
            risk_assessment['factors']['max_drawdown'] = round(max_drawdown * 100, 2)
            
            if max_drawdown > self.risk_thresholds['max_drawdown']:
                risk_assessment['risk_score'] += 25
                risk_assessment['warnings'].append(f'Significant drawdown: {max_drawdown:.1%}')
            
            # Calculate VaR
            var_95 = self._calculate_var(market_data, confidence=0.95)
            risk_assessment['factors']['var_95'] = round(var_95 * 100, 2)
            
            if abs(var_95) > self.risk_thresholds['var_95']:
                risk_assessment['risk_score'] += 15
                risk_assessment['warnings'].append(f'High VaR: {var_95:.1%}')
            
            # Determine risk level
            if risk_assessment['risk_score'] >= 80:
                risk_assessment['risk_level'] = 'very_high'
                risk_assessment['recommendations'].append('Consider reducing position size')
            elif risk_assessment['risk_score'] >= 65:
                risk_assessment['risk_level'] = 'high'
                risk_assessment['recommendations'].append('Use tight stop-losses')
            elif risk_assessment['risk_score'] >= 35:
                risk_assessment['risk_level'] = 'medium'
                risk_assessment['recommendations'].append('Monitor position closely')
            else:
                risk_assessment['risk_level'] = 'low'
                risk_assessment['recommendations'].append('Risk within acceptable range')
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                'symbol': symbol,
                'risk_score': 100,
                'risk_level': 'unknown',
                'error': str(e)
            }
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate historical volatility"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.02  # Default volatility
        
        returns = data['close'].pct_change().dropna()
        return returns.std()
    
    def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        cumulative = (1 + data['close'].pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_var(self, data: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        var = np.percentile(returns, (1 - confidence) * 100)
        
        return var
