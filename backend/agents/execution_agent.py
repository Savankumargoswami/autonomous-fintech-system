"""
Execution Agent - Optimal trade execution strategies
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ExecutionAgent:
    """
    AI agent for optimal trade execution
    """
    
    def __init__(self):
        self.execution_strategies = {
            'market': self._market_order_strategy,
            'limit': self._limit_order_strategy,
            'twap': self._twap_strategy,
            'vwap': self._vwap_strategy,
            'iceberg': self._iceberg_strategy
        }
        
    def optimize_execution(self, order: Dict, market_conditions: Dict) -> Dict:
        """
        Optimize trade execution based on market conditions
        """
        try:
            # Analyze order size and market impact
            impact_analysis = self._analyze_market_impact(order, market_conditions)
            
            # Select optimal execution strategy
            strategy = self._select_strategy(order, impact_analysis)
            
            # Generate execution plan
            execution_plan = self.execution_strategies[strategy](order, market_conditions)
            
            return {
                'strategy': strategy,
                'impact_analysis': impact_analysis,
                'execution_plan': execution_plan,
                'estimated_cost': self._calculate_execution_cost(order, execution_plan),
                'recommendations': self._generate_recommendations(order, impact_analysis)
            }
            
        except Exception as e:
            logger.error(f"Execution optimization error: {e}")
            return {
                'strategy': 'market',
                'error': str(e)
            }
    
    def _analyze_market_impact(self, order: Dict, market_conditions: Dict) -> Dict:
        """Analyze potential market impact of order"""
        order_size = order.get('quantity', 0)
        avg_volume = market_conditions.get('avg_volume', 1000000)
        current_spread = market_conditions.get('spread', 0.01)
        
        # Calculate participation rate
        participation_rate = order_size / avg_volume if avg_volume > 0 else 0
        
        # Estimate price impact (simplified model)
        price_impact = participation_rate * 0.1  # 10% impact per 100% participation
        
        return {
            'participation_rate': round(participation_rate * 100, 2),
            'estimated_price_impact': round(price_impact * 100, 2),
            'current_spread': round(current_spread, 4),
            'liquidity_score': self._calculate_liquidity_score(avg_volume, current_spread)
        }
    
    def _select_strategy(self, order: Dict, impact_analysis: Dict) -> str:
        """Select optimal execution strategy"""
        if impact_analysis['participation_rate'] < 1:
            return 'market'
        elif impact_analysis['participation_rate'] < 5:
            return 'limit'
        elif impact_analysis['participation_rate'] < 10:
            return 'twap'
        elif impact_analysis['participation_rate'] < 20:
            return 'vwap'
        else:
            return 'iceberg'
    
    def _market_order_strategy(self, order: Dict, market_conditions: Dict) -> Dict:
        """Market order execution strategy"""
        return {
            'type': 'market',
            'slices': [order],  # Single execution
            'timing': 'immediate',
            'expected_fill_price': market_conditions.get('current_price', 100)
        }
    
    def _limit_order_strategy(self, order: Dict, market_conditions: Dict) -> Dict:
        """Limit order execution strategy"""
        current_price = market_conditions.get('current_price', 100)
        spread = market_conditions.get('spread', 0.01)
        
        if order['side'] == 'buy':
            limit_price = current_price - (spread / 2)
        else:
            limit_price = current_price + (spread / 2)
        
        return {
            'type': 'limit',
            'slices': [{**order, 'limit_price': round(limit_price, 2)}],
            'timing': 'passive',
            'expected_fill_price': limit_price
        }
    
    def _twap_strategy(self, order: Dict, market_conditions: Dict) -> Dict:
        """Time-Weighted Average Price strategy"""
        num_slices = 10
        slice_size = order['quantity'] / num_slices
        
        slices = []
        for i in range(num_slices):
            slices.append({
                'quantity': slice_size,
                'time_offset': i * 60,  # 1 minute intervals
                'side': order['side']
            })
        
        return {
            'type': 'twap',
            'slices': slices,
            'timing': 'distributed',
            'duration_minutes': num_slices,
            'expected_fill_price': market_conditions.get('current_price', 100)
        }
    
    def _vwap_strategy(self, order: Dict, market_conditions: Dict) -> Dict:
        """Volume-Weighted Average Price strategy"""
        # Use typical volume distribution
        volume_profile = [0.15, 0.10, 0.08, 0.07, 0.10, 0.10, 0.10, 0.08, 0.07, 0.15]
        
        slices = []
        for i, vol_pct in enumerate(volume_profile):
            slices.append({
                'quantity': order['quantity'] * vol_pct,
                'time_offset': i * 30,  # 30 second intervals
                'side': order['side']
            })
        
        return {
            'type': 'vwap',
            'slices': slices,
            'timing': 'volume_based',
            'duration_minutes': 5,
            'expected_fill_price': market_conditions.get('vwap', 100)
        }
    
    def _iceberg_strategy(self, order: Dict, market_conditions: Dict) -> Dict:
        """Iceberg order strategy"""
        visible_size = min(order['quantity'] * 0.1, 1000)  # Show only 10% or 1000 shares
        
        return {
            'type': 'iceberg',
            'total_quantity': order['quantity'],
            'visible_quantity': visible_size,
            'refresh_quantity': visible_size,
            'timing': 'hidden',
            'expected_fill_price': market_conditions.get('current_price', 100)
        }
    
    def _calculate_liquidity_score(self, volume: float, spread: float) -> float:
        """Calculate liquidity score (0-100)"""
        volume_score = min(volume / 10000000 * 100, 100)  # Normalize to 10M volume
        spread_score = max(0, 100 - (spread * 10000))  # Lower spread = higher score
        
        return round((volume_score + spread_score) / 2, 2)
    
    def _calculate_execution_cost(self, order: Dict, execution_plan: Dict) -> Dict:
        """Calculate estimated execution cost"""
        base_commission = 0.001  # 0.1%
        
        if execution_plan['type'] == 'market':
            slippage = 0.0005  # 0.05% slippage
        elif execution_plan['type'] == 'limit':
            slippage = 0  # No slippage for limit orders
        else:
            slippage = 0.0002  # Reduced slippage for algorithmic execution
        
        total_value = order['quantity'] * execution_plan.get('expected_fill_price', 100)
        commission_cost = total_value * base_commission
        slippage_cost = total_value * slippage
        
        return {
            'commission': round(commission_cost, 2),
            'estimated_slippage': round(slippage_cost, 2),
            'total_cost': round(commission_cost + slippage_cost, 2)
        }
    
    def _generate_recommendations(self, order: Dict, impact_analysis: Dict) -> List[str]:
        """Generate execution recommendations"""
        recommendations = []
        
        if impact_analysis['participation_rate'] > 10:
            recommendations.append('Consider splitting order over multiple days')
        
        if impact_analysis['liquidity_score'] < 50:
            recommendations.append('Low liquidity - use limit orders')
        
        if impact_analysis['estimated_price_impact'] > 1:
            recommendations.append('Significant price impact expected - use algorithmic execution')
        
        return recommendations
