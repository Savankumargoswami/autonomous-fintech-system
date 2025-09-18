from .base_agent import BaseAgent
import numpy as np
from typing import Dict, Any, List
import asyncio

class RiskAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.risk_metrics = {}
        self.risk_limits = config.get('risk_limits', {
            'max_portfolio_var': 0.02,  # 2% daily VaR
            'max_position_weight': 0.2,  # 20% max position
            'max_sector_exposure': 0.3,  # 30% max sector exposure
            'min_diversification_ratio': 0.5
        })
    
    async def initialize(self):
        """Initialize risk models and limits"""
        pass
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics from portfolio data"""
        try:
            portfolio_data = data.get('portfolio', {})
            market_data = data.get('market_data', {})
            
            risk_metrics = await self._calculate_risk_metrics(portfolio_data, market_data)
            self.risk_metrics = risk_metrics
            
            return {
                'agent_id': self.agent_id,
                'risk_metrics': risk_metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Risk agent error: {e}")
            return {}
    
    async def _calculate_risk_metrics(self, portfolio: Dict, market_data: Dict) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        if not portfolio or not market_data:
            return {}
        
        positions = portfolio.get('positions', {})
        if not positions:
            return {}
        
        # Portfolio Value at Risk (VaR)
        var_95 = await self._calculate_var(positions, market_data)
        
        # Maximum Drawdown
        max_drawdown = await self._calculate_max_drawdown(portfolio.get('history', []))
        
        # Beta and correlation metrics
        beta = await self._calculate_portfolio_beta(positions, market_data)
        
        # Concentration risk
        concentration_risk = self._calculate_concentration_risk(positions)
        
        # Liquidity risk
        liquidity_score = self._calculate_liquidity_score(positions, market_data)
        
        return {
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'beta': beta,
            'concentration_risk': concentration_risk,
            'liquidity_score': liquidity_score,
            'risk_score': self._calculate_overall_risk_score(var_95, max_drawdown, concentration_risk)
        }
    
    async def _calculate_var(self, positions: Dict, market_data: Dict, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        portfolio_returns = []
        
        for symbol, position in positions.items():
            if symbol in market_data:
                price_data = market_data[symbol].get('close', [])
                if len(price_data) > 1:
                    returns = np.diff(price_data) / price_data[:-1]
                    weighted_returns = returns * position.get('weight', 0)
                    portfolio_returns.extend(weighted_returns)
        
        if not portfolio_returns:
            return 0.0
        
        return float(np.percentile(portfolio_returns, (1 - confidence) * 100))
    
    async def _calculate_max_drawdown(self, portfolio_history: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_history) < 2:
            return 0.0
        
        values = [entry.get('total_value', 0) for entry in portfolio_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    async def _calculate_portfolio_beta(self, positions: Dict, market_data: Dict) -> float:
        """Calculate portfolio beta relative to market"""
        # Simplified beta calculation
        weighted_betas = []
        
        for symbol, position in positions.items():
            weight = position.get('weight', 0)
            # Assume beta of 1.0 for simplification (in production, fetch real betas)
            beta = 1.0
            weighted_betas.append(beta * weight)
        
        return sum(weighted_betas)
    
    def _calculate_concentration_risk(self, positions: Dict) -> float:
        """Calculate concentration risk using Herfindahl index"""
        weights = [pos.get('weight', 0) for pos in positions.values()]
        herfindahl_index = sum(w**2 for w in weights)
        return herfindahl_index
    
    def _calculate_liquidity_score(self, positions: Dict, market_data: Dict) -> float:
        """Calculate portfolio liquidity score"""
        liquidity_scores = []
        
        for symbol, position in positions.items():
            weight = position.get('weight', 0)
            volume = market_data.get(symbol, {}).get('volume', 0)
            
            # Simple liquidity score based on volume
            if volume > 1000000:  # High volume
                score = 1.0
            elif volume > 100000:  # Medium volume
                score = 0.7
            else:  # Low volume
                score = 0.3
            
            liquidity_scores.append(score * weight)
        
        return sum(liquidity_scores)
    
    def _calculate_overall_risk_score(self, var: float, max_dd: float, concentration: float) -> float:
        """Calculate overall risk score (0-1, where 1 is highest risk)"""
        var_score = min(abs(var) * 50, 1.0)  # Scale VaR
        dd_score = min(max_dd, 1.0)
        concentration_score = min(concentration, 1.0)
        
        return (var_score + dd_score + concentration_score) / 3
    
    async def make_decision(self) -> Dict[str, Any]:
        """Make risk management decisions"""
        if not self.risk_metrics:
            return {'action': 'monitor', 'adjustments': {}}
        
        risk_score = self.risk_metrics.get('risk_score', 0)
        adjustments = {}
        
        # Risk limit checks
        if risk_score > 0.8:  # High risk
            adjustments['reduce_exposure'] = 0.3
            action = 'reduce_risk'
        elif risk_score > 0.6:  # Medium risk
            adjustments['hedge_positions'] = 0.2
            action = 'hedge'
        else:
            action = 'monitor'
        
        # Check specific limits
        concentration = self.risk_metrics.get('concentration_risk', 0)
        if concentration > self.risk_limits['max_portfolio_var']:
            adjustments['rebalance'] = True
        
        return {
            'action': action,
            'adjustments': adjustments,
            'risk_score': risk_score,
            'alerts': self._generate_risk_alerts()
        }
    
    def _generate_risk_alerts(self) -> List[str]:
        """Generate risk alerts based on current metrics"""
        alerts = []
        
        if self.risk_metrics.get('risk_score', 0) > 0.8:
            alerts.append("HIGH RISK: Overall risk score exceeds threshold")
        
        if self.risk_metrics.get('max_drawdown', 0) > 0.2:
            alerts.append("WARNING: Maximum drawdown exceeds 20%")
        
        if self.risk_metrics.get('concentration_risk', 0) > 0.5:
            alerts.append("CONCENTRATION: Portfolio highly concentrated")
        
        return alerts

