"""
Risk Manager - Portfolio risk management and monitoring
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from bson import ObjectId

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Comprehensive risk management system for portfolio protection
    """
    
    def __init__(self, db):
        self.db = db
        self.risk_limits = {
            'max_position_size': 0.2,  # 20% of portfolio
            'max_daily_loss': 0.05,     # 5% daily loss limit
            'max_leverage': 1.0,         # No leverage for paper trading
            'stop_loss_percentage': 0.05,  # 5% stop loss
            'take_profit_percentage': 0.1,  # 10% take profit
            'max_open_positions': 10,
            'min_trade_amount': 100,
            'max_trade_amount': 50000
        }
        
    def assess_trade_risk(self, user_id: str, symbol: str, 
                         side: str, quantity: float, price: float) -> Dict:
        """
        Assess risk for a proposed trade
        
        Returns:
            Risk assessment with approval status and warnings
        """
        try:
            # Get user portfolio
            user = self.db.users.find_one({'_id': ObjectId(user_id)})
            if not user:
                return {'approved': False, 'error': 'User not found'}
            
            portfolio = user.get('portfolio', {})
            balance = portfolio.get('balance', 0)
            positions = portfolio.get('positions', [])
            
            # Calculate trade value
            trade_value = price * quantity
            
            # Calculate portfolio value
            total_portfolio_value = balance
            for position in positions:
                total_portfolio_value += position.get('market_value', 0)
            
            # Risk checks
            risk_assessment = {
                'approved': True,
                'warnings': [],
                'risk_score': 0,
                'recommendations': []
            }
            
            # Check 1: Position size limit
            position_percentage = trade_value / total_portfolio_value if total_portfolio_value > 0 else 0
            if position_percentage > self.risk_limits['max_position_size']:
                risk_assessment['approved'] = False
                risk_assessment['warnings'].append(
                    f"Position size ({position_percentage:.1%}) exceeds limit ({self.risk_limits['max_position_size']:.1%})"
                )
                risk_assessment['risk_score'] += 50
            
            # Check 2: Daily loss limit
            daily_loss = self._calculate_daily_loss(portfolio)
            if daily_loss > self.risk_limits['max_daily_loss']:
                risk_assessment['approved'] = False
                risk_assessment['warnings'].append(
                    f"Daily loss limit reached ({daily_loss:.1%})"
                )
                risk_assessment['risk_score'] += 40
            
            # Check 3: Number of open positions
            if len(positions) >= self.risk_limits['max_open_positions']:
                risk_assessment['warnings'].append(
                    f"Maximum open positions reached ({self.risk_limits['max_open_positions']})"
                )
                risk_assessment['risk_score'] += 20
            
            # Check 4: Concentration risk
            concentration = self._calculate_concentration_risk(positions, symbol, trade_value)
            if concentration > 0.3:  # 30% in single asset
                risk_assessment['warnings'].append(
                    f"High concentration risk in {symbol} ({concentration:.1%})"
                )
                risk_assessment['risk_score'] += 30
            
            # Check 5: Volatility risk
            volatility_risk = self._assess_volatility_risk(symbol)
            if volatility_risk > 0.7:
                risk_assessment['warnings'].append(
                    f"High volatility detected for {symbol}"
                )
                risk_assessment['risk_score'] += 25
            
            # Check 6: Correlation risk
            correlation_risk = self._assess_correlation_risk(positions, symbol)
            if correlation_risk > 0.8:
                risk_assessment['warnings'].append(
                    "High correlation with existing positions"
                )
                risk_assessment['risk_score'] += 15
            
            # Generate recommendations
            risk_assessment['recommendations'] = self._generate_risk_recommendations(
                risk_assessment['risk_score'],
                position_percentage,
                volatility_risk
            )
            
            # Calculate suggested position size
            risk_assessment['suggested_position_size'] = self._calculate_safe_position_size(
                total_portfolio_value,
                volatility_risk,
                len(positions)
            )
            
            # Add risk metrics
            risk_assessment['metrics'] = {
                'position_size_pct': round(position_percentage * 100, 2),
                'daily_loss_pct': round(daily_loss * 100, 2),
                'concentration_risk': round(concentration * 100, 2),
                'volatility_score': round(volatility_risk, 2),
                'correlation_score': round(correlation_risk, 2),
                'overall_risk_score': min(risk_assessment['risk_score'], 100)
            }
            
            # Set stop loss and take profit levels
            if side == 'buy':
                risk_assessment['suggested_stop_loss'] = price * (1 - self.risk_limits['stop_loss_percentage'])
                risk_assessment['suggested_take_profit'] = price * (1 + self.risk_limits['take_profit_percentage'])
            else:
                risk_assessment['suggested_stop_loss'] = price * (1 + self.risk_limits['stop_loss_percentage'])
                risk_assessment['suggested_take_profit'] = price * (1 - self.risk_limits['take_profit_percentage'])
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                'approved': False,
                'error': str(e),
                'risk_score': 100
            }
    
    def monitor_portfolio_risk(self, user_id: str) -> Dict:
        """
        Monitor overall portfolio risk metrics
        """
        try:
            user = self.db.users.find_one({'_id': ObjectId(user_id)})
            if not user:
                return {'error': 'User not found'}
            
            portfolio = user.get('portfolio', {})
            positions = portfolio.get('positions', [])
            transactions = portfolio.get('transactions', [])
            
            # Calculate portfolio metrics
            metrics = {
                'total_value': portfolio.get('balance', 0),
                'positions_value': 0,
                'var_95': 0,  # Value at Risk
                'expected_shortfall': 0,
                'beta': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'current_drawdown': 0
            }
            
            # Calculate positions value
            for position in positions:
                metrics['positions_value'] += position.get('market_value', 0)
            
            metrics['total_value'] += metrics['positions_value']
            
            # Calculate VaR (95% confidence)
            if transactions:
                returns = self._calculate_returns_from_transactions(transactions)
                if len(returns) > 1:
                    metrics['var_95'] = np.percentile(returns, 5) * metrics['total_value']
                    metrics['expected_shortfall'] = np.mean([r for r in returns if r < np.percentile(returns, 5)]) * metrics['total_value']
            
            # Calculate risk ratios
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(transactions)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(transactions)
            metrics['max_drawdown'] = self._calculate_max_drawdown(transactions)
            
            # Risk alerts
            alerts = []
            
            if metrics['max_drawdown'] > 15:
                alerts.append({
                    'level': 'warning',
                    'message': f"High drawdown detected: {metrics['max_drawdown']:.1f}%"
                })
            
            if metrics['sharpe_ratio'] < 0:
                alerts.append({
                    'level': 'info',
                    'message': "Negative Sharpe ratio - consider adjusting strategy"
                })
            
            # Position-level risks
            position_risks = []
            for position in positions:
                risk = self._assess_position_risk(position)
                position_risks.append(risk)
                
                if risk['unrealized_loss_pct'] > 10:
                    alerts.append({
                        'level': 'warning',
                        'message': f"{position['symbol']}: Unrealized loss {risk['unrealized_loss_pct']:.1f}%"
                    })
            
            return {
                'metrics': metrics,
                'position_risks': position_risks,
                'alerts': alerts,
                'risk_status': self._determine_risk_status(metrics, alerts),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio monitoring error: {e}")
            return {'error': str(e)}
    
    def _calculate_daily_loss(self, portfolio: Dict) -> float:
        """Calculate today's loss percentage"""
        transactions = portfolio.get('transactions', [])
        if not transactions:
            return 0.0
        
        today = datetime.utcnow().date()
        daily_pnl = 0
        initial_balance = 100000  # Starting balance
        
        for transaction in transactions:
            if transaction.get('timestamp'):
                tx_date = transaction['timestamp'].date() if hasattr(transaction['timestamp'], 'date') else datetime.fromisoformat(str(transaction['timestamp'])).date()
                if tx_date == today:
                    if transaction['side'] == 'sell':
                        # Calculate P&L for sells
                        daily_pnl += transaction.get('value', 0) - transaction.get('commission', 0)
                    else:
                        daily_pnl -= transaction.get('value', 0) + transaction.get('commission', 0)
        
        return abs(min(daily_pnl / initial_balance, 0))
    
    def _calculate_concentration_risk(self, positions: List[Dict], 
                                     symbol: str, trade_value: float) -> float:
        """Calculate concentration risk for a symbol"""
        total_value = sum(p.get('market_value', 0) for p in positions)
        symbol_value = sum(p.get('market_value', 0) for p in positions if p.get('symbol') == symbol)
        
        new_concentration = (symbol_value + trade_value) / (total_value + trade_value) if (total_value + trade_value) > 0 else 0
        return new_concentration
    
    def _assess_volatility_risk(self, symbol: str) -> float:
        """Assess volatility risk for a symbol"""
        # In production, fetch actual volatility data
        # For now, return simulated volatility score (0-1)
        import random
        return random.uniform(0.2, 0.8)
    
    def _assess_correlation_risk(self, positions: List[Dict], symbol: str) -> float:
        """Assess correlation risk with existing positions"""
        # In production, calculate actual correlations
        # For now, return simulated correlation score (0-1)
        if not positions:
            return 0.0
        return min(len(positions) * 0.1, 0.9)
    
    def _generate_risk_recommendations(self, risk_score: float, 
                                      position_pct: float, volatility: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_score > 70:
            recommendations.append("Consider reducing position size")
        
        if position_pct > 0.15:
            recommendations.append(f"Position size is large ({position_pct:.1%}), consider scaling in")
        
        if volatility > 0.6:
            recommendations.append("High volatility - use tighter stop losses")
            recommendations.append("Consider reducing position size due to volatility")
        
        if risk_score < 30:
            recommendations.append("Risk level acceptable for this trade")
        
        return recommendations
    
    def _calculate_safe_position_size(self, portfolio_value: float, 
                                     volatility: float, num_positions: int) -> float:
        """Calculate safe position size based on Kelly Criterion"""
        # Simplified Kelly Criterion
        base_size = portfolio_value * self.risk_limits['max_position_size']
        
        # Adjust for volatility
        volatility_adjustment = 1 - (volatility * 0.5)
        
        # Adjust for number of positions
        diversification_adjustment = max(0.5, 1 - (num_positions * 0.05))
        
        safe_size = base_size * volatility_adjustment * diversification_adjustment
        
        return round(safe_size, 2)
    
    def _calculate_returns_from_transactions(self, transactions: List[Dict]) -> List[float]:
        """Calculate returns from transaction history"""
        returns = []
        
        for i in range(1, len(transactions)):
            if transactions[i].get('side') == 'sell' and transactions[i-1].get('side') == 'buy':
                if transactions[i].get('symbol') == transactions[i-1].get('symbol'):
                    ret = (transactions[i].get('price', 0) - transactions[i-1].get('price', 0)) / transactions[i-1].get('price', 1)
                    returns.append(ret)
        
        return returns
    
    def _calculate_sharpe_ratio(self, transactions: List[Dict]) -> float:
        """Calculate Sharpe ratio from transactions"""
        returns = self._calculate_returns_from_transactions(transactions)
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        
        if std_return == 0:
            return 0.0
        
        sharpe = (avg_return - risk_free_rate) / std_return * np.sqrt(252)
        return round(sharpe, 2)
    
    def _calculate_sortino_ratio(self, transactions: List[Dict]) -> float:
        """Calculate Sortino ratio (downside risk adjusted)"""
        returns = self._calculate_returns_from_transactions(transactions)
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return 0.0
        
        downside_std = np.std(downside_returns)
        risk_free_rate = 0.02 / 252
        
        if downside_std == 0:
            return 0.0
        
        sortino = (avg_return - risk_free_rate) / downside_std * np.sqrt(252)
        return round(sortino, 2)
    
    def _calculate_max_drawdown(self, transactions: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        if not transactions:
            return 0.0
        
        values = []
        running_value = 100000  # Starting balance
        
        for t in transactions:
            if t.get('side') == 'buy':
                running_value -= t.get('value', 0)
            else:
                running_value += t.get('value', 0)
            values.append(running_value)
        
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return round(max_dd, 2)
    
    def _assess_position_risk(self, position: Dict) -> Dict:
        """Assess risk for individual position"""
        current_value = position.get('market_value', 0)
        avg_price = position.get('avg_price', 0)
        current_price = position.get('current_price', avg_price)
        quantity = position.get('quantity', 0)
        
        cost_basis = avg_price * quantity
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        return {
            'symbol': position.get('symbol'),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'unrealized_pnl_pct': round(unrealized_pnl_pct, 2),
            'unrealized_loss_pct': abs(min(unrealized_pnl_pct, 0)),
            'position_size': current_value,
            'days_held': (datetime.utcnow() - position.get('entry_date', datetime.utcnow())).days if position.get('entry_date') else 0
        }
    
    def _determine_risk_status(self, metrics: Dict, alerts: List[Dict]) -> str:
        """Determine overall risk status"""
        if any(a['level'] == 'critical' for a in alerts):
            return 'critical'
        elif any(a['level'] == 'warning' for a in alerts):
            return 'warning'
        elif metrics.get('sharpe_ratio', 0) < 0:
            return 'caution'
        else:
            return 'normal'
    
    def get_risk_limits(self) -> Dict:
        """Get current risk limits"""
        return self.risk_limits
    
    def update_risk_limits(self, new_limits: Dict) -> bool:
        """Update risk limits"""
        try:
            self.risk_limits.update(new_limits)
            return True
        except Exception as e:
            logger.error(f"Failed to update risk limits: {e}")
            return False
