"""
Portfolio model for managing user portfolios
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from bson import ObjectId

class Portfolio:
    """Portfolio model class for managing trading portfolios"""
    
    def __init__(self, db):
        self.db = db
        self.collection = db.portfolios
        
    def create_portfolio(self, user_id: str, initial_balance: float = 100000.0) -> str:
        """Create a new portfolio for a user"""
        portfolio_data = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'balance': {
                'cash': initial_balance,
                'invested': 0.0,
                'total': initial_balance,
                'initial': initial_balance,
                'available': initial_balance
            },
            'positions': [],
            'pending_orders': [],
            'closed_positions': [],
            'watchlist': [],
            'performance': {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'day_change': 0.0,
                'day_change_pct': 0.0,
                'week_change': 0.0,
                'week_change_pct': 0.0,
                'month_change': 0.0,
                'month_change_pct': 0.0,
                'year_change': 0.0,
                'year_change_pct': 0.0,
                'all_time_high': initial_balance,
                'all_time_low': initial_balance,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'avg_holding_period': 0.0,
                'total_commission': 0.0
            },
            'risk_metrics': {
                'portfolio_beta': 0.0,
                'portfolio_alpha': 0.0,
                'value_at_risk': 0.0,
                'conditional_var': 0.0,
                'correlation_spy': 0.0,
                'volatility': 0.0,
                'downside_deviation': 0.0
            },
            'allocation': {
                'by_sector': {},
                'by_asset_class': {},
                'by_region': {},
                'by_market_cap': {}
            },
            'daily_snapshots': [],
            'settings': {
                'risk_tolerance': 'medium',
                'rebalance_frequency': 'monthly',
                'stop_loss_enabled': True,
                'take_profit_enabled': True,
                'max_position_size': 0.2,
                'max_daily_loss': 0.05
            }
        }
        
        result = self.collection.insert_one(portfolio_data)
        return str(result.inserted_id)
    
    def add_position(self, user_id: str, position: Dict) -> Optional[str]:
        """Add a new position to portfolio"""
        try:
            position_data = {
                'id': str(ObjectId()),
                'symbol': position['symbol'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': position.get('current_price', position['entry_price']),
                'entry_date': datetime.utcnow(),
                'position_type': position.get('position_type', 'long'),
                'stop_loss': position.get('stop_loss'),
                'take_profit': position.get('take_profit'),
                'unrealized_pnl': 0.0,
                'unrealized_pnl_pct': 0.0,
                'market_value': position['quantity'] * position['entry_price'],
                'cost_basis': position['quantity'] * position['entry_price'],
                'commission': position.get('commission', 0.0),
                'notes': position.get('notes', ''),
                'tags': position.get('tags', [])
            }
            
            # Calculate total cost
            total_cost = position_data['cost_basis'] + position_data['commission']
            
            # Update portfolio
            result = self.collection.update_one(
                {'user_id': user_id},
                {
                    '$push': {'positions': position_data},
                    '$inc': {
                        'balance.cash': -total_cost,
                        'balance.invested': position_data['cost_basis'],
                        'performance.total_commission': position_data['commission']
                    },
                    '$set': {
                        'updated_at': datetime.utcnow(),
                        'balance.available': {'$subtract': ['$balance.cash', total_cost]}
                    }
                }
            )
            
            if result.modified_count > 0:
                self._update_performance_metrics(user_id)
                return position_data['id']
            return None
            
        except Exception as e:
            print(f"Error adding position: {e}")
            return None
    
    def update_position(self, user_id: str, position_id: str, updates: Dict) -> bool:
        """Update an existing position"""
        try:
            # Get current portfolio
            portfolio = self.collection.find_one({'user_id': user_id})
            if not portfolio:
                return False
            
            positions = portfolio.get('positions', [])
            position_index = None
            
            # Find position
            for i, pos in enumerate(positions):
                if pos['id'] == position_id:
                    position_index = i
                    break
            
            if position_index is None:
                return False
            
            # Update position
            current_position = positions[position_index]
            current_position.update(updates)
            
            # Recalculate metrics
            current_price = updates.get('current_price', current_position['current_price'])
            entry_price = current_position['entry_price']
            quantity = current_position['quantity']
            
            # Calculate unrealized P&L
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            current_position['unrealized_pnl'] = round(unrealized_pnl, 2)
            current_position['unrealized_pnl_pct'] = round(unrealized_pnl_pct, 2)
            current_position['market_value'] = round(current_price * quantity, 2)
            current_position['current_price'] = current_price
            
            # Update in database
            result = self.collection.update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        f'positions.{position_index}': current_position,
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"Error updating position: {e}")
            return False
    
    def close_position(self, user_id: str, position_id: str, 
                      exit_price: float, commission: float = 0) -> Optional[Dict]:
        """Close a position and record the trade"""
        try:
            portfolio = self.collection.find_one({'user_id': user_id})
            if not portfolio:
                return None
            
            positions = portfolio.get('positions', [])
            position_to_close = None
            position_index = None
            
            # Find position
            for i, pos in enumerate(positions):
                if pos['id'] == position_id:
                    position_to_close = pos
                    position_index = i
                    break
            
            if not position_to_close:
                return None
            
            # Calculate realized P&L
            entry_price = position_to_close['entry_price']
            quantity = position_to_close['quantity']
            entry_commission = position_to_close.get('commission', 0)
            
            gross_pnl = (exit_price - entry_price) * quantity
            total_commission = entry_commission + commission
            net_pnl = gross_pnl - total_commission
            pnl_pct = (net_pnl / position_to_close['cost_basis']) * 100 if position_to_close['cost_basis'] > 0 else 0
            
            holding_period = (datetime.utcnow() - position_to_close['entry_date']).days
            
            # Create closed position record
            closed_position = {
                **position_to_close,
                'exit_price': exit_price,
                'exit_date': datetime.utcnow(),
                'holding_period_days': holding_period,
                'realized_pnl': round(net_pnl, 2),
                'realized_pnl_pct': round(pnl_pct, 2),
                'gross_pnl': round(gross_pnl, 2),
                'exit_commission': commission,
                'total_commission': total_commission,
                'trade_result': 'win' if net_pnl > 0 else 'loss'
            }
            
            # Remove from positions
            positions.pop(position_index)
            
            # Calculate exit value
            exit_value = (exit_price * quantity) - commission
            
            # Update portfolio
            update_result = self.collection.update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        'positions': positions,
                        'updated_at': datetime.utcnow()
                    },
                    '$push': {
                        'closed_positions': closed_position
                    },
                    '$inc': {
                        'balance.cash': exit_value,
                        'balance.invested': -position_to_close['cost_basis'],
                        'performance.total_trades': 1,
                        'performance.winning_trades': 1 if net_pnl > 0 else 0,
                        'performance.losing_trades': 1 if net_pnl <= 0 else 0,
                        'performance.total_commission': commission
                    }
                }
            )
            
            if update_result.modified_count > 0:
                self._update_performance_metrics(user_id)
                return closed_position
            
            return None
            
        except Exception as e:
            print(f"Error closing position: {e}")
            return None
    
    def _update_performance_metrics(self, user_id: str):
        """Update portfolio performance metrics"""
        try:
            portfolio = self.collection.find_one({'user_id': user_id})
            if not portfolio:
                return
            
            closed_positions = portfolio.get('closed_positions', [])
            if not closed_positions:
                return
            
            # Calculate performance metrics
            total_pnl = sum(pos['realized_pnl'] for pos in closed_positions)
            winning_trades = [pos for pos in closed_positions if pos['realized_pnl'] > 0]
            losing_trades = [pos for pos in closed_positions if pos['realized_pnl'] <= 0]
            
            # Basic metrics
            metrics = {
                'total_return': total_pnl,
                'total_return_pct': (total_pnl / portfolio['balance']['initial']) * 100,
                'total_trades': len(closed_positions),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades)
            }
            
            # Win rate
            if closed_positions:
                metrics['win_rate'] = (len(winning_trades) / len(closed_positions)) * 100
            else:
                metrics['win_rate'] = 0
            
            # Average win/loss
            if winning_trades:
                metrics['avg_win'] = np.mean([t['realized_pnl'] for t in winning_trades])
            else:
                metrics['avg_win'] = 0
                
            if losing_trades:
                metrics['avg_loss'] = np.mean([t['realized_pnl'] for t in losing_trades])
            else:
                metrics['avg_loss'] = 0
            
            # Best and worst trades
            if closed_positions:
                metrics['best_trade'] = max(pos['realized_pnl'] for pos in closed_positions)
                metrics['worst_trade'] = min(pos['realized_pnl'] for pos in closed_positions)
            else:
                metrics['best_trade'] = 0
                metrics['worst_trade'] = 0
            
            # Profit factor
            if losing_trades:
                gross_profit = sum(t['realized_pnl'] for t in winning_trades) if winning_trades else 0
                gross_loss = abs(sum(t['realized_pnl'] for t in losing_trades))
                metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
            else:
                metrics['profit_factor'] = 0
            
            # Average holding period
            if closed_positions:
                holding_periods = [pos.get('holding_period_days', 0) for pos in closed_positions]
                metrics['avg_holding_period'] = np.mean(holding_periods)
            else:
                metrics['avg_holding_period'] = 0
            
            # Calculate Sharpe ratio
            if len(closed_positions) > 1:
                returns = [pos['realized_pnl_pct'] / 100 for pos in closed_positions]
                if np.std(returns) > 0:
                    metrics['sharpe_ratio'] = (np.mean(returns) - 0.02/252) / np.std(returns) * np.sqrt(252)
                else:
                    metrics['sharpe_ratio'] = 0
            
            # Calculate max drawdown
            metrics['max_drawdown'] = self._calculate_max_drawdown(closed_positions)
            
            # Update portfolio performance
            self.collection.update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        f'performance.{key}': round(value, 2) if isinstance(value, float) else value
                        for key, value in metrics.items()
                    }
                }
            )
            
        except Exception as e:
            print(f"Error updating performance metrics: {e}")
    
    def _calculate_max_drawdown(self, closed_positions: List[Dict]) -> float:
        """Calculate maximum drawdown from closed positions"""
        if not closed_positions:
            return 0.0
        
        cumulative_pnl = []
        running_total = 0
        
        for pos in sorted(closed_positions, key=lambda x: x.get('exit_date', datetime.utcnow())):
            running_total += pos.get('realized_pnl', 0)
            cumulative_pnl.append(running_total)
        
        if not cumulative_pnl:
            return 0.0
        
        peak = cumulative_pnl[0]
        max_dd = 0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            drawdown = ((peak - value) / abs(peak)) * 100 if peak != 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return round(max_dd, 2)
    
    def get_portfolio(self, user_id: str) -> Optional[Dict]:
        """Get complete portfolio data for a user"""
        return self.collection.find_one({'user_id': user_id})
    
    def get_positions(self, user_id: str) -> List[Dict]:
        """Get all open positions for a user"""
        portfolio = self.collection.find_one({'user_id': user_id})
        return portfolio.get('positions', []) if portfolio else []
    
    def get_performance(self, user_id: str) -> Dict:
        """Get portfolio performance metrics"""
        portfolio = self.collection.find_one({'user_id': user_id})
        return portfolio.get('performance', {}) if portfolio else {}
    
    def get_closed_positions(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get closed positions history"""
        portfolio = self.collection.find_one({'user_id': user_id})
        if portfolio:
            closed = portfolio.get('closed_positions', [])
            # Sort by exit date (most recent first)
            closed.sort(key=lambda x: x.get('exit_date', datetime.min), reverse=True)
            return closed[:limit]
        return []
    
    def add_to_watchlist(self, user_id: str, symbol: str) -> bool:
        """Add symbol to watchlist"""
        result = self.collection.update_one(
            {'user_id': user_id},
            {
                '$addToSet': {'watchlist': symbol},
                '$set': {'updated_at': datetime.utcnow()}
            }
        )
        return result.modified_count > 0
    
    def remove_from_watchlist(self, user_id: str, symbol: str) -> bool:
        """Remove symbol from watchlist"""
        result = self.collection.update_one(
            {'user_id': user_id},
            {
                '$pull': {'watchlist': symbol},
                '$set': {'updated_at': datetime.utcnow()}
            }
        )
        return result.modified_count > 0
    
    def get_watchlist(self, user_id: str) -> List[str]:
        """Get user's watchlist"""
        portfolio = self.collection.find_one({'user_id': user_id})
        return portfolio.get('watchlist', []) if portfolio else []
    
    def take_daily_snapshot(self, user_id: str):
        """Take a daily snapshot of portfolio value"""
        try:
            portfolio = self.collection.find_one({'user_id': user_id})
            if not portfolio:
                return
            
            # Calculate total portfolio value
            total_value = portfolio['balance']['cash']
            for position in portfolio.get('positions', []):
                total_value += position.get('market_value', 0)
            
            snapshot = {
                'date': datetime.utcnow().date(),
                'total_value': total_value,
                'cash_balance': portfolio['balance']['cash'],
                'invested_value': portfolio['balance']['invested'],
                'num_positions': len(portfolio.get('positions', [])),
                'daily_pnl': 0  # Calculate from previous snapshot
            }
            
            # Calculate daily P&L if previous snapshot exists
            snapshots = portfolio.get('daily_snapshots', [])
            if snapshots:
                prev_value = snapshots[-1].get('total_value', total_value)
                snapshot['daily_pnl'] = total_value - prev_value
            
            # Add snapshot
            self.collection.update_one(
                {'user_id': user_id},
                {
                    '$push': {
                        'daily_snapshots': {
                            '$each': [snapshot],
                            '$slice': -365  # Keep last 365 days
                        }
                    }
                }
            )
            
        except Exception as e:
            print(f"Error taking daily snapshot: {e}")
    
    def get_portfolio_value_history(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get portfolio value history"""
        portfolio = self.collection.find_one({'user_id': user_id})
        if portfolio:
            snapshots = portfolio.get('daily_snapshots', [])
            return snapshots[-days:] if len(snapshots) > days else snapshots
        return []
