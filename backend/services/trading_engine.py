"""
Trading Engine - Core trading execution and management system
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid
import numpy as np
import pandas as pd
from bson import ObjectId

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Main trading engine for order execution and management
    """
    
    def __init__(self, db, redis_client):
        self.db = db
        self.redis = redis_client
        self.active_orders = {}
        self.market_subscriptions = {}
        
    def execute_trade(self, user_id: str, symbol: str, side: str, 
                     quantity: float, order_type: str = 'market', 
                     price: Optional[float] = None) -> Dict:
        """
        Execute a paper trade
        
        Args:
            user_id: User identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Number of shares/units
            order_type: 'market' or 'limit'
            price: Limit price (optional)
            
        Returns:
            Trade execution result
        """
        try:
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Get current market price
            market_price = self._get_market_price(symbol)
            if not market_price:
                return {'status': 'failed', 'error': 'Unable to fetch market price'}
            
            # Calculate execution price
            execution_price = price if order_type == 'limit' and price else market_price
            
            # Check if limit order should execute
            if order_type == 'limit':
                if side == 'buy' and price < market_price:
                    # Limit buy below market - add to pending
                    self._add_pending_order(user_id, order_id, symbol, side, 
                                           quantity, price)
                    return {
                        'status': 'pending',
                        'order_id': order_id,
                        'message': 'Limit order placed'
                    }
                elif side == 'sell' and price > market_price:
                    # Limit sell above market - add to pending
                    self._add_pending_order(user_id, order_id, symbol, side, 
                                           quantity, price)
                    return {
                        'status': 'pending',
                        'order_id': order_id,
                        'message': 'Limit order placed'
                    }
            
            # Get user portfolio
            user = self.db.users.find_one({'_id': ObjectId(user_id)})
            if not user:
                return {'status': 'failed', 'error': 'User not found'}
            
            portfolio = user.get('portfolio', {})
            balance = portfolio.get('balance', 0)
            positions = portfolio.get('positions', [])
            
            # Calculate trade value
            trade_value = execution_price * quantity
            commission = self._calculate_commission(trade_value)
            total_cost = trade_value + commission
            
            # Validate trade
            if side == 'buy':
                if balance < total_cost:
                    return {
                        'status': 'failed',
                        'error': 'Insufficient balance',
                        'required': total_cost,
                        'available': balance
                    }
            else:  # sell
                position = self._get_position(positions, symbol)
                if not position or position['quantity'] < quantity:
                    return {
                        'status': 'failed',
                        'error': 'Insufficient position',
                        'required': quantity,
                        'available': position['quantity'] if position else 0
                    }
            
            # Execute trade
            if side == 'buy':
                # Update balance
                new_balance = balance - total_cost
                
                # Update or create position
                position = self._get_position(positions, symbol)
                if position:
                    # Update existing position
                    avg_price = ((position['avg_price'] * position['quantity']) + 
                                trade_value) / (position['quantity'] + quantity)
                    position['quantity'] += quantity
                    position['avg_price'] = avg_price
                    position['current_price'] = market_price
                    position['market_value'] = position['quantity'] * market_price
                else:
                    # Create new position
                    positions.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'avg_price': execution_price,
                        'current_price': market_price,
                        'market_value': quantity * market_price,
                        'entry_date': datetime.utcnow()
                    })
            else:  # sell
                # Update balance
                new_balance = balance + trade_value - commission
                
                # Update position
                position = self._get_position(positions, symbol)
                position['quantity'] -= quantity
                if position['quantity'] == 0:
                    positions.remove(position)
                else:
                    position['market_value'] = position['quantity'] * market_price
            
            # Create transaction record
            transaction = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': execution_price,
                'value': trade_value,
                'commission': commission,
                'timestamp': datetime.utcnow(),
                'status': 'executed'
            }
            
            # Update portfolio
            portfolio['balance'] = new_balance
            portfolio['positions'] = positions
            portfolio['transactions'] = portfolio.get('transactions', [])
            portfolio['transactions'].append(transaction)
            
            # Update performance metrics
            portfolio['performance'] = self._calculate_performance(portfolio)
            
            # Save to database
            self.db.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'portfolio': portfolio}}
            )
            
            # Cache in Redis
            if self.redis:
                self.redis.setex(
                    f'portfolio:{user_id}',
                    300,  # 5 minute cache
                    json.dumps(portfolio, default=str)
                )
            
            # Log trade
            logger.info(f"Trade executed: {order_id} - {symbol} {side} {quantity}@{execution_price}")
            
            return {
                'status': 'executed',
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': execution_price,
                'value': trade_value,
                'commission': commission,
                'new_balance': new_balance,
                'timestamp': transaction['timestamp'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            # Check Redis cache first
            if self.redis:
                cached_price = self.redis.get(f'price:{symbol}')
                if cached_price:
                    return float(cached_price)
            
            # Simulate market price (in production, fetch from API)
            import random
            base_price = 100.0  # Default base
            volatility = 0.02
            price = base_price * (1 + random.uniform(-volatility, volatility))
            
            # Cache price
            if self.redis:
                self.redis.setex(f'price:{symbol}', 10, str(price))
            
            return price
            
        except Exception as e:
            logger.error(f"Error fetching market price: {e}")
            return None
    
    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate trading commission"""
        # Flat rate + percentage
        flat_fee = 0.0  # No flat fee for paper trading
        percentage = 0.001  # 0.1% commission
        return flat_fee + (trade_value * percentage)
    
    def _get_position(self, positions: List[Dict], symbol: str) -> Optional[Dict]:
        """Get position for symbol"""
        for position in positions:
            if position['symbol'] == symbol:
                return position
        return None
    
    def _add_pending_order(self, user_id: str, order_id: str, 
                          symbol: str, side: str, quantity: float, price: float):
        """Add order to pending orders"""
        pending_order = {
            'order_id': order_id,
            'user_id': user_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.utcnow(),
            'status': 'pending'
        }
        
        # Store in database
        if self.db:
            self.db.pending_orders.insert_one(pending_order)
        
        # Store in active orders
        self.active_orders[order_id] = pending_order
    
    def _calculate_performance(self, portfolio: Dict) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            transactions = portfolio.get('transactions', [])
            if not transactions:
                return {
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Calculate returns
            initial_balance = 100000.0  # Starting balance
            current_value = portfolio['balance']
            for position in portfolio.get('positions', []):
                current_value += position['market_value']
            
            total_return = ((current_value - initial_balance) / initial_balance) * 100
            
            # Calculate win rate
            completed_trades = [t for t in transactions if t.get('side') == 'sell']
            if completed_trades:
                profitable_trades = sum(1 for t in completed_trades 
                                       if self._is_profitable_trade(t, transactions))
                win_rate = (profitable_trades / len(completed_trades)) * 100
            else:
                win_rate = 0.0
            
            # Simple Sharpe ratio calculation
            returns = self._calculate_daily_returns(portfolio)
            if len(returns) > 1:
                sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio)
            
            return {
                'total_return': round(total_return, 2),
                'win_rate': round(win_rate, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2)
            }
            
        except Exception as e:
            logger.error(f"Performance calculation error: {e}")
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
    
    def _is_profitable_trade(self, sell_trade: Dict, all_transactions: List[Dict]) -> bool:
        """Check if a sell trade was profitable"""
        # Find corresponding buy trade
        symbol = sell_trade['symbol']
        buy_trades = [t for t in all_transactions 
                     if t['symbol'] == symbol and t['side'] == 'buy' 
                     and t['timestamp'] < sell_trade['timestamp']]
        
        if buy_trades:
            avg_buy_price = np.mean([t['price'] for t in buy_trades])
            return sell_trade['price'] > avg_buy_price
        return False
    
    def _calculate_daily_returns(self, portfolio: Dict) -> List[float]:
        """Calculate daily returns for Sharpe ratio"""
        # Simplified calculation - in production, use actual daily values
        transactions = portfolio.get('transactions', [])
        if len(transactions) < 2:
            return []
        
        daily_returns = []
        for i in range(1, len(transactions)):
            if transactions[i]['side'] == 'sell' and transactions[i-1]['side'] == 'buy':
                ret = (transactions[i]['price'] - transactions[i-1]['price']) / transactions[i-1]['price']
                daily_returns.append(ret)
        
        return daily_returns
    
    def _calculate_max_drawdown(self, portfolio: Dict) -> float:
        """Calculate maximum drawdown"""
        # Simplified calculation
        transactions = portfolio.get('transactions', [])
        if not transactions:
            return 0.0
        
        values = []
        running_balance = 100000.0
        
        for t in transactions:
            if t['side'] == 'buy':
                running_balance -= t['value']
            else:
                running_balance += t['value']
            values.append(running_balance)
        
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def cancel_order(self, user_id: str, order_id: str) -> Dict:
        """Cancel a pending order"""
        try:
            # Check if order exists and belongs to user
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                if order['user_id'] != user_id:
                    return {'status': 'failed', 'error': 'Unauthorized'}
                
                # Remove from active orders
                del self.active_orders[order_id]
                
                # Update in database
                if self.db:
                    self.db.pending_orders.update_one(
                        {'order_id': order_id},
                        {'$set': {'status': 'cancelled'}}
                    )
                
                return {'status': 'success', 'message': 'Order cancelled'}
            else:
                return {'status': 'failed', 'error': 'Order not found'}
                
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_pending_orders(self, user_id: str) -> List[Dict]:
        """Get all pending orders for user"""
        try:
            if self.db:
                orders = list(self.db.pending_orders.find(
                    {'user_id': user_id, 'status': 'pending'}
                ))
                return orders
            return []
            
        except Exception as e:
            logger.error(f"Error fetching pending orders: {e}")
            return []
    
    def subscribe_to_market(self, user_id: str, symbol: str):
        """Subscribe user to real-time market data"""
        if user_id not in self.market_subscriptions:
            self.market_subscriptions[user_id] = []
        
        if symbol not in self.market_subscriptions[user_id]:
            self.market_subscriptions[user_id].append(symbol)
            logger.info(f"User {user_id} subscribed to {symbol}")
    
    def process_pending_orders(self):
        """Process all pending limit orders"""
        try:
            for order_id, order in self.active_orders.items():
                if order['status'] == 'pending':
                    market_price = self._get_market_price(order['symbol'])
                    
                    # Check if order should execute
                    should_execute = False
                    if order['side'] == 'buy' and market_price <= order['price']:
                        should_execute = True
                    elif order['side'] == 'sell' and market_price >= order['price']:
                        should_execute = True
                    
                    if should_execute:
                        # Execute the order
                        result = self.execute_trade(
                            order['user_id'],
                            order['symbol'],
                            order['side'],
                            order['quantity'],
                            'market'
                        )
                        
                        if result['status'] == 'executed':
                            # Update order status
                            order['status'] = 'executed'
                            if self.db:
                                self.db.pending_orders.update_one(
                                    {'order_id': order_id},
                                    {'$set': {'status': 'executed'}}
                                )
                            
                            logger.info(f"Limit order {order_id} executed")
                            
        except Exception as e:
            logger.error(f"Error processing pending orders: {e}")
