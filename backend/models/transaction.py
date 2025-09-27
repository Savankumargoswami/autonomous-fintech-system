"""
Transaction model for recording all trading activities
"""
from datetime import datetime, timedelta
from bson import ObjectId
from typing import Dict, List, Optional

class Transaction:
    """Transaction model class"""
    
    def __init__(self, db):
        self.db = db
        self.collection = db.transactions
    
    def create_transaction(self, user_id: str, transaction_data: Dict):
        """Create a new transaction record"""
        transaction = {
            'user_id': user_id,
            'transaction_id': str(ObjectId()),
            'timestamp': datetime.utcnow(),
            'type': transaction_data['type'],  # 'buy', 'sell', 'deposit', 'withdraw'
            'symbol': transaction_data.get('symbol'),
            'quantity': transaction_data.get('quantity', 0),
            'price': transaction_data.get('price', 0),
            'total_value': transaction_data.get('total_value', 0),
            'commission': transaction_data.get('commission', 0),
            'order_type': transaction_data.get('order_type', 'market'),  # 'market', 'limit', 'stop'
            'status': transaction_data.get('status', 'completed'),  # 'pending', 'completed', 'cancelled', 'failed'
            'side': transaction_data.get('side'),  # 'long', 'short'
            'notes': transaction_data.get('notes', ''),
            'metadata': {
                'ip_address': transaction_data.get('ip_address'),
                'user_agent': transaction_data.get('user_agent'),
                'platform': transaction_data.get('platform', 'web'),
                'api_version': transaction_data.get('api_version', 'v1')
            },
            'execution_details': {
                'slippage': transaction_data.get('slippage', 0),
                'execution_time_ms': transaction_data.get('execution_time', 0),
                'fills': transaction_data.get('fills', []),
                'rejected_reason': transaction_data.get('rejected_reason')
            },
            'balance_after': transaction_data.get('balance_after', 0),
            'position_after': transaction_data.get('position_after', {}),
            'risk_metrics': {
                'position_size_pct': transaction_data.get('position_size_pct', 0),
                'portfolio_risk': transaction_data.get('portfolio_risk', 0),
                'var_impact': transaction_data.get('var_impact', 0)
            }
        }
        
        result = self.collection.insert_one(transaction)
        
        # Also update user's transaction history
        self.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {
                '$push': {
                    'portfolio.transactions': {
                        '$each': [transaction],
                        '$position': 0  # Add to beginning
                    }
                }
            }
        )
        
        return transaction['transaction_id']
    
    def get_transactions(self, user_id: str, filters: Dict = None, limit: int = 100):
        """Get transactions for a user with optional filters"""
        query = {'user_id': user_id}
        
        if filters:
            # Add filter conditions
            if 'symbol' in filters:
                query['symbol'] = filters['symbol']
            if 'type' in filters:
                query['type'] = filters['type']
            if 'status' in filters:
                query['status'] = filters['status']
            if 'start_date' in filters:
                query['timestamp'] = {'$gte': filters['start_date']}
            if 'end_date' in filters:
                if 'timestamp' in query:
                    query['timestamp']['$lte'] = filters['end_date']
                else:
                    query['timestamp'] = {'$lte': filters['end_date']}
        
        transactions = list(
            self.collection.find(query)
            .sort('timestamp', -1)
            .limit(limit)
        )
        
        # Convert ObjectId to string for JSON serialization
        for tx in transactions:
            tx['_id'] = str(tx['_id'])
            
        return transactions
    
    def get_transaction_by_id(self, transaction_id: str):
        """Get a specific transaction by ID"""
        transaction = self.collection.find_one({'transaction_id': transaction_id})
        if transaction:
            transaction['_id'] = str(transaction['_id'])
        return transaction
    
    def update_transaction_status(self, transaction_id: str, status: str, details: Dict = None):
        """Update transaction status"""
        update_data = {
            'status': status,
            'updated_at': datetime.utcnow()
        }
        
        if details:
            update_data['execution_details'] = details
        
        return self.collection.update_one(
            {'transaction_id': transaction_id},
            {'$set': update_data}
        )
    
    def cancel_transaction(self, transaction_id: str, reason: str = ''):
        """Cancel a pending transaction"""
        return self.update_transaction_status(
            transaction_id,
            'cancelled',
            {'cancelled_at': datetime.utcnow(), 'cancel_reason': reason}
        )
    
    def get_trading_summary(self, user_id: str, period_days: int = 30):
        """Get trading summary statistics"""
        start_date = datetime.utcnow() - timedelta(days=period_days)
        
        pipeline = [
            {
                '$match': {
                    'user_id': user_id,
                    'timestamp': {'$gte': start_date},
                    'type': {'$in': ['buy', 'sell']}
                }
            },
            {
                '$group': {
                    '_id': '$type',
                    'count': {'$sum': 1},
                    'total_value': {'$sum': '$total_value'},
                    'total_commission': {'$sum': '$commission'},
                    'avg_value': {'$avg': '$total_value'}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        summary = {
            'period_days': period_days,
            'total_transactions': 0,
            'buy_transactions': 0,
            'sell_transactions': 0,
            'total_volume': 0,
            'total_commission': 0,
            'avg_transaction_size': 0
        }
        
        for result in results:
            if result['_id'] == 'buy':
                summary['buy_transactions'] = result['count']
            elif result['_id'] == 'sell':
                summary['sell_transactions'] = result['count']
            
            summary['total_transactions'] += result['count']
            summary['total_volume'] += result['total_value']
            summary['total_commission'] += result['total_commission']
        
        if summary['total_transactions'] > 0:
            summary['avg_transaction_size'] = summary['total_volume'] / summary['total_transactions']
        
        return summary
    
    def get_profit_loss(self, user_id: str, symbol: str = None):
        """Calculate profit/loss from transactions"""
        query = {
            'user_id': user_id,
            'type': {'$in': ['buy', 'sell']},
            'status': 'completed'
        }
        
        if symbol:
            query['symbol'] = symbol
        
        transactions = list(self.collection.find(query).sort('timestamp', 1))
        
        if not transactions:
            return {'realized_pnl': 0, 'trades': []}
        
        trades = []
        positions = {}
        
        for tx in transactions:
            symbol = tx['symbol']
            
            if symbol not in positions:
                positions[symbol] = {'quantity': 0, 'cost_basis': 0}
            
            if tx['type'] == 'buy':
                # Add to position
                positions[symbol]['quantity'] += tx['quantity']
                positions[symbol]['cost_basis'] += tx['total_value']
                
            elif tx['type'] == 'sell':
                # Calculate P&L for this sale
                if positions[symbol]['quantity'] > 0:
                    avg_cost = positions[symbol]['cost_basis'] / positions[symbol]['quantity']
                    sale_cost_basis = avg_cost * tx['quantity']
                    pnl = tx['total_value'] - sale_cost_basis - tx['commission']
                    
                    trades.append({
                        'symbol': symbol,
                        'quantity': tx['quantity'],
                        'buy_price': avg_cost,
                        'sell_price': tx['price'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / sale_cost_basis) * 100 if sale_cost_basis > 0 else 0,
                        'sell_date': tx['timestamp']
                    })
                    
                    # Update position
                    positions[symbol]['quantity'] -= tx['quantity']
                    positions[symbol]['cost_basis'] -= sale_cost_basis
        
        total_pnl = sum(trade['pnl'] for trade in trades)
        
        return {
            'realized_pnl': round(total_pnl, 2),
            'trades': trades,
            'open_positions': positions
        }
    
    def get_daily_transactions(self, user_id: str, date: datetime = None):
        """Get all transactions for a specific day"""
        if date is None:
            date = datetime.utcnow().date()
        else:
            date = date.date()
        
        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = datetime.combine(date, datetime.max.time())
        
        return self.get_transactions(
            user_id,
            filters={
                'start_date': start_of_day,
                'end_date': end_of_day
            }
        )
