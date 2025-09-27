"""
User model for MongoDB
"""
from datetime import datetime
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

class User:
    """User model class"""
    
    def __init__(self, db):
        self.db = db
        self.collection = db.users
    
    def create_user(self, username, email, password):
        """Create a new user"""
        user_data = {
            'username': username,
            'email': email,
            'password': generate_password_hash(password),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'is_active': True,
            'portfolio': {
                'balance': 100000.0,  # Starting paper trading balance
                'initial_balance': 100000.0,
                'positions': [],
                'transactions': [],
                'performance': {
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }
            },
            'settings': {
                'risk_tolerance': 'medium',
                'trading_mode': 'paper',
                'notifications': True,
                'two_factor': False
            },
            'api_keys': {
                'paper_trading': True,
                'live_trading': False
            }
        }
        
        result = self.collection.insert_one(user_data)
        return str(result.inserted_id)
    
    def find_by_username(self, username):
        """Find user by username"""
        return self.collection.find_one({'username': username})
    
    def find_by_email(self, email):
        """Find user by email"""
        return self.collection.find_one({'email': email})
    
    def find_by_id(self, user_id):
        """Find user by ID"""
        return self.collection.find_one({'_id': ObjectId(user_id)})
    
    def update_user(self, user_id, update_data):
        """Update user data"""
        update_data['updated_at'] = datetime.utcnow()
        return self.collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': update_data}
        )
    
    def verify_password(self, user, password):
        """Verify user password"""
        return check_password_hash(user['password'], password)
    
    def update_portfolio(self, user_id, portfolio_data):
        """Update user portfolio"""
        return self.collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'portfolio': portfolio_data, 'updated_at': datetime.utcnow()}}
        )
    
    def add_transaction(self, user_id, transaction):
        """Add a transaction to user's history"""
        return self.collection.update_one(
            {'_id': ObjectId(user_id)},
            {
                '$push': {'portfolio.transactions': transaction},
                '$set': {'updated_at': datetime.utcnow()}
            }
        )
    
    def update_balance(self, user_id, new_balance):
        """Update user's account balance"""
        return self.collection.update_one(
            {'_id': ObjectId(user_id)},
            {
                '$set': {
                    'portfolio.balance': new_balance,
                    'updated_at': datetime.utcnow()
                }
            }
        )
    
    def get_portfolio(self, user_id):
        """Get user's portfolio"""
        user = self.find_by_id(user_id)
        return user.get('portfolio', {}) if user else None
    
    def delete_user(self, user_id):
        """Delete a user (soft delete)"""
        return self.collection.update_one(
            {'_id': ObjectId(user_id)},
            {
                '$set': {
                    'is_active': False,
                    'deleted_at': datetime.utcnow()
                }
            }
        )
