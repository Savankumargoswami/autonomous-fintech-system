"""
Portfolio management routes
"""
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from bson import ObjectId
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('portfolio', __name__)

@bp.route('/', methods=['GET'])
@jwt_required()
def get_portfolio():
    """Get complete portfolio data"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        db = current_app.config.get('db')
        
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        # Get user portfolio
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        portfolio = user.get('portfolio', {})
        
        # Calculate current total value
        total_value = portfolio.get('balance', 0)
        for position in portfolio.get('positions', []):
            total_value += position.get('market_value', 0)
        
        # Format response
        response = {
            'balance': portfolio.get('balance', 0),
            'total_value': total_value,
            'positions': portfolio.get('positions', []),
            'performance': portfolio.get('performance', {}),
            'initial_balance': 100000.0
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Portfolio fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/performance', methods=['GET'])
@jwt_required()
def get_performance():
    """Get detailed portfolio performance metrics"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        db = current_app.config.get('db')
        
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        portfolio = user.get('portfolio', {})
        performance = portfolio.get('performance', {})
        
        # Calculate additional metrics
        balance = portfolio.get('balance', 0)
        positions_value = sum(p.get('market_value', 0) for p in portfolio.get('positions', []))
        total_value = balance + positions_value
        initial_balance = 100000.0
        
        # Add calculated metrics
        performance['current_value'] = total_value
        performance['total_return_amount'] = total_value - initial_balance
        performance['total_return_pct'] = ((total_value - initial_balance) / initial_balance) * 100
        
        return jsonify(performance), 200
        
    except Exception as e:
        logger.error(f"Performance fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/transactions', methods=['GET'])
@jwt_required()
def get_transactions():
    """Get transaction history"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        db = current_app.config.get('db')
        
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        symbol = request.args.get('symbol')
        
        # Build query
        query = {'user_id': user_id}
        if symbol:
            query['symbol'] = symbol
        
        # Get transactions
        transactions = list(
            db.transactions.find(query)
            .sort('timestamp', -1)
            .skip(offset)
            .limit(limit)
        )
        
        # Format transactions
        formatted_transactions = []
        for tx in transactions:
            formatted_transactions.append({
                'transaction_id': str(tx.get('transaction_id', '')),
                'timestamp': tx.get('timestamp').isoformat() if tx.get('timestamp') else None,
                'type': tx.get('type'),
                'symbol': tx.get('symbol'),
                'quantity': tx.get('quantity'),
                'price': tx.get('price'),
                'value': tx.get('total_value'),
                'commission': tx.get('commission'),
                'status': tx.get('status')
            })
        
        return jsonify(formatted_transactions), 200
        
    except Exception as e:
        logger.error(f"Transactions fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/summary', methods=['GET'])
@jwt_required()
def get_summary():
    """Get portfolio summary with risk metrics"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        
        # Get risk manager
        risk_manager = current_app.config.get('risk_manager')
        if not risk_manager:
            return jsonify({'error': 'Risk manager not available'}), 500
        
        # Get portfolio risk summary
        summary = risk_manager.monitor_portfolio_risk(user_id)
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Summary fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/history', methods=['GET'])
@jwt_required()
def get_portfolio_history():
    """Get portfolio value history"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        db = current_app.config.get('db')
        
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        # Get period from query params
        period = request.args.get('period', '30d')
        
        # Calculate date range
        if period == '7d':
            start_date = datetime.utcnow() - timedelta(days=7)
        elif period == '30d':
            start_date = datetime.utcnow() - timedelta(days=30)
        elif period == '90d':
            start_date = datetime.utcnow() - timedelta(days=90)
        elif period == '1y':
            start_date = datetime.utcnow() - timedelta(days=365)
        else:
            start_date = datetime.utcnow() - timedelta(days=30)
        
        # Get transactions in period
        transactions = list(
            db.transactions.find({
                'user_id': user_id,
                'timestamp': {'$gte': start_date}
            }).sort('timestamp', 1)
        )
        
        # Calculate portfolio value over time
        history = []
        running_balance = 100000.0  # Initial balance
        
        for tx in transactions:
            if tx['type'] == 'buy':
                running_balance -= tx.get('total_value', 0) + tx.get('commission', 0)
            elif tx['type'] == 'sell':
                running_balance += tx.get('total_value', 0) - tx.get('commission', 0)
            
            history.append({
                'date': tx['timestamp'].isoformat(),
                'value': running_balance,
                'transaction': tx['type']
            })
        
        return jsonify(history), 200
        
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/analytics', methods=['GET'])
@jwt_required()
def get_analytics():
    """Get detailed portfolio analytics"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        db = current_app.config.get('db')
        
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        # Get user and transactions
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        portfolio = user.get('portfolio', {})
        
        # Get all transactions
        transactions = list(
            db.transactions.find({
                'user_id': user_id,
                'type': {'$in': ['buy', 'sell']}
            })
        )
        
        # Calculate analytics
        analytics = {
            'total_trades': len(transactions),
            'buy_orders': len([t for t in transactions if t['type'] == 'buy']),
            'sell_orders': len([t for t in transactions if t['type'] == 'sell']),
            'average_trade_size': sum(t.get('total_value', 0) for t in transactions) / len(transactions) if transactions else 0,
            'total_commission': sum(t.get('commission', 0) for t in transactions),
            'most_traded': self._get_most_traded_symbol(transactions),
            'best_performer': self._get_best_performer(portfolio),
            'worst_performer': self._get_worst_performer(portfolio),
            'portfolio_diversity': len(portfolio.get('positions', [])),
            'risk_score': self._calculate_risk_score(portfolio),
            'daily_stats': self._get_daily_stats(transactions)
        }
        
        return jsonify(analytics), 200
        
    except Exception as e:
        logger.error(f"Analytics fetch error: {e}")
        return jsonify({'error': str(e)}), 500

def _get_most_traded_symbol(transactions):
    """Get most frequently traded symbol"""
    from collections import Counter
    symbols = [t['symbol'] for t in transactions if t.get('symbol')]
    if symbols:
        counter = Counter(symbols)
        return counter.most_common(1)[0][0]
    return None

def _get_best_performer(portfolio):
    """Get best performing position"""
    positions = portfolio.get('positions', [])
    if not positions:
        return None
    
    best = max(positions, key=lambda p: 
               (p.get('current_price', 0) - p.get('avg_price', 0)) / p.get('avg_price', 1) if p.get('avg_price') else 0)
    
    pnl_pct = ((best.get('current_price', 0) - best.get('avg_price', 0)) / best.get('avg_price', 1)) * 100 if best.get('avg_price') else 0
    
    return {
        'symbol': best.get('symbol'),
        'return_pct': round(pnl_pct, 2)
    }

def _get_worst_performer(portfolio):
    """Get worst performing position"""
    positions = portfolio.get('positions', [])
    if not positions:
        return None
    
    worst = min(positions, key=lambda p: 
                (p.get('current_price', 0) - p.get('avg_price', 0)) / p.get('avg_price', 1) if p.get('avg_price') else 0)
    
    pnl_pct = ((worst.get('current_price', 0) - worst.get('avg_price', 0)) / worst.get('avg_price', 1)) * 100 if worst.get('avg_price') else 0
    
    return {
        'symbol': worst.get('symbol'),
        'return_pct': round(pnl_pct, 2)
    }

def _calculate_risk_score(portfolio):
    """Calculate overall portfolio risk score"""
    positions = portfolio.get('positions', [])
    if not positions:
        return 0
    
    # Simple risk calculation based on concentration
    total_value = sum(p.get('market_value', 0) for p in positions)
    if total_value == 0:
        return 0
    
    max_position = max(p.get('market_value', 0) for p in positions)
    concentration = (max_position / total_value) * 100
    
    # Risk score from 0-100
    if concentration > 50:
        return 80
    elif concentration > 30:
        return 60
    elif concentration > 20:
        return 40
    else:
        return 20

def _get_daily_stats(transactions):
    """Get daily trading statistics"""
    from collections import defaultdict
    daily = defaultdict(lambda: {'trades': 0, 'volume': 0})
    
    for tx in transactions:
        if tx.get('timestamp'):
            date = tx['timestamp'].date()
            daily[date]['trades'] += 1
            daily[date]['volume'] += tx.get('total_value', 0)
    
    if not daily:
        return {'average_daily_trades': 0, 'average_daily_volume': 0}
    
    avg_trades = sum(d['trades'] for d in daily.values()) / len(daily)
    avg_volume = sum(d['volume'] for d in daily.values()) / len(daily)
    
    return {
        'average_daily_trades': round(avg_trades, 1),
        'average_daily_volume': round(avg_volume, 2)
    }
