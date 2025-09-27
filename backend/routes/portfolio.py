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
