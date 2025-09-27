"""
Trading routes for order execution and management
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from bson import ObjectId
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

bp = Blueprint('trading', __name__)

@bp.route('/execute', methods=['POST'])
@jwt_required()
def execute_trade():
    """Execute a paper trade"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Get trading engine
        trading_engine = current_app.config.get('trading_engine')
        if not trading_engine:
            return jsonify({'error': 'Trading engine not available'}), 500
        
        # Validate trade data
        symbol = data.get('symbol')
        side = data.get('side')  # 'buy' or 'sell'
        quantity = data.get('quantity')
        order_type = data.get('order_type', 'market')
        price = data.get('price')  # For limit orders
        
        if not all([symbol, side, quantity]):
            return jsonify({'error': 'Missing required trade parameters'}), 400
        
        # Validate parameters
        if side not in ['buy', 'sell']:
            return jsonify({'error': 'Side must be buy or sell'}), 400
        
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be positive'}), 400
        
        # Execute trade through trading engine
        result = trading_engine.execute_trade(
            user_id=user_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price
        )
        
        # Emit trade update via WebSocket if available
        socketio = current_app.config.get('socketio')
        if socketio and result['status'] == 'executed':
            socketio.emit('trade_executed', result, room=user_id)
        
        return jsonify(result), 200 if result['status'] == 'executed' else 400
        
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/orders', methods=['GET'])
@jwt_required()
def get_orders():
    """Get user's orders (pending and recent)"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        
        # Get trading engine and database
        trading_engine = current_app.config.get('trading_engine')
        db = current_app.config.get('db')
        
        if not trading_engine or not db:
            return jsonify({'error': 'Services not available'}), 500
        
        # Get pending orders
        pending_orders = trading_engine.get_pending_orders(user_id)
        
        # Get recent completed orders from transactions
        recent_transactions = list(
            db.transactions.find({
                'user_id': user_id,
                'type': {'$in': ['buy', 'sell']}
            }).sort('timestamp', -1).limit(20)
        )
        
        # Format response
        orders = {
            'pending': pending_orders,
            'recent': [
                {
                    'order_id': str(tx.get('transaction_id', '')),
                    'symbol': tx.get('symbol'),
                    'side': tx.get('type'),
                    'quantity': tx.get('quantity'),
                    'price': tx.get('price'),
                    'status': tx.get('status'),
                    'timestamp': tx.get('timestamp').isoformat() if tx.get('timestamp') else None
                }
                for tx in recent_transactions
            ]
        }
        
        return jsonify(orders), 200
        
    except Exception as e:
        logger.error(f"Orders fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/order/<order_id>', methods=['DELETE'])
@jwt_required()
def cancel_order(order_id):
    """Cancel a pending order"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        
        # Get trading engine
        trading_engine = current_app.config.get('trading_engine')
        if not trading_engine:
            return jsonify({'error': 'Trading engine not available'}), 500
        
        # Cancel order
        result = trading_engine.cancel_order(user_id, order_id)
        
        if result.get('status') == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        logger.error(f"Order cancellation error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/positions', methods=['GET'])
@jwt_required()
def get_positions():
    """Get current open positions"""
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
        
        positions = user.get('portfolio', {}).get('positions', [])
        
        # Format positions for response
        formatted_positions = []
        for pos in positions:
            formatted_positions.append({
                'symbol': pos.get('symbol'),
                'quantity': pos.get('quantity'),
                'avg_price': pos.get('avg_price'),
                'current_price': pos.get('current_price'),
                'market_value': pos.get('market_value'),
                'unrealized_pnl': (pos.get('current_price', 0) - pos.get('avg_price', 0)) * pos.get('quantity', 0),
                'unrealized_pnl_pct': ((pos.get('current_price', 0) - pos.get('avg_price', 0)) / pos.get('avg_price', 1)) * 100 if pos.get('avg_price') else 0,
                'entry_date': pos.get('entry_date').isoformat() if pos.get('entry_date') else None
            })
        
        return jsonify(formatted_positions), 200
        
    except Exception as e:
        logger.error(f"Positions fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/position/<symbol>/close', methods=['POST'])
@jwt_required()
def close_position(symbol):
    """Close a specific position"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Get quantity to close (partial or full)
        quantity = data.get('quantity')  # If not specified, close entire position
        
        # Get trading engine
        trading_engine = current_app.config.get('trading_engine')
        if not trading_engine:
            return jsonify({'error': 'Trading engine not available'}), 500
        
        # Get user's position
        db = current_app.config.get('db')
        user = db.users.find_one({'_id': ObjectId(user_id)})
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        positions = user.get('portfolio', {}).get('positions', [])
        position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if not position:
            return jsonify({'error': f'No position found for {symbol}'}), 404
        
        # If quantity not specified, close entire position
        if not quantity:
            quantity = position['quantity']
        
        # Execute sell order to close position
        result = trading_engine.execute_trade(
            user_id=user_id,
            symbol=symbol,
            side='sell',
            quantity=quantity,
            order_type='market'
        )
        
        return jsonify(result), 200 if result['status'] == 'executed' else 400
        
    except Exception as e:
        logger.error(f"Position close error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/risk/check', methods=['POST'])
@jwt_required()
def check_trade_risk():
    """Check risk for a proposed trade"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Get risk manager
        risk_manager = current_app.config.get('risk_manager')
        if not risk_manager:
            return jsonify({'error': 'Risk manager not available'}), 500
        
        # Get trade parameters
        symbol = data.get('symbol')
        side = data.get('side')
        quantity = data.get('quantity')
        price = data.get('price', 100)  # Default price if not provided
        
        if not all([symbol, side, quantity]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Assess trade risk
        risk_assessment = risk_manager.assess_trade_risk(
            user_id=user_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )
        
        return jsonify(risk_assessment), 200
        
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/strategies', methods=['GET'])
@jwt_required()
def get_trading_strategies():
    """Get available trading strategies"""
    try:
        strategies = [
            {
                'id': 'momentum',
                'name': 'Momentum Trading',
                'description': 'Trade based on price momentum and trend strength',
                'risk_level': 'medium',
                'recommended_for': 'trending markets'
            },
            {
                'id': 'mean_reversion',
                'name': 'Mean Reversion',
                'description': 'Trade on the assumption that prices will revert to mean',
                'risk_level': 'low',
                'recommended_for': 'ranging markets'
            },
            {
                'id': 'breakout',
                'name': 'Breakout Trading',
                'description': 'Trade on price breakouts from consolidation',
                'risk_level': 'high',
                'recommended_for': 'volatile markets'
            },
            {
                'id': 'ai_ensemble',
                'name': 'AI Ensemble',
                'description': 'Use machine learning to combine multiple strategies',
                'risk_level': 'medium',
                'recommended_for': 'all market conditions'
            }
        ]
        
        return jsonify(strategies), 200
        
    except Exception as e:
        logger.error(f"Strategies fetch error: {e}")
        return jsonify({'error': str(e)}), 500
