"""
Market data routes
"""
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('market_data', __name__)

@bp.route('/search', methods=['GET'])
@jwt_required()
def search_symbols():
    """Search for trading symbols"""
    try:
        query = request.args.get('q', '')
        
        # Sample symbols for demo
        symbols = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'type': 'stock'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'type': 'stock'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corp.', 'type': 'stock'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'type': 'stock'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'type': 'stock'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'type': 'stock'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'type': 'stock'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'type': 'stock'},
            {'symbol': 'V', 'name': 'Visa Inc.', 'type': 'stock'},
            {'symbol': 'WMT', 'name': 'Walmart Inc.', 'type': 'stock'}
        ]
        
        if query:
            symbols = [s for s in symbols if query.upper() in s['symbol'] or query.lower() in s['name'].lower()]
        
        return jsonify(symbols), 200
        
    except Exception as e:
        logger.error(f"Symbol search error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/quote/<symbol>', methods=['GET'])
@jwt_required()
def get_quote(symbol):
    """Get real-time quote for a symbol"""
    try:
        from flask import current_app
        data_fetcher = current_app.config.get('data_fetcher')
        
        if data_fetcher:
            quote = data_fetcher.get_quote(symbol)
            if quote:
                return jsonify(quote), 200
            else:
                return jsonify({'error': 'Failed to fetch quote'}), 404
        
        return jsonify({'error': 'Service not available'}), 500
        
    except Exception as e:
        logger.error(f"Quote fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/news/<symbol>', methods=['GET'])
@jwt_required()
def get_news(symbol):
    """Get news for a symbol"""
    try:
        from flask import current_app
        data_fetcher = current_app.config.get('data_fetcher')
        
        if data_fetcher:
            news = data_fetcher.get_market_news(symbol)
            return jsonify(news), 200
        
        return jsonify({'error': 'Service not available'}), 500
        
    except Exception as e:
        logger.error(f"News fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/historical/<symbol>', methods=['GET'])
@jwt_required()
def get_historical(symbol):
    """Get historical data"""
    try:
        from flask import current_app
        days = request.args.get('days', 30, type=int)
        data_fetcher = current_app.config.get('data_fetcher')
        
        if data_fetcher:
            data = data_fetcher.get_historical_data(symbol, days)
            # Convert DataFrame to JSON-serializable format
            if not data.empty:
                result = {
                    'symbol': symbol,
                    'data': data.reset_index().to_dict('records')
                }
                return jsonify(result), 200
        
        return jsonify({'error': 'Data not available'}), 500
        
    except Exception as e:
        logger.error(f"Historical data fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/analysis/<symbol>', methods=['GET'])
@jwt_required()
def get_market_analysis(symbol):
    """Get AI-driven market analysis"""
    try:
        from flask import current_app
        
        # Get services
        data_fetcher = current_app.config.get('data_fetcher')
        market_analyzer = current_app.config.get('market_analyzer')
        strategy_agent = current_app.config.get('strategy_agent')
        risk_agent = current_app.config.get('risk_agent')
        sentiment_agent = current_app.config.get('sentiment_agent')
        
        if not all([data_fetcher, market_analyzer, strategy_agent, risk_agent, sentiment_agent]):
            return jsonify({'error': 'Analysis services not available'}), 500
        
        # Get market data
        market_data = data_fetcher.get_historical_data(symbol)
        
        # Run AI analysis
        strategy_recommendation = strategy_agent.analyze(market_data)
        risk_assessment = risk_agent.assess_risk(symbol, market_data)
        sentiment = sentiment_agent.analyze_sentiment(symbol)
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'strategy': strategy_recommendation,
            'risk': risk_assessment,
            'sentiment': sentiment,
            'recommendation': market_analyzer.get_recommendation(
                strategy_recommendation,
                risk_assessment,
                sentiment
            )
        }
        
        return jsonify(analysis), 200
        
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/watchlist', methods=['GET', 'POST', 'DELETE'])
@jwt_required()
def manage_watchlist():
    """Manage user's watchlist"""
    try:
        from flask import current_app
        from bson import ObjectId
        user_id = get_jwt_identity()
        db = current_app.config.get('db')
        
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        if request.method == 'GET':
            # Get watchlist
            user = db.users.find_one({'_id': ObjectId(user_id)})
            watchlist = user.get('watchlist', []) if user else []
            return jsonify(watchlist), 200
        
        elif request.method == 'POST':
            # Add to watchlist
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
            
            db.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$addToSet': {'watchlist': symbol}}
            )
            
            return jsonify({'message': f'{symbol} added to watchlist'}), 200
        
        elif request.method == 'DELETE':
            # Remove from watchlist
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
            
            db.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$pull': {'watchlist': symbol}}
            )
            
            return jsonify({'message': f'{symbol} removed from watchlist'}), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/indicators/<symbol>', methods=['GET'])
@jwt_required()
def get_technical_indicators(symbol):
    """Get technical indicators for a symbol"""
    try:
        from flask import current_app
        data_fetcher = current_app.config.get('data_fetcher')
        
        if data_fetcher:
            # Get historical data with indicators
            data = data_fetcher.get_historical_data(symbol, days=60)
            
            # Extract latest indicators
            if not data.empty:
                latest = data.iloc[-1]
                indicators = {
                    'symbol': symbol,
                    'rsi': float(latest.get('rsi', 50)),
                    'macd': float(latest.get('macd', 0)),
                    'signal': float(latest.get('signal', 0)),
                    'bb_upper': float(latest.get('bb_upper', 0)),
                    'bb_lower': float(latest.get('bb_lower', 0)),
                    'sma_20': float(latest.get('sma_20', 0)),
                    'sma_50': float(latest.get('sma_50', 0)),
                    'ema_12': float(latest.get('ema_12', 0)),
                    'ema_26': float(latest.get('ema_26', 0)),
                    'volume': int(latest.get('volume', 0))
                }
                return jsonify(indicators), 200
        
        return jsonify({'error': 'Data not available'}), 500
        
    except Exception as e:
        logger.error(f"Indicators fetch error: {e}")
        return jsonify({'error': str(e)}), 500
