Main Flask Application for Autonomous Financial Risk Management System
"""
import os
import sys
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
import redis
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import custom modules
from backend.config import Config
from backend.routes import auth, trading, portfolio, market_data
from backend.services.trading_engine import TradingEngine
from backend.services.risk_manager import RiskManager
from backend.services.market_analyzer import MarketAnalyzer
from backend.services.data_fetcher import DataFetcher
from backend.agents.strategy_agent import StrategyAgent
from backend.agents.risk_agent import RiskAgent
from backend.agents.execution_agent import ExecutionAgent
from backend.agents.sentiment_agent import SentimentAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
jwt = JWTManager(app)

# Initialize MongoDB
try:
    mongo_client = MongoClient(os.getenv('MONGODB_URI'), serverSelectionTimeoutMS=5000)
    db = mongo_client['fintech_trading']
    mongo_client.server_info()  # Test connection
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    db = None

# Initialize Redis
try:
    redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
    redis_client.ping()
    logger.info("Connected to Redis successfully")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# Initialize services
trading_engine = TradingEngine(db, redis_client)
risk_manager = RiskManager(db)
market_analyzer = MarketAnalyzer()
data_fetcher = DataFetcher()

# Initialize AI agents
strategy_agent = StrategyAgent()
risk_agent = RiskAgent()
execution_agent = ExecutionAgent()
sentiment_agent = SentimentAgent()

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'mongodb': 'connected' if db is not None else 'disconnected',
            'redis': 'connected' if redis_client is not None else 'disconnected',
            'trading_engine': 'active',
            'risk_manager': 'active'
        }
    }
    return jsonify(status), 200

# Authentication endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Check if user exists
        if db and db.users.find_one({'$or': [{'username': username}, {'email': email}]}):
            return jsonify({'error': 'User already exists'}), 409
        
        # Create new user
        hashed_password = generate_password_hash(password)
        user = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.utcnow(),
            'portfolio': {
                'balance': 100000.0,  # Starting balance for paper trading
                'positions': [],
                'transactions': [],
                'performance': {
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0
                }
            }
        }
        
        if db:
            result = db.users.insert_one(user)
            user_id = str(result.inserted_id)
            
            # Create access token
            access_token = create_access_token(identity=user_id)
            
            return jsonify({
                'message': 'User registered successfully',
                'access_token': access_token,
                'user': {
                    'id': user_id,
                    'username': username,
                    'email': email
                }
            }), 201
        else:
            return jsonify({'error': 'Database connection error'}), 500
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not all([username, password]):
            return jsonify({'error': 'Username and password required'}), 400
        
        if db:
            user = db.users.find_one({'username': username})
            
            if user and check_password_hash(user['password'], password):
                user_id = str(user['_id'])
                access_token = create_access_token(identity=user_id)
                
                return jsonify({
                    'access_token': access_token,
                    'user': {
                        'id': user_id,
                        'username': user['username'],
                        'email': user['email']
                    }
                }), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
        else:
            return jsonify({'error': 'Database connection error'}), 500
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500

# Trading endpoints
@app.route('/api/trading/execute', methods=['POST'])
@jwt_required()
def execute_trade():
    """Execute a paper trade"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Validate trade data
        symbol = data.get('symbol')
        side = data.get('side')  # 'buy' or 'sell'
        quantity = data.get('quantity')
        order_type = data.get('order_type', 'market')
        
        if not all([symbol, side, quantity]):
            return jsonify({'error': 'Missing required trade parameters'}), 400
        
        # Execute trade through trading engine
        result = trading_engine.execute_trade(
            user_id=user_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type
        )
        
        # Emit trade update via WebSocket
        socketio.emit('trade_executed', result, room=user_id)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
@jwt_required()
def get_portfolio():
    """Get user portfolio"""
    try:
        user_id = get_jwt_identity()
        
        if db:
            user = db.users.find_one({'_id': user_id})
            if user:
                portfolio = user.get('portfolio', {})
                return jsonify(portfolio), 200
            else:
                return jsonify({'error': 'User not found'}), 404
        else:
            return jsonify({'error': 'Database connection error'}), 500
            
    except Exception as e:
        logger.error(f"Portfolio fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market/quote/<symbol>', methods=['GET'])
@jwt_required()
def get_quote(symbol):
    """Get real-time market quote"""
    try:
        quote = data_fetcher.get_quote(symbol)
        if quote:
            return jsonify(quote), 200
        else:
            return jsonify({'error': 'Failed to fetch quote'}), 404
            
    except Exception as e:
        logger.error(f"Quote fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market/analysis/<symbol>', methods=['GET'])
@jwt_required()
def get_market_analysis(symbol):
    """Get AI-driven market analysis"""
    try:
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

# WebSocket events
@socketio.on('connect')
@jwt_required()
def handle_connect():
    """Handle WebSocket connection"""
    user_id = get_jwt_identity()
    logger.info(f"User {user_id} connected via WebSocket")
    emit('connected', {'message': 'Connected to trading server'})

@socketio.on('subscribe_market_data')
@jwt_required()
def handle_market_subscription(data):
    """Subscribe to real-time market data"""
    symbols = data.get('symbols', [])
    user_id = get_jwt_identity()
    
    # Start streaming market data
    for symbol in symbols:
        trading_engine.subscribe_to_market(user_id, symbol)
    
    emit('subscription_confirmed', {'symbols': symbols})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info("Client disconnected")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Register blueprints
app.register_blueprint(auth.bp, url_prefix='/api/auth')
app.register_blueprint(trading.bp, url_prefix='/api/trading')
app.register_blueprint(portfolio.bp, url_prefix='/api/portfolio')
app.register_blueprint(market_data.bp, url_prefix='/api/market')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
