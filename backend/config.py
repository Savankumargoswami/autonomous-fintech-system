"""
Application configuration module
"""
import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = FLASK_ENV == 'development'
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    JWT_ALGORITHM = 'HS256'
    
    # Database Configuration
    MONGODB_URI = os.getenv('MONGODB_URI')
    MONGODB_DB_NAME = 'fintech_trading'
    
    # Redis Configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_DECODE_RESPONSES = True
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    
    # Trading Configuration
    DEFAULT_PAPER_TRADING_BALANCE = 100000.0
    DEFAULT_COMMISSION_RATE = 0.001  # 0.1%
    MAX_POSITION_SIZE = 0.2  # 20% of portfolio
    STOP_LOSS_PERCENTAGE = 0.05  # 5%
    TAKE_PROFIT_PERCENTAGE = 0.1  # 10%
    
    # Rate Limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100/hour"
    RATELIMIT_STORAGE_URL = REDIS_URL
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'logs/fintech.log'
    
    # WebSocket Configuration
    SOCKETIO_ASYNC_MODE = 'eventlet'
    SOCKETIO_CORS_ALLOWED_ORIGINS = '*'
    
    # ML Model Configuration
    MODEL_PATH = 'ml_models/trained_models'
    MODEL_UPDATE_FREQUENCY = 86400  # 24 hours in seconds
    
    # Market Data Configuration
    MARKET_DATA_CACHE_TTL = 60  # seconds
    HISTORICAL_DATA_DAYS = 365
    REALTIME_UPDATE_INTERVAL = 5  # seconds
    
    # Risk Management
    MAX_DAILY_TRADES = 50
    MAX_DAILY_LOSS = 0.05  # 5% of portfolio
    MARGIN_REQUIREMENT = 0.25  # 25%
    
    # Performance Metrics
    SHARPE_RATIO_RISK_FREE_RATE = 0.02  # 2% annual
    BENCHMARK_SYMBOL = 'SPY'  # S&P 500 ETF
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Override with production values
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        syslog = SysLogHandler()
        syslog.setLevel(logging.WARNING)
        app.logger.addHandler(syslog)

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    MONGODB_URI = 'mongodb://localhost:27017/test_fintech'
    
    # Disable rate limiting in tests
    RATELIMIT_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, DevelopmentConfig)
