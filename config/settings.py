import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/fintech_db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # API Keys
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    iex_cloud_api_key: str = os.getenv("IEX_CLOUD_API_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "dev_secret_key")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Application
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Trading Parameters
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "100000"))
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    risk_free_rate: float = float(os.getenv("RISK_FREE_RATE", "0.02"))
    
    # Allowed Origins for CORS
    allowed_origins: List[str] = ["http://localhost:3000", "https://your-domain.com"]
    
    class Config:
        env_file = ".env"

settings = Settings()
