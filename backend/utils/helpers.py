"""
Helper functions for calculations and validations
"""
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

def calculate_returns(prices: List[float]) -> List[float]:
    """Calculate percentage returns from price series"""
    if len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
    
    return returns

def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    # Basic validation for stock symbols
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, symbol))

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format amount as currency string"""
    if currency == 'USD':
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns"""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return round(sharpe, 2)

def calculate_max_drawdown(values: List[float]) -> float:
    """Calculate maximum drawdown from value series"""
    if not values:
        return 0.0
    
    peak = values[0]
    max_dd = 0.0
    
    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak != 0 else 0
        if drawdown > max_dd:
            max_dd = drawdown
    
    return round(max_dd * 100, 2)

def calculate_position_size(
    portfolio_value: float,
    risk_per_trade: float = 0.02,
    stop_loss_pct: float = 0.05
) -> float:
    """
    Calculate position size based on risk management rules
    
    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Maximum risk per trade (2% default)
        stop_loss_pct: Stop loss percentage (5% default)
    
    Returns:
        Maximum position size
    """
    risk_amount = portfolio_value * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    
    # Cap at 20% of portfolio
    max_position = portfolio_value * 0.2
    
    return min(position_size, max_position)

def validate_trade_params(
    symbol: str,
    quantity: float,
    side: str,
    order_type: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate trade parameters
    
    Returns:
        (is_valid, error_message)
    """
    # Validate symbol
    if not validate_symbol(symbol):
        return False, "Invalid symbol format"
    
    # Validate quantity
    if quantity <= 0:
        return False, "Quantity must be positive"
    
    if quantity > 10000:
        return False, "Quantity exceeds maximum limit"
    
    # Validate side
    if side not in ['buy', 'sell']:
        return False, "Side must be 'buy' or 'sell'"
    
    # Validate order type
    if order_type not in ['market', 'limit', 'stop']:
        return False, "Invalid order type"
    
    return True, None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to price dataframe"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

def get_market_hours() -> Dict[str, bool]:
    """Check if markets are open"""
    now = datetime.utcnow()
    weekday = now.weekday()
    hour = now.hour
    
    # US market hours (UTC): 14:30 - 21:00 (9:30 AM - 4:00 PM EST)
    us_market_open = weekday < 5 and 14 <= hour < 21
    
    # Forex: 24/5
    forex_open = weekday < 5 or (weekday == 6 and hour < 22)
    
    # Crypto: 24/7
    crypto_open = True
    
    return {
        'us_stocks': us_market_open,
        'forex': forex_open,
        'crypto': crypto_open
    }

def calculate_risk_metrics(
    positions: List[Dict],
    portfolio_value: float
) -> Dict[str, float]:
    """Calculate portfolio risk metrics"""
    if not positions:
        return {
            'total_exposure': 0.0,
            'largest_position': 0.0,
            'concentration_risk': 0.0,
            'leverage': 0.0
        }
    
    total_exposure = sum(p.get('market_value', 0) for p in positions)
    largest_position = max(p.get('market_value', 0) for p in positions)
    
    concentration_risk = (largest_position / portfolio_value * 100) if portfolio_value > 0 else 0
    leverage = (total_exposure / portfolio_value) if portfolio_value > 0 else 0
    
    return {
        'total_exposure': round(total_exposure, 2),
        'largest_position': round(largest_position, 2),
        'concentration_risk': round(concentration_risk, 2),
        'leverage': round(leverage, 2)
    }

def generate_trade_id() -> str:
    """Generate unique trade ID"""
    from uuid import uuid4
    return str(uuid4())

def calculate_slippage(
    order_size: float,
    avg_volume: float,
    spread: float
) -> float:
    """Estimate slippage for order"""
    participation_rate = order_size / avg_volume if avg_volume > 0 else 0
    
    # Simple slippage model
    slippage = spread * (1 + participation_rate * 10)
    
    return min(slippage, spread * 5)  # Cap at 5x spread
