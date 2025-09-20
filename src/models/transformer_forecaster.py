import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

class TransformerForecaster(nn.Module):
    def __init__(self, 
                 input_dim: int = 5,  # OHLCV
                 model_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 sequence_length: int = 100,
                 prediction_horizon: int = 5):
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim, sequence_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, prediction_horizon)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer
        transformer_out = self.transformer(x)
        
        # Use the last time step for prediction
        last_hidden = transformer_out[:, -1, :]
        
        # Project to prediction horizon
        predictions = self.output_projection(last_hidden)
        
        return predictions

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, model_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() *
                           -(np.log(10000.0) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MarketForecaster:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def initialize(self):
        """Initialize the forecasting model"""
        self.model = TransformerForecaster(
            input_dim=self.config.get('input_dim', 5),
            model_dim=self.config.get('model_dim', 128),
            num_heads=self.config.get('num_heads', 8),
            num_layers=self.config.get('num_layers', 6),
            sequence_length=self.config.get('sequence_length', 100),
            prediction_horizon=self.config.get('prediction_horizon', 5)
        ).to(self.device)
        
        # Initialize scaler for normalization
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        
    async def train_model(self, training_data: List[Dict]):
        """Train the forecasting model"""
        if not training_data:
            return
        
        # Prepare training data
        X, y = self._prepare_training_data(training_data)
        
        if len(X) == 0:
            return
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.model.train()
        num_epochs = self.config.get('num_epochs', 100)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_tensor)
            loss = criterion(predictions, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    def _prepare_training_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        sequences = []
        targets = []
        
        # Combine all symbols data
        for item in data:
            symbol_data = item.get('data', {})
            if not symbol or quantity <= 0:
            raise HTTPException(status_code=400, detail="Invalid order parameters")
        
        # Get current market data
        market_data = await data_pipeline.get_market_data(symbol)
        if not market_data:
            raise HTTPException(status_code=404, detail=f"No market data for {symbol}")
        
        current_price = market_data.get('price', 0)
        
        # Simulate order execution
        execution_price = current_price
        if order_type == 'limit':
            limit_price = order_data.get('limit_price', current_price)
            execution_price = limit_price
        
        # Calculate costs
        market_value = execution_price * quantity
        commission = market_value * 0.001  # 0.1% commission
        
        simulation_result = {
            "order_id": f"SIM_{int(datetime.utcnow().timestamp())}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "execution_price": execution_price,
            "market_value": market_value,
            "commission": commission,
            "total_cost": market_value + commission if side == 'buy' else market_value - commission,
            "status": "simulated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return simulation_result
        
    except Exception as e:
        logger.error(f"Error simulating order: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/performance")
async def get_performance_analytics(current_user: dict = Depends(get_current_user)):
    """Get portfolio performance analytics"""
    try:
        # Mock performance data - in production, calculate from actual trades
        performance_data = {
            "total_return": 0.25,  # 25%
            "annualized_return": 0.18,  # 18%
            "volatility": 0.12,  # 12%
            "sharpe_ratio": 1.33,
            "max_drawdown": -0.08,  # -8%
            "win_rate": 0.65,  # 65%
            "profit_factor": 1.85,
            "daily_returns": [0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012],  # Last 7 days
            "monthly_returns": [0.05, 0.03, -0.02, 0.08, 0.04, 0.06],  # Last 6 months
            "benchmark_comparison": {
                "portfolio_return": 0.25,
                "spy_return": 0.15,
                "outperformance": 0.10
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error fetching performance analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws/real-time-data")
async def websocket_real_time_data(websocket):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    
    try:
        while True:
            # Send real-time market data
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            real_time_data = {}
            
            for symbol in symbols:
                market_data = await data_pipeline.get_market_data(symbol)
                if market_data:
                    real_time_data[symbol] = market_data
            
            # Send strategy signals
            if strategy_agent and strategy_agent.signals:
                real_time_data['signals'] = strategy_agent.signals
            
            # Send risk alerts
            if risk_agent:
                real_time_data['risk_alerts'] = risk_agent._generate_risk_alerts()
            
            await websocket.send_json({
                "type": "market_update",
                "data": real_time_data,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
