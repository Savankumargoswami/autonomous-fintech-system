# ml_models/transformer_forecasting.py
"""
Transformer model for time-series price forecasting
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Add positional encoding to embeddings"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerForecaster(nn.Module):
    """
    Transformer model for financial time series forecasting
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 4, 
                 forecast_horizon: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Project input to hidden dimension
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use the last encoded state for prediction
        last_state = encoded[:, -1, :]
        
        # Project to forecast
        forecast = self.output_projection(last_state)
        
        return forecast
    
    def prepare_data(self, data: pd.DataFrame, sequence_length: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for transformer model
        """
        features = []
        targets = []
        
        # Normalize data
        data_norm = data.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data_norm.columns:
                data_norm[col] = (data_norm[col] - data_norm[col].mean()) / data_norm[col].std()
        
        for i in range(sequence_length, len(data) - self.forecast_horizon):
            # Extract features
            seq_features = []
            for j in range(i - sequence_length, i):
                row_features = [
                    data_norm['open'].iloc[j],
                    data_norm['high'].iloc[j],
                    data_norm['low'].iloc[j],
                    data_norm['close'].iloc[j],
                    data_norm['volume'].iloc[j] / 1e6,
                ]
                
                # Add technical indicators if available
                if 'rsi' in data.columns:
                    row_features.append(data['rsi'].iloc[j] / 100)
                if 'macd' in data.columns:
                    row_features.append(np.tanh(data['macd'].iloc[j]))
                
                # Pad to input_dim
                while len(row_features) < self.input_dim:
                    row_features.append(0)
                
                seq_features.append(row_features[:self.input_dim])
            
            features.append(seq_features)
            
            # Target is next N closing prices
            target = data_norm['close'].iloc[i:i + self.forecast_horizon].values
            targets.append(target)
        
        return torch.FloatTensor(features), torch.FloatTensor(targets)
    
    def train_model(self, train_data: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """
        Train the transformer model
        """
        X, y = self.prepare_data(train_data)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                predictions = self(batch_X)
                loss = criterion(predictions, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Make price predictions
        """
        self.eval()
        with torch.no_grad():
            X, _ = self.prepare_data(data)
            if len(X) > 0:
                # Use last sequence
                last_seq = X[-1:, :, :]
                forecast = self(last_seq).numpy()[0]
                
                # Denormalize predictions
                mean = data['close'].mean()
                std = data['close'].std()
                forecast = forecast * std + mean
                
                return {
                    'forecast': forecast.tolist(),
                    'forecast_dates': pd.date_range(
                        start=data.index[-1] + pd.Timedelta(days=1),
                        periods=self.forecast_horizon
                    ).tolist()
                }
        
        return {'forecast': [], 'forecast_dates': []}
