# ml_models/reinforcement_learning.py
"""
Reinforcement Learning Trading Agent using Deep Q-Learning and PPO
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DQN(nn.Module):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class RLTradingAgent:
    """
    Reinforcement Learning agent for autonomous trading
    """
    
    def __init__(self, state_size: int = 30, action_size: int = 3, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size  # Buy, Hold, Sell
        self.memory = []
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def act(self, state: np.ndarray, is_eval: bool = False) -> int:
        """
        Choose action based on current state
        
        Returns:
            0: Hold
            1: Buy
            2: Sell
        """
        if not is_eval and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.detach().numpy())
    
    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            current_q = self.q_network(state_tensor)
            next_q = self.target_network(next_state_tensor)
            
            target = current_q.clone()
            
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * torch.max(next_q)
            
            loss = nn.MSELoss()(current_q, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def prepare_state(self, data: pd.DataFrame, lookback: int = 30) -> np.ndarray:
        """
        Prepare state vector from market data
        """
        if len(data) < lookback:
            return np.zeros(self.state_size)
        
        recent_data = data.tail(lookback)
        
        # Calculate features
        features = []
        
        # Price features
        returns = recent_data['close'].pct_change().fillna(0)
        features.extend([
            returns.mean(),
            returns.std(),
            returns.min(),
            returns.max(),
            returns.iloc[-1]
        ])
        
        # Volume features
        volume_mean = recent_data['volume'].mean()
        volume_std = recent_data['volume'].std()
        features.extend([
            volume_mean,
            volume_std,
            recent_data['volume'].iloc[-1] / volume_mean if volume_mean > 0 else 1
        ])
        
        # Technical indicators
        if 'rsi' in recent_data.columns:
            features.append(recent_data['rsi'].iloc[-1] / 100)
        else:
            features.append(0.5)
        
        if 'macd' in recent_data.columns:
            features.append(np.tanh(recent_data['macd'].iloc[-1]))
        else:
            features.append(0)
        
        # Moving averages
        sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
        current_price = recent_data['close'].iloc[-1]
        features.append((current_price - sma_20) / sma_20 if sma_20 > 0 else 0)
        
        # Pad or truncate to state_size
        if len(features) < self.state_size:
            features.extend([0] * (self.state_size - len(features)))
        else:
            features = features[:self.state_size]
        
        return np.array(features, dtype=np.float32)
    
    def train_on_history(self, historical_data: pd.DataFrame, episodes: int = 100):
        """
        Train the agent on historical data
        """
        for episode in range(episodes):
            total_reward = 0
            state = self.prepare_state(historical_data.iloc[:30])
            
            for i in range(30, len(historical_data) - 1):
                action = self.act(state)
                
                # Calculate reward
                price_change = (historical_data['close'].iloc[i + 1] - 
                               historical_data['close'].iloc[i]) / historical_data['close'].iloc[i]
                
                if action == 1:  # Buy
                    reward = price_change * 100
                elif action == 2:  # Sell
                    reward = -price_change * 100
                else:  # Hold
                    reward = 0
                
                next_state = self.prepare_state(historical_data.iloc[:i+1])
                done = i == len(historical_data) - 2
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(self.memory) > 32:
                    self.replay(32)
            
            if episode % 10 == 0:
                self.update_target_network()
                logger.info(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
    
    def predict(self, market_data: pd.DataFrame) -> Dict:
        """
        Make trading prediction based on current market state
        """
        state = self.prepare_state(market_data)
        action = self.act(state, is_eval=True)
        
        # Get Q-values for confidence
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor).detach().numpy()[0]
        
        action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
        confidence = np.exp(q_values[action]) / np.sum(np.exp(q_values))  # Softmax
        
        return {
            'action': action_map[action],
            'confidence': float(confidence),
            'q_values': q_values.tolist(),
            'state_features': state.tolist()[:10]  # Return first 10 features for debugging
        }
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
