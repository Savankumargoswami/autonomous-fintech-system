"""
Graph Neural Network for market relationship modeling
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class GraphConvolutionLayer(nn.Module):
    """Graph Convolution Layer for GNN"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        return output + self.bias

class MarketGraphNN(nn.Module):
    """
    Graph Neural Network for modeling market relationships
    """
    
    def __init__(self, num_features: int = 10, hidden_dim: int = 128, 
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.gc1 = GraphConvolutionLayer(num_features, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, hidden_dim // 2)
        self.gc3 = GraphConvolutionLayer(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, x, adj):
        # First GCN layer
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        
        # Second GCN layer
        x = F.relu(self.gc2(x, adj))
        x = self.dropout(x)
        
        # Final layer
        x = self.gc3(x, adj)
        
        return F.log_softmax(x, dim=1)
    
    def build_market_graph(self, symbols: List[str], correlations: pd.DataFrame) -> nx.Graph:
        """
        Build a graph representing market relationships
        """
        G = nx.Graph()
        
        # Add nodes for each symbol
        for symbol in symbols:
            G.add_node(symbol)
        
        # Add edges based on correlations
        threshold = 0.5  # Correlation threshold for edge creation
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j and sym1 in correlations.index and sym2 in correlations.columns:
                    corr = correlations.loc[sym1, sym2]
                    if abs(corr) > threshold:
                        G.add_edge(sym1, sym2, weight=abs(corr))
        
        return G
    
    def prepare_graph_data(self, market_data: Dict[str, pd.DataFrame], 
                           symbols: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for GNN training
        """
        # Calculate correlations
        prices = pd.DataFrame()
        for symbol in symbols:
            if symbol in market_data:
                prices[symbol] = market_data[symbol]['close']
        
        correlations = prices.corr()
        
        # Build graph
        G = self.build_market_graph(symbols, correlations)
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(G)
        adj_tensor = torch.FloatTensor(adj_matrix.todense())
        
        # Create feature matrix
        features = []
        for symbol in symbols:
            if symbol in market_data:
                data = market_data[symbol]
                symbol_features = [
                    data['close'].pct_change().mean(),
                    data['close'].pct_change().std(),
                    data['volume'].mean() / 1e6,
                    data['high'].mean(),
                    data['low'].mean(),
                    data.get('rsi', pd.Series([50])).iloc[-1] / 100,
                    np.tanh(data.get('macd', pd.Series([0])).iloc[-1]),
                ]
                # Pad to num_features
                while len(symbol_features) < 10:
                    symbol_features.append(0)
                features.append(symbol_features[:10])
            else:
                features.append([0] * 10)
        
        feature_tensor = torch.FloatTensor(features)
        
        return feature_tensor, adj_tensor
    
    def predict_market_relationships(self, market_data: Dict[str, pd.DataFrame], 
                                    symbols: List[str]) -> Dict:
        """
        Predict market relationships and identify key influencers
        """
        self.eval()
        with torch.no_grad():
            features, adj = self.prepare_graph_data(market_data, symbols)
            
            # Get node embeddings
            embeddings = F.relu(self.gc1(features, adj))
            
            # Analyze relationships
            G = self.build_market_graph(symbols, pd.DataFrame())
            
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Identify key symbols
            key_symbols = sorted(degree_centrality.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'key_influencers': [s[0] for s in key_symbols],
                'network_density': nx.density(G),
                'average_clustering': nx.average_clustering(G),
                'centrality_scores': dict(key_symbols)
            }
