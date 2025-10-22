"""
Temporal Graph Attention Network (Section 3.7)
Implements Equations 45-52 with:
- Dynamic graph construction
- Multi-head graph attention
- Temporal encoding
- Hierarchical pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class TemporalGraphAttentionNetwork(nn.Module):
    """
    Temporal GAT for capturing inter-asset dependencies
    
    Paper Equations 45-52:
    - Dynamic edge construction based on correlation and spillover
    - Multi-head attention aggregation
    - Temporal positional encoding
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        edge_threshold_corr: float = 0.3,
        edge_threshold_spillover: float = 0.15,
        hierarchical_pooling: bool = True,
        temporal_encoding: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.edge_threshold_corr = edge_threshold_corr
        self.edge_threshold_spillover = edge_threshold_spillover
        self.hierarchical_pooling = hierarchical_pooling
        self.temporal_encoding = temporal_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATLayer(
                hidden_dim if i == 0 else hidden_dim,
                hidden_dim,
                n_heads,
                dropout
            ) for i in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Temporal positional encoding
        if temporal_encoding:
            self.temporal_encoder = nn.Linear(1, hidden_dim)
        
        # Hierarchical pooling
        if hierarchical_pooling:
            self.pool_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None,
        temporal_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through temporal GAT
        
        Args:
            x: (batch_size, input_dim) node features
            edge_index: (2, num_edges) edge connectivity (optional, will construct if None)
            edge_weights: (num_edges,) edge weights (optional)
            temporal_positions: (batch_size,) temporal positions (optional)
            
        Returns:
            output: (batch_size, output_dim) graph-enhanced representations
        """
        batch_size = x.size(0)
        device = x.device
        
        # === STEP 1: Construct dynamic graph (Equations 45-46) ===
        if edge_index is None or edge_weights is None:
            edge_index, edge_weights = self.construct_dynamic_graph(x)
        
        # === STEP 2: Input projection ===
        h = self.input_projection(x)  # (batch_size, hidden_dim)
        
        # === STEP 3: Add temporal encoding (Equation 47) ===
        if self.temporal_encoding and temporal_positions is not None:
            temporal_pos = temporal_positions.unsqueeze(-1).float()
            temporal_emb = self.temporal_encoder(temporal_pos)
            h = h + temporal_emb
        
        # === STEP 4: Graph attention layers (Equations 48-51) ===
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_weights)
        
        # === STEP 5: Hierarchical pooling (Equation 52) ===
        if self.hierarchical_pooling:
            h_expanded = h.unsqueeze(0)  # (1, batch_size, hidden_dim)
            h_pooled, _ = self.pool_attn(h_expanded, h_expanded, h_expanded)
            h = h_pooled.squeeze(0)
        
        # === STEP 6: Output projection ===
        output = self.output_projection(h)
        output = self.norm(output)
        
        return output
    
    def construct_dynamic_graph(
        self,
        node_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct dynamic graph based on node feature similarity
        
        Args:
            node_features: (batch_size, feature_dim)
            
        Returns:
            edge_index: (2, num_edges)
            edge_weights: (num_edges,)
        """
        batch_size = node_features.size(0)
        device = node_features.device
        
        # Compute pairwise cosine similarity (correlation proxy)
        node_features_norm = F.normalize(node_features, p=2, dim=-1)
        similarity_matrix = torch.matmul(node_features_norm, node_features_norm.T)
        
        # Threshold to create edges
        edge_mask = (similarity_matrix > self.edge_threshold_corr) & (similarity_matrix < 0.99)
        
        # Extract edge indices
        edge_index = edge_mask.nonzero(as_tuple=False).T  # (2, num_edges)
        edge_weights = similarity_matrix[edge_mask]  # (num_edges,)
        
        # Normalize edge weights
        edge_weights = torch.clamp(edge_weights, 0, 1)
        
        return edge_index, edge_weights


class GATLayer(nn.Module):
    """Single Graph Attention Layer"""
    
    def __init__(self, input_dim: int, output_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.head_dim = output_dim // n_heads
        
        assert output_dim % n_heads == 0, "output_dim must be divisible by n_heads"
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        
        # Attention weights
        self.attn = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GAT layer
        
        Args:
            x: (batch_size, input_dim) node features
            edge_index: (2, num_edges) edge connectivity
            edge_weights: (num_edges,) edge weights
            
        Returns:
            output: (batch_size, output_dim) updated node features
        """
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, self.n_heads, self.head_dim)
        
        # Compute attention scores for edges
        if edge_index.size(1) > 0:
            src_idx, dst_idx = edge_index[0], edge_index[1]
            
            # Concatenate source and destination features
            edge_features = torch.cat([Q[src_idx], K[dst_idx]], dim=-1)  # (num_edges, n_heads, 2*head_dim)
            
            # Compute attention scores
            attn_scores = torch.sum(edge_features * self.attn.unsqueeze(0), dim=-1)  # (num_edges, n_heads)
            attn_scores = F.leaky_relu(attn_scores, negative_slope=0.2)
            
            # Apply edge weights if provided
            if edge_weights is not None:
                attn_scores = attn_scores * edge_weights.unsqueeze(-1)
            
            # Softmax over neighbors
            attn_weights = torch.zeros(batch_size, batch_size, self.n_heads, device=x.device)
            attn_weights[dst_idx, src_idx] = attn_scores
            
            # Normalize per destination node
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Aggregate messages
            messages = torch.einsum('ijk,jhd->ihd', attn_weights, V)  # (batch_size, n_heads, head_dim)
        else:
            # No edges: self-attention only
            messages = V
        
        # Concatenate heads
        output = messages.reshape(batch_size, self.output_dim)
        
        # Residual connection and normalization
        output = self.norm(output + x if x.size(-1) == self.output_dim else output)
        output = self.dropout(output)
        
        return output
