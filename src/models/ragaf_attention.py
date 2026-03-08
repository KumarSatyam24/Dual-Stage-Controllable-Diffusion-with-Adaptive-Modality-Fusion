"""
RAGAF Attention: Region-Adaptive Graph-Attention Fusion

This is the core innovation of RAGAF-Diffusion. It implements region-aware
graph attention to fuse sketch structure and text semantics during generation.

Key concepts:
1. Graph-aware attention over sketch regions
2. Region-text token association
3. Spatial-semantic alignment
4. Dynamic region importance weighting

Author: RAGAF-Diffusion Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from data.region_graph import RegionGraph


class RegionGraphAttention(nn.Module):
    """
    Graph attention over sketch regions.
    
    Computes attention scores between regions based on:
    - Graph connectivity (adjacency)
    - Region features (spatial, geometric)
    - Learned attention weights
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize graph attention.
        
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.k_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.v_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature projection (for edge weights)
        self.edge_proj = nn.Linear(1, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute graph attention.
        
        Args:
            node_features: Node features (N, node_feature_dim)
            edge_index: Edge connectivity (2, E)
            edge_weights: Edge weights (E,), optional
        
        Returns:
            Updated node features (N, hidden_dim)
        """
        N = node_features.shape[0]
        
        # Project to Q, K, V
        Q = self.q_proj(node_features)  # (N, hidden_dim)
        K = self.k_proj(node_features)
        V = self.v_proj(node_features)
        
        # Reshape for multi-head attention
        Q = Q.view(N, self.num_heads, self.head_dim)  # (N, heads, head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)
        
        # Compute attention scores for connected nodes
        # This is a simplified graph attention (full GAT would use more complex edge features)
        
        if edge_index.shape[1] == 0:
            # No edges, return input features projected
            return self.out_proj(node_features)
        
        # Get source and target nodes
        src_nodes, tgt_nodes = edge_index[0], edge_index[1]
        
        # Compute attention scores
        Q_tgt = Q[tgt_nodes]  # (E, heads, head_dim)
        K_src = K[src_nodes]
        
        attn_scores = (Q_tgt * K_src).sum(dim=-1) / self.scale  # (E, heads)
        
        # Add edge weights if available
        if edge_weights is not None:
            edge_weights_proj = self.edge_proj(edge_weights.unsqueeze(-1))  # (E, heads)
            attn_scores = attn_scores + edge_weights_proj
        
        # Aggregate attention per target node using scatter softmax
        # For each target node, softmax over all incoming edges
        attn_probs = self._scatter_softmax(attn_scores, tgt_nodes, N)  # (E, heads)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        V_src = V[src_nodes]  # (E, heads, head_dim)
        attn_output = attn_probs.unsqueeze(-1) * V_src  # (E, heads, head_dim)
        
        # Aggregate outputs per target node
        output = torch.zeros(N, self.num_heads, self.head_dim, device=node_features.device)
        output.index_add_(0, tgt_nodes, attn_output)
        
        # Reshape and project
        output = output.view(N, self.hidden_dim)
        output = self.out_proj(output)
        
        return output
    
    def _scatter_softmax(
        self, 
        scores: torch.Tensor, 
        indices: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute softmax over scattered indices.
        
        Args:
            scores: Scores (E, heads)
            indices: Target node indices (E,)
            num_nodes: Total number of nodes
        
        Returns:
            Softmax probabilities (E, heads)
        """
        # Compute max per target node for numerical stability
        max_scores = torch.zeros(
            num_nodes, scores.shape[1], device=scores.device
        ).fill_(-1e9)
        max_scores.scatter_reduce_(0, indices.unsqueeze(-1).expand_as(scores), scores, reduce="amax")
        
        # Subtract max
        scores_normalized = scores - max_scores[indices]
        
        # Exp
        exp_scores = torch.exp(scores_normalized)
        
        # Sum per target node
        sum_exp = torch.zeros(num_nodes, scores.shape[1], device=scores.device)
        sum_exp.index_add_(0, indices, exp_scores)
        
        # Divide
        probs = exp_scores / (sum_exp[indices] + 1e-8)
        
        return probs


class RegionTextCrossAttention(nn.Module):
    """
    Cross-attention between sketch regions and text tokens.
    
    Associates text tokens with relevant sketch regions based on
    semantic similarity and spatial context.
    """
    
    def __init__(
        self,
        region_dim: int,
        text_dim: int = 768,  # CLIP text embedding dim
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize region-text cross-attention.
        
        Args:
            region_dim: Region feature dimension
            text_dim: Text embedding dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.region_dim = region_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Project regions and text to same dimension
        self.region_proj = nn.Linear(region_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # Query from regions
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)  # Key from text
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)  # Value from text
        
        self.out_proj = nn.Linear(hidden_dim, region_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        region_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute region-text cross-attention.
        
        Args:
            region_features: Region features (N, region_dim)
            text_embeddings: Text embeddings (T, text_dim)
            attention_mask: Optional mask for text tokens (T,)
        
        Returns:
            - Updated region features (N, region_dim)
            - Attention map (N, T) showing region-text associations
        """
        N = region_features.shape[0]
        T = text_embeddings.shape[0]
        
        # Project to common space
        region_proj = self.region_proj(region_features)  # (N, hidden_dim)
        text_proj = self.text_proj(text_embeddings)      # (T, hidden_dim)
        
        # Compute Q, K, V
        Q = self.q_proj(region_proj)  # (N, hidden_dim)
        K = self.k_proj(text_proj)    # (T, hidden_dim)
        V = self.v_proj(text_proj)    # (T, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, N, head_dim)
        K = K.view(T, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, T, head_dim)
        V = V.view(T, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, T, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (heads, N, T)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.unsqueeze(0).unsqueeze(1), float('-inf')
            )
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)  # (heads, N, T)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, V)  # (heads, N, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(0, 1).contiguous()  # (N, heads, head_dim)
        attn_output = attn_output.view(N, self.hidden_dim)
        attn_output = self.out_proj(attn_output)  # (N, region_dim)
        
        # Average attention across heads for visualization
        attn_map = attn_probs.mean(dim=0)  # (N, T)
        
        return attn_output, attn_map


class RAGAFAttentionModule(nn.Module):
    """
    Complete RAGAF (Region-Adaptive Graph-Attention Fusion) module.
    
    Combines:
    1. Graph attention over regions
    2. Region-text cross-attention
    3. Residual connections and normalization
    """
    
    def __init__(
        self,
        node_feature_dim: int = 6,  # Spatial features from region extraction
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_graph_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize RAGAF attention module.
        
        Args:
            node_feature_dim: Input node feature dimension
            text_dim: Text embedding dimension
            hidden_dim: Hidden dimension
            num_graph_layers: Number of graph attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        
        # Initial node feature projection
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # Stack of graph attention layers
        self.graph_layers = nn.ModuleList([
            RegionGraphAttention(
                node_feature_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_graph_layers)
        ])
        
        # Layer norms
        self.graph_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_graph_layers)
        ])
        
        # Region-text cross-attention
        self.cross_attention = RegionTextCrossAttention(
            region_dim=hidden_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        region_graph: RegionGraph,
        text_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RAGAF attention.
        
        Args:
            region_graph: RegionGraph object
            text_embeddings: Text embeddings (T, text_dim)
        
        Returns:
            - Region features enriched with text information (N, hidden_dim)
            - Region-text attention map (N, T)
        """
        # Initial embedding
        node_features = self.node_embedding(region_graph.node_features)  # (N, hidden_dim)
        
        # Apply graph attention layers
        for graph_layer, norm in zip(self.graph_layers, self.graph_norms):
            # Graph attention with residual
            graph_output = graph_layer(
                node_features,
                region_graph.edge_index,
                region_graph.edge_weights
            )
            node_features = norm(node_features + graph_output)
        
        # Cross-attention with text
        cross_output, attn_map = self.cross_attention(
            node_features,
            text_embeddings
        )
        
        # Residual and norm
        node_features = self.cross_attn_norm(node_features + cross_output)
        
        # Final projection
        output_features = self.output_proj(node_features)
        
        return output_features, attn_map


if __name__ == "__main__":
    # Example usage
    print("RAGAF Attention Module for Region-Text Fusion")
    print("=" * 60)
    
    # Create dummy region graph
    from data.region_graph import RegionGraph
    
    num_nodes = 10
    node_features = torch.randn(num_nodes, 6)  # Spatial features
    edge_index = torch.randint(0, num_nodes, (2, 30))  # Random edges
    edge_weights = torch.rand(30)
    
    region_graph = RegionGraph(
        num_nodes=num_nodes,
        node_features=node_features,
        edge_index=edge_index,
        edge_weights=edge_weights,
        region_masks=[],
        adjacency_matrix=None
    )
    
    # Create dummy text embeddings
    text_embeddings = torch.randn(77, 768)  # CLIP embeddings
    
    # Create RAGAF module
    ragaf = RAGAFAttentionModule(
        node_feature_dim=6,
        text_dim=768,
        hidden_dim=512,
        num_graph_layers=2
    )
    
    # Forward pass
    output_features, attn_map = ragaf(region_graph, text_embeddings)
    
    print(f"Input node features: {node_features.shape}")
    print(f"Text embeddings: {text_embeddings.shape}")
    print(f"Output node features: {output_features.shape}")
    print(f"Attention map: {attn_map.shape}")
    print(f"\nAttention map statistics:")
    print(f"  Min: {attn_map.min().item():.4f}")
    print(f"  Max: {attn_map.max().item():.4f}")
    print(f"  Mean: {attn_map.mean().item():.4f}")
