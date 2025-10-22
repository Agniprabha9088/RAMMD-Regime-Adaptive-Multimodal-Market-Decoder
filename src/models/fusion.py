"""
Dynamic Multimodal Fusion with Regime-Aware Gating (Section 3.9)
Implements Equations 67-74
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class RegimeConditionedFusion(nn.Module):
    """
    Regime-conditioned gating and dynamic fusion
    
    Paper Equations 67-74:
    - Modality-specific transformations
    - Regime-aware gating mechanism
    - Dynamic weight assignment
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        regime_feature_dim: int = 4,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        num_regimes: int = 4,
        dropout: float = 0.2,
        modality_dropout: float = 0.1
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.regime_feature_dim = regime_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_regimes = num_regimes
        
        # === Modality-specific transformations (Equation 67-68) ===
        self.modality_transforms = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for modality, dim in modality_dims.items()
        })
        
        # === Regime-conditioned gating network (Equations 69-71) ===
        total_dim = hidden_dim * self.num_modalities
        
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim + regime_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # === Fusion projection (Equation 72-73) ===
        self.fusion_projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # === GNN integration (Equation 74) ===
        self.gnn_gate = nn.Linear(output_dim * 2, output_dim)
        
        # Modality dropout for robustness
        self.modality_dropout = nn.Dropout(modality_dropout)
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        regime_features: torch.Tensor,
        gnn_features: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass with regime-conditioned fusion
        
        Args:
            modality_features: Dict of {modality: tensor (batch, modality_dim)}
            regime_features: (batch, regime_feature_dim) regime encoding
            gnn_features: (batch, gnn_dim) graph-enhanced features (optional)
            
        Returns:
            fused: (batch, output_dim) fused representation
            weights: (batch, num_modalities) modality weights
        """
        batch_size = list(modality_features.values())[0].size(0)
        device = list(modality_features.values())[0].device
        
        # === STEP 1: Modality-specific transformations (Equations 67-68) ===
        transformed = {}
        for modality, features in modality_features.items():
            transformed[modality] = self.modality_transforms[modality](features)
        
        # Concatenate all modality features
        concat_features = torch.cat([transformed[m] for m in sorted(transformed.keys())], dim=-1)
        
        # === STEP 2: Regime-conditioned gating (Equations 69-71) ===
        # Concatenate with regime features
        gate_input = torch.cat([concat_features, regime_features], dim=-1)
        
        # Compute modality weights
        modality_weights = self.gate_network(gate_input)  # (batch, num_modalities)
        
        # === STEP 3: Weighted modality features ===
        # Apply weights to each modality's transformed features
        weighted_features = []
        for i, modality in enumerate(sorted(transformed.keys())):
            weight = modality_weights[:, i:i+1]  # (batch, 1)
            weighted = weight * transformed[modality]
            weighted_features.append(weighted)
        
        # Apply modality dropout (randomly zero out entire modalities for robustness)
        if self.training:
            for i in range(len(weighted_features)):
                weighted_features[i] = self.modality_dropout(weighted_features[i])
        
        # Concatenate weighted features
        weighted_concat = torch.cat(weighted_features, dim=-1)
        
        # === STEP 4: Fusion projection (Equations 72-73) ===
        fused = self.fusion_projection(weighted_concat)  # (batch, output_dim)
        
        # === STEP 5: Integrate GNN features (Equation 74) ===
        if gnn_features is not None:
            # Adaptive gating between fused features and GNN features
            combined = torch.cat([fused, gnn_features], dim=-1)
            gate = torch.sigmoid(self.gnn_gate(combined))
            fused = gate * fused + (1 - gate) * gnn_features
        
        # === STEP 6: Layer normalization ===
        fused = self.norm(fused)
        
        return fused, modality_weights
