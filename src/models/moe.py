"""
Regime-Conditioned Mixture-of-Experts (Section 3.5)
Implements Equations 26-33 with:
- Per-regime expert routing
- Top-k sparse gating
- Group attention
- Load balancing loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class Expert(nn.Module):
    """Single expert network with residual connection"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Residual connection
        if input_dim != output_dim:
            self.residual = nn.Linear(input_dim, output_dim)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) + self.residual(x)


class RegimeConditionedMoE(nn.Module):
    """
    Mixture of Experts with regime-conditioned routing
    
    Paper Equations 26-33:
    - Expert routing: w_e = softmax(Router(h, z))
    - Top-k gating: select top-k experts
    - Group attention: aggregate expert outputs
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        num_regimes: int = 4,
        top_k: int = 2,
        group_attention: bool = True,
        attention_heads: int = 4,
        expert_dropout: float = 0.1,
        load_balancing_loss: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_regimes = num_regimes
        self.top_k = top_k
        self.group_attention = group_attention
        self.load_balancing_loss = load_balancing_loss
        
        # Create experts for each regime
        self.experts = nn.ModuleDict()
        for regime_id in range(num_regimes):
            regime_experts = nn.ModuleList([
                Expert(input_dim, hidden_dim, output_dim, expert_dropout)
                for _ in range(num_experts)
            ])
            self.experts[f"regime_{regime_id}"] = regime_experts
        
        # Routing network (conditioned on regime and input)
        # Equation 26: w_e = Router(h, z)
        self.routing_network = nn.ModuleDict()
        for regime_id in range(num_regimes):
            self.routing_network[f"regime_{regime_id}"] = nn.Sequential(
                nn.Linear(input_dim + num_regimes, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_experts)
            )
        
        # Group attention for expert aggregation
        if group_attention:
            self.group_attn = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=attention_heads,
                dropout=0.1,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(output_dim)
        
        # For load balancing loss tracking
        self.register_buffer('expert_usage', torch.zeros(num_regimes, num_experts))
        
    def forward(
        self,
        x: torch.Tensor,
        regime: torch.Tensor,
        regime_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with regime-conditioned routing
        
        Args:
            x: (batch_size, input_dim) input features
            regime: (batch_size,) regime labels
            regime_features: (batch_size, num_regimes) one-hot regime encoding
            
        Returns:
            output: (batch_size, output_dim) MoE output
            routing_info: Dict with routing weights and auxiliary losses
        """
        batch_size = x.size(0)
        device = x.device
        
        outputs = []
        routing_weights_list = []
        expert_outputs_all = []
        
        for i in range(batch_size):
            regime_id = regime[i].item()
            input_i = x[i:i+1]  # (1, input_dim)
            regime_feat_i = regime_features[i:i+1]  # (1, num_regimes)
            
            # === Compute routing weights (Equation 26-27) ===
            routing_input = torch.cat([input_i, regime_feat_i], dim=-1)
            routing_logits = self.routing_network[f"regime_{regime_id}"](routing_input)
            routing_weights = F.softmax(routing_logits, dim=-1)  # (1, num_experts)
            
            # === Top-k sparse gating (Equation 28) ===
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # Renormalize
            
            # === Get expert outputs (Equation 29-30) ===
            experts = self.experts[f"regime_{regime_id}"]
            expert_outputs = []
            for expert in experts:
                expert_output = expert(input_i)  # (1, output_dim)
                expert_outputs.append(expert_output)
            expert_outputs = torch.stack(expert_outputs, dim=1)  # (1, num_experts, output_dim)
            
            # === Group attention (Equation 31-32) ===
            if self.group_attention:
                # Self-attention among expert outputs
                expert_outputs_attn, _ = self.group_attn(
                    expert_outputs, expert_outputs, expert_outputs
                )
                expert_outputs = self.attn_norm(expert_outputs + expert_outputs_attn)
            
            # === Weighted combination of top-k experts (Equation 33) ===
            output_i = torch.zeros(1, self.output_dim, device=device)
            for k in range(self.top_k):
                expert_idx = top_k_indices[0, k]
                weight = top_k_weights[0, k]
                output_i += weight * expert_outputs[0, expert_idx]
            
            outputs.append(output_i)
            routing_weights_list.append(routing_weights)
            expert_outputs_all.append(expert_outputs)
            
            # Track expert usage for load balancing
            if self.training:
                for k in range(self.top_k):
                    expert_idx = top_k_indices[0, k]
                    self.expert_usage[regime_id, expert_idx] += 1
        
        # Stack outputs
        output = torch.cat(outputs, dim=0)  # (batch_size, output_dim)
        routing_weights_batch = torch.cat(routing_weights_list, dim=0)  # (batch_size, num_experts)
        
        # === Compute load balancing loss ===
        load_balance_loss = torch.tensor(0.0, device=device)
        if self.load_balancing_loss and self.training:
            # Encourage uniform expert usage
            expert_usage_norm = self.expert_usage / (self.expert_usage.sum(dim=-1, keepdim=True) + 1e-8)
            target_usage = 1.0 / self.num_experts
            load_balance_loss = F.mse_loss(expert_usage_norm, torch.full_like(expert_usage_norm, target_usage))
        
        routing_info = {
            "routing_weights": routing_weights_batch,
            "top_k_indices": top_k_indices,
            "load_balance_loss": load_balance_loss,
            "expert_usage": self.expert_usage.clone()
        }
        
        return output, routing_info
    
    def reset_expert_usage(self):
        """Reset expert usage statistics"""
        self.expert_usage.zero_()
