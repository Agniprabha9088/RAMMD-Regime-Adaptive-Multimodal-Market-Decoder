"""
FOCAL: Factorized Orthogonal Contrastive Learning (Section 3.6)
Implements Equations 34-42 with:
- Shared/Private space decomposition
- InfoNCE loss with regime-aware sampling
- Transformation-invariant loss
- Temporal structural constraint
- Orthogonality regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FOCALContrastiveLearning(nn.Module):
    """
    Cross-Modal Contrastive Learning in Factorized Latent Space
    
    Paper Equations 34-42:
    - Eq 34: h = h_shared + h_private (factorization)
    - Eq 35: h_shared ⊥ h_private (orthogonality)
    - Eq 36-38: InfoNCE with regime-aware negatives
    - Eq 40: Transformation-invariant loss
    - Eq 41: Temporal structural constraint
    - Eq 42: Combined loss
    """
    
    def __init__(
        self,
        input_dim: int,
        shared_dim: int = 256,
        private_dim: int = 256,
        num_modalities: int = 4,
        temperature: float = 0.07,
        lambda_private: float = 0.5,
        lambda_temporal: float = 0.3,
        lambda_ortho: float = 1.0,
        regime_aware_sampling: bool = True,
        num_regimes: int = 4,
        hard_negative_mining: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.num_modalities = num_modalities
        self.temperature = temperature
        self.lambda_private = lambda_private
        self.lambda_temporal = lambda_temporal
        self.lambda_ortho = lambda_ortho
        self.regime_aware_sampling = regime_aware_sampling
        self.num_regimes = num_regimes
        self.hard_negative_mining = hard_negative_mining
        
        # Projection matrices for shared/private decomposition
        modalities = ['price', 'news', 'social', 'macro']
        
        self.shared_projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(input_dim, shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, shared_dim)
            ) for m in modalities
        })
        
        self.private_projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(input_dim, private_dim),
                nn.ReLU(),
                nn.Linear(private_dim, private_dim)
            ) for m in modalities
        })
        
    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        regime_labels: Optional[torch.Tensor] = None,
        temporal_adjacency: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass with FOCAL decomposition and contrastive losses
        
        Args:
            modality_embeddings: Dict of {modality: tensor (batch, input_dim)}
            regime_labels: (batch,) regime assignments for regime-aware sampling
            temporal_adjacency: (batch, batch) temporal proximity matrix
            
        Returns:
            Dict containing shared/private embeddings and contrastive loss
        """
        batch_size = list(modality_embeddings.values())[0].size(0)
        device = list(modality_embeddings.values())[0].device
        
        outputs = {}
        
        # === STEP 1: Factorization into Shared and Private Spaces (Eq 34) ===
        for modality, embedding in modality_embeddings.items():
            # Project to shared space
            h_shared = self.shared_projections[modality](embedding)
            h_shared = F.normalize(h_shared, p=2, dim=-1)
            
            # Project to private space
            h_private = self.private_projections[modality](embedding)
            h_private = F.normalize(h_private, p=2, dim=-1)
            
            # === Enforce orthogonality (Eq 35) ===
            # Gram-Schmidt orthogonalization: remove shared component from private
            inner_product = torch.sum(h_private * h_shared, dim=-1, keepdim=True)
            h_private = h_private - inner_product * h_shared
            h_private = F.normalize(h_private, p=2, dim=-1)
            
            # Combined embedding
            h_combined = torch.cat([h_shared, h_private], dim=-1)
            
            outputs[modality] = {
                'shared': h_shared,
                'private': h_private,
                'combined': h_combined
            }
        
        # === STEP 2: Modal-Matching Contrastive Loss (Shared Space) (Eq 36-38) ===
        shared_loss = 0.0
        modality_pairs = [
            ('price', 'news'),
            ('price', 'social'),
            ('news', 'social'),
            ('price', 'macro'),
            ('news', 'macro'),
            ('social', 'macro')
        ]
        
        for mod1, mod2 in modality_pairs:
            h1_shared = outputs[mod1]['shared']
            h2_shared = outputs[mod2]['shared']
            
            # Compute similarity matrix (Eq 36)
            sim_matrix = torch.matmul(h1_shared, h2_shared.T) / self.temperature
            
            # Positive pairs: diagonal elements (same sample across modalities)
            labels = torch.arange(batch_size, device=device)
            
            # === Regime-aware negative sampling (Eq 38) ===
            if self.regime_aware_sampling and regime_labels is not None:
                # Create regime similarity matrix
                regime_matrix = regime_labels.unsqueeze(0) == regime_labels.unsqueeze(1)
                
                # Hard negatives: samples from different regimes
                gamma = 0.5  # margin for cross-regime samples
                hard_negative_mask = ~regime_matrix
                
                # Apply margin to cross-regime similarities (make them harder negatives)
                sim_matrix = torch.where(
                    hard_negative_mask,
                    sim_matrix - gamma,
                    sim_matrix
                )
            
            # === Hard negative mining (Eq 38 extension) ===
            if self.hard_negative_mining:
                # For each positive pair, find hardest negatives (highest similarity)
                with torch.no_grad():
                    # Mask out positive pairs
                    mask = torch.eye(batch_size, device=device).bool()
                    sim_matrix_masked = sim_matrix.masked_fill(mask, -1e9)
                    
                    # Find hardest negatives
                    hard_negs = torch.topk(sim_matrix_masked, k=min(5, batch_size-1), dim=-1)[0]
                    
                    # Weight matrix: higher weight for harder negatives
                    weights = torch.ones_like(sim_matrix)
                    top_k_indices = torch.topk(sim_matrix_masked, k=min(5, batch_size-1), dim=-1)[1]
                    for i in range(batch_size):
                        weights[i, top_k_indices[i]] = 2.0  # 2x weight for hard negatives
                
                # Apply weights to similarity matrix
                sim_matrix = sim_matrix * weights.detach()
            
            # InfoNCE loss (Eq 36)
            loss_12 = F.cross_entropy(sim_matrix, labels)
            loss_21 = F.cross_entropy(sim_matrix.T, labels)
            shared_loss += (loss_12 + loss_21) / 2
        
        shared_loss = shared_loss / len(modality_pairs)
        
        # === STEP 3: Transformation-Invariant Loss (Private Space) (Eq 40) ===
        private_loss = 0.0
        
        for modality, embedding in modality_embeddings.items():
            h_private = outputs[modality]['private']
            
            # Apply data augmentation (Eq 40)
            # Augmentation 1: Gaussian noise
            noise = torch.randn_like(embedding) * 0.1
            embedding_aug1 = embedding + noise
            h_private_aug1 = self.private_projections[modality](embedding_aug1)
            h_private_aug1 = F.normalize(h_private_aug1, p=2, dim=-1)
            
            # Augmentation 2: Dropout
            dropout_mask = torch.bernoulli(torch.full_like(embedding, 0.9))
            embedding_aug2 = embedding * dropout_mask
            h_private_aug2 = self.private_projections[modality](embedding_aug2)
            h_private_aug2 = F.normalize(h_private_aug2, p=2, dim=-1)
            
            # Contrastive loss: original vs augmented views
            sim_matrix_aug1 = torch.matmul(h_private, h_private_aug1.T) / self.temperature
            sim_matrix_aug2 = torch.matmul(h_private, h_private_aug2.T) / self.temperature
            
            labels = torch.arange(batch_size, device=device)
            loss_aug1 = F.cross_entropy(sim_matrix_aug1, labels)
            loss_aug2 = F.cross_entropy(sim_matrix_aug2, labels)
            
            private_loss += (loss_aug1 + loss_aug2) / 2
        
        private_loss = private_loss / len(modality_embeddings)
        
        # === STEP 4: Temporal Structural Constraint (Eq 41) ===
        temporal_loss = torch.tensor(0.0, device=device)
        
        if temporal_adjacency is not None:
            for modality in modality_embeddings.keys():
                h_combined = outputs[modality]['combined']
                
                # Compute pairwise distances in embedding space
                dist_matrix = torch.cdist(h_combined, h_combined, p=2)
                
                # Temporal constraint: temporally close samples should be close in embedding space
                # temporal_adjacency[i,j] = 1 if |i-j| <= window, else 0
                temporal_loss += torch.mean(dist_matrix * temporal_adjacency)
        
        # === STEP 5: Orthogonality Regularization (Eq 42) ===
        ortho_loss = 0.0
        
        for modality in modality_embeddings.keys():
            h_shared = outputs[modality]['shared']
            h_private = outputs[modality]['private']
            
            # Frobenius norm of inner product matrix (should be zero for orthogonality)
            inner_product = torch.matmul(h_shared.T, h_private)
            ortho_loss += torch.norm(inner_product, p='fro') ** 2
        
        ortho_loss = ortho_loss / len(modality_embeddings)
        
        # === STEP 6: Combined Contrastive Loss (Eq 42) ===
        contrastive_loss = (
            shared_loss + 
            self.lambda_private * private_loss + 
            self.lambda_temporal * temporal_loss + 
            self.lambda_ortho * ortho_loss
        )
        
        outputs['contrastive_loss'] = contrastive_loss
        outputs['loss_components'] = {
            'shared': shared_loss.item(),
            'private': private_loss.item(),
            'temporal': temporal_loss.item(),
            'ortho': ortho_loss.item()
        }
        
        return outputs
