"""
RAMMD: Complete Integration
All 7 architectural components combined (Sections 3.3-3.10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .regime_detection import RegimeDetector
from .encoders import PriceEncoder, NewsEncoder, SocialEncoder, MacroEncoder
from .moe import RegimeConditionedMoE
from .contrastive_learning import FOCALContrastiveLearning
from .gnn import TemporalGraphAttentionNetwork
from .wavelet_attention import MultiScaleWaveletAttention
from .fusion import RegimeConditionedFusion


class RAMMD(nn.Module):
    """
    Complete RAMMD Architecture
    
    Components:
    1. Regime Detection (GMM + HMM + Drift)
    2. Modality Encoders (PatchTST, FinBERT, DistilBERT, MLP)
    3. Mixture-of-Experts (per regime, per modality)
    4. FOCAL Contrastive Learning (shared/private decomposition)
    5. Temporal Graph Neural Network
    6. Multi-Scale Wavelet Attention
    7. Dynamic Fusion with Regime Gating
    + Prediction Heads (regression, classification, volatility, regime)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_regimes = config.get('num_regimes', 4)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # ====================================================================
        # MODULE 1: Regime Detection
        # ====================================================================
        regime_config = config.get('regime_detection', {})
        self.regime_detector = RegimeDetector(
            n_regimes=regime_config.get('num_regimes', 4),
            feature_dim=regime_config.get('feature_dim', 32),
            covariance_type=regime_config.get('covariance_type', 'full'),
            hmm_smoothing=regime_config.get('hmm_smoothing', True),
            drift_detection=regime_config.get('drift_detection', True)
        )
        
        # ====================================================================
        # MODULE 2: Modality Encoders
        # ====================================================================
        encoders_config = config.get('encoders', {})
        
        # Price Encoder (PatchTST)
        price_config = encoders_config.get('price', {})
        self.price_encoder = PriceEncoder(
            window_size=price_config.get('window_size', 252),
            input_channels=price_config.get('input_channels', 23),
            patch_length=price_config.get('patch_length', 16),
            stride=price_config.get('stride', 16),
            d_model=price_config.get('d_model', 512),
            n_heads=price_config.get('n_heads', 8),
            n_layers=price_config.get('n_layers', 4),
            dropout=price_config.get('dropout', 0.1),
            pretrained_path=price_config.get('pretrained_path', None)
        )
        
        # News Encoder (FinBERT)
        news_config = encoders_config.get('news', {})
        self.news_encoder = NewsEncoder(
            model_name=news_config.get('model_name', 'yiyanghkust/finbert-tone'),
            pretrained_path=news_config.get('pretrained_path', 'checkpoints/pretrained/finbert_tone'),
            max_length=news_config.get('max_length', 512),
            d_model=news_config.get('d_model', 768)
        )
        
        # Social Encoder (DistilBERT)
        social_config = encoders_config.get('social', {})
        self.social_encoder = SocialEncoder(
            model_name=social_config.get('model_name', 'distilbert-base-uncased'),
            pretrained_path=social_config.get('pretrained_path', 'checkpoints/pretrained/distilbert'),
            max_length=social_config.get('max_length', 280),
            d_model=social_config.get('d_model', 768)
        )
        
        # Macro Encoder (MLP)
        macro_config = encoders_config.get('macro', {})
        self.macro_encoder = MacroEncoder(
            input_dim=macro_config.get('input_dim', 24),
            hidden_dims=macro_config.get('hidden_dims', [128, 256, 512]),
            output_dim=macro_config.get('output_dim', 256),
            dropout=macro_config.get('dropout', 0.1)
        )
        
        # ====================================================================
        # MODULE 3: Mixture-of-Experts (per modality)
        # ====================================================================
        moe_config = config.get('moe', {})
        
        self.moe_price = RegimeConditionedMoE(
            input_dim=price_config.get('d_model', 512),
            hidden_dim=moe_config.get('hidden_dim', 512),
            output_dim=moe_config.get('output_dim', 512),
            num_experts=moe_config.get('num_experts_per_modality', 4),
            num_regimes=self.num_regimes,
            top_k=moe_config.get('top_k', 2),
            group_attention=moe_config.get('group_attention', True),
            attention_heads=moe_config.get('group_attention_heads', 4)
        )
        
        self.moe_news = RegimeConditionedMoE(
            input_dim=news_config.get('d_model', 768),
            hidden_dim=moe_config.get('hidden_dim', 512),
            output_dim=moe_config.get('output_dim', 512),
            num_experts=moe_config.get('num_experts_per_modality', 4),
            num_regimes=self.num_regimes,
            top_k=moe_config.get('top_k', 2)
        )
        
        self.moe_social = RegimeConditionedMoE(
            input_dim=social_config.get('d_model', 768),
            hidden_dim=moe_config.get('hidden_dim', 512),
            output_dim=moe_config.get('output_dim', 512),
            num_experts=moe_config.get('num_experts_per_modality', 4),
            num_regimes=self.num_regimes,
            top_k=moe_config.get('top_k', 2)
        )
        
        self.moe_macro = RegimeConditionedMoE(
            input_dim=macro_config.get('output_dim', 256),
            hidden_dim=moe_config.get('hidden_dim', 512),
            output_dim=moe_config.get('output_dim', 512),
            num_experts=moe_config.get('num_experts_per_modality', 4),
            num_regimes=self.num_regimes,
            top_k=moe_config.get('top_k', 2)
        )
        
        # ====================================================================
        # MODULE 4: FOCAL Contrastive Learning
        # ====================================================================
        contrastive_config = config.get('contrastive', {})
        self.focal_contrastive = FOCALContrastiveLearning(
            input_dim=moe_config.get('output_dim', 512),
            shared_dim=contrastive_config.get('shared_dim', 256),
            private_dim=contrastive_config.get('private_dim', 256),
            num_modalities=4,
            temperature=contrastive_config.get('temperature', 0.07),
            lambda_private=contrastive_config.get('lambda_private', 0.5),
            lambda_temporal=contrastive_config.get('lambda_temporal', 0.3),
            lambda_ortho=contrastive_config.get('lambda_ortho', 1.0),
            regime_aware_sampling=contrastive_config.get('regime_aware_sampling', True),
            num_regimes=self.num_regimes
        )
        
        # ====================================================================
        # MODULE 5: Temporal Graph Neural Network
        # ====================================================================
        gnn_config = config.get('gnn', {})
        total_modality_dim = moe_config.get('output_dim', 512) * 4
        
        self.gnn = TemporalGraphAttentionNetwork(
            input_dim=gnn_config.get('input_dim', total_modality_dim),
            hidden_dim=gnn_config.get('hidden_dim', 512),
            output_dim=gnn_config.get('output_dim', 512),
            n_layers=gnn_config.get('n_layers', 2),
            n_heads=gnn_config.get('n_heads', 4),
            dropout=gnn_config.get('dropout', 0.1)
        )
        
        # ====================================================================
        # MODULE 6: Multi-Scale Wavelet Attention
        # ====================================================================
        wavelet_config = config.get('wavelet', {})
        self.wavelet_attention = MultiScaleWaveletAttention(
            input_dim=moe_config.get('output_dim', 512),
            wavelet_type=wavelet_config.get('wavelet_type', 'db8'),
            decomposition_levels=wavelet_config.get('decomposition_levels', 5),
            attention_heads=wavelet_config.get('attention_heads', 4),
            regime_dependent=wavelet_config.get('regime_dependent', True),
            num_regimes=self.num_regimes
        )
        
        # ====================================================================
        # MODULE 7: Dynamic Fusion
        # ====================================================================
        fusion_config = config.get('fusion', {})
        self.fusion_module = RegimeConditionedFusion(
            modality_dims={
                'price': moe_config.get('output_dim', 512),
                'news': moe_config.get('output_dim', 512),
                'social': moe_config.get('output_dim', 512),
                'macro': moe_config.get('output_dim', 512)
            },
            regime_feature_dim=self.num_regimes,
            hidden_dim=fusion_config.get('hidden_dim', 1024),
            output_dim=fusion_config.get('output_dim', 512),
            num_regimes=self.num_regimes,
            dropout=fusion_config.get('dropout', 0.2)
        )
        
        # ====================================================================
        # PREDICTION HEADS
        # ====================================================================
        pred_input_dim = fusion_config.get('output_dim', 512)
        pred_config = config.get('prediction', {})
        shared_hidden = pred_config.get('shared_hidden', 256)
        
        # Regression Head (next-day return)
        if pred_config.get('regression_head', True):
            self.regression_head = nn.Sequential(
                nn.Linear(pred_input_dim, shared_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(shared_hidden, 1)
            )
        
        # Classification Head (direction: down/flat/up)
        if pred_config.get('classification_head', True):
            self.classification_head = nn.Sequential(
                nn.Linear(pred_input_dim, shared_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(shared_hidden, 3)
            )
        
        # Volatility Head
        if pred_config.get('volatility_head', True):
            self.volatility_head = nn.Sequential(
                nn.Linear(pred_input_dim, shared_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(shared_hidden, 1),
                nn.Softplus()  # Ensure positive
            )
        
        # Regime Prediction Head
        if pred_config.get('regime_head', True):
            self.regime_head = nn.Sequential(
                nn.Linear(pred_input_dim, shared_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(shared_hidden, self.num_regimes)
            )
        
        self.output_norm = nn.LayerNorm(pred_input_dim)
        
    def forward(
        self,
        price_data: torch.Tensor,
        news_texts: Optional[List[List[str]]] = None,
        social_texts: Optional[List[List[str]]] = None,
        macro_data: torch.Tensor = None,
        graph_data: Optional[Dict] = None,
        regime_features: Optional[torch.Tensor] = None,
        return_explanations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through RAMMD
        
        Args:
            price_data: (batch, window, features) price time series
            news_texts: List[List[str]] news articles per asset
            social_texts: List[List[str]] social posts per asset
            macro_data: (batch, macro_dim) macroeconomic indicators
            graph_data: Optional graph structure
            regime_features: Optional pre-computed regime features
            return_explanations: Whether to compute SHAP values
            
        Returns:
            outputs: Dict with predictions and auxiliary information
        """
        batch_size = price_data.size(0)
        device = price_data.device
        outputs = {}
        
        # ====================================================================
        # STEP 1: Regime Detection
        # ====================================================================
        if regime_features is None:
            regime_features = self._extract_regime_features(price_data)
        
        regime_labels, regime_probs = self.regime_detector.predict(
            regime_features.cpu().numpy()
        )
        regime_labels = torch.tensor(regime_labels, device=device, dtype=torch.long)
        regime_probs = torch.tensor(regime_probs, device=device, dtype=torch.float32)
        regime_onehot = F.one_hot(regime_labels, num_classes=self.num_regimes).float()
        
        outputs['regime_labels'] = regime_labels
        outputs['regime_probs'] = regime_probs
        
        # ====================================================================
        # STEP 2: Modality Encoding
        # ====================================================================
        h_price = self.price_encoder(price_data)
        
        # Handle news (may be None)
        if news_texts and len(news_texts) > 0 and any(len(texts) > 0 for texts in news_texts):
            h_news_list = []
            for asset_news in news_texts:
                if len(asset_news) > 0:
                    h_news = self.news_encoder(asset_news)
                    h_news_list.append(h_news)
                else:
                    h_news_list.append(torch.zeros(1, 768, device=device))
            h_news = torch.cat(h_news_list, dim=0)
        else:
            h_news = torch.zeros(batch_size, 768, device=device)
        
        # Handle social media (may be None)
        if social_texts and len(social_texts) > 0 and any(len(texts) > 0 for texts in social_texts):
            h_social_list = []
            for asset_social in social_texts:
                if len(asset_social) > 0:
                    h_social = self.social_encoder(asset_social)
                    h_social_list.append(h_social)
                else:
                    h_social_list.append(torch.zeros(1, 768, device=device))
            h_social = torch.cat(h_social_list, dim=0)
        else:
            h_social = torch.zeros(batch_size, 768, device=device)
        
        # Macro encoding
        if macro_data is not None:
            h_macro = self.macro_encoder(macro_data)
        else:
            h_macro = torch.zeros(batch_size, 256, device=device)
        
        # ====================================================================
        # STEP 3: Mixture-of-Experts Processing
        # ====================================================================
        h_price_moe, routing_price = self.moe_price(h_price, regime_labels, regime_onehot)
        h_news_moe, routing_news = self.moe_news(h_news, regime_labels, regime_onehot)
        h_social_moe, routing_social = self.moe_social(h_social, regime_labels, regime_onehot)
        h_macro_moe, routing_macro = self.moe_macro(h_macro, regime_labels, regime_onehot)
        
        outputs['routing_weights'] = {
            'price': routing_price['routing_weights'],
            'news': routing_news['routing_weights'],
            'social': routing_social['routing_weights'],
            'macro': routing_macro['routing_weights']
        }
        
        # ====================================================================
        # STEP 4: Cross-Modal Contrastive Learning (FOCAL)
        # ====================================================================
        modality_embeddings = {
            'price': h_price_moe,
            'news': h_news_moe,
            'social': h_social_moe,
            'macro': h_macro_moe
        }
        
        focal_output = self.focal_contrastive(
            modality_embeddings,
            regime_labels=regime_labels
        )
        
        h_price_focal = focal_output['price']['combined']
        h_news_focal = focal_output['news']['combined']
        h_social_focal = focal_output['social']['combined']
        h_macro_focal = focal_output['macro']['combined']
        
        outputs['contrastive_loss'] = focal_output['contrastive_loss']
        outputs['contrastive_components'] = focal_output['loss_components']
        
        # ====================================================================
        # STEP 5: Temporal Graph Neural Network
        # ====================================================================
        node_features = torch.cat([
            h_price_focal, h_news_focal, h_social_focal, h_macro_focal
        ], dim=-1)
        
        if graph_data is not None:
            h_gnn = self.gnn(
                node_features,
                graph_data.get('edge_index'),
                graph_data.get('edge_weights')
            )
        else:
            h_gnn = self.gnn(node_features, None, None)
        
        outputs['gnn_embeddings'] = h_gnn
        
        # ====================================================================
        # STEP 6: Multi-Scale Wavelet Attention
        # ====================================================================
        h_wavelet = self.wavelet_attention(h_price_focal, regime_labels=regime_labels)
        outputs['wavelet_enhanced'] = h_wavelet
        
        # ====================================================================
        # STEP 7: Dynamic Multimodal Fusion
        # ====================================================================
        modality_features_final = {
            'price': h_price_focal,
            'news': h_news_focal,
            'social': h_social_focal,
            'macro': h_macro_focal
        }
        
        h_fused, modality_weights = self.fusion_module(
            modality_features_final,
            regime_onehot,
            h_gnn
        )
        
        outputs['modality_weights'] = modality_weights
        
        # Layer normalization
        h_fused = self.output_norm(h_fused)
        
        # ====================================================================
        # STEP 8: Prediction Heads
        # ====================================================================
        if hasattr(self, 'regression_head'):
            regression_output = self.regression_head(h_fused)
            outputs['regression_output'] = regression_output.squeeze(-1)
        
        if hasattr(self, 'classification_head'):
            classification_logits = self.classification_head(h_fused)
            outputs['classification_output'] = F.softmax(classification_logits, dim=-1)
        
        if hasattr(self, 'volatility_head'):
            volatility_output = self.volatility_head(h_fused)
            outputs['volatility_output'] = volatility_output.squeeze(-1)
        
        if hasattr(self, 'regime_head'):
            regime_pred_logits = self.regime_head(h_fused)
            outputs['regime_pred'] = F.softmax(regime_pred_logits, dim=-1)
        
        return outputs
    
    def _extract_regime_features(self, price_data: torch.Tensor) -> torch.Tensor:
        """Extract regime detection features from price data"""
        # Compute returns from close prices (assuming index 3 is close)
        returns = torch.diff(torch.log(price_data[:, :, 3] + 1e-8), dim=1)
        
        # Basic features
        mean_return = returns.mean(dim=1).mean()
        volatility = returns.std(dim=1).mean()
        
        regime_features = torch.tensor(
            [[mean_return.item(), volatility.item()]],
            device=price_data.device
        )
        
        return regime_features
    
    def fit_regime_detector(self, historical_data: Dict, verbose: bool = True):
        """Fit GMM regime detector on historical data"""
        price_data = historical_data['price_data']
        
        if isinstance(price_data, torch.Tensor):
            price_data_np = price_data.cpu().numpy()
        else:
            price_data_np = price_data
        
        # Compute returns
        returns = np.diff(np.log(price_data_np[:, :, 3] + 1e-8), axis=1)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Extract regime features
        regime_features = self.regime_detector.extract_regime_features(
            returns_tensor,
            volumes=historical_data.get('volumes'),
            vix=historical_data.get('vix')
        )
        
        # Fit detector
        regime_labels = self.regime_detector.fit(regime_features, verbose=verbose)
        
        return regime_labels
