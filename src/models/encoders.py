"""
Modality-Specific Encoders (Section 3.4)
- PriceEncoder: PatchTST for price time series (Equations 14-18)
- NewsEncoder: FinBERT for news text (Equations 19-20)
- SocialEncoder: DistilBERT for social media (Equations 21-24)
- MacroEncoder: MLP for macroeconomic indicators (Equation 25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional
from pathlib import Path


class PriceEncoder(nn.Module):
    """
    PatchTST-based encoder for price time series
    
    Paper Equations 14-18:
    - Patching: P_i = {p_{i,1}, ..., p_{i,L_p}}
    - Embedding: e_i = W_e · P_i + PE_i
    - Transformer: H^price = PatchTST(E)
    """
    
    def __init__(
        self,
        window_size: int = 252,
        input_channels: int = 23,  # OHLCV + 18 indicators
        patch_length: int = 16,
        stride: int = 16,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        self.window_size = window_size
        self.input_channels = input_channels
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        
        # Patch embedding layer
        self.patch_embedding = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=patch_length,
            stride=stride
        )
        
        # Positional encoding
        num_patches = (window_size - patch_length) // stride + 1
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, window_size, input_channels) price data
            
        Returns:
            embedding: (batch_size, d_model) encoded representation
        """
        batch_size = x.size(0)
        
        # Transpose for Conv1d: (batch, channels, time)
        x = x.permute(0, 2, 1)
        
        # Patch embedding (Equation 14-15)
        patches = self.patch_embedding(x)  # (batch, d_model, num_patches)
        patches = patches.permute(0, 2, 1)  # (batch, num_patches, d_model)
        
        # Add positional encoding (Equation 16)
        patches = patches + self.positional_encoding[:, :patches.size(1), :]
        
        # Transformer encoding (Equation 17-18)
        encoded = self.transformer(patches)  # (batch, num_patches, d_model)
        
        # Global average pooling
        pooled = torch.mean(encoded, dim=1)  # (batch, d_model)
        
        # Layer normalization
        output = self.norm(pooled)
        
        return output


class NewsEncoder(nn.Module):
    """
    FinBERT-based encoder for news text
    
    Paper Equations 19-20:
    - h_i^news = FinBERT(text_i)
    - Aggregation with recency weighting
    """
    
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        pretrained_path: Optional[str] = "checkpoints/pretrained/finbert_tone",
        max_length: int = 512,
        d_model: int = 768,
        aggregation_window: int = 5,
        recency_weight_decay: float = 0.95
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.d_model = d_model
        self.aggregation_window = aggregation_window
        self.recency_weight_decay = recency_weight_decay
        
        # Load FinBERT from local checkpoint
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading FinBERT from {pretrained_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
        else:
            print(f"Loading FinBERT from HuggingFace: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Freeze lower layers for efficiency
        for param in list(self.model.parameters())[:-24]:  # Freeze all but last 6 layers
            param.requires_grad = False
        
        # Sentiment dimension (positive, negative, neutral)
        self.sentiment_dim = 3
        
        # Projection layer
        self.projection = nn.Linear(d_model + self.sentiment_dim, d_model)
        
    def forward(
        self,
        texts: List[str],
        recency_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            texts: List of news articles
            recency_weights: (num_articles,) time decay weights
            
        Returns:
            embedding: (1, d_model) aggregated news embedding
        """
        if not texts or len(texts) == 0:
            return torch.zeros(1, self.d_model, device=self.model.device)
        
        # Tokenize (Equation 19)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Forward through FinBERT
        outputs = self.model(**encoded, output_hidden_states=True)
        
        # Extract [CLS] embeddings from last hidden state
        cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # (num_articles, d_model)
        
        # Get sentiment scores
        sentiment_probs = torch.softmax(outputs.logits, dim=-1)  # (num_articles, 3)
        
        # Concatenate embeddings and sentiment
        combined = torch.cat([cls_embeddings, sentiment_probs], dim=-1)
        
        # Project to unified dimension
        embeddings = self.projection(combined)
        
        # Apply recency weighting (Equation 20)
        if recency_weights is None:
            # Exponential decay: more recent = higher weight
            recency_weights = torch.tensor([
                self.recency_weight_decay ** i for i in range(len(texts) - 1, -1, -1)
            ], device=embeddings.device)
        
        recency_weights = recency_weights.to(embeddings.device)
        recency_weights = recency_weights / recency_weights.sum()
        
        # Weighted aggregation
        aggregated = torch.sum(embeddings * recency_weights.unsqueeze(1), dim=0, keepdim=True)
        
        return aggregated  # (1, d_model)


class SocialEncoder(nn.Module):
    """
    DistilBERT-based encoder for social media posts
    
    Paper Equations 21-24:
    - h_i^social = DistilBERT(post_i)
    - Author influence weighting
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        pretrained_path: Optional[str] = "checkpoints/pretrained/distilbert",
        max_length: int = 280,
        d_model: int = 768,
        author_weighting: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.d_model = d_model
        self.author_weighting = author_weighting
        
        # Load DistilBERT from local checkpoint
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading DistilBERT from {pretrained_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            self.model = AutoModel.from_pretrained(pretrained_path)
        else:
            print(f"Loading DistilBERT from HuggingFace: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze most layers
        for param in list(self.model.parameters())[:-16]:  # Freeze all but last 4 layers
            param.requires_grad = False
    
    def forward(
        self,
        texts: List[str],
        author_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            texts: List of social media posts
            author_weights: (num_posts,) author influence weights
            
        Returns:
            embedding: (1, d_model) aggregated social embedding
        """
        if not texts or len(texts) == 0:
            return torch.zeros(1, self.d_model, device=self.model.device)
        
        # Tokenize (Equation 21)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Forward through DistilBERT (Equations 22-23)
        outputs = self.model(**encoded)
        
        # Extract [CLS] embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (num_posts, d_model)
        
        # Apply author weighting (Equation 24)
        if self.author_weighting and author_weights is not None:
            author_weights = author_weights.to(cls_embeddings.device)
            author_weights = author_weights / author_weights.sum()
            aggregated = torch.sum(cls_embeddings * author_weights.unsqueeze(1), dim=0, keepdim=True)
        else:
            # Uniform weighting
            aggregated = torch.mean(cls_embeddings, dim=0, keepdim=True)
        
        return aggregated  # (1, d_model)


class MacroEncoder(nn.Module):
    """
    MLP encoder for macroeconomic indicators
    
    Paper Equation 25:
    - h^macro = MLP(M_t)
    """
    
    def __init__(
        self,
        input_dim: int = 24,
        hidden_dims: List[int] = [128, 256, 512],
        output_dim: int = 256,
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) macro indicators
            
        Returns:
            embedding: (batch_size, output_dim) encoded representation
        """
        return self.encoder(x)
