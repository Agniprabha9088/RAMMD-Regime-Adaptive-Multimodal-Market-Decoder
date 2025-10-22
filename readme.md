# RAMMD: Regime-Adaptive Multimodal Market Decoder with Mixture-of-Experts, Cross-Modal Contrastive Learning, and Graph-Structured Attention for Dynamically Weighted Prediction of Financial Asset Prices

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Paper](https://img.shields.io/badge/Paper-PDF-green.svg)](https://doi.org/10.5281/zenodo.17418880)


**Official PyTorch Implementation**

[📄 Paper](https://doi.org/10.5281/zenodo.17418880) | [🚀 Quick Start](#-quick-start) | [📊 Benchmarks](#-performance-benchmarks) 

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Performance Benchmarks](#-performance-benchmarks)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Advanced Usage](#-advanced-usage)
- [Pre-trained Models](#-pre-trained-models)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Configuration](#-configuration)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🌟 Overview

**RAMMD** (Regime-Adaptive Multimodal Market Decoder) is a state-of-the-art deep learning architecture for financial market prediction that dynamically adapts to changing market regimes. By integrating **7 novel architectural components**, RAMMD achieves unprecedented accuracy in multi-horizon forecasting across diverse market conditions.

### 🎯 Research Paper

**Title:** *RAMMD: Regime-Adaptive Multimodal Market Decoder with Mixture-of-Experts, Cross-Modal Contrastive Learning, and Graph-Structured Attention for Dynamically Weighted Prediction of Financial Asset Prices*

**Authors:** Agniprabha Chakraborty¹, Anindya Jana², Manideep Das³  
**Affiliation:** Jadavpur University, West Bengal, India  
**Conference:** IEEE Transactions on Neural Networks and Learning Systems (2025)



---

## ✨ Key Features

### 🧠 **Novel Architectural Innovations**

| Component | Innovation | Impact |
|-----------|------------|--------|
| **Regime Detection** | GMM + HMM + 3-way drift ensemble (ADWIN, KSWIN, Page-Hinkley) | +2.1% accuracy |
| **Mixture-of-Experts** | Regime-conditioned routing with group attention | +1.8% accuracy |
| **FOCAL Contrastive** | Factorized orthogonal latent spaces (shared + private) | +1.5% accuracy |
| **Temporal GAT** | Dynamic graph construction with spillover indices | +1.3% accuracy |
| **Wavelet Attention** | MODWT with 5-level decomposition + regime-dependent scales | +1.7% accuracy |
| **Dynamic Fusion** | Regime-aware gating with modality dropout | Robust to missing data |
| **Explainability** | Multi-level SHAP attribution (modality + expert + temporal) | Regulatory compliance |

### 🚀 **Performance Highlights**

- **59.8%** Directional Accuracy (vs. 51.4% baseline)
- **Sharpe Ratio: 1.68** (vs. 1.12 baseline)
- **MAE: 0.0094** (12.7% improvement)
- **Max Drawdown: 18.2%** (23% reduction)
- **Information Ratio: 0.84** (vs. 0.52 baseline)

### 🔥 **Production-Ready Features**

✅ Multi-modal input support (Price, News, Social Media, Macro)  
✅ Real-time regime adaptation with online drift detection  
✅ Automatic model recalibration on distribution shift  
✅ Mixed precision training (FP16/BF16)  
✅ Distributed training support (DDP, FSDP)  
✅ Checkpoint resuming & early stopping  
✅ W&B / TensorBoard integration  
✅ SHAP-based explainability module  
✅ Walk-forward validation  
✅ Transaction cost modeling  

---

## 🏗️ Architecture

RAMMD integrates 7 interconnected modules in a hierarchical pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAMMD Architecture Pipeline                       │
└─────────────────────────────────────────────────────────────────────┘

Input Data:
├─ Price Time Series (OHLCV + 18 Technical Indicators)
├─ News Articles (FinBERT embeddings)
├─ Social Media Posts (DistilBERT embeddings)
└─ Macroeconomic Indicators (24 features)

                            ↓

┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 1: Regime Detection (GMM + HMM)                              │
│ -  Gaussian Mixture Model (K=4 regimes)                              │
│ -  Hidden Markov Model (temporal smoothing)                          │
│ -  Drift Detection: ADWIN + KSWIN + Page-Hinkley                    │
│ Output: z_t ∈ {Trending, Mean-Rev, Volatile, Crisis}               │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 2: Modality Encoders                                         │
│ ├─ PatchTST (Price): 252-day window → 512-dim                      │
│ ├─ FinBERT (News): Sentiment + embeddings → 768-dim                │
│ ├─ DistilBERT (Social): Author-weighted → 768-dim                  │
│ └─ MLP (Macro): 24 indicators → 256-dim                            │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 3: Regime-Conditioned Mixture-of-Experts                    │
│ -  4 experts per modality per regime (64 total experts)             │
│ -  Top-k sparse gating (k=2)                                         │
│ -  Group attention for inter-expert communication                    │
│ Output: h^(m)_MoE for each modality m                              │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 4: FOCAL Contrastive Learning                                │
│ -  Factorization: h = h_shared + h_private                          │
│ -  Shared space: InfoNCE loss (cross-modal consistency)             │
│ -  Private space: Transformation-invariant loss                      │
│ -  Regime-aware negative sampling                                    │
│ -  Orthogonality constraint: h_shared ⊥ h_private                   │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 5: Temporal Graph Attention Network                          │
│ -  Dynamic graph construction (correlation + spillover)              │
│ -  Multi-head GAT (4 heads, 2 layers)                               │
│ -  Hierarchical pooling (asset → sector → market)                   │
│ Output: h_GNN capturing inter-asset dependencies                    │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 6: Multi-Scale Wavelet Attention (MODWT)                    │
│ -  5-level decomposition (db8 wavelet)                              │
│   - Level 1: 2-4 days (ultra-short)                                │
│   - Level 2: 4-8 days (short)                                       │
│   - Level 3: 8-16 days (medium)                                     │
│   - Level 4: 16-32 days (monthly)                                   │
│   - Level 5: 32-64 days (quarterly)                                 │
│ -  Regime-dependent scale attention                                  │
│ Output: h_wavelet with multi-scale temporal features               │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MODULE 7: Dynamic Multimodal Fusion                                 │
│ -  Regime-conditioned gating network                                 │
│ -  Adaptive modality weighting                                       │
│ -  Integration with GNN features                                     │
│ Output: h_fused (512-dim unified representation)                    │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PREDICTION HEADS                                                     │
│ ├─ Regression: Next-day return prediction                          │
│ ├─ Classification: Direction (up/down/flat)                        │
│ ├─ Volatility: Realized volatility forecast                        │
│ └─ Regime: Next regime prediction                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Benchmarks

### **Forecasting Accuracy (15+ years, 2008-2025)**

| Model | MAE ↓ | RMSE ↓ | Dir. Acc. ↑ | Sharpe ↑ | Max DD ↓ | Info Ratio ↑ |
|-------|-------|---------|-------------|----------|----------|--------------|
| ARIMA | 0.0145 | 0.0234 | 49.8% | 0.42 | 32.4% | 0.21 |
| LSTM | 0.0119 | 0.0203 | 53.4% | 0.89 | 26.8% | 0.41 |
| Transformer | 0.0113 | 0.0195 | 54.8% | 0.97 | 25.1% | 0.45 |
| PatchTST | 0.0109 | 0.0189 | 55.3% | 1.02 | 24.3% | 0.48 |
| TIC-FusionNet | 0.0107 | 0.0186 | 56.2% | 1.08 | 23.7% | 0.50 |
| MIGA (MoE) | 0.0102 | 0.0180 | 57.4% | 1.12 | 20.9% | 0.52 |
| **RAMMD (Ours)** | **0.0094** | **0.0167** | **59.8%** | **1.68** | **18.2%** | **0.84** |

### **Ablation Study**

| Configuration | Dir. Acc. | Sharpe | MAE |
|--------------|-----------|--------|-----|
| RAMMD (Full) | **59.8%** | **1.68** | **0.0094** |
| w/o Regime Detection | 57.7% | 1.52 | 0.0102 |
| w/o MoE | 57.9% | 1.54 | 0.0099 |
| w/o FOCAL | 58.3% | 1.59 | 0.0097 |
| w/o GNN | 58.5% | 1.61 | 0.0096 |
| w/o Wavelet | 58.1% | 1.56 | 0.0098 |
| w/o Drift Detection | 58.9% | 1.63 | 0.0095 |

### **Multi-Market Validation**

| Market | Assets | Dir. Acc. | Sharpe | MAE |
|--------|--------|-----------|--------|-----|
| US (S&P 500) | 500 | 60.2% | 1.71 | 0.0092 |
| US (NASDAQ) | 100 | 59.5% | 1.65 | 0.0095 |
| Europe (STOXX 600) | 600 | 58.9% | 1.58 | 0.0097 |
| Asia (Nikkei 225) | 225 | 58.4% | 1.54 | 0.0099 |
| India (NSE Nifty 50) | 50 | 59.1% | 1.61 | 0.0096 |
| China (SSE Composite) | 300 | 57.8% | 1.49 | 0.0101 |

---

## 🚀 Installation

### **Prerequisites**

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (32GB recommended)
- 10GB+ disk space

### **Method 1: Quick Install (Recommended)**

```bash
# Clone repository
git clone https://github.com/yourusername/RAMMD.git
cd RAMMD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .

# Download pre-trained models
python scripts/download_pretrained_models.py
```

### **Method 2: Manual Install**

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### **Method 3: Docker**

```bash
docker pull rammd/rammd:latest
docker run -it --gpus all rammd/rammd:latest
```

### **Verify Installation**

```python
import torch
from src.models.rammd import RAMMD

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📁 Project Structure

```
RAMMD_Project/
│
├── 📄 README.md                      # This file
├── 📄 requirements.txt               # Dependencies
├── 📄 setup.py                       # Package configuration
├── 📄 .env.example                   # Environment variables template
├── 📄 .gitignore                     # Git ignore rules
│
├── 📁 config/                        # ⚙️ Configuration Files
│   ├── model_config.yaml            # Model architecture settings
│   ├── training_config.yaml         # Training hyperparameters
│   └── data_config.yaml             # Data sources & preprocessing
│
├── 📁 src/                           # 💻 Source Code
│   ├── 📁 data/                     # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py           # RAMMDDataset class
│   │   ├── preprocessor.py          # Data cleaning & normalization
│   │   └── augmentations.py         # Financial data augmentation
│   │
│   ├── 📁 models/                   # 🧠 Neural Network Modules
│   │   ├── __init__.py
│   │   ├── rammd.py                 # Main RAMMD architecture
│   │   ├── regime_detection.py     # GMM + HMM + Drift detection
│   │   ├── encoders.py              # PatchTST, FinBERT, DistilBERT, MLP
│   │   ├── moe.py                   # Mixture-of-Experts
│   │   ├── contrastive_learning.py # FOCAL implementation
│   │   ├── gnn.py                   # Temporal Graph Attention Network
│   │   ├── wavelet_attention.py    # MODWT + Attention
│   │   └── fusion.py                # Dynamic multimodal fusion
│   │
│   ├── 📁 training/                 # 🏋️ Training Infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py               # Advanced trainer (mixed precision, DDP)
│   │   ├── loss_functions.py       # Multi-task losses
│   │   └── optimizer.py             # Custom optimizers & schedulers
│   │
│   ├── 📁 evaluation/               # 📊 Evaluation & Metrics
│   │   ├── __init__.py
│   │   ├── metrics.py               # Financial metrics (Sharpe, MAE, etc.)
│   │   ├── backtesting.py           # Trading simulation engine
│   │   └── explainability.py       # SHAP attribution
│   │
│   └── 📁 utils/                    # 🛠️ Utility Functions
│       ├── __init__.py
│       ├── helpers.py               # General utilities
│       ├── drift_detection.py      # Drift detector implementations
│       └── graph_construction.py   # Dynamic graph building
│
├── 📁 scripts/                       # 🚀 Executable Scripts
│   ├── download_pretrained_models.py # Download FinBERT, DistilBERT, etc.
│   ├── train.py                     # Main training script
│   ├── evaluate.py                  # Evaluation script
│   └── inference.py                 # Real-time inference
│
├── 📁 notebooks/                     # 📓 Jupyter Notebooks
│   ├── 01_data_exploration.ipynb    # Data analysis
│   ├── 02_training_walkthrough.ipynb # Training tutorial
│   └── 03_evaluation_analysis.ipynb # Results visualization
│
├── 📁 tests/                         # ✅ Unit Tests
│   ├── __init__.py
│   ├── test_models.py               # Model tests
│   └── test_data.py                 # Data pipeline tests
│
├── 📁 checkpoints/                   # 💾 Model Checkpoints
│   ├── .gitkeep
│   └── pretrained/                  # Pre-trained models (auto-downloaded)
│       ├── finbert_tone/            # yiyanghkust/finbert-tone
│       ├── finbert_pretrain/        # ProsusAI/finbert
│       ├── distilbert/              # distilbert-base-uncased
│       └── roberta_financial/       # Ensemble model
│
├── 📁 data/                          # 📊 Data Storage
│   ├── raw/                         # Raw financial data
│   │   └── .gitkeep
│   └── processed/                   # Preprocessed & cached data
│       └── .gitkeep
│
└── 📁 logs/                          # 📝 Training Logs
    └── .gitkeep
```

**Total Files:** 50+ Python files, 3 YAML configs, 3 Jupyter notebooks

---

## 🎯 Quick Start

### **1. Download Pre-trained Models**

```bash
python scripts/download_pretrained_models.py
```

**Downloaded Models:**
- `yiyanghkust/finbert-tone` (438 MB) - Financial sentiment analysis
- `ProsusAI/finbert` (420 MB) - General financial text
- `distilbert-base-uncased` (256 MB) - Social media encoding
- `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` (310 MB)

### **2. Prepare Data**

```bash
# Download sample data (SPY, AAPL, GOOGL, MSFT, TSLA)
python scripts/download_data.py --assets SPY,AAPL,GOOGL,MSFT,TSLA \
                                 --start-date 2015-01-01 \
                                 --end-date 2024-12-31
```

### **3. Train Model**

```bash
# Train with default configuration
python scripts/train.py

# Train with custom config
python scripts/train.py --config config/training_config.yaml \
                        --batch-size 64 \
                        --epochs 200 \
                        --learning-rate 3e-4
```

### **4. Evaluate**

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best.pth

# Run backtest with transaction costs
python scripts/evaluate.py --checkpoint checkpoints/best.pth \
                          --backtest \
                          --transaction-cost 0.001
```

### **5. Inference**

```bash
# Predict next-day returns
python scripts/inference.py --checkpoint checkpoints/best.pth \
                           --assets AAPL,GOOGL,MSFT \
                           --date 2024-10-22 \
                           --explain
```

---

## 🔧 Advanced Usage

### **Multi-GPU Training**

```bash
# Distributed Data Parallel (DDP)
torchrun --nproc_per_node=4 scripts/train.py

# Fully Sharded Data Parallel (FSDP)
python scripts/train.py --strategy fsdp
```

### **Mixed Precision Training**

```yaml
# Enabled by default in config/training_config.yaml
training:
  mixed_precision: true  # FP16/BF16
```

### **Hyperparameter Tuning with Optuna**

```bash
python scripts/tune_hyperparameters.py --n-trials 100
```

### **Custom Configuration**

```python
import yaml
from src.models.rammd import RAMMD

# Load and modify config
with open('config/model_config.yaml') as f:
    config = yaml.safe_load(f)

config['model']['num_regimes'] = 6  # Change to 6 regimes
config['model']['moe']['num_experts'] = 8  # More experts

# Initialize model
model = RAMMD(config['model'])
```

---

## 🤖 Pre-trained Models

### **Direct Download Links**

| Model | HuggingFace | Size | Purpose |
|-------|-------------|------|---------|
| FinBERT-Tone | [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone) | 438 MB | News sentiment (3-class) |
| FinBERT-Pretrain | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | 420 MB | Financial text encoding |
| DistilBERT | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) | 256 MB | Social media posts |
| RoBERTa-Financial | [mrm8488/distilroberta-finetuned](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) | 310 MB | Backup ensemble model |

### **Checkpoint Structure**

```
checkpoints/pretrained/finbert_tone/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
└── metadata.pt
```

---

## 📊 Dataset Preparation

### **Supported Data Sources**

1. **Price Data:**
   - Yahoo Finance (`yfinance`)
   - Alpha Vantage
   - Tiingo
   - Polygon.io

2. **News Data:**
   - NewsAPI
   - Finnhub
   - Alpha Vantage News

3. **Social Media:**
   - Twitter API v2
   - Reddit (PRAW)
   - StockTwits

4. **Macroeconomic:**
   - FRED (Federal Reserve Economic Data)
   - World Bank API

### **Data Format**

```python
data = {
    'price': np.array([T, N_assets, 23]),  # OHLCV + 18 indicators
    'news': List[List[str]],                # News articles per asset
    'social': List[List[str]],              # Social posts per asset
    'macro': np.array([T, 24])             # Macro indicators
}
```

### **Technical Indicators (18)**

- Moving Averages: SMA(5, 20, 50), EMA(12, 26)
- Momentum: RSI(14), MACD, Stochastic(K, D)
- Volatility: Bollinger Bands (upper, middle, lower), ATR(14), ADX(14)
- Volume: OBV

---

## 🏋️ Training

### **Training Pipeline**

```bash
# Stage 1: Contrastive pre-training (50 epochs)
python scripts/train.py --stage pretrain --epochs 50

# Stage 2: End-to-end training (200 epochs)
python scripts/train.py --stage train --epochs 200

# Stage 3: Fine-tuning on specific market
python scripts/train.py --stage finetune --market NSE --epochs 50
```

### **Training Configuration**

```yaml
# config/training_config.yaml
training:
  optimizer: AdamW
  learning_rate: 0.0003
  batch_size: 64
  num_epochs: 200
  gradient_clip: 1.0

  loss_weights:
    regression: 1.0
    classification: 2.0
    volatility: 0.5
    regime: 0.3
    contrastive: 0.8
```

### **Monitoring**

```bash
# TensorBoard
tensorboard --logdir logs/

# W&B
wandb login
python scripts/train.py --use-wandb
```

---

## 📈 Evaluation

### **Metrics**

- **Regression:** MAE, RMSE, R², MAPE
- **Classification:** Accuracy, Precision, Recall, F1
- **Financial:** Sharpe Ratio, Information Ratio, Sortino Ratio
- **Risk:** Max Drawdown, Value-at-Risk (VaR), Conditional VaR
- **Trading:** Win Rate, Profit Factor, Calmar Ratio

### **Evaluation Script**

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --test-data data/processed/test_2024.pkl \
    --metrics all \
    --save-predictions predictions.csv
```

---

## 🔮 Inference

### **Real-time Prediction**

```python
from src.models.rammd import RAMMD
import torch

# Load model
model = RAMMD.from_pretrained('checkpoints/best.pth')
model.eval()

# Prepare input
price_data = fetch_latest_prices(['AAPL', 'GOOGL'])
news = fetch_latest_news(['AAPL', 'GOOGL'])
macro = fetch_macro_indicators()

# Predict
with torch.no_grad():
    outputs = model(
        price_data=price_data,
        news_texts=news,
        macro_data=macro
    )

print(f"Predicted return: {outputs['regression_output']}")
print(f"Direction: {outputs['classification_output'].argmax()}")
print(f"Regime: {outputs['regime_labels']}")
```

---

## ⚙️ Configuration

### **Model Architecture**

Edit `config/model_config.yaml` to customize:

```yaml
model:
  num_regimes: 4                    # Number of market regimes

  moe:
    num_experts_per_modality: 4     # Experts per modality
    top_k: 2                        # Active experts

  contrastive:
    temperature: 0.07               # InfoNCE temperature
    lambda_private: 0.5             # Private loss weight

  wavelet:
    decomposition_levels: 5         # MODWT levels
    wavelet_type: "db8"            # Daubechies-8
```

---

## 📖 Citation

If you use RAMMD in your research, please cite:

```bibtex
@misc{chakraborty2025rammd,
  author       = {Chakraborty, Agniprabha and Dubey, Daipayan},
  title        = {RAMMD: Regime-Adaptive Multimodal Market Decoder with Mixture-of-Experts, Cross-Modal Contrastive Learning, and Graph-Structured Attention},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17418880},
  url          = {https://zenodo.org/records/17418880}
}
```

---

## 🤝 Contributing

We welcome contributions! Feel free to reach out to us.

### **Development Setup**

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/ scripts/
flake8 src/ scripts/

# Type checking
mypy src/
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

### **Key Points:**

- ✅ You may use, modify, and distribute this software
- ✅ Source code must be made available when distributing
- ✅ Modifications must be released under the same license
- ✅ Changes to the code must be documented
- ❌ The software is provided "as is" without warranty
- ❌ The license and copyright notice must be preserved

For the full license text, see [LICENSE](LICENSE) or visit:
https://www.gnu.org/licenses/gpl-3.0.en.html

### **Why GPL-3.0?**

We chose GPL-3.0 to ensure that:
1. The research community benefits from improvements
2. Derivative works remain open source
3. Commercial applications contribute back to the community
4. Academic integrity and reproducibility are maintained

### **Commercial Use**

If you wish to use RAMMD in proprietary software without GPL restrictions, please contact us for alternative licensing arrangements.

---

## 🙏 Acknowledgments

- **Pre-trained Models:**
  - FinBERT: [Huang et al. (2022)](https://arxiv.org/abs/2006.08097)
  - PatchTST: [Nie et al. (2023)](https://arxiv.org/abs/2211.14730)
  - DistilBERT: [Sanh et al. (2019)](https://arxiv.org/abs/1910.01108)

- **Baseline Models:**
  - MIGA: [Chen et al. (2023)](https://arxiv.org/abs/2306.xxxxx)
  - TIC-FusionNet: [Wang et al. (2024)](https://arxiv.org/abs/2401.xxxxx)

- **Datasets:**
  - Yahoo Finance, Alpha Vantage, Tiingo, FRED

- **Frameworks:**
  - PyTorch, Hugging Face Transformers, PyTorch Geometric


---

## 📧 Contact

**Lead Author:** Agniprabha Chakraborty  
📧 Email: [agniprabhac.power.ug@jadavpuruniversity.in](mailto:agniprabhac.power.ug@jadavpuruniversity.in)  
🏛️ Affiliation: Department of Power Engineering, Jadavpur University  
🐙 GitHub: [@Agniprabha9088](https://github.com/Agniprabha9088)  
🔗 LinkedIn: [Agniprabha Chakraborty](https://linkedin.com/in/yourprofile)

**Co-Author:**
- Daipayan Dubey: [dubeydaipayan@gmail.com](mailto:dubeydaipayan@gmail.com)

## 📢 Updates & Changelog

### **Version 1.0.0** (October 2025)
- 🎉 Initial release
- ✅ 7 core modules implementation
- ✅ Pre-trained model integration
- ✅ Comprehensive documentation
- ✅ Multi-market validation

### **Roadmap**

- [ ] **v1.1.0** (December 2025): Add support for cryptocurrency markets
- [ ] **v1.2.0** (Q1 2026): Implement reinforcement learning-based trading agent
- [ ] **v1.3.0** (Q2 2026): Web API for real-time inference
- [ ] **v2.0.0** (Q3 2026): Multi-asset portfolio optimization module

---

## 🎓 Educational Resources

### **Tutorials**

1. [Understanding Market Regimes](docs/tutorials/01_market_regimes.md)
2. [Implementing Mixture-of-Experts](docs/tutorials/02_moe.md)
3. [FOCAL Contrastive Learning Explained](docs/tutorials/03_focal.md)
4. [Graph Neural Networks for Finance](docs/tutorials/04_gnn.md)
5. [Wavelet Decomposition in Trading](docs/tutorials/05_wavelets.md)



---

## 🔒 Security & Privacy

### **Data Handling**

- All financial data is processed locally
- No data is transmitted to external servers without explicit user consent
- API keys and credentials are stored in encrypted environment files

### **Model Security**

- Checkpoints are signed with SHA-256 hashes
- Pre-trained models verified against official repositories
- Regular security audits of dependencies

### **Reporting Vulnerabilities**

If you discover a security vulnerability, please email:
📧 agniprovo9088@gmail.com

---

## ❓ FAQ

### **Q: Can RAMMD be used for live trading?**
A: RAMMD is designed for research purposes. While it can provide predictions, we recommend thorough backtesting and risk management before live deployment.

### **Q: What markets does RAMMD support?**
A: RAMMD has been validated on US equities, European stocks, Asian markets, and Indian markets. It can be adapted to other asset classes.

### **Q: How much GPU memory is required?**
A: Minimum 8GB VRAM for inference, 16GB+ recommended for training.

### **Q: Can I use RAMMD without GPU?**
A: Yes, but training will be significantly slower. Inference can run on CPU with acceptable latency.

### **Q: How do I add custom data sources?**
A: Extend the `DataLoader` class in `src/data/data_loader.py` to integrate new data sources.

### **Q: Is the model interpretable?**
A: Yes! RAMMD includes SHAP-based explainability to understand prediction drivers.

---

## 🐛 Known Issues

### **Current Limitations**

1. **Memory Usage:** Full model requires ~12GB VRAM during training
   - **Workaround:** Use gradient checkpointing or reduce batch size

2. **News Data:** Some news APIs have rate limits
   - **Workaround:** Cache news data or use alternative sources

3. **Real-time Inference:** Initial load time ~5 seconds
   - **Workaround:** Keep model in memory for repeated predictions

4. **Multi-language Support:** Currently optimized for English text
   - **Roadmap:** Multilingual support planned for v1.2.0



---



## 📱 Support



### **Stay Updated**

- ⭐ Star this repository for updates
- 👀 Watch releases for new versions


---

<div align="center">

**Made with ❤️ by our team**

**© 2025 RAMMD Project - Licensed under GPL-3.0**

</div>
