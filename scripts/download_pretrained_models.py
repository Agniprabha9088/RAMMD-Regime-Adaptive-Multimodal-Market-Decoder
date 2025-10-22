"""
Download all pre-trained models for RAMMD
Run: python scripts/download_pretrained_models.py
"""
import os
from pathlib import Path
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig
)
import torch
from tqdm import tqdm

CHECKPOINT_DIR = Path("checkpoints/pretrained")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "finbert_tone": {
        "hf_name": "yiyanghkust/finbert-tone",
        "type": "classification",
        "description": "FinBERT fine-tuned for financial sentiment (3-class)"
    },
    "finbert_pretrain": {
        "hf_name": "ProsusAI/finbert",
        "type": "base",
        "description": "FinBERT pre-trained on financial corpus"
    },
    "distilbert": {
        "hf_name": "distilbert-base-uncased",
        "type": "base",
        "description": "DistilBERT for social media encoding"
    },
    "roberta_financial": {
        "hf_name": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "type": "classification",
        "description": "RoBERTa fine-tuned on financial news"
    }
}

def download_model(model_key: str, model_info: dict):
    """Download a single model"""
    print(f"\n{'='*80}")
    print(f"Downloading: {model_key}")
    print(f"Description: {model_info['description']}")
    print(f"{'='*80}")
    
    model_name = model_info["hf_name"]
    model_type = model_info["type"]
    model_dir = CHECKPOINT_DIR / model_key
    model_dir.mkdir(exist_ok=True)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(model_dir))
        
        # Download model
        print("Downloading model...")
        if model_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        
        model.save_pretrained(str(model_dir))
        
        # Save metadata
        metadata = {
            "model_key": model_key,
            "hf_name": model_name,
            "type": model_type,
            "description": model_info["description"],
            "num_parameters": sum(p.numel() for p in model.parameters())
        }
        torch.save(metadata, model_dir / "metadata.pt")
        
        print(f"✓ Successfully downloaded {model_key}")
        print(f"  Saved to: {model_dir}")
        print(f"  Parameters: {metadata['num_parameters']:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {model_key}: {str(e)}")
        return False

def main():
    print("="*80)
    print("RAMMD PRE-TRAINED MODEL DOWNLOADER")
    print("="*80)
    print(f"\nTotal models to download: {len(MODELS)}")
    print(f"Cache directory: {CHECKPOINT_DIR.absolute()}")
    
    success_count = 0
    for model_key, model_info in MODELS.items():
        if download_model(model_key, model_info):
            success_count += 1
    
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully downloaded: {success_count}/{len(MODELS)} models")
    
    if success_count == len(MODELS):
        print("\n✓ All models downloaded successfully!")
        print("\nYou can now run: python scripts/train.py")
    else:
        print("\n⚠ Some models failed to download")
        print("Please check your internet connection or download manually from HuggingFace")

if __name__ == "__main__":
    main()
