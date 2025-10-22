"""
Evaluation script for RAMMD
Usage: python scripts/evaluate.py --checkpoint checkpoints/best.pth
"""

import torch
import yaml
import argparse
import sys
from pathlib import Path

sys.path.append('src')

from models.rammd import RAMMD
from evaluation.metrics import compute_metrics, print_metrics
from evaluation.backtesting import Backtester


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAMMD model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/model_config.yaml')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    args = parser.parse_args()
    
    print("="*80)
    print("RAMMD EVALUATION")
    print("="*80)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RAMMD(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Val loss: {checkpoint.get('val_loss', 'unknown')}")
    
    # TODO: Load test data and run evaluation
    print("\n⚠ Test data loading not implemented yet")
    print("Please implement test data loading in src/data/data_loader.py")
    
    if args.backtest:
        print("\nRunning backtest...")
        backtester = Backtester()
        # results = backtester.run_backtest(predictions, actual_returns, dates)


if __name__ == "__main__":
    main()
