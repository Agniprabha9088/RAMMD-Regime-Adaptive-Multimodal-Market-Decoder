"""
Inference script for RAMMD
Usage: python scripts/inference.py --checkpoint checkpoints/best.pth --date 2024-10-22
"""

import torch
import yaml
import argparse
import sys
from datetime import datetime

sys.path.append('src')

from models.rammd import RAMMD


def main():
    parser = argparse.ArgumentParser(description='RAMMD Inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/model_config.yaml')
    parser.add_argument('--assets', type=str, default='SPY,AAPL,GOOGL')
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()
    
    print("="*80)
    print("RAMMD INFERENCE")
    print("="*80)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RAMMD(config['model'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nPredicting for: {args.assets}")
    print(f"Date: {args.date}")
    
    # TODO: Implement real-time data fetching and prediction
    print("\n⚠ Real-time inference not implemented yet")
    print("Please implement data fetching in this script")


if __name__ == "__main__":
    main()
