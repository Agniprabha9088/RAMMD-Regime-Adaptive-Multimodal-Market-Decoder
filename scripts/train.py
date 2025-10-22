"""
Main Training Script for RAMMD
Usage: python scripts/train.py
"""

import torch
import yaml
import sys
from pathlib import Path
import wandb
from torch.utils.data import DataLoader

# Add src to path
sys.path.append('src')

from models.rammd import RAMMD
from training.trainer import RAMMDTrainer


def load_config(config_path='config/model_config.yaml'):
    """Load model configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open('config/training_config.yaml', 'r') as f:
        training_config = yaml.safe_load(f)
    
    config.update(training_config)
    return config


def create_dummy_dataset(batch_size=64, num_batches=100):
    """Create dummy dataset for testing"""
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return {
                'price': torch.randn(252, 23),
                'macro': torch.randn(24),
                'news': None,
                'social': None,
                'targets': {
                    'returns': torch.randn(1),
                    'direction': torch.randint(0, 3, (1,)).item(),
                    'volatility': torch.rand(1)
                }
            }
    
    dataset = DummyDataset(batch_size * num_batches)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def main():
    print("="*80)
    print("RAMMD TRAINING SCRIPT")
    print("="*80)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['model']['device'] = str(device)
    print(f"   Device: {device}")
    
    # Initialize W&B
    if config.get('training', {}).get('use_wandb', False):
        wandb.init(
            project=config['training'].get('wandb_project', 'RAMMD'),
            config=config,
            name=f"rammd_run_{config['training'].get('experiment_name', 'default')}"
        )
    
    # Initialize model
    print("\n2. Initializing RAMMD model...")
    model = RAMMD(config['model'])
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create data loaders (dummy for now)
    print("\n3. Creating data loaders...")
    train_loader = create_dummy_dataset(batch_size=config['training']['batch_size'])
    val_loader = create_dummy_dataset(batch_size=config['training']['batch_size'], num_batches=20)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        outputs = model(
            price_data=sample_batch['price'].unsqueeze(0).to(device),
            macro_data=sample_batch['macro'].unsqueeze(0).to(device)
        )
    print("   ✓ Forward pass successful!")
    print(f"   Output keys: {list(outputs.keys())}")
    
    # Initialize trainer
    print("\n5. Initializing trainer...")
    trainer = RAMMDTrainer(model, config, device=str(device))
    
    # Start training
    print("\n6. Starting training...")
    print("="*80)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest model saved to: checkpoints/best.pth")
    print(f"Latest model saved to: checkpoints/latest.pth")


if __name__ == "__main__":
    main()
