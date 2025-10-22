"""
Advanced Trainer for RAMMD with:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Checkpoint management
- W&B/TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional


class RAMMDTrainer:
    """Advanced trainer for RAMMD model"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        training_config = config.get('training', {})
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=training_config.get('learning_rate', 0.0003),
            weight_decay=training_config.get('weight_decay', 0.0001),
            betas=training_config.get('betas', [0.9, 0.999])
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=training_config.get('warmup_epochs', 10),
            T_mult=2,
            eta_min=training_config.get('lr_min', 1e-6)
        )
        
        # Mixed precision
        self.use_mixed_precision = training_config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Gradient clipping
        self.gradient_clip = training_config.get('gradient_clip', 1.0)
        
        # Early stopping
        self.early_stopping_patience = training_config.get('early_stopping_patience', 30)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.use_wandb = training_config.get('use_wandb', False)
        self.log_interval = training_config.get('log_interval', 10)
        
        # Checkpoints
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss weights
        self.loss_weights = training_config.get('loss_weights', {
            'regression': 1.0,
            'classification': 2.0,
            'volatility': 0.5,
            'regime': 0.3,
            'contrastive': 0.8
        })
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        losses_dict = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            price_data = batch['price'].to(self.device)
            macro_data = batch['macro'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                outputs = self.model(
                    price_data=price_data,
                    news_texts=batch.get('news'),
                    social_texts=batch.get('social'),
                    macro_data=macro_data
                )
                
                # Compute losses
                loss, loss_components = self.compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for k, v in loss_components.items():
                losses_dict[k] = losses_dict.get(k, 0.0) + v
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to W&B
            if self.use_wandb and batch_idx % self.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    **{f'train/{k}': v for k, v in loss_components.items()}
                })
        
        avg_loss = total_loss / len(train_loader)
        avg_losses = {k: v / len(train_loader) for k, v in losses_dict.items()}
        
        return avg_loss, avg_losses
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        losses_dict = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                price_data = batch['price'].to(self.device)
                macro_data = batch['macro'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                with autocast(enabled=self.use_mixed_precision):
                    outputs = self.model(
                        price_data=price_data,
                        news_texts=batch.get('news'),
                        social_texts=batch.get('social'),
                        macro_data=macro_data
                    )
                    
                    loss, loss_components = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                for k, v in loss_components.items():
                    losses_dict[k] = losses_dict.get(k, 0.0) + v
        
        avg_loss = total_loss / len(val_loader)
        avg_losses = {k: v / len(val_loader) for k, v in losses_dict.items()}
        
        return avg_loss, avg_losses
    
    def compute_loss(self, outputs: Dict, targets: Dict) -> tuple:
        """Compute multi-task loss"""
        losses = {}
        
        # Regression loss (Huber)
        if 'regression_output' in outputs and 'returns' in targets:
            delta = 1.0
            error = torch.abs(outputs['regression_output'] - targets['returns'])
            regression_loss = torch.where(
                error <= delta,
                0.5 * error ** 2,
                delta * (error - 0.5 * delta)
            ).mean()
            losses['regression'] = self.loss_weights['regression'] * regression_loss
        
        # Classification loss
        if 'classification_output' in outputs and 'direction' in targets:
            classification_loss = F.cross_entropy(
                outputs['classification_output'],
                targets['direction']
            )
            losses['classification'] = self.loss_weights['classification'] * classification_loss
        
        # Volatility loss
        if 'volatility_output' in outputs and 'volatility' in targets:
            volatility_loss = F.mse_loss(
                outputs['volatility_output'],
                targets['volatility']
            )
            losses['volatility'] = self.loss_weights['volatility'] * volatility_loss
        
        # Contrastive loss
        if 'contrastive_loss' in outputs:
            losses['contrastive'] = self.loss_weights['contrastive'] * outputs['contrastive_loss']
        
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss, {k: v.item() for k, v in losses.items()}
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        print(f"\n{'='*80}")
        print(f"Starting RAMMD Training")
        print(f"{'='*80}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_losses = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': train_loss,
                    'val/total_loss': val_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                print(f"  Best val loss: {self.best_val_loss:.6f}")
                break
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}\n")
