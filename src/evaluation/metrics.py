"""
Evaluation metrics for RAMMD
"""

import torch
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(predictions: Dict, targets: Dict) -> Dict:
    """
    Compute all evaluation metrics
    
    Returns:
        metrics: Dict with MAE, RMSE, Directional Accuracy, Sharpe Ratio, etc.
    """
    metrics = {}
    
    # Regression metrics
    if 'regression_output' in predictions and 'returns' in targets:
        pred_returns = predictions['regression_output'].cpu().numpy()
        true_returns = targets['returns'].cpu().numpy()
        
        # MAE
        metrics['MAE'] = np.mean(np.abs(pred_returns - true_returns))
        
        # RMSE
        metrics['RMSE'] = np.sqrt(np.mean((pred_returns - true_returns) ** 2))
        
        # R²
        ss_res = np.sum((true_returns - pred_returns) ** 2)
        ss_tot = np.sum((true_returns - np.mean(true_returns)) ** 2)
        metrics['R2'] = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Classification metrics (directional accuracy)
    if 'classification_output' in predictions and 'direction' in targets:
        pred_direction = torch.argmax(predictions['classification_output'], dim=-1).cpu().numpy()
        true_direction = targets['direction'].cpu().numpy()
        
        metrics['Directional_Accuracy'] = accuracy_score(true_direction, pred_direction)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_direction, pred_direction, average='weighted', zero_division=0
        )
        metrics['Precision'] = precision
        metrics['Recall'] = recall
        metrics['F1'] = f1
    
    # Sharpe Ratio
    if 'regression_output' in predictions:
        returns = predictions['regression_output'].cpu().numpy()
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        metrics['Sharpe_Ratio'] = sharpe
    
    # Maximum Drawdown
    if 'regression_output' in predictions:
        returns = predictions['regression_output'].cpu().numpy()
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['Max_Drawdown'] = np.min(drawdown)
    
    return metrics


def print_metrics(metrics: Dict):
    """Pretty print metrics"""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Accuracy' in key or 'Precision' in key or 'Recall' in key or 'F1' in key:
                print(f"{key:25s}: {value*100:6.2f}%")
            else:
                print(f"{key:25s}: {value:8.4f}")
    
    print("="*60 + "\n")
