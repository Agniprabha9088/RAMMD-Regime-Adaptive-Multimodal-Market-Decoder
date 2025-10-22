"""
Data preprocessing utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Optional


class DataPreprocessor:
    """Preprocess financial data"""
    
    def __init__(self, method: str = 'robust'):
        self.method = method
        self.scalers = {}
        
    def fit_transform(self, data: np.ndarray, name: str = 'default'):
        """Fit scaler and transform data"""
        if self.method == 'robust':
            scaler = RobustScaler()
        elif self.method == 'standard':
            scaler = StandardScaler()
        else:
            return data
        
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        
        transformed = scaler.fit_transform(data_flat)
        self.scalers[name] = scaler
        
        return transformed.reshape(original_shape)
    
    def transform(self, data: np.ndarray, name: str = 'default'):
        """Transform data using fitted scaler"""
        if name not in self.scalers:
            return data
        
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        
        transformed = self.scalers[name].transform(data_flat)
        
        return transformed.reshape(original_shape)
    
    @staticmethod
    def handle_missing(data: np.ndarray, method: str = 'forward_fill'):
        """Handle missing values"""
        if method == 'forward_fill':
            # Forward fill then backward fill
            mask = np.isnan(data)
            idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            data = data[idx, np.arange(idx.shape[1])]
        elif method == 'zero':
            data = np.nan_to_num(data, nan=0.0)
        
        return data
    
    @staticmethod
    def remove_outliers(data: np.ndarray, threshold: float = 3.0):
        """Remove outliers using z-score"""
        z_scores = np.abs((data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8))
        data[z_scores > threshold] = np.nan
        return DataPreprocessor.handle_missing(data)
