"""
RAMMD Dataset and DataLoader
Handles multi-modal financial data loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import yfinance as yf
from datetime import datetime, timedelta


class RAMMDDataset(Dataset):
    """
    Multi-modal financial dataset
    
    Loads:
    - Price data (OHLCV + technical indicators)
    - News articles
    - Social media posts
    - Macroeconomic indicators
    """
    
    def __init__(
        self,
        data_dir: str,
        assets: List[str],
        start_date: str,
        end_date: str,
        window_size: int = 252,
        split: str = 'train',
        config: Optional[Dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.split = split
        self.config = config or {}
        
        # Load data
        self.price_data = self._load_price_data()
        self.macro_data = self._load_macro_data()
        self.news_data = self._load_news_data()
        self.social_data = self._load_social_data()
        
        # Compute targets
        self.targets = self._compute_targets()
        
        # Valid indices (where we have enough history)
        self.valid_indices = list(range(window_size, len(self.price_data) - 1))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get price window
        price_window = self.price_data[actual_idx - self.window_size:actual_idx]
        
        # Get macro data
        macro = self.macro_data[actual_idx]
        
        # Get news (last 5 days)
        news_texts = self._get_news_window(actual_idx, window=5)
        
        # Get social media (last 3 days)
        social_texts = self._get_social_window(actual_idx, window=3)
        
        # Get targets
        targets = {
            'returns': self.targets['returns'][actual_idx],
            'direction': self.targets['direction'][actual_idx],
            'volatility': self.targets['volatility'][actual_idx]
        }
        
        return {
            'price': torch.tensor(price_window, dtype=torch.float32),
            'macro': torch.tensor(macro, dtype=torch.float32),
            'news': news_texts,
            'social': social_texts,
            'targets': targets
        }
    
    def _load_price_data(self):
        """Load price data with technical indicators"""
        print(f"Loading price data for {len(self.assets)} assets...")
        
        # Try loading from cache
        cache_file = self.data_dir / 'processed' / f'price_data_{self.split}.npy'
        if cache_file.exists():
            return np.load(cache_file)
        
        # Download from Yahoo Finance
        all_data = []
        for asset in self.assets:
            try:
                df = yf.download(asset, start=self.start_date, end=self.end_date, progress=False)
                
                # Add technical indicators
                df = self._add_technical_indicators(df)
                
                all_data.append(df.values)
            except Exception as e:
                print(f"Warning: Failed to load {asset}: {e}")
                # Use zeros as fallback
                all_data.append(np.zeros((100, 23)))
        
        # Stack and pad if needed
        min_length = min(len(d) for d in all_data)
        price_data = np.stack([d[:min_length] for d in all_data])
        
        # Transpose to (T, N_assets, features)
        price_data = np.transpose(price_data, (1, 0, 2))
        
        # Save cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, price_data)
        
        return price_data
    
    def _add_technical_indicators(self, df):
        """Add technical indicators to price data"""
        import ta
        
        # Original OHLCV (5 features)
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Moving averages (5 features)
        features['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        features['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        features['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        features['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        features['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # RSI (1 feature)
        features['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD (3 features)
        macd = ta.trend.MACD(df['Close'])
        features['MACD'] = macd.macd()
        features['MACD_signal'] = macd.macd_signal()
        features['MACD_hist'] = macd.macd_diff()
        
        # Bollinger Bands (3 features)
        bollinger = ta.volatility.BollingerBands(df['Close'])
        features['BB_upper'] = bollinger.bollinger_hband()
        features['BB_middle'] = bollinger.bollinger_mavg()
        features['BB_lower'] = bollinger.bollinger_lband()
        
        # ATR (1 feature)
        features['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # ADX (1 feature)
        features['ADX_14'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        # Stochastic (2 features)
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        features['Stoch_K'] = stoch.stoch()
        features['Stoch_D'] = stoch.stoch_signal()
        
        # OBV (1 feature)
        features['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Fill NaN with forward fill then backward fill
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    def _load_macro_data(self):
        """Load macroeconomic indicators"""
        # Placeholder: return dummy data
        # In production, load from FRED or other sources
        T = len(self.price_data)
        return np.random.randn(T, 24).astype(np.float32)
    
    def _load_news_data(self):
        """Load news articles"""
        # Placeholder: return empty
        # In production, load from NewsAPI or stored data
        return {}
    
    def _load_social_data(self):
        """Load social media posts"""
        # Placeholder: return empty
        # In production, load from Twitter API or stored data
        return {}
    
    def _get_news_window(self, idx, window=5):
        """Get news articles from last N days"""
        # Placeholder: return None
        return None
    
    def _get_social_window(self, idx, window=3):
        """Get social media posts from last N days"""
        # Placeholder: return None
        return None
    
    def _compute_targets(self):
        """Compute prediction targets"""
        # Returns (next day)
        close_prices = self.price_data[:, :, 3]  # Close price
        returns = np.diff(np.log(close_prices + 1e-8), axis=0)
        returns = np.mean(returns, axis=1)  # Average across assets
        returns = np.concatenate([returns, [0]])  # Pad last
        
        # Direction (0=down, 1=flat, 2=up)
        direction = np.zeros_like(returns, dtype=np.int64)
        direction[returns > 0.001] = 2  # Up
        direction[returns < -0.001] = 0  # Down
        direction[(returns >= -0.001) & (returns <= 0.001)] = 1  # Flat
        
        # Volatility (rolling std)
        volatility = np.zeros_like(returns)
        window = 20
        for i in range(window, len(returns)):
            volatility[i] = np.std(returns[i-window:i])
        
        return {
            'returns': torch.tensor(returns, dtype=torch.float32),
            'direction': torch.tensor(direction, dtype=torch.long),
            'volatility': torch.tensor(volatility, dtype=torch.float32)
        }


class RAMMDDataLoader:
    """Factory for creating train/val/test dataloaders"""
    
    @staticmethod
    def create_loaders(config: Dict):
        """Create train, validation, and test dataloaders"""
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        
        train_dataset = RAMMDDataset(
            data_dir='data',
            assets=data_config.get('assets', ['SPY']),
            start_date=training_config.get('train_start', '2010-01-01'),
            end_date=training_config.get('train_end', '2019-12-31'),
            split='train',
            config=config
        )
        
        val_dataset = RAMMDDataset(
            data_dir='data',
            assets=data_config.get('assets', ['SPY']),
            start_date=training_config.get('val_start', '2020-01-01'),
            end_date=training_config.get('val_end', '2021-12-31'),
            split='val',
            config=config
        )
        
        test_dataset = RAMMDDataset(
            data_dir='data',
            assets=data_config.get('assets', ['SPY']),
            start_date=training_config.get('test_start', '2022-01-01'),
            end_date=training_config.get('test_end', '2024-12-31'),
            split='test',
            config=config
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.get('batch_size', 64),
            shuffle=True,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.get('batch_size', 64),
            shuffle=False,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config.get('batch_size', 64),
            shuffle=False,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
