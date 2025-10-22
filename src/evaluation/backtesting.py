"""
Backtesting engine for RAMMD
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class Backtester:
    """Simple backtesting engine"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
    def run_backtest(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        dates: List
    ) -> Dict:
        """
        Run backtest simulation
        
        Args:
            predictions: Predicted returns
            actual_returns: Actual returns
            dates: Trading dates
            
        Returns:
            results: Dict with performance metrics
        """
        capital = self.initial_capital
        portfolio_values = [capital]
        positions = []
        
        for i in range(len(predictions)):
            pred = predictions[i]
            actual = actual_returns[i]
            
            # Simple strategy: long if predicted positive, short if negative
            if pred > 0.001:
                position = 1.0  # Long
            elif pred < -0.001:
                position = -1.0  # Short
            else:
                position = 0.0  # Flat
            
            # Apply transaction costs
            if i > 0 and position != positions[-1]:
                capital *= (1 - self.transaction_cost)
            
            # Apply slippage
            adjusted_return = actual - np.sign(position) * self.slippage
            
            # Update capital
            capital *= (1 + position * adjusted_return)
            
            portfolio_values.append(capital)
            positions.append(position)
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        results = {
            'Final_Portfolio_Value': capital,
            'Total_Return': (capital - self.initial_capital) / self.initial_capital,
            'Sharpe_Ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'Max_Drawdown': self._calculate_max_drawdown(portfolio_values),
            'Win_Rate': np.mean(np.array(returns) > 0),
            'Portfolio_Values': portfolio_values,
            'Positions': positions
        }
        
        return results
    
    @staticmethod
    def _calculate_max_drawdown(portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
