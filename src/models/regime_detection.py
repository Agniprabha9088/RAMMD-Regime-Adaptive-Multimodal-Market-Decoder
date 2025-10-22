"""
Hierarchical Regime Detection Module
Implements GMM + HMM + Online Drift Detection (Equations 11-13)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from river.drift import ADWIN, KSWIN, PageHinkley
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RegimeDetector(nn.Module):
    """
    Hierarchical Regime Detection with GMM + HMM + Drift Detection
    
    Paper Section 3.3:
    - Equation 11: z_t ~ GMM(θ_K) where K=4 regimes
    - Equation 12: P(z_t | z_{t-1}) ~ HMM for temporal smoothing
    - Equation 13: Ensemble drift detection (ADWIN + KSWIN + Page-Hinkley)
    
    Regimes:
    0 - Trending Bull Market (low vol, positive returns)
    1 - Mean-Reverting Market (moderate vol, range-bound)
    2 - High Volatility Market (high vol, uncertain)
    3 - Crisis / Bear Market (high vol, negative returns)
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        feature_dim: int = 32,
        covariance_type: str = "full",
        hmm_smoothing: bool = True,
        drift_detection: bool = True
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.feature_dim = feature_dim
        self.covariance_type = covariance_type
        self.hmm_smoothing = hmm_smoothing
        self.drift_detection = drift_detection
        
        # GMM for regime clustering (Equation 11)
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type=covariance_type,
            max_iter=200,
            n_init=10,
            random_state=42,
            verbose=0
        )
        
        # HMM for temporal smoothing (Equation 12)
        if hmm_smoothing:
            self.hmm_model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type=covariance_type,
                n_iter=100,
                random_state=42,
                verbose=False
            )
        
        # Drift detectors ensemble (Equation 13)
        if drift_detection:
            self.adwin = ADWIN(delta=0.002)
            self.kswin = KSWIN(alpha=0.005, window_size=100, stat_size=30)
            self.page_hinkley = PageHinkley(delta=0.005, threshold=50, alpha=0.9999)
        
        self.is_fitted = False
        self.regime_history = []
        self.regime_statistics = {i: {'count': 0, 'duration': [], 'transitions': 0} 
                                  for i in range(n_regimes)}
        
    def extract_regime_features(
        self,
        returns: torch.Tensor,
        volumes: Optional[torch.Tensor] = None,
        vix: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Extract comprehensive market-wide regime features
        
        Features (Equation 13):
        - Market statistics: μ, σ, skew, kurt
        - Cross-sectional dispersion
        - Correlation structure
        - Volume metrics
        - Extreme values
        - Trend indicators
        - Market breadth
        
        Args:
            returns: (N_assets, T) returns matrix
            volumes: (N_assets, T) volume data (optional)
            vix: (T,) VIX index (optional)
            
        Returns:
            features: (T', feature_dim) regime feature matrix
        """
        # Convert to numpy
        if isinstance(returns, torch.Tensor):
            returns_np = returns.cpu().numpy()
        else:
            returns_np = returns
            
        if volumes is not None and isinstance(volumes, torch.Tensor):
            volumes_np = volumes.cpu().numpy()
        else:
            volumes_np = volumes
        
        N, T = returns_np.shape
        features = []
        window = 20  # 20-day rolling window
        
        for t in range(window, T):
            window_returns = returns_np[:, t-window:t]
            feat_vec = []
            
            # === 1. Market Return Statistics ===
            mean_return = np.mean(window_returns)
            median_return = np.median(window_returns)
            feat_vec.extend([mean_return, median_return])
            
            # === 2. Volatility Measures ===
            volatility = np.std(window_returns)
            downside_vol = np.std(window_returns[window_returns < 0]) if np.any(window_returns < 0) else 0.0
            upside_vol = np.std(window_returns[window_returns > 0]) if np.any(window_returns > 0) else 0.0
            feat_vec.extend([volatility, downside_vol, upside_vol])
            
            # === 3. Higher Moments ===
            flat_returns = window_returns.flatten()
            skewness = self._compute_skewness(flat_returns)
            kurtosis = self._compute_kurtosis(flat_returns)
            feat_vec.extend([skewness, kurtosis])
            
            # === 4. Cross-Sectional Dispersion ===
            asset_means = np.mean(window_returns, axis=1)
            dispersion = np.std(asset_means)
            iqr = np.percentile(asset_means, 75) - np.percentile(asset_means, 25)
            feat_vec.extend([dispersion, iqr])
            
            # === 5. Correlation Structure ===
            avg_corr = self._compute_avg_correlation(window_returns)
            max_corr = self._compute_max_correlation(window_returns)
            min_corr = self._compute_min_correlation(window_returns)
            feat_vec.extend([avg_corr, max_corr, min_corr])
            
            # === 6. Extreme Returns ===
            max_return = np.max(window_returns)
            min_return = np.min(window_returns)
            range_return = max_return - min_return
            feat_vec.extend([max_return, min_return, range_return])
            
            # === 7. Volume Metrics (if available) ===
            if volumes_np is not None:
                window_volumes = volumes_np[:, t-window:t]
                avg_volume = np.mean(window_volumes)
                volume_vol = np.std(window_volumes)
                volume_trend = np.mean(window_volumes[:, -5:]) / (np.mean(window_volumes[:, :5]) + 1e-8)
                feat_vec.extend([avg_volume, volume_vol, volume_trend])
            else:
                feat_vec.extend([0.0, 0.0, 1.0])
            
            # === 8. VIX Features (if available) ===
            if vix is not None:
                vix_val = vix[t].item() if isinstance(vix[t], torch.Tensor) else vix[t]
                vix_change = vix_val - (vix[t-1].item() if isinstance(vix[t-1], torch.Tensor) else vix[t-1])
                vix_ma = np.mean(vix[t-window:t]) if isinstance(vix, np.ndarray) else 0.0
                feat_vec.extend([vix_val, vix_change, vix_ma])
            else:
                feat_vec.extend([0.0, 0.0, 0.0])
            
            # === 9. Trend Indicators ===
            recent_mean = np.mean(window_returns[:, -5:])
            trend_strength = (recent_mean - mean_return) / (volatility + 1e-8)
            momentum = np.mean(window_returns[:, -5:]) - np.mean(window_returns[:, :5])
            feat_vec.extend([trend_strength, momentum])
            
            # === 10. Market Breadth ===
            positive_returns = np.sum(np.mean(window_returns, axis=1) > 0) / N
            strong_performers = np.sum(np.mean(window_returns, axis=1) > 0.01) / N
            feat_vec.extend([positive_returns, strong_performers])
            
            # === 11. Risk Metrics ===
            sharpe_approx = mean_return / (volatility + 1e-8)
            sortino_approx = mean_return / (downside_vol + 1e-8)
            feat_vec.extend([sharpe_approx, sortino_approx])
            
            features.append(feat_vec)
        
        features_array = np.array(features, dtype=np.float32)
        
        # Handle NaN/Inf
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # Normalize features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        features_array = scaler.fit_transform(features_array)
        
        return features_array
    
    def fit(self, features: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Fit GMM and HMM on historical regime features
        
        Args:
            features: (T, feature_dim) regime features
            verbose: Print fitting progress
            
        Returns:
            regime_labels: (T,) regime assignments
        """
        if verbose:
            print(f"Fitting regime detector on {len(features)} samples...")
        
        # Step 1: Fit GMM (Equation 11)
        self.gmm.fit(features)
        regime_probs = self.gmm.predict_proba(features)
        regime_labels = self.gmm.predict(features)
        
        if verbose:
            print(f"✓ GMM fitted with {self.n_regimes} components")
            print(f"  BIC: {self.gmm.bic(features):.2f}")
            print(f"  AIC: {self.gmm.aic(features):.2f}")
            print(f"  Converged: {self.gmm.converged_}")
        
        # Step 2: Apply HMM smoothing (Equation 12)
        if self.hmm_smoothing:
            try:
                # Initialize HMM with GMM parameters
                self.hmm_model.means_ = self.gmm.means_
                self.hmm_model.covars_ = self.gmm.covariances_
                
                # Transition matrix with diagonal dominance (persistence)
                # P(z_t = k | z_{t-1} = k) = 0.9
                trans_matrix = np.eye(self.n_regimes) * 0.9
                trans_matrix += (1 - np.eye(self.n_regimes)) * 0.1 / (self.n_regimes - 1)
                self.hmm_model.transmat_ = trans_matrix
                
                # Uniform start probability
                self.hmm_model.startprob_ = np.ones(self.n_regimes) / self.n_regimes
                
                # Fit and predict with HMM
                self.hmm_model.fit(features)
                regime_labels = self.hmm_model.predict(features)
                
                if verbose:
                    print(f"✓ HMM smoothing applied")
                    print(f"  Convergence iterations: {self.hmm_model.monitor_.iter}")
            
            except Exception as e:
                if verbose:
                    print(f"⚠ HMM smoothing failed: {e}, using GMM labels")
        
        self.is_fitted = True
        self.regime_history = list(regime_labels)
        
        # Compute regime statistics
        for regime_id in range(self.n_regimes):
            mask = (regime_labels == regime_id)
            count = np.sum(mask)
            percentage = 100 * count / len(regime_labels)
            self.regime_statistics[regime_id]['count'] = count
            
            # Compute average duration
            durations = []
            current_duration = 0
            for label in regime_labels:
                if label == regime_id:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0
            if current_duration > 0:
                durations.append(current_duration)
            
            avg_duration = np.mean(durations) if durations else 0
            self.regime_statistics[regime_id]['duration'] = durations
            
            if verbose:
                print(f"  Regime {regime_id} ({self.get_regime_description(regime_id)}): "
                      f"{count} samples ({percentage:.1f}%), avg duration: {avg_duration:.1f} days")
        
        return regime_labels
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime for new features
        
        Args:
            features: (1, feature_dim) or (T, feature_dim)
            
        Returns:
            regime_labels: (T,) regime assignments
            regime_probs: (T, n_regimes) regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # GMM prediction
        regime_probs = self.gmm.predict_proba(features)
        regime_labels = self.gmm.predict(features)
        
        return regime_labels, regime_probs
    
    def detect_drift(self, feature_value: float) -> bool:
        """
        Detect concept drift using ensemble of detectors (Equation 13)
        
        Ensemble voting: drift detected if >= 2 out of 3 detectors signal drift
        
        Args:
            feature_value: Single scalar observation (typically first principal component)
            
        Returns:
            drift_detected: True if ensemble detects drift
        """
        if not self.drift_detection:
            return False
        
        drift_signals = []
        
        # ADWIN detector
        self.adwin.update(feature_value)
        if self.adwin.drift_detected:
            drift_signals.append(True)
            self.adwin = ADWIN(delta=0.002)
        
        # KSWIN detector
        self.kswin.update(feature_value)
        if self.kswin.drift_detected:
            drift_signals.append(True)
            self.kswin = KSWIN(alpha=0.005, window_size=100, stat_size=30)
        
        # Page-Hinkley detector
        self.page_hinkley.update(feature_value)
        if self.page_hinkley.drift_detected:
            drift_signals.append(True)
            self.page_hinkley = PageHinkley(delta=0.005, threshold=50, alpha=0.9999)
        
        # Ensemble decision
        drift_detected = sum(drift_signals) >= 2
        
        if drift_detected:
            print("⚠ CONCEPT DRIFT DETECTED! Model adaptation required.")
            print(f"  Detectors triggered: {sum(drift_signals)}/3")
        
        return drift_detected
    
    def adapt_to_drift(self, recent_features: np.ndarray, mode: str = "incremental"):
        """
        Adapt model when drift is detected
        
        Args:
            recent_features: (T_recent, feature_dim) recent observations
            mode: "incremental" (fast) or "full_retrain" (comprehensive)
        """
        print(f"🔄 Adapting regime detector (mode={mode})...")
        
        if mode == "incremental":
            # Incremental update: refit GMM on recent data
            self.gmm.fit(recent_features)
            print("✓ Incremental adaptation complete")
        
        elif mode == "full_retrain":
            # Full retraining with HMM
            self.fit(recent_features, verbose=False)
            print("✓ Full retraining complete")
        
        # Reset drift detectors
        self.adwin = ADWIN(delta=0.002)
        self.kswin = KSWIN(alpha=0.005, window_size=100, stat_size=30)
        self.page_hinkley = PageHinkley(delta=0.005, threshold=50, alpha=0.9999)
    
    # === Helper Methods ===
    
    @staticmethod
    def _compute_skewness(x: np.ndarray) -> float:
        """Compute skewness of distribution"""
        x = x[~np.isnan(x)]
        if len(x) < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return np.mean(((x - mean) / std) ** 3)
    
    @staticmethod
    def _compute_kurtosis(x: np.ndarray) -> float:
        """Compute excess kurtosis"""
        x = x[~np.isnan(x)]
        if len(x) < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return np.mean(((x - mean) / std) ** 4) - 3
    
    @staticmethod
    def _compute_avg_correlation(returns: np.ndarray) -> float:
        """Compute average pairwise correlation"""
        if returns.shape[0] < 2:
            return 0.0
        corr_matrix = np.corrcoef(returns)
        n = corr_matrix.shape[0]
        if n < 2:
            return 0.0
        upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
        upper_triangle = upper_triangle[~np.isnan(upper_triangle)]
        return np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
    
    @staticmethod
    def _compute_max_correlation(returns: np.ndarray) -> float:
        """Compute maximum pairwise correlation"""
        if returns.shape[0] < 2:
            return 0.0
        corr_matrix = np.corrcoef(returns)
        n = corr_matrix.shape[0]
        if n < 2:
            return 0.0
        upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
        upper_triangle = upper_triangle[~np.isnan(upper_triangle)]
        return np.max(upper_triangle) if len(upper_triangle) > 0 else 0.0
    
    @staticmethod
    def _compute_min_correlation(returns: np.ndarray) -> float:
        """Compute minimum pairwise correlation"""
        if returns.shape[0] < 2:
            return 0.0
        corr_matrix = np.corrcoef(returns)
        n = corr_matrix.shape[0]
        if n < 2:
            return 0.0
        upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
        upper_triangle = upper_triangle[~np.isnan(upper_triangle)]
        return np.min(upper_triangle) if len(upper_triangle) > 0 else 0.0
    
    def get_regime_description(self, regime_id: int) -> str:
        """Get human-readable regime description"""
        descriptions = {
            0: "Trending Bull Market",
            1: "Mean-Reverting Market",
            2: "High Volatility Market",
            3: "Crisis / Bear Market"
        }
        return descriptions.get(regime_id, f"Regime {regime_id}")
    
    def get_regime_characteristics(self, regime_id: int) -> Dict:
        """Get detailed characteristics of a regime"""
        if regime_id not in self.regime_statistics:
            return {}
        
        stats = self.regime_statistics[regime_id]
        return {
            'description': self.get_regime_description(regime_id),
            'count': stats['count'],
            'average_duration': np.mean(stats['duration']) if stats['duration'] else 0,
            'max_duration': np.max(stats['duration']) if stats['duration'] else 0,
            'min_duration': np.min(stats['duration']) if stats['duration'] else 0
        }
