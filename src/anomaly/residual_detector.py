"""
Residual-Based Anomaly Detection

Detects anomalies by analyzing prediction residuals (errors) from ML models.
Uses statistical methods to identify when system behavior deviates from expected.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class ResidualAnomalyDetector:
    """
    Residual-based anomaly detection using statistical methods.
    
    Detects anomalies by analyzing the distribution of prediction residuals
    and flagging observations that fall outside expected bounds.
    
    Methods:
    - Z-score: Standard deviations from mean
    - IQR: Interquartile range outlier detection
    - MAD: Median Absolute Deviation
    - EWMA: Exponentially Weighted Moving Average
    """
    
    def __init__(
        self,
        method: str = 'zscore',
        threshold: float = 3.0,
        window_size: int = 50,
        contamination: float = 0.1
    ):
        """
        Initialize residual anomaly detector.
        
        Args:
            method: Detection method ('zscore', 'iqr', 'mad', 'ewma')
            threshold: Threshold for anomaly detection
                - zscore: number of standard deviations (default: 3.0)
                - iqr: IQR multiplier (default: 1.5)
                - mad: MAD multiplier (default: 3.0)
                - ewma: number of control limits (default: 3.0)
            window_size: Window size for rolling statistics (default: 50)
            contamination: Expected proportion of anomalies (0.0-0.5)
        """
        self.method = method.lower()
        self.threshold = threshold
        self.window_size = window_size
        self.contamination = contamination
        
        # Statistics for normalization
        self.residual_mean_ = None
        self.residual_std_ = None
        self.residual_median_ = None
        self.residual_mad_ = None
        self.residual_q1_ = None
        self.residual_q3_ = None
        
        # EWMA parameters
        self.ewma_alpha_ = 0.2  # Smoothing factor
        
        self.is_fitted_ = False
        
        logger.info(f"Initialized ResidualAnomalyDetector with method={method}, threshold={threshold}")
    
    def fit(self, residuals: np.ndarray) -> 'ResidualAnomalyDetector':
        """
        Fit detector on training residuals to learn normal behavior.
        
        Args:
            residuals: Training residuals (predictions - actuals)
        
        Returns:
            self
        """
        residuals = np.asarray(residuals).flatten()
        
        # Calculate statistics
        self.residual_mean_ = np.mean(residuals)
        self.residual_std_ = np.std(residuals)
        self.residual_median_ = np.median(residuals)
        
        # MAD (Median Absolute Deviation)
        self.residual_mad_ = np.median(np.abs(residuals - self.residual_median_))
        
        # IQR
        self.residual_q1_ = np.percentile(residuals, 25)
        self.residual_q3_ = np.percentile(residuals, 75)
        
        self.is_fitted_ = True
        
        logger.info(f"Fitted on {len(residuals)} residuals")
        logger.info(f"  Mean: {self.residual_mean_:.4f}, Std: {self.residual_std_:.4f}")
        logger.info(f"  Median: {self.residual_median_:.4f}, MAD: {self.residual_mad_:.4f}")
        
        return self
    
    def detect(self, residuals: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in residuals.
        
        Args:
            residuals: Residuals to check for anomalies
        
        Returns:
            Binary array: 1 = anomaly, 0 = normal
        """
        if not self.is_fitted_:
            raise ValueError("Detector must be fitted before calling detect()")
        
        residuals = np.asarray(residuals).flatten()
        
        if self.method == 'zscore':
            anomalies = self._detect_zscore(residuals)
        elif self.method == 'iqr':
            anomalies = self._detect_iqr(residuals)
        elif self.method == 'mad':
            anomalies = self._detect_mad(residuals)
        elif self.method == 'ewma':
            anomalies = self._detect_ewma(residuals)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        anomaly_rate = np.mean(anomalies)
        logger.info(f"Detected {np.sum(anomalies)} anomalies ({anomaly_rate:.2%} of data)")
        
        return anomalies
    
    def _detect_zscore(self, residuals: np.ndarray) -> np.ndarray:
        """Z-score method: |residual - mean| > threshold * std"""
        z_scores = np.abs((residuals - self.residual_mean_) / self.residual_std_)
        return (z_scores > self.threshold).astype(int)
    
    def _detect_iqr(self, residuals: np.ndarray) -> np.ndarray:
        """IQR method: outside [Q1 - threshold*IQR, Q3 + threshold*IQR]"""
        iqr = self.residual_q3_ - self.residual_q1_
        lower_bound = self.residual_q1_ - self.threshold * iqr
        upper_bound = self.residual_q3_ + self.threshold * iqr
        return ((residuals < lower_bound) | (residuals > upper_bound)).astype(int)
    
    def _detect_mad(self, residuals: np.ndarray) -> np.ndarray:
        """MAD method: |residual - median| > threshold * MAD"""
        deviations = np.abs(residuals - self.residual_median_) / self.residual_mad_
        return (deviations > self.threshold).astype(int)
    
    def _detect_ewma(self, residuals: np.ndarray) -> np.ndarray:
        """EWMA control chart method"""
        # Calculate EWMA
        ewma = np.zeros(len(residuals))
        ewma[0] = residuals[0]
        for i in range(1, len(residuals)):
            ewma[i] = self.ewma_alpha_ * residuals[i] + (1 - self.ewma_alpha_) * ewma[i-1]
        
        # Control limits
        std_ewma = self.residual_std_ * np.sqrt(self.ewma_alpha_ / (2 - self.ewma_alpha_))
        upper_limit = self.residual_mean_ + self.threshold * std_ewma
        lower_limit = self.residual_mean_ - self.threshold * std_ewma
        
        return ((ewma < lower_limit) | (ewma > upper_limit)).astype(int)
    
    def get_anomaly_scores(self, residuals: np.ndarray) -> np.ndarray:
        """
        Get continuous anomaly scores (not binary).
        
        Args:
            residuals: Residuals to score
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted_:
            raise ValueError("Detector must be fitted before calling get_anomaly_scores()")
        
        residuals = np.asarray(residuals).flatten()
        
        if self.method == 'zscore':
            scores = np.abs((residuals - self.residual_mean_) / self.residual_std_)
        elif self.method == 'iqr':
            iqr = self.residual_q3_ - self.residual_q1_
            lower_bound = self.residual_q1_ - self.threshold * iqr
            upper_bound = self.residual_q3_ + self.threshold * iqr
            # Distance from bounds
            scores = np.maximum(
                (lower_bound - residuals) / iqr,
                (residuals - upper_bound) / iqr
            )
            scores = np.maximum(scores, 0)  # Clip negative
        elif self.method == 'mad':
            scores = np.abs(residuals - self.residual_median_) / self.residual_mad_
        elif self.method == 'ewma':
            # Use EWMA deviation
            ewma = np.zeros(len(residuals))
            ewma[0] = residuals[0]
            for i in range(1, len(residuals)):
                ewma[i] = self.ewma_alpha_ * residuals[i] + (1 - self.ewma_alpha_) * ewma[i-1]
            scores = np.abs(ewma - self.residual_mean_) / self.residual_std_
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return scores
    
    def detect_with_context(
        self, 
        residuals: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        engine_ids: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Detect anomalies with contextual information.
        
        Args:
            residuals: Residuals to check
            timestamps: Optional timestamps/cycle numbers
            engine_ids: Optional engine identifiers
        
        Returns:
            DataFrame with anomaly information
        """
        residuals = np.asarray(residuals).flatten()
        anomalies = self.detect(residuals)
        scores = self.get_anomaly_scores(residuals)
        
        df = pd.DataFrame({
            'residual': residuals,
            'is_anomaly': anomalies,
            'anomaly_score': scores
        })
        
        if timestamps is not None:
            df['timestamp'] = timestamps
        
        if engine_ids is not None:
            df['engine_id'] = engine_ids
        
        return df
    
    def plot_residuals(
        self,
        residuals: np.ndarray,
        anomalies: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Residual Anomaly Detection",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Visualize residuals and detected anomalies.
        
        Args:
            residuals: Residuals to plot
            anomalies: Binary anomaly labels (computed if None)
            timestamps: Optional timestamps for x-axis
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if anomalies is None:
            anomalies = self.detect(residuals)
        
        residuals = np.asarray(residuals).flatten()
        
        if timestamps is None:
            timestamps = np.arange(len(residuals))
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Time series with anomalies highlighted
        ax = axes[0, 0]
        ax.plot(timestamps, residuals, 'b-', alpha=0.6, label='Residuals')
        ax.scatter(
            timestamps[anomalies == 1],
            residuals[anomalies == 1],
            c='red', s=50, alpha=0.8, label='Anomalies', zorder=5
        )
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Histogram with threshold
        ax = axes[0, 1]
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        if self.method == 'zscore':
            threshold_val = self.residual_mean_ + self.threshold * self.residual_std_
            ax.axvline(x=threshold_val, color='r', linestyle='--', label=f'+{self.threshold}σ')
            ax.axvline(x=self.residual_mean_ - self.threshold * self.residual_std_,
                      color='r', linestyle='--', label=f'-{self.threshold}σ')
        ax.axvline(x=self.residual_mean_, color='g', linestyle='-', label='Mean')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Anomaly scores
        ax = axes[1, 0]
        scores = self.get_anomaly_scores(residuals)
        ax.plot(timestamps, scores, 'b-', alpha=0.6)
        ax.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        ax.scatter(
            timestamps[anomalies == 1],
            scores[anomalies == 1],
            c='red', s=50, alpha=0.8, zorder=5
        )
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality Check)')
        ax.grid(True, alpha=0.3)
        
        # 5. Rolling statistics
        ax = axes[2, 0]
        if len(residuals) >= self.window_size:
            rolling_mean = pd.Series(residuals).rolling(window=self.window_size).mean()
            rolling_std = pd.Series(residuals).rolling(window=self.window_size).std()
            ax.plot(timestamps, rolling_mean, 'g-', label=f'Rolling Mean (w={self.window_size})')
            ax.fill_between(
                timestamps,
                rolling_mean - 2*rolling_std,
                rolling_mean + 2*rolling_std,
                alpha=0.3, label='±2σ band'
            )
            ax.plot(timestamps, residuals, 'b-', alpha=0.3, label='Residuals')
            ax.set_xlabel('Time/Cycle')
            ax.set_ylabel('Residual')
            ax.set_title('Rolling Statistics')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Statistics summary
        ax = axes[2, 1]
        ax.axis('off')
        
        stats_text = f"""
        Detection Method: {self.method.upper()}
        Threshold: {self.threshold}
        
        Residual Statistics:
        • Mean: {self.residual_mean_:.4f}
        • Std Dev: {self.residual_std_:.4f}
        • Median: {self.residual_median_:.4f}
        • MAD: {self.residual_mad_:.4f}
        
        Anomaly Statistics:
        • Total Points: {len(residuals)}
        • Anomalies: {np.sum(anomalies)}
        • Anomaly Rate: {np.mean(anomalies):.2%}
        • Max Score: {np.max(scores):.2f}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def get_statistics(self) -> Dict:
        """
        Get detector statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.is_fitted_:
            raise ValueError("Detector must be fitted first")
        
        return {
            'method': self.method,
            'threshold': self.threshold,
            'window_size': self.window_size,
            'residual_mean': self.residual_mean_,
            'residual_std': self.residual_std_,
            'residual_median': self.residual_median_,
            'residual_mad': self.residual_mad_,
            'residual_q1': self.residual_q1_,
            'residual_q3': self.residual_q3_,
            'iqr': self.residual_q3_ - self.residual_q1_
        }
    
    def save(self, filepath: str):
        """Save detector to disk"""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Saved detector to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ResidualAnomalyDetector':
        """Load detector from disk"""
        import joblib
        detector = joblib.load(filepath)
        logger.info(f"Loaded detector from {filepath}")
        return detector
