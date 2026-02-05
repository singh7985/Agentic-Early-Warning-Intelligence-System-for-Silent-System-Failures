"""
Change-Point Detection

Detects abrupt changes in system behavior that may indicate silent degradation
or shift to failure modes. Uses multiple algorithms for robust detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """
    Change-point detection using multiple algorithms.
    
    Detects points in time series where statistical properties change,
    indicating potential degradation onset or mode shifts.
    
    Methods:
    - CUSUM: Cumulative Sum Control Chart
    - EWMA: Exponentially Weighted Moving Average
    - Bayesian: Bayesian Online Changepoint Detection
    - Mann-Kendall: Trend detection test
    """
    
    def __init__(
        self,
        method: str = 'cusum',
        threshold: float = 3.0,
        drift: float = 0.5,
        min_distance: int = 10
    ):
        """
        Initialize change-point detector.
        
        Args:
            method: Detection method ('cusum', 'ewma', 'bayesian', 'mann_kendall')
            threshold: Detection threshold
            drift: Drift parameter for CUSUM (default: 0.5)
            min_distance: Minimum samples between detected change points
        """
        self.method = method.lower()
        self.threshold = threshold
        self.drift = drift
        self.min_distance = min_distance
        
        self.mean_ = None
        self.std_ = None
        
        logger.info(f"Initialized ChangePointDetector with method={method}, threshold={threshold}")
    
    def fit(self, X: np.ndarray) -> 'ChangePointDetector':
        """
        Fit detector on baseline data to learn normal behavior.
        
        Args:
            X: Baseline time series data
        
        Returns:
            self
        """
        X = np.asarray(X).flatten()
        
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        
        logger.info(f"Fitted on {len(X)} samples: mean={self.mean_:.4f}, std={self.std_:.4f}")
        
        return self
    
    def detect(self, X: np.ndarray) -> np.ndarray:
        """
        Detect change points in time series.
        
        Args:
            X: Time series data
        
        Returns:
            Array of change point indices
        """
        X = np.asarray(X).flatten()
        
        if self.method == 'cusum':
            change_points = self._detect_cusum(X)
        elif self.method == 'ewma':
            change_points = self._detect_ewma(X)
        elif self.method == 'bayesian':
            change_points = self._detect_bayesian(X)
        elif self.method == 'mann_kendall':
            change_points = self._detect_mann_kendall(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Enforce minimum distance between change points
        if len(change_points) > 1:
            change_points = self._filter_by_distance(change_points)
        
        logger.info(f"Detected {len(change_points)} change points")
        
        return change_points
    
    def _detect_cusum(self, X: np.ndarray) -> np.ndarray:
        """
        CUSUM (Cumulative Sum) change point detection.
        
        Detects shifts in mean by accumulating deviations.
        """
        if self.mean_ is None:
            self.fit(X[:min(100, len(X))])
        
        # Standardize
        X_std = (X - self.mean_) / self.std_
        
        # CUSUM statistics
        cusum_pos = np.zeros(len(X))
        cusum_neg = np.zeros(len(X))
        
        for i in range(1, len(X)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + X_std[i] - self.drift)
            cusum_neg[i] = max(0, cusum_neg[i-1] - X_std[i] - self.drift)
        
        # Detect change points where CUSUM exceeds threshold
        changes_pos = np.where(cusum_pos > self.threshold)[0]
        changes_neg = np.where(cusum_neg > self.threshold)[0]
        
        # Combine and get unique change points
        change_points = np.unique(np.concatenate([changes_pos, changes_neg]))
        
        # Get first occurrence in consecutive runs
        if len(change_points) > 0:
            change_points = self._get_run_starts(change_points)
        
        return change_points
    
    def _detect_ewma(self, X: np.ndarray) -> np.ndarray:
        """
        EWMA (Exponentially Weighted Moving Average) change point detection.
        """
        if self.mean_ is None:
            self.fit(X[:min(100, len(X))])
        
        # EWMA calculation
        alpha = 0.2
        ewma = np.zeros(len(X))
        ewma[0] = X[0]
        
        for i in range(1, len(X)):
            ewma[i] = alpha * X[i] + (1 - alpha) * ewma[i-1]
        
        # Control limits
        std_ewma = self.std_ * np.sqrt(alpha / (2 - alpha))
        upper_limit = self.mean_ + self.threshold * std_ewma
        lower_limit = self.mean_ - self.threshold * std_ewma
        
        # Detect violations
        violations = np.where((ewma < lower_limit) | (ewma > upper_limit))[0]
        
        if len(violations) > 0:
            change_points = self._get_run_starts(violations)
        else:
            change_points = np.array([])
        
        return change_points
    
    def _detect_bayesian(self, X: np.ndarray) -> np.ndarray:
        """
        Simplified Bayesian Online Changepoint Detection.
        
        Uses running statistics to detect distribution changes.
        """
        if self.mean_ is None:
            self.fit(X[:min(100, len(X))])
        
        window = 20
        change_probs = np.zeros(len(X))
        
        for i in range(window, len(X)):
            # Compare recent window to baseline
            recent = X[i-window:i]
            
            # Use t-test to detect mean shift
            t_stat, p_value = stats.ttest_1samp(recent, self.mean_)
            
            # Higher change probability for lower p-values
            change_probs[i] = 1 - p_value
        
        # Detect peaks in change probability
        peaks, _ = find_peaks(change_probs, height=1-0.05, distance=self.min_distance)
        
        return peaks
    
    def _detect_mann_kendall(self, X: np.ndarray) -> np.ndarray:
        """
        Mann-Kendall trend test for detecting monotonic trends.
        
        Applies test in sliding windows to detect trend onset.
        """
        window = 30
        change_points = []
        
        for i in range(window, len(X), window//2):
            window_data = X[i-window:i]
            
            # Mann-Kendall test
            trend_result = self._mann_kendall_test(window_data)
            
            if trend_result['significant'] and abs(trend_result['z_score']) > self.threshold:
                change_points.append(i - window)  # Mark start of trend
        
        return np.array(change_points)
    
    def _mann_kendall_test(self, X: np.ndarray) -> Dict:
        """
        Perform Mann-Kendall trend test.
        
        Returns:
            Dictionary with test results
        """
        n = len(X)
        
        # Calculate S statistic
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(X[j] - X[i])
        
        # Variance
        var_s = n * (n-1) * (2*n+5) / 18
        
        # Z-score
        if s > 0:
            z_score = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z_score = (s + 1) / np.sqrt(var_s)
        else:
            z_score = 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            's_statistic': s,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'trend': 'increasing' if s > 0 else ('decreasing' if s < 0 else 'no trend')
        }
    
    def _get_run_starts(self, indices: np.ndarray) -> np.ndarray:
        """Get first index of each consecutive run"""
        if len(indices) == 0:
            return indices
        
        # Find where consecutive indices break
        breaks = np.where(np.diff(indices) > 1)[0]
        
        # Start indices are first element and elements after breaks
        starts = np.concatenate([[indices[0]], indices[breaks + 1]])
        
        return starts
    
    def _filter_by_distance(self, change_points: np.ndarray) -> np.ndarray:
        """Filter change points by minimum distance"""
        if len(change_points) == 0:
            return change_points
        
        filtered = [change_points[0]]
        for cp in change_points[1:]:
            if cp - filtered[-1] >= self.min_distance:
                filtered.append(cp)
        
        return np.array(filtered)
    
    def get_change_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get continuous change-point scores for each time step.
        
        Args:
            X: Time series data
        
        Returns:
            Change scores (higher = more likely change point)
        """
        X = np.asarray(X).flatten()
        
        if self.method == 'cusum':
            scores = self._cusum_scores(X)
        elif self.method == 'ewma':
            scores = self._ewma_scores(X)
        elif self.method == 'bayesian':
            scores = self._bayesian_scores(X)
        else:
            # Default: use rolling variance ratio
            scores = self._variance_ratio_scores(X)
        
        return scores
    
    def _cusum_scores(self, X: np.ndarray) -> np.ndarray:
        """CUSUM-based change scores"""
        if self.mean_ is None:
            self.fit(X[:min(100, len(X))])
        
        X_std = (X - self.mean_) / self.std_
        
        cusum_pos = np.zeros(len(X))
        cusum_neg = np.zeros(len(X))
        
        for i in range(1, len(X)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + X_std[i] - self.drift)
            cusum_neg[i] = max(0, cusum_neg[i-1] - X_std[i] - self.drift)
        
        scores = np.maximum(cusum_pos, cusum_neg)
        return scores
    
    def _ewma_scores(self, X: np.ndarray) -> np.ndarray:
        """EWMA-based change scores"""
        if self.mean_ is None:
            self.fit(X[:min(100, len(X))])
        
        alpha = 0.2
        ewma = np.zeros(len(X))
        ewma[0] = X[0]
        
        for i in range(1, len(X)):
            ewma[i] = alpha * X[i] + (1 - alpha) * ewma[i-1]
        
        scores = np.abs(ewma - self.mean_) / self.std_
        return scores
    
    def _bayesian_scores(self, X: np.ndarray) -> np.ndarray:
        """Bayesian change probability scores"""
        if self.mean_ is None:
            self.fit(X[:min(100, len(X))])
        
        window = 20
        scores = np.zeros(len(X))
        
        for i in range(window, len(X)):
            recent = X[i-window:i]
            _, p_value = stats.ttest_1samp(recent, self.mean_)
            scores[i] = 1 - p_value
        
        return scores
    
    def _variance_ratio_scores(self, X: np.ndarray) -> np.ndarray:
        """Variance ratio change scores"""
        window = 20
        scores = np.zeros(len(X))
        
        baseline_var = np.var(X[:min(100, len(X))])
        
        for i in range(window, len(X)):
            recent_var = np.var(X[i-window:i])
            scores[i] = abs(np.log(recent_var / baseline_var)) if baseline_var > 0 else 0
        
        return scores
    
    def detect_with_context(
        self,
        X: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        engine_ids: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Detect change points with context.
        
        Args:
            X: Time series data
            timestamps: Optional timestamps
            engine_ids: Optional engine IDs
        
        Returns:
            DataFrame with change point information
        """
        X = np.asarray(X).flatten()
        change_points = self.detect(X)
        scores = self.get_change_scores(X)
        
        df = pd.DataFrame({
            'value': X,
            'change_score': scores,
            'is_change_point': 0
        })
        
        df.loc[change_points, 'is_change_point'] = 1
        
        if timestamps is not None:
            df['timestamp'] = timestamps
        
        if engine_ids is not None:
            df['engine_id'] = engine_ids
        
        return df
    
    def plot_change_points(
        self,
        X: np.ndarray,
        change_points: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Change Point Detection",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Visualize change point detection.
        
        Args:
            X: Time series data
            change_points: Change point indices (computed if None)
            timestamps: Optional timestamps
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        X = np.asarray(X).flatten()
        
        if change_points is None:
            change_points = self.detect(X)
        
        scores = self.get_change_scores(X)
        
        if timestamps is None:
            timestamps = np.arange(len(X))
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Time series with change points
        ax = axes[0]
        ax.plot(timestamps, X, 'b-', alpha=0.7, label='Time Series')
        
        # Highlight change points
        for cp in change_points:
            ax.axvline(x=timestamps[cp], color='r', linestyle='--', alpha=0.7)
        
        if len(change_points) > 0:
            ax.scatter(
                timestamps[change_points],
                X[change_points],
                c='red', s=100, alpha=0.8, label='Change Points',
                edgecolor='black', linewidths=2, zorder=5
            )
        
        if self.mean_ is not None:
            ax.axhline(y=self.mean_, color='g', linestyle=':', alpha=0.5, label='Baseline Mean')
        
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Value')
        ax.set_title('Time Series with Detected Change Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Change scores
        ax = axes[1]
        ax.plot(timestamps, scores, 'purple', alpha=0.7, label='Change Score')
        ax.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        
        if len(change_points) > 0:
            ax.scatter(
                timestamps[change_points],
                scores[change_points],
                c='red', s=100, alpha=0.8, zorder=5
            )
        
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Change Score')
        ax.set_title(f'Change Scores ({self.method.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Segment statistics
        ax = axes[2]
        
        # Divide into segments by change points
        segments = np.split(X, change_points) if len(change_points) > 0 else [X]
        segment_means = [np.mean(seg) for seg in segments]
        segment_stds = [np.std(seg) for seg in segments]
        
        segment_starts = [0] + list(change_points)
        segment_ends = list(change_points) + [len(X)]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        
        for i, (start, end, mean, std, color) in enumerate(zip(
            segment_starts, segment_ends, segment_means, segment_stds, colors
        )):
            ax.fill_between(
                timestamps[start:end],
                mean - std, mean + std,
                alpha=0.3, color=color, label=f'Segment {i+1}'
            )
            ax.plot(timestamps[start:end], [mean]*(end-start), color=color, linewidth=2)
        
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Value')
        ax.set_title('Segment Statistics (Mean ± Std)')
        if len(segments) <= 10:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'drift': self.drift,
            'min_distance': self.min_distance,
            'baseline_mean': self.mean_,
            'baseline_std': self.std_
        }
    
    def save(self, filepath: str):
        """Save detector to disk"""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Saved detector to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ChangePointDetector':
        """Load detector from disk"""
        import joblib
        detector = joblib.load(filepath)
        logger.info(f"Loaded detector from {filepath}")
        return detector
