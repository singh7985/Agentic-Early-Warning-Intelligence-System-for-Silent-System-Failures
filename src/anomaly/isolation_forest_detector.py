"""
Isolation Forest Anomaly Detection

Uses Isolation Forest algorithm to detect anomalies in multivariate sensor data.
Particularly effective for detecting anomalies in high-dimensional feature spaces.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detection for multivariate data.
    
    Uses the Isolation Forest algorithm which isolates anomalies by randomly
    selecting features and split values. Anomalies are easier to isolate and
    thus have shorter path lengths in the trees.
    
    Best for:
    - High-dimensional sensor data
    - Multivariate anomaly detection
    - Unknown anomaly patterns
    - Silent degradation in feature space
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[int, str] = 'auto',
        max_features: float = 1.0,
        random_state: int = 42,
        normalize: bool = True
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0-0.5, default: 0.1)
            n_estimators: Number of isolation trees (default: 100)
            max_samples: Number of samples to train each tree (default: 'auto')
            max_features: Proportion of features to use (default: 1.0)
            random_state: Random seed for reproducibility
            normalize: Whether to standardize features (default: True)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.normalize = normalize
        
        # Initialize model
        self.model_ = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Scaler for normalization
        self.scaler_ = StandardScaler() if normalize else None
        
        self.is_fitted_ = False
        self.feature_names_ = None
        
        logger.info(f"Initialized IsolationForestDetector with contamination={contamination}, "
                   f"n_estimators={n_estimators}")
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> 'IsolationForestDetector':
        """
        Fit Isolation Forest on training data.
        
        Args:
            X: Training features (n_samples, n_features)
            feature_names: Optional feature names
        
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        elif feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        X = np.asarray(X)
        
        # Normalize if requested
        if self.normalize:
            X = self.scaler_.fit_transform(X)
        
        # Fit model
        self.model_.fit(X)
        
        self.is_fitted_ = True
        
        logger.info(f"Fitted on {X.shape[0]} samples with {X.shape[1]} features")
        
        return self
    
    def detect(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Detect anomalies in data.
        
        Args:
            X: Features to check (n_samples, n_features)
        
        Returns:
            Binary array: 1 = anomaly, 0 = normal
        """
        if not self.is_fitted_:
            raise ValueError("Detector must be fitted before calling detect()")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.asarray(X)
        
        # Normalize if needed
        if self.normalize:
            X = self.scaler_.transform(X)
        
        # Predict (-1 = anomaly, 1 = normal)
        predictions = self.model_.predict(X)
        
        # Convert to binary (1 = anomaly, 0 = normal)
        anomalies = (predictions == -1).astype(int)
        
        anomaly_rate = np.mean(anomalies)
        logger.info(f"Detected {np.sum(anomalies)} anomalies ({anomaly_rate:.2%} of data)")
        
        return anomalies
    
    def get_anomaly_scores(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get continuous anomaly scores.
        
        Args:
            X: Features to score
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted_:
            raise ValueError("Detector must be fitted before calling get_anomaly_scores()")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = np.asarray(X)
        
        # Normalize if needed
        if self.normalize:
            X = self.scaler_.transform(X)
        
        # Get decision function scores (negative = more anomalous)
        scores = self.model_.decision_function(X)
        
        # Convert to positive scores (higher = more anomalous)
        anomaly_scores = -scores
        
        return anomaly_scores
    
    def detect_with_context(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        timestamps: Optional[np.ndarray] = None,
        engine_ids: Optional[np.ndarray] = None,
        include_features: bool = False
    ) -> pd.DataFrame:
        """
        Detect anomalies with contextual information.
        
        Args:
            X: Features to check
            timestamps: Optional timestamps/cycle numbers
            engine_ids: Optional engine identifiers
            include_features: Whether to include feature values in output
        
        Returns:
            DataFrame with anomaly information
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.asarray(X)
            feature_names = self.feature_names_
        
        anomalies = self.detect(X_array)
        scores = self.get_anomaly_scores(X_array)
        
        df = pd.DataFrame({
            'is_anomaly': anomalies,
            'anomaly_score': scores
        })
        
        if timestamps is not None:
            df['timestamp'] = timestamps
        
        if engine_ids is not None:
            df['engine_id'] = engine_ids
        
        if include_features:
            for i, name in enumerate(feature_names):
                df[name] = X_array[:, i]
        
        return df
    
    def get_feature_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        n_top: int = 10
    ) -> pd.DataFrame:
        """
        Estimate feature importance for anomaly detection.
        
        Uses permutation importance to measure how much each feature
        contributes to anomaly detection.
        
        Args:
            X: Sample data for importance estimation
            n_top: Number of top features to return
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.asarray(X)
            feature_names = self.feature_names_
        
        # Get baseline scores
        baseline_scores = self.get_anomaly_scores(X_array)
        
        # Permutation importance
        importance = []
        for i in range(X_array.shape[1]):
            X_permuted = X_array.copy()
            np.random.shuffle(X_permuted[:, i])
            
            permuted_scores = self.get_anomaly_scores(X_permuted)
            
            # Importance = change in mean anomaly score
            importance.append(np.abs(np.mean(permuted_scores) - np.mean(baseline_scores)))
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(n_top)
        
        return df
    
    def plot_anomalies(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        anomalies: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        title: str = "Isolation Forest Anomaly Detection",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Visualize anomaly detection results.
        
        Args:
            X: Features
            anomalies: Binary anomaly labels (computed if None)
            timestamps: Optional timestamps for x-axis
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = np.asarray(X)
            feature_names = self.feature_names_
        
        if anomalies is None:
            anomalies = self.detect(X_array)
        
        scores = self.get_anomaly_scores(X_array)
        
        if timestamps is None:
            timestamps = np.arange(len(X_array))
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Anomaly scores over time
        ax = axes[0, 0]
        ax.plot(timestamps, scores, 'b-', alpha=0.6, label='Anomaly Score')
        ax.scatter(
            timestamps[anomalies == 1],
            scores[anomalies == 1],
            c='red', s=50, alpha=0.8, label='Anomalies', zorder=5
        )
        ax.set_xlabel('Time/Cycle')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Score distribution
        ax = axes[0, 1]
        ax.hist(scores[anomalies == 0], bins=50, alpha=0.7, label='Normal', edgecolor='black')
        ax.hist(scores[anomalies == 1], bins=30, alpha=0.7, label='Anomalies', edgecolor='black', color='red')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Anomaly Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. PCA projection (if enough features)
        if X_array.shape[1] >= 2:
            ax = axes[1, 0]
            from sklearn.decomposition import PCA
            
            X_normalized = self.scaler_.transform(X_array) if self.normalize else X_array
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_normalized)
            
            scatter_normal = ax.scatter(
                X_pca[anomalies == 0, 0],
                X_pca[anomalies == 0, 1],
                c='blue', alpha=0.5, s=30, label='Normal'
            )
            scatter_anomaly = ax.scatter(
                X_pca[anomalies == 1, 0],
                X_pca[anomalies == 1, 1],
                c='red', alpha=0.8, s=50, label='Anomalies', edgecolor='black'
            )
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax.set_title('PCA Projection')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Feature importance
        ax = axes[1, 1]
        importance_df = self.get_feature_importance(X_array, n_top=min(10, X_array.shape[1]))
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # 5. Anomaly rate over time (rolling window)
        ax = axes[2, 0]
        window = min(50, len(anomalies) // 10)
        if window > 1:
            rolling_rate = pd.Series(anomalies).rolling(window=window).mean()
            ax.plot(timestamps, rolling_rate, 'r-', linewidth=2)
            ax.axhline(y=self.contamination, color='k', linestyle='--',
                      label=f'Expected Rate ({self.contamination:.1%})')
            ax.set_xlabel('Time/Cycle')
            ax.set_ylabel('Anomaly Rate')
            ax.set_title(f'Rolling Anomaly Rate (window={window})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Statistics summary
        ax = axes[2, 1]
        ax.axis('off')
        
        stats_text = f"""
        Model Parameters:
        • Contamination: {self.contamination:.1%}
        • N Estimators: {self.n_estimators}
        • Max Samples: {self.max_samples}
        • Normalization: {self.normalize}
        
        Detection Results:
        • Total Points: {len(anomalies)}
        • Anomalies: {np.sum(anomalies)}
        • Anomaly Rate: {np.mean(anomalies):.2%}
        • Mean Score: {np.mean(scores):.4f}
        • Max Score: {np.max(scores):.4f}
        
        Features: {X_array.shape[1]}
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
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'normalize': self.normalize,
            'n_features': len(self.feature_names_),
            'feature_names': self.feature_names_
        }
    
    def save(self, filepath: str):
        """Save detector to disk"""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Saved detector to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'IsolationForestDetector':
        """Load detector from disk"""
        import joblib
        detector = joblib.load(filepath)
        logger.info(f"Loaded detector from {filepath}")
        return detector
