"""
Drift Detection: Monitor data and prediction drift with Evidently

Purpose:
- Detect data drift in input features
- Detect prediction drift in model outputs
- Track distribution changes
- Alert on significant drifts
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection"""
    drift_detected: bool
    p_value: float
    drift_type: str  # 'kolmogorov_smirnov', 'statistical'
    severity: str  # 'none', 'low', 'medium', 'high'
    affected_features: List[str]
    drift_score: float  # 0-1 where 1 is complete drift
    timestamp: datetime


@dataclass
class DataDriftMetrics:
    """Metrics for data drift detection"""
    mean_shift: Dict[str, float]
    std_shift: Dict[str, float]
    ks_statistics: Dict[str, float]
    drift_features: List[str]
    overall_drift_score: float


@dataclass
class PredictionDriftMetrics:
    """Metrics for prediction drift detection"""
    predicted_mean_shift: float
    predicted_std_shift: float
    prediction_range_shift: float
    confidence_degradation: float
    drift_features: List[str]
    overall_drift_score: float


class DriftDetector:
    """
    Detect data and prediction drift using statistical methods.
    
    Methods:
    - Kolmogorov-Smirnov test for distribution comparison
    - Mean/variance shift detection
    - Multivariate drift detection
    """

    def __init__(self, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            threshold: P-value threshold for statistical significance
        """
        self.threshold = threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.drift_history: List[DriftResult] = []
        logger.info(f"DriftDetector initialized with threshold: {threshold}")

    def set_reference_data(self, data: pd.DataFrame):
        """Set reference data for drift comparison."""
        self.reference_data = data
        logger.info(f"Reference data set: {data.shape[0]} samples, {data.shape[1]} features")

    def set_reference_predictions(self, predictions: np.ndarray, 
                                 confidences: Optional[np.ndarray] = None):
        """Set reference predictions for drift comparison."""
        self.reference_predictions = predictions
        self.reference_confidences = confidences
        logger.info(f"Reference predictions set: {len(predictions)} samples")

    def detect_data_drift(self, current_data: pd.DataFrame, 
                         features: Optional[List[str]] = None) -> DriftResult:
        """
        Detect data drift using Kolmogorov-Smirnov test.
        
        Args:
            current_data: Current batch of data
            features: Features to check (None = all)
            
        Returns:
            DriftResult with drift detection information
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        features = features or self.reference_data.columns.tolist()
        drift_features = []
        ks_stats = {}
        p_values = []

        # Check each feature
        for feature in features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue

            ref_values = self.reference_data[feature].dropna().values
            curr_values = current_data[feature].dropna().values

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
            ks_stats[feature] = ks_stat
            p_values.append(p_value)

            if p_value < self.threshold:
                drift_features.append(feature)
                logger.warning(f"Drift detected in {feature}: p={p_value:.4f}, KS={ks_stat:.4f}")

        # Determine severity
        drift_detected = len(drift_features) > 0
        mean_p_value = float(np.mean(p_values)) if p_values else 1.0
        drift_score = 1.0 - mean_p_value  # Higher = more drift

        severity = self._determine_severity(drift_score, len(drift_features), len(features))

        result = DriftResult(
            drift_detected=drift_detected,
            p_value=mean_p_value,
            drift_type='kolmogorov_smirnov',
            severity=severity,
            affected_features=drift_features,
            drift_score=drift_score,
            timestamp=datetime.now(),
        )

        self.drift_history.append(result)
        
        if drift_detected:
            logger.warning(f"Data drift detected: {severity}, {len(drift_features)}/{len(features)} features affected")
        else:
            logger.info("No data drift detected")

        return result

    def detect_prediction_drift(self, current_predictions: np.ndarray,
                               current_confidences: Optional[np.ndarray] = None) -> DriftResult:
        """
        Detect prediction drift comparing prediction distributions.
        
        Args:
            current_predictions: Current predictions
            current_confidences: Current model confidences
            
        Returns:
            DriftResult with drift detection information
        """
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set. Call set_reference_predictions() first.")

        ref_preds = self.reference_predictions
        curr_preds = current_predictions

        # Mean shift detection
        ref_mean = np.mean(ref_preds)
        curr_mean = np.mean(curr_preds)
        mean_shift = abs(curr_mean - ref_mean) / (ref_mean + 1e-6)

        # Variance shift detection
        ref_std = np.std(ref_preds)
        curr_std = np.std(curr_preds)
        std_shift = abs(curr_std - ref_std) / (ref_std + 1e-6)

        # KS test on predictions
        ks_stat, p_value = stats.ks_2samp(ref_preds, curr_preds)

        # Confidence degradation
        confidence_degradation = 0.0
        if self.reference_confidences is not None and current_confidences is not None:
            ref_conf_mean = np.mean(self.reference_confidences)
            curr_conf_mean = np.mean(current_confidences)
            confidence_degradation = (ref_conf_mean - curr_conf_mean) / (ref_conf_mean + 1e-6)

        # Determine drift severity
        drift_score = (mean_shift + std_shift + ks_stat + confidence_degradation) / 4
        drift_detected = p_value < self.threshold or drift_score > 0.3

        severity = self._determine_severity(drift_score, 1, 1)

        result = DriftResult(
            drift_detected=drift_detected,
            p_value=p_value,
            drift_type='prediction_drift',
            severity=severity,
            affected_features=['predictions'],
            drift_score=drift_score,
            timestamp=datetime.now(),
        )

        self.drift_history.append(result)

        if drift_detected:
            logger.warning(f"Prediction drift detected: {severity}, score={drift_score:.3f}")
        else:
            logger.info(f"No prediction drift detected: score={drift_score:.3f}")

        return result

    def _determine_severity(self, drift_score: float, affected_count: int, 
                           total_count: int) -> str:
        """Determine severity level based on drift score and affected features."""
        affected_ratio = affected_count / max(total_count, 1)

        if drift_score < 0.1 or affected_ratio < 0.1:
            return 'none'
        elif drift_score < 0.25 or affected_ratio < 0.25:
            return 'low'
        elif drift_score < 0.5 or affected_ratio < 0.5:
            return 'medium'
        else:
            return 'high'

    def get_data_drift_metrics(self, current_data: pd.DataFrame) -> DataDriftMetrics:
        """Get detailed data drift metrics."""
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        mean_shift = {}
        std_shift = {}
        ks_statistics = {}
        drift_features = []

        for feature in self.reference_data.columns:
            if feature not in current_data.columns:
                continue

            ref_vals = self.reference_data[feature].dropna().values
            curr_vals = current_data[feature].dropna().values

            # Mean and std shift
            ref_mean, curr_mean = np.mean(ref_vals), np.mean(curr_vals)
            ref_std, curr_std = np.std(ref_vals), np.std(curr_vals)

            mean_shift[feature] = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-6)
            std_shift[feature] = abs(curr_std - ref_std) / (ref_std + 1e-6)

            # KS statistic
            ks_stat, _ = stats.ks_2samp(ref_vals, curr_vals)
            ks_statistics[feature] = ks_stat

            if ks_stat > 0.1:
                drift_features.append(feature)

        overall_drift_score = float(np.mean(list(ks_statistics.values()))) if ks_statistics else 0.0

        return DataDriftMetrics(
            mean_shift=mean_shift,
            std_shift=std_shift,
            ks_statistics=ks_statistics,
            drift_features=drift_features,
            overall_drift_score=overall_drift_score,
        )

    def get_prediction_drift_metrics(self, current_predictions: np.ndarray,
                                    current_confidences: Optional[np.ndarray] = None
                                    ) -> PredictionDriftMetrics:
        """Get detailed prediction drift metrics."""
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set")

        ref_mean, curr_mean = np.mean(self.reference_predictions), np.mean(current_predictions)
        ref_std, curr_std = np.std(self.reference_predictions), np.std(current_predictions)
        ref_range = np.max(self.reference_predictions) - np.min(self.reference_predictions)
        curr_range = np.max(current_predictions) - np.min(current_predictions)

        predicted_mean_shift = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-6)
        predicted_std_shift = abs(curr_std - ref_std) / (ref_std + 1e-6)
        prediction_range_shift = abs(curr_range - ref_range) / (ref_range + 1e-6)

        confidence_degradation = 0.0
        if self.reference_confidences is not None and current_confidences is not None:
            ref_conf_mean = np.mean(self.reference_confidences)
            curr_conf_mean = np.mean(current_confidences)
            confidence_degradation = (ref_conf_mean - curr_conf_mean) / (ref_conf_mean + 1e-6)

        overall_drift_score = (predicted_mean_shift + predicted_std_shift + 
                              prediction_range_shift + confidence_degradation) / 4

        drift_features = []
        if predicted_mean_shift > 0.1:
            drift_features.append('mean_shift')
        if predicted_std_shift > 0.1:
            drift_features.append('std_shift')
        if confidence_degradation > 0.05:
            drift_features.append('confidence_degradation')

        return PredictionDriftMetrics(
            predicted_mean_shift=predicted_mean_shift,
            predicted_std_shift=predicted_std_shift,
            prediction_range_shift=prediction_range_shift,
            confidence_degradation=confidence_degradation,
            drift_features=drift_features,
            overall_drift_score=overall_drift_score,
        )

    def get_drift_report(self) -> Dict[str, any]:
        """Get report of all detected drifts."""
        if not self.drift_history:
            return {'total_drifts': 0, 'drifts': []}

        high_severity = sum(1 for d in self.drift_history if d.severity == 'high')
        medium_severity = sum(1 for d in self.drift_history if d.severity == 'medium')

        return {
            'total_drifts': len(self.drift_history),
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'latest_drift': self.drift_history[-1].timestamp.isoformat(),
            'drifts': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'type': d.drift_type,
                    'severity': d.severity,
                    'score': d.drift_score,
                    'affected_features': d.affected_features,
                }
                for d in self.drift_history[-10:]  # Last 10
            ]
        }

    def reset(self):
        """Reset drift detector."""
        self.reference_data = None
        self.reference_predictions = None
        self.drift_history = []
        logger.info("DriftDetector reset")
