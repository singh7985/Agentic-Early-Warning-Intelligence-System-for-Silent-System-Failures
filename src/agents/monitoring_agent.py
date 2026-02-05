"""
Monitoring Agent: Runs ML inference and detects anomalies/drift

Responsibilities:
- Load trained ML models (CNN-LSTM, FFN)
- Run inference on sensor data
- Detect anomalies using multiple detectors
- Monitor feature drift
- Report confidence scores
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from ML prediction"""
    engine_id: int
    cycle: int
    predicted_rul: float
    confidence: float
    timestamp: str
    model_type: str
    uncertainty: float  # Standard deviation or confidence interval


@dataclass
class AnomalyDetection:
    """Result from anomaly detection"""
    engine_id: int
    cycle: int
    is_anomaly: bool
    anomaly_score: float  # Normalized 0-1
    anomaly_type: str  # 'sensor_anomaly', 'pattern_anomaly', 'none'
    affected_sensors: List[str]
    timestamp: str
    severity: str  # 'low', 'medium', 'high'


@dataclass
class DriftDetection:
    """Result from drift detection"""
    engine_id: int
    cycle: int
    is_drift: bool
    drift_score: float  # Normalized 0-1
    drifted_features: List[str]
    drift_type: str  # 'feature_drift', 'distribution_drift', 'none'
    timestamp: str


@dataclass
class MonitoringReport:
    """Complete monitoring report from agent"""
    engine_id: int
    cycle: int
    timestamp: str
    prediction: PredictionResult
    anomaly: AnomalyDetection
    drift: DriftDetection
    overall_confidence: float  # Min confidence across all signals
    alert_flag: bool  # True if any signal indicates concern


class MonitoringAgent:
    """
    Agent responsible for continuous monitoring of sensor data
    and ML-based predictions with anomaly/drift detection.
    
    Features:
    - Multi-model inference (CNN-LSTM, FFN ensembles)
    - Anomaly detection with configurable thresholds
    - Data drift monitoring
    - Confidence scoring
    - Alert generation
    """

    def __init__(
        self,
        anomaly_detector: Optional[Any] = None,
        drift_detector: Optional[Any] = None,
        models: Optional[Dict[str, Any]] = None,
        anomaly_threshold: float = 0.6,
        drift_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize Monitoring Agent.
        
        Args:
            anomaly_detector: Anomaly detection model (e.g., IsolationForest)
            drift_detector: Data drift detection model
            models: Dict of trained ML models {model_name: model_instance}
            anomaly_threshold: Threshold for anomaly detection (0-1)
            drift_threshold: Threshold for drift detection (0-1)
            confidence_threshold: Minimum confidence for predictions (0-1)
        """
        self.anomaly_detector = anomaly_detector
        self.drift_detector = drift_detector
        self.models = models or {}
        
        self.anomaly_threshold = anomaly_threshold
        self.drift_threshold = drift_threshold
        self.confidence_threshold = confidence_threshold
        
        # Monitoring history
        self.prediction_history = []
        self.anomaly_history = []
        self.drift_history = []
        
        logger.info(
            f"MonitoringAgent initialized with {len(self.models)} models. "
            f"Thresholds: anomaly={anomaly_threshold}, "
            f"drift={drift_threshold}, confidence={confidence_threshold}"
        )

    def predict_rul(
        self,
        sensor_data: np.ndarray,
        engine_id: int,
        cycle: int,
        use_ensemble: bool = True,
    ) -> PredictionResult:
        """
        Run ML inference to predict Remaining Useful Life.
        
        Args:
            sensor_data: Input features [n_samples, n_features]
            engine_id: Engine identifier
            cycle: Current cycle number
            use_ensemble: Use ensemble averaging if multiple models
        
        Returns:
            PredictionResult with RUL prediction and confidence
        """
        if not self.models:
            logger.warning("No models available for RUL prediction")
            return PredictionResult(
                engine_id=engine_id,
                cycle=cycle,
                predicted_rul=0.0,
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                model_type="none",
                uncertainty=1.0,
            )

        try:
            predictions = []
            uncertainties = []

            # Run inference with all available models
            for model_name, model in self.models.items():
                try:
                    # Reshape if needed
                    if len(sensor_data.shape) == 1:
                        input_data = sensor_data.reshape(1, -1, 1)
                    else:
                        input_data = sensor_data

                    # Get prediction
                    pred = model.predict(input_data, verbose=0)
                    rul_pred = float(pred[0, 0])
                    
                    # Estimate uncertainty from dropout or ensemble variance
                    uncertainty = self._estimate_uncertainty(model_name, rul_pred)
                    
                    predictions.append(rul_pred)
                    uncertainties.append(uncertainty)
                    
                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {e}")
                    continue

            if not predictions:
                return PredictionResult(
                    engine_id=engine_id,
                    cycle=cycle,
                    predicted_rul=0.0,
                    confidence=0.0,
                    timestamp=datetime.now().isoformat(),
                    model_type="failed",
                    uncertainty=1.0,
                )

            # Ensemble averaging
            if use_ensemble and len(predictions) > 1:
                predicted_rul = float(np.mean(predictions))
                uncertainty = float(np.mean(uncertainties))
                model_type = "ensemble"
            else:
                predicted_rul = predictions[0]
                uncertainty = uncertainties[0]
                model_type = list(self.models.keys())[0]

            # Confidence based on uncertainty and model agreement
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                confidence = 1.0 / (1.0 + prediction_std + uncertainty)
            else:
                confidence = 1.0 / (1.0 + uncertainty)

            confidence = float(np.clip(confidence, 0, 1))

            result = PredictionResult(
                engine_id=engine_id,
                cycle=cycle,
                predicted_rul=max(0, predicted_rul),
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                model_type=model_type,
                uncertainty=uncertainty,
            )

            self.prediction_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Error in RUL prediction: {e}")
            return PredictionResult(
                engine_id=engine_id,
                cycle=cycle,
                predicted_rul=0.0,
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                model_type="error",
                uncertainty=1.0,
            )

    def detect_anomalies(
        self,
        sensor_data: np.ndarray,
        engine_id: int,
        cycle: int,
        sensor_names: Optional[List[str]] = None,
    ) -> AnomalyDetection:
        """
        Detect anomalies in sensor data.
        
        Args:
            sensor_data: Sensor readings [n_features] or [n_samples, n_features]
            engine_id: Engine identifier
            cycle: Current cycle number
            sensor_names: Names of sensors
        
        Returns:
            AnomalyDetection result with anomaly type and severity
        """
        if self.anomaly_detector is None:
            return AnomalyDetection(
                engine_id=engine_id,
                cycle=cycle,
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type='none',
                affected_sensors=[],
                timestamp=datetime.now().isoformat(),
                severity='low',
            )

        try:
            # Flatten if needed
            if len(sensor_data.shape) > 1:
                flat_data = sensor_data.flatten().reshape(1, -1)
            else:
                flat_data = sensor_data.reshape(1, -1)

            # Get anomaly score (-1 for anomaly, 1 for normal in isolation forest)
            # Convert to 0-1 scale
            if hasattr(self.anomaly_detector, 'decision_function'):
                scores = self.anomaly_detector.decision_function(flat_data)
                anomaly_score = float(np.clip(1.0 / (1.0 + np.exp(scores[0])), 0, 1))
            else:
                prediction = self.anomaly_detector.predict(flat_data)[0]
                anomaly_score = 0.8 if prediction == -1 else 0.2

            is_anomaly = anomaly_score >= self.anomaly_threshold

            # Determine affected sensors
            affected_sensors = []
            if is_anomaly and sensor_names:
                # Simple heuristic: sensors with highest deviation from mean
                mean = np.mean(sensor_data) if len(sensor_data.shape) == 1 else np.mean(sensor_data, axis=0)
                std = np.std(sensor_data) if len(sensor_data.shape) == 1 else np.std(sensor_data, axis=0)
                deviations = np.abs((sensor_data - mean) / (std + 1e-6))
                
                if len(sensor_data.shape) == 1:
                    top_indices = np.argsort(deviations)[-3:]  # Top 3
                else:
                    top_indices = np.argsort(np.max(deviations, axis=0))[-3:]
                
                affected_sensors = [sensor_names[i] for i in top_indices if i < len(sensor_names)]

            # Determine severity
            if anomaly_score >= 0.8:
                severity = 'high'
            elif anomaly_score >= 0.6:
                severity = 'medium'
            else:
                severity = 'low'

            result = AnomalyDetection(
                engine_id=engine_id,
                cycle=cycle,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                anomaly_type='sensor_anomaly' if is_anomaly else 'none',
                affected_sensors=affected_sensors,
                timestamp=datetime.now().isoformat(),
                severity=severity,
            )

            self.anomaly_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return AnomalyDetection(
                engine_id=engine_id,
                cycle=cycle,
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type='none',
                affected_sensors=[],
                timestamp=datetime.now().isoformat(),
                severity='low',
            )

    def detect_drift(
        self,
        current_data: np.ndarray,
        reference_data: Optional[np.ndarray] = None,
        engine_id: int = 0,
        cycle: int = 0,
        feature_names: Optional[List[str]] = None,
    ) -> DriftDetection:
        """
        Detect data distribution drift.
        
        Args:
            current_data: Current sensor readings
            reference_data: Reference/baseline data for comparison
            engine_id: Engine identifier
            cycle: Current cycle number
            feature_names: Names of features
        
        Returns:
            DriftDetection result
        """
        if self.drift_detector is None:
            return DriftDetection(
                engine_id=engine_id,
                cycle=cycle,
                is_drift=False,
                drift_score=0.0,
                drifted_features=[],
                drift_type='none',
                timestamp=datetime.now().isoformat(),
            )

        try:
            # Use drift detector if available
            is_drift = False
            drift_score = 0.0
            drifted_features = []

            if hasattr(self.drift_detector, 'detect'):
                drift_result = self.drift_detector.detect(
                    current_data,
                    reference_data
                )
                is_drift = drift_result.get('is_drift', False)
                drift_score = drift_result.get('score', 0.0)
                drifted_features = drift_result.get('drifted_features', [])

            else:
                # Simple statistical drift detection
                if reference_data is not None:
                    current_mean = np.mean(current_data, axis=0) if len(current_data.shape) > 1 else np.mean(current_data)
                    ref_mean = np.mean(reference_data, axis=0) if len(reference_data.shape) > 1 else np.mean(reference_data)
                    
                    drift_score = float(np.mean(np.abs(current_mean - ref_mean)))
                    is_drift = drift_score >= self.drift_threshold

            result = DriftDetection(
                engine_id=engine_id,
                cycle=cycle,
                is_drift=is_drift,
                drift_score=drift_score,
                drifted_features=drifted_features,
                drift_type='feature_drift' if is_drift else 'none',
                timestamp=datetime.now().isoformat(),
            )

            self.drift_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return DriftDetection(
                engine_id=engine_id,
                cycle=cycle,
                is_drift=False,
                drift_score=0.0,
                drifted_features=[],
                drift_type='none',
                timestamp=datetime.now().isoformat(),
            )

    def generate_report(
        self,
        sensor_data: np.ndarray,
        engine_id: int,
        cycle: int,
        sensor_names: Optional[List[str]] = None,
        reference_data: Optional[np.ndarray] = None,
    ) -> MonitoringReport:
        """
        Generate comprehensive monitoring report.
        
        Args:
            sensor_data: Current sensor readings
            engine_id: Engine identifier
            cycle: Current cycle number
            sensor_names: Names of sensors
            reference_data: Reference data for drift detection
        
        Returns:
            MonitoringReport with all monitoring signals
        """
        # Get predictions
        prediction = self.predict_rul(sensor_data, engine_id, cycle)

        # Detect anomalies
        anomaly = self.detect_anomalies(sensor_data, engine_id, cycle, sensor_names)

        # Detect drift
        drift = self.detect_drift(
            sensor_data,
            reference_data,
            engine_id,
            cycle,
            sensor_names,
        )

        # Calculate overall confidence
        overall_confidence = min(
            prediction.confidence,
            1.0 - anomaly.anomaly_score,
            1.0 - drift.drift_score if drift.drift_score > 0 else 1.0,
        )

        # Generate alert if any signal is concerning
        alert_flag = (
            prediction.confidence < self.confidence_threshold
            or anomaly.is_anomaly
            or drift.is_drift
        )

        report = MonitoringReport(
            engine_id=engine_id,
            cycle=cycle,
            timestamp=datetime.now().isoformat(),
            prediction=prediction,
            anomaly=anomaly,
            drift=drift,
            overall_confidence=overall_confidence,
            alert_flag=alert_flag,
        )

        return report

    def _estimate_uncertainty(self, model_name: str, rul_pred: float) -> float:
        """Estimate prediction uncertainty."""
        # Simple heuristic: lower uncertainty for mid-range RUL values
        # Higher uncertainty at extremes
        if rul_pred < 10 or rul_pred > 200:
            return 0.3
        elif rul_pred < 30 or rul_pred > 150:
            return 0.2
        else:
            return 0.1

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'n_predictions': len(self.prediction_history),
            'n_anomalies': len([a for a in self.anomaly_history if a.is_anomaly]),
            'n_drifts': len([d for d in self.drift_history if d.is_drift]),
            'avg_prediction_confidence': np.mean([p.confidence for p in self.prediction_history]) if self.prediction_history else 0.0,
            'avg_anomaly_score': np.mean([a.anomaly_score for a in self.anomaly_history]) if self.anomaly_history else 0.0,
            'avg_drift_score': np.mean([d.drift_score for d in self.drift_history]) if self.drift_history else 0.0,
        }

    def reset_history(self):
        """Clear monitoring history."""
        self.prediction_history = []
        self.anomaly_history = []
        self.drift_history = []
        logger.info("Monitoring history cleared")
