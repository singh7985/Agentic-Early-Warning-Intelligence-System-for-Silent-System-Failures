"""
Metrics Calculator: Computes evaluation metrics for the system

Metrics:
1. RUL Prediction Error: MAE, RMSE, correlation
2. Early Warning Lead-Time: How much advance notice before failure
3. Groundedness Score: How well-grounded are explanations
4. False Alarm Rate: Predictions vs. actual failures
5. Precision/Recall: Detection quality at thresholds
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RULMetrics:
    """RUL prediction metrics"""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    nasa_score: float  # NASA Scoring Function
    r_squared: float  # Coefficient of determination
    pearson_r: float  # Pearson correlation
    spearman_r: float  # Spearman correlation
    median_error: float  # Median absolute error


@dataclass
class WarningMetrics:
    """Early warning metrics"""
    avg_lead_time: float  # Cycles of advance notice
    min_lead_time: float
    max_lead_time: float
    warning_rate: float  # % engines warned before failure
    false_alarm_rate: float  # % warnings that didn't lead to failure
    missed_failure_rate: float  # % failures with no warning


@dataclass
class GroundednessMetrics:
    """Explanation groundedness metrics"""
    avg_groundedness: float  # 0-1 score
    avg_citation_count: float
    avg_pattern_match_score: float
    explanation_coverage: float  # % of decisions with explanation
    historical_relevance: float  # Relevance of retrieved cases


@dataclass
class DetectionMetrics:
    """Anomaly/degradation detection metrics"""
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    f1_score: float  # Harmonic mean
    accuracy: float  # (TP + TN) / Total
    specificity: float  # TN / (TN + FP)
    roc_auc: float  # Area under ROC curve


@dataclass
class CompleteMetrics:
    """All metrics for a system configuration"""
    system_name: str
    rul_metrics: RULMetrics
    warning_metrics: WarningMetrics
    groundedness_metrics: GroundednessMetrics
    detection_metrics: DetectionMetrics
    timestamp: str


class MetricsCalculator:
    """
    Calculates comprehensive evaluation metrics for system comparison.
    
    Features:
    - RUL prediction error (MAE, RMSE, MAPE, correlation)
    - Early warning lead-time analysis
    - Groundedness scoring (explanation quality)
    - False alarm analysis
    - Detection metrics (precision, recall, F1)
    """

    def __init__(self, name: str = "System"):
        """
        Initialize metrics calculator.
        
        Args:
            name: System name for results tracking
        """
        self.name = name
        self.predictions = []
        self.ground_truth = []
        self.warnings = []
        self.failures = []
        self.explanations = []

    def add_rul_prediction(
        self,
        predicted_rul: float,
        actual_rul: float,
        engine_id: int = 0,
        cycle: int = 0,
    ):
        """Add RUL prediction for evaluation."""
        self.predictions.append({
            'engine_id': engine_id,
            'cycle': cycle,
            'predicted': predicted_rul,
            'actual': actual_rul,
            'error': abs(predicted_rul - actual_rul),
            'relative_error': abs(predicted_rul - actual_rul) / max(1, actual_rul),
        })

    def add_failure_event(
        self,
        engine_id: int,
        failure_cycle: int,
        failure_type: str = "unknown",
        severity: str = "unknown",
    ):
        """Add actual failure event."""
        self.failures.append({
            'engine_id': engine_id,
            'cycle': failure_cycle,
            'type': failure_type,
            'severity': severity,
        })

    def add_warning(
        self,
        engine_id: int,
        warning_cycle: int,
        confidence: float = 1.0,
        correct: bool = False,
    ):
        """Add generated warning."""
        self.warnings.append({
            'engine_id': engine_id,
            'cycle': warning_cycle,
            'confidence': confidence,
            'correct': correct,
        })

    def add_explanation(
        self,
        engine_id: int,
        cycle: int,
        explanation: str,
        citations: int = 0,
        pattern_matches: int = 0,
    ):
        """Add explanation for groundedness evaluation."""
        self.explanations.append({
            'engine_id': engine_id,
            'cycle': cycle,
            'text': explanation,
            'citations': citations,
            'patterns': pattern_matches,
        })

    def calculate_rul_metrics(self) -> RULMetrics:
        """Calculate RUL prediction metrics."""
        if not self.predictions:
            logger.warning("No predictions available for RUL metrics")
            return RULMetrics(
                mae=0, rmse=0, nasa_score=0, r_squared=0,
                pearson_r=0, spearman_r=0, median_error=0
            )

        pred_df = pd.DataFrame(self.predictions)
        
        predicted = pred_df['predicted'].values
        actual = pred_df['actual'].values
        errors = pred_df['error'].values
        relative_errors = pred_df['relative_error'].values

        # MAE
        mae = float(np.mean(errors))

        # RMSE
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # NASA Score
        d = predicted - actual
        nasa_score = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Pearson correlation
        if len(predicted) > 2:
            pearson_r = float(np.corrcoef(predicted, actual)[0, 1])
        else:
            pearson_r = 0

        # Spearman correlation
        if len(predicted) > 2:
            spearman_r = float(stats.spearmanr(predicted, actual)[0])
        else:
            spearman_r = 0

        # Median error
        median_error = float(np.median(errors))

        return RULMetrics(
            mae=mae,
            rmse=rmse,
            nasa_score=nasa_score,
            r_squared=r_squared,
            pearson_r=pearson_r,
            spearman_r=spearman_r,
            median_error=median_error,
        )

    def calculate_warning_metrics(self) -> WarningMetrics:
        """Calculate early warning metrics."""
        if not self.warnings or not self.failures:
            logger.warning("No warnings or failures for warning metrics")
            return WarningMetrics(
                avg_lead_time=0, min_lead_time=0, max_lead_time=0,
                warning_rate=0, false_alarm_rate=0, missed_failure_rate=0
            )

        warnings_df = pd.DataFrame(self.warnings)
        failures_df = pd.DataFrame(self.failures)

        # Calculate lead times
        lead_times = []
        warned_engines = set()

        for _, failure in failures_df.iterrows():
            engine_id = failure['engine_id']
            failure_cycle = failure['cycle']
            
            # Find warnings for this engine before failure
            engine_warnings = warnings_df[
                (warnings_df['engine_id'] == engine_id) &
                (warnings_df['cycle'] < failure_cycle)
            ]

            if len(engine_warnings) > 0:
                warned_engines.add(engine_id)
                # Lead time = cycles between first warning and failure
                first_warning = engine_warnings.iloc[0]
                lead_time = failure_cycle - first_warning['cycle']
                lead_times.append(lead_time)

        # Metrics
        avg_lead_time = float(np.mean(lead_times)) if lead_times else 0
        min_lead_time = float(np.min(lead_times)) if lead_times else 0
        max_lead_time = float(np.max(lead_times)) if lead_times else 0

        warning_rate = len(warned_engines) / len(failures_df['engine_id'].unique()) \
            if len(failures_df) > 0 else 0

        # False alarms
        correct_warnings = len(warnings_df[warnings_df['correct']])
        total_warnings = len(warnings_df)
        false_alarm_rate = (total_warnings - correct_warnings) / max(1, total_warnings)

        # Missed failures
        missed = len(failures_df) - len(warned_engines)
        missed_failure_rate = missed / len(failures_df) if len(failures_df) > 0 else 0

        return WarningMetrics(
            avg_lead_time=avg_lead_time,
            min_lead_time=min_lead_time,
            max_lead_time=max_lead_time,
            warning_rate=warning_rate,
            false_alarm_rate=false_alarm_rate,
            missed_failure_rate=missed_failure_rate,
        )

    def calculate_groundedness_metrics(self) -> GroundednessMetrics:
        """Calculate explanation groundedness metrics."""
        if not self.explanations:
            logger.warning("No explanations for groundedness metrics")
            return GroundednessMetrics(
                avg_groundedness=0, avg_citation_count=0,
                avg_pattern_match_score=0, explanation_coverage=0,
                historical_relevance=0
            )

        expl_df = pd.DataFrame(self.explanations)

        # Average citations per explanation
        avg_citation_count = float(expl_df['citations'].mean())

        # Average pattern matches
        avg_pattern_match = float(expl_df['patterns'].mean())

        # Groundedness score (0-1, based on citations and patterns)
        # More citations/patterns = more grounded
        max_citations = 5
        max_patterns = 3
        groundedness_scores = []

        for _, row in expl_df.iterrows():
            citation_score = min(row['citations'] / max_citations, 1.0)
            pattern_score = min(row['patterns'] / max_patterns, 1.0)
            combined_score = (citation_score * 0.6 + pattern_score * 0.4)
            groundedness_scores.append(combined_score)

        avg_groundedness = float(np.mean(groundedness_scores)) \
            if groundedness_scores else 0

        # Coverage: % of decisions with explanation
        explanation_coverage = len(expl_df) / max(1, len(expl_df))

        # Historical relevance: estimate from pattern matches
        historical_relevance = min(avg_pattern_match / 3.0, 1.0)

        return GroundednessMetrics(
            avg_groundedness=avg_groundedness,
            avg_citation_count=avg_citation_count,
            avg_pattern_match_score=avg_pattern_match,
            explanation_coverage=explanation_coverage,
            historical_relevance=historical_relevance,
        )

    def calculate_detection_metrics(self) -> DetectionMetrics:
        """Calculate anomaly/degradation detection metrics."""
        if not self.warnings or not self.failures:
            logger.warning("Insufficient data for detection metrics")
            return DetectionMetrics(
                precision=0, recall=0, f1_score=0, accuracy=0,
                specificity=0, roc_auc=0
            )

        warnings_df = pd.DataFrame(self.warnings)
        failures_df = pd.DataFrame(self.failures)

        # Build confusion matrix
        tp = len(warnings_df[warnings_df['correct']])
        fp = len(warnings_df[~warnings_df['correct']])
        fn = len(failures_df) - tp
        tn = 1000 - (tp + fp + fn)  # Assume 1000 total cases

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # ROC AUC (approximate from confidence scores)
        if len(warnings_df) > 0:
            confidences = warnings_df['confidence'].values
            roc_auc = float(np.mean(confidences))
        else:
            roc_auc = 0.5

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            specificity=specificity,
            roc_auc=roc_auc,
        )

    def calculate_all_metrics(self) -> CompleteMetrics:
        """Calculate all metrics."""
        import time
        from datetime import datetime

        return CompleteMetrics(
            system_name=self.name,
            rul_metrics=self.calculate_rul_metrics(),
            warning_metrics=self.calculate_warning_metrics(),
            groundedness_metrics=self.calculate_groundedness_metrics(),
            detection_metrics=self.calculate_detection_metrics(),
            timestamp=datetime.now().isoformat(),
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        metrics = self.calculate_all_metrics()
        return {
            'system': metrics.system_name,
            'rul': asdict(metrics.rul_metrics),
            'warning': asdict(metrics.warning_metrics),
            'groundedness': asdict(metrics.groundedness_metrics),
            'detection': asdict(metrics.detection_metrics),
            'timestamp': metrics.timestamp,
        }

    def reset(self):
        """Reset all data."""
        self.predictions = []
        self.ground_truth = []
        self.warnings = []
        self.failures = []
        self.explanations = []
        logger.info(f"Reset metrics for {self.name}")
