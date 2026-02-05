"""
System Comparison: Compare three system configurations

Configurations:
1. ML only: Baseline with just RUL prediction
2. ML + RAG: With vector database retrieval
3. ML + RAG + Agents: Full system with agents
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .metrics import MetricsCalculator, CompleteMetrics

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """System configuration"""
    name: str
    description: str
    has_ml: bool = True
    has_rag: bool = False
    has_agents: bool = False


@dataclass
class ComparisonResult:
    """Result from system comparison"""
    ml_only: CompleteMetrics
    ml_rag: CompleteMetrics
    ml_rag_agents: CompleteMetrics
    
    # Improvement metrics
    rag_improvement: Dict[str, float]  # ML+RAG vs ML
    agents_improvement: Dict[str, float]  # ML+RAG+Agents vs ML+RAG
    
    timestamp: str


class SystemComparison:
    """
    Compares three system configurations on key metrics.
    
    Features:
    - Evaluate all three systems on same dataset
    - Calculate improvement metrics
    - Identify which components contribute most
    - Generate comparison tables and visualizations
    """

    def __init__(self):
        """Initialize comparison framework."""
        self.ml_only = MetricsCalculator("ML Only")
        self.ml_rag = MetricsCalculator("ML + RAG")
        self.ml_rag_agents = MetricsCalculator("ML + RAG + Agents")
        
        logger.info("SystemComparison initialized")

    def add_ml_only_result(
        self,
        predicted_rul: float,
        actual_rul: float,
        engine_id: int = 0,
        cycle: int = 0,
    ):
        """Add result for ML-only system."""
        self.ml_only.add_rul_prediction(predicted_rul, actual_rul, engine_id, cycle)

    def add_ml_rag_result(
        self,
        predicted_rul: float,
        actual_rul: float,
        explanation: str,
        citations: int = 0,
        engine_id: int = 0,
        cycle: int = 0,
    ):
        """Add result for ML+RAG system."""
        self.ml_rag.add_rul_prediction(predicted_rul, actual_rul, engine_id, cycle)
        self.ml_rag.add_explanation(engine_id, cycle, explanation, citations)

    def add_ml_rag_agents_result(
        self,
        predicted_rul: float,
        actual_rul: float,
        explanation: str,
        citations: int = 0,
        patterns: int = 0,
        engine_id: int = 0,
        cycle: int = 0,
    ):
        """Add result for ML+RAG+Agents system."""
        self.ml_rag_agents.add_rul_prediction(predicted_rul, actual_rul, engine_id, cycle)
        self.ml_rag_agents.add_explanation(
            engine_id, cycle, explanation, citations, patterns
        )

    def add_failure_event(
        self,
        engine_id: int,
        failure_cycle: int,
        failure_type: str = "unknown",
    ):
        """Add actual failure event (for all systems)."""
        self.ml_only.add_failure_event(engine_id, failure_cycle, failure_type)
        self.ml_rag.add_failure_event(engine_id, failure_cycle, failure_type)
        self.ml_rag_agents.add_failure_event(engine_id, failure_cycle, failure_type)

    def add_warning(
        self,
        system: str,  # 'ml', 'rag', 'agents'
        engine_id: int,
        warning_cycle: int,
        confidence: float = 1.0,
        correct: bool = False,
    ):
        """Add warning for specific system."""
        if system == 'ml':
            self.ml_only.add_warning(engine_id, warning_cycle, confidence, correct)
        elif system == 'rag':
            self.ml_rag.add_warning(engine_id, warning_cycle, confidence, correct)
        elif system == 'agents':
            self.ml_rag_agents.add_warning(engine_id, warning_cycle, confidence, correct)

    def compare(self) -> ComparisonResult:
        """Compare all three systems."""
        import time
        from datetime import datetime

        logger.info("Comparing ML, ML+RAG, and ML+RAG+Agents systems...")

        # Get metrics for all systems
        ml_metrics = self.ml_only.calculate_all_metrics()
        rag_metrics = self.ml_rag.calculate_all_metrics()
        agents_metrics = self.ml_rag_agents.calculate_all_metrics()

        # Calculate improvements
        rag_improvement = self._calculate_improvement(ml_metrics, rag_metrics)
        agents_improvement = self._calculate_improvement(rag_metrics, agents_metrics)

        result = ComparisonResult(
            ml_only=ml_metrics,
            ml_rag=rag_metrics,
            ml_rag_agents=agents_metrics,
            rag_improvement=rag_improvement,
            agents_improvement=agents_improvement,
            timestamp=datetime.now().isoformat(),
        )

        logger.info("Comparison complete")
        return result

    def _calculate_improvement(
        self,
        baseline: CompleteMetrics,
        system: CompleteMetrics,
    ) -> Dict[str, float]:
        """Calculate percentage improvement from baseline to system."""
        improvements = {}

        # RUL metrics (lower is better)
        improvements['rul_mae'] = (baseline.rul_metrics.mae - system.rul_metrics.mae) \
            / max(baseline.rul_metrics.mae, 0.1) * 100
        improvements['rul_rmse'] = (baseline.rul_metrics.rmse - system.rul_metrics.rmse) \
            / max(baseline.rul_metrics.rmse, 0.1) * 100
        improvements['rul_r2'] = (system.rul_metrics.r_squared - baseline.rul_metrics.r_squared) \
            / max(abs(baseline.rul_metrics.r_squared), 0.1) * 100

        # Warning metrics (higher is better)
        improvements['warning_rate'] = (system.warning_metrics.warning_rate - baseline.warning_metrics.warning_rate) \
            / max(baseline.warning_metrics.warning_rate, 0.01) * 100
        improvements['lead_time'] = (system.warning_metrics.avg_lead_time - baseline.warning_metrics.avg_lead_time) \
            / max(baseline.warning_metrics.avg_lead_time, 1) * 100
        improvements['false_alarm'] = (baseline.warning_metrics.false_alarm_rate - system.warning_metrics.false_alarm_rate) \
            / max(baseline.warning_metrics.false_alarm_rate, 0.01) * 100

        # Groundedness (higher is better)
        improvements['groundedness'] = (system.groundedness_metrics.avg_groundedness - baseline.groundedness_metrics.avg_groundedness) \
            / max(baseline.groundedness_metrics.avg_groundedness, 0.1) * 100
        improvements['citations'] = (system.groundedness_metrics.avg_citation_count - baseline.groundedness_metrics.avg_citation_count) \
            / max(baseline.groundedness_metrics.avg_citation_count, 0.1) * 100

        # Detection metrics (higher is better)
        improvements['f1_score'] = (system.detection_metrics.f1_score - baseline.detection_metrics.f1_score) \
            / max(baseline.detection_metrics.f1_score, 0.1) * 100
        improvements['precision'] = (system.detection_metrics.precision - baseline.detection_metrics.precision) \
            / max(baseline.detection_metrics.precision, 0.1) * 100
        improvements['recall'] = (system.detection_metrics.recall - baseline.detection_metrics.recall) \
            / max(baseline.detection_metrics.recall, 0.1) * 100

        return improvements

    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison as dataframe."""
        comparison = self.compare()

        data = {
            'Metric': [
                'RUL MAE',
                'RUL RMSE',
                'RUL R²',
                'Warning Rate (%)',
                'Lead Time (cycles)',
                'False Alarm Rate (%)',
                'Groundedness',
                'Citations/Explanation',
                'Precision',
                'Recall',
                'F1 Score',
            ],
            'ML Only': [
                f"{comparison.ml_only.rul_metrics.mae:.1f}",
                f"{comparison.ml_only.rul_metrics.rmse:.1f}",
                f"{comparison.ml_only.rul_metrics.r_squared:.2f}",
                f"{comparison.ml_only.warning_metrics.warning_rate*100:.1f}",
                f"{comparison.ml_only.warning_metrics.avg_lead_time:.0f}",
                f"{comparison.ml_only.warning_metrics.false_alarm_rate*100:.1f}",
                f"{comparison.ml_only.groundedness_metrics.avg_groundedness:.2f}",
                f"{comparison.ml_only.groundedness_metrics.avg_citation_count:.1f}",
                f"{comparison.ml_only.detection_metrics.precision:.2f}",
                f"{comparison.ml_only.detection_metrics.recall:.2f}",
                f"{comparison.ml_only.detection_metrics.f1_score:.2f}",
            ],
            'ML + RAG': [
                f"{comparison.ml_rag.rul_metrics.mae:.1f}",
                f"{comparison.ml_rag.rul_metrics.rmse:.1f}",
                f"{comparison.ml_rag.rul_metrics.r_squared:.2f}",
                f"{comparison.ml_rag.warning_metrics.warning_rate*100:.1f}",
                f"{comparison.ml_rag.warning_metrics.avg_lead_time:.0f}",
                f"{comparison.ml_rag.warning_metrics.false_alarm_rate*100:.1f}",
                f"{comparison.ml_rag.groundedness_metrics.avg_groundedness:.2f}",
                f"{comparison.ml_rag.groundedness_metrics.avg_citation_count:.1f}",
                f"{comparison.ml_rag.detection_metrics.precision:.2f}",
                f"{comparison.ml_rag.detection_metrics.recall:.2f}",
                f"{comparison.ml_rag.detection_metrics.f1_score:.2f}",
            ],
            'ML + RAG + Agents': [
                f"{comparison.ml_rag_agents.rul_metrics.mae:.1f}",
                f"{comparison.ml_rag_agents.rul_metrics.rmse:.1f}",
                f"{comparison.ml_rag_agents.rul_metrics.r_squared:.2f}",
                f"{comparison.ml_rag_agents.warning_metrics.warning_rate*100:.1f}",
                f"{comparison.ml_rag_agents.warning_metrics.avg_lead_time:.0f}",
                f"{comparison.ml_rag_agents.warning_metrics.false_alarm_rate*100:.1f}",
                f"{comparison.ml_rag_agents.groundedness_metrics.avg_groundedness:.2f}",
                f"{comparison.ml_rag_agents.groundedness_metrics.avg_citation_count:.1f}",
                f"{comparison.ml_rag_agents.detection_metrics.precision:.2f}",
                f"{comparison.ml_rag_agents.detection_metrics.recall:.2f}",
                f"{comparison.ml_rag_agents.detection_metrics.f1_score:.2f}",
            ],
        }

        return pd.DataFrame(data)

    def get_improvement_summary(self) -> Dict[str, Dict[str, float]]:
        """Get improvement summary."""
        comparison = self.compare()
        return {
            'rag_vs_ml': comparison.rag_improvement,
            'agents_vs_rag': comparison.agents_improvement,
        }

    def reset(self):
        """Reset all systems."""
        self.ml_only.reset()
        self.ml_rag.reset()
        self.ml_rag_agents.reset()
        logger.info("All systems reset")
