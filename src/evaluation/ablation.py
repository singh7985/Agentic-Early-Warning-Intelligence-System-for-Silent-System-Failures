"""
Ablation Study: Isolate contribution of each component

Framework tests:
1. ML only (baseline)
2. ML + RAG (add retrieval)
3. ML + RAG - Agent 1 (remove monitoring agent)
4. ML + RAG - Agent 2 (remove retrieval agent)
5. ML + RAG - Agent 3 (remove reasoning agent)
6. ML + RAG - Agent 4 (remove action agent)
7. ML + RAG + All Agents (full system)

This isolates the contribution of each component.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .metrics import MetricsCalculator, CompleteMetrics

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Ablation study configuration"""
    name: str
    has_ml: bool
    has_rag: bool
    has_monitoring_agent: bool
    has_retrieval_agent: bool
    has_reasoning_agent: bool
    has_action_agent: bool


@dataclass
class AblationResult:
    """Result for single ablation configuration"""
    config: AblationConfig
    metrics: CompleteMetrics


class AblationStudy:
    """
    Performs ablation study to isolate component contributions.
    
    Features:
    - Test 7 different configurations
    - Measure impact of each component
    - Identify which components matter most
    - Generate contribution analysis
    """

    def __init__(self):
        """Initialize ablation study."""
        # Define 7 configurations
        self.configs = {
            'ml_only': AblationConfig(
                name='ML Only',
                has_ml=True,
                has_rag=False,
                has_monitoring_agent=False,
                has_retrieval_agent=False,
                has_reasoning_agent=False,
                has_action_agent=False,
            ),
            'ml_rag': AblationConfig(
                name='ML + RAG',
                has_ml=True,
                has_rag=True,
                has_monitoring_agent=False,
                has_retrieval_agent=False,
                has_reasoning_agent=False,
                has_action_agent=False,
            ),
            'ml_rag_no_monitoring': AblationConfig(
                name='ML + RAG (no Monitoring Agent)',
                has_ml=True,
                has_rag=True,
                has_monitoring_agent=False,  # REMOVED
                has_retrieval_agent=True,
                has_reasoning_agent=True,
                has_action_agent=True,
            ),
            'ml_rag_no_retrieval': AblationConfig(
                name='ML + RAG (no Retrieval Agent)',
                has_ml=True,
                has_rag=True,
                has_monitoring_agent=True,
                has_retrieval_agent=False,  # REMOVED
                has_reasoning_agent=True,
                has_action_agent=True,
            ),
            'ml_rag_no_reasoning': AblationConfig(
                name='ML + RAG (no Reasoning Agent)',
                has_ml=True,
                has_rag=True,
                has_monitoring_agent=True,
                has_retrieval_agent=True,
                has_reasoning_agent=False,  # REMOVED
                has_action_agent=True,
            ),
            'ml_rag_no_action': AblationConfig(
                name='ML + RAG (no Action Agent)',
                has_ml=True,
                has_rag=True,
                has_monitoring_agent=True,
                has_retrieval_agent=True,
                has_reasoning_agent=True,
                has_action_agent=False,  # REMOVED
            ),
            'ml_rag_all_agents': AblationConfig(
                name='ML + RAG + All Agents',
                has_ml=True,
                has_rag=True,
                has_monitoring_agent=True,
                has_retrieval_agent=True,
                has_reasoning_agent=True,
                has_action_agent=True,
            ),
        }

        # Initialize metric calculators for each config
        self.results = {}
        for key, config in self.configs.items():
            self.results[key] = {
                'calculator': MetricsCalculator(config.name),
                'config': config,
            }

        logger.info(f"Ablation study initialized with {len(self.configs)} configurations")

    def add_result(
        self,
        config_key: str,
        predicted_rul: float,
        actual_rul: float,
        explanation: str = "",
        citations: int = 0,
        patterns: int = 0,
        engine_id: int = 0,
        cycle: int = 0,
    ):
        """Add prediction result for configuration."""
        if config_key not in self.results:
            logger.warning(f"Unknown configuration: {config_key}")
            return

        calculator = self.results[config_key]['calculator']
        calculator.add_rul_prediction(predicted_rul, actual_rul, engine_id, cycle)
        
        if explanation:
            calculator.add_explanation(engine_id, cycle, explanation, citations, patterns)

    def add_failure(
        self,
        engine_id: int,
        failure_cycle: int,
        failure_type: str = "unknown",
    ):
        """Add failure event (for all configurations)."""
        for key in self.results.keys():
            self.results[key]['calculator'].add_failure_event(
                engine_id, failure_cycle, failure_type
            )

    def add_warning(
        self,
        config_key: str,
        engine_id: int,
        warning_cycle: int,
        confidence: float = 1.0,
        correct: bool = False,
    ):
        """Add warning for configuration."""
        if config_key not in self.results:
            logger.warning(f"Unknown configuration: {config_key}")
            return

        self.results[config_key]['calculator'].add_warning(
            engine_id, warning_cycle, confidence, correct
        )

    def compute_ablation(self) -> Dict[str, AblationResult]:
        """Compute results for all configurations."""
        logger.info("Computing ablation study results...")

        ablation_results = {}
        for key, result_dict in self.results.items():
            calculator = result_dict['calculator']
            config = result_dict['config']
            
            metrics = calculator.calculate_all_metrics()
            ablation_results[key] = AblationResult(
                config=config,
                metrics=metrics,
            )

        logger.info("Ablation study complete")
        return ablation_results

    def get_ablation_table(self) -> pd.DataFrame:
        """Get ablation results as table."""
        results = self.compute_ablation()

        data = []
        for key, result in results.items():
            data.append({
                'Configuration': result.config.name,
                'RUL MAE': f"{result.metrics.rul_metrics.mae:.1f}",
                'RUL RMSE': f"{result.metrics.rul_metrics.rmse:.1f}",
                'RUL R²': f"{result.metrics.rul_metrics.r_squared:.2f}",
                'Warning Rate': f"{result.metrics.warning_metrics.warning_rate*100:.1f}%",
                'Lead Time': f"{result.metrics.warning_metrics.avg_lead_time:.0f}",
                'False Alarms': f"{result.metrics.warning_metrics.false_alarm_rate*100:.1f}%",
                'Groundedness': f"{result.metrics.groundedness_metrics.avg_groundedness:.2f}",
                'F1 Score': f"{result.metrics.detection_metrics.f1_score:.2f}",
            })

        return pd.DataFrame(data)

    def calculate_component_contribution(self) -> Dict[str, float]:
        """Calculate contribution of each component."""
        results = self.compute_ablation()

        # Use full system as baseline
        full_system = results['ml_rag_all_agents'].metrics

        contributions = {}

        # Calculate impact of each agent removal
        if 'ml_rag_no_monitoring' in results:
            monitoring_impact = self._calculate_impact(
                results['ml_rag_no_monitoring'].metrics,
                full_system,
            )
            contributions['Monitoring Agent'] = monitoring_impact

        if 'ml_rag_no_retrieval' in results:
            retrieval_impact = self._calculate_impact(
                results['ml_rag_no_retrieval'].metrics,
                full_system,
            )
            contributions['Retrieval Agent'] = retrieval_impact

        if 'ml_rag_no_reasoning' in results:
            reasoning_impact = self._calculate_impact(
                results['ml_rag_no_reasoning'].metrics,
                full_system,
            )
            contributions['Reasoning Agent'] = reasoning_impact

        if 'ml_rag_no_action' in results:
            action_impact = self._calculate_impact(
                results['ml_rag_no_action'].metrics,
                full_system,
            )
            contributions['Action Agent'] = action_impact

        # Calculate impact of RAG vs ML only
        rag_impact = self._calculate_impact(
            results['ml_only'].metrics,
            results['ml_rag'].metrics,
        )
        contributions['RAG System'] = rag_impact

        return contributions

    def _calculate_impact(
        self,
        without: CompleteMetrics,
        with_: CompleteMetrics,
    ) -> float:
        """
        Calculate overall impact of a component.
        Positive = component helps performance
        """
        # Average normalized improvement across metrics
        improvements = []

        # RUL error (lower is better)
        if without.rul_metrics.mae > 0:
            rul_improvement = (without.rul_metrics.mae - with_.rul_metrics.mae) \
                / without.rul_metrics.mae
            improvements.append(rul_improvement)

        # Warning rate (higher is better)
        if without.warning_metrics.warning_rate > 0:
            warning_improvement = (with_.warning_metrics.warning_rate - without.warning_metrics.warning_rate) \
                / without.warning_metrics.warning_rate
            improvements.append(warning_improvement)

        # False alarm rate (lower is better)
        if without.warning_metrics.false_alarm_rate > 0:
            fa_improvement = (without.warning_metrics.false_alarm_rate - with_.warning_metrics.false_alarm_rate) \
                / without.warning_metrics.false_alarm_rate
            improvements.append(fa_improvement)

        # Groundedness (higher is better)
        if without.groundedness_metrics.avg_groundedness > 0:
            groundedness_improvement = (with_.groundedness_metrics.avg_groundedness - without.groundedness_metrics.avg_groundedness) \
                / without.groundedness_metrics.avg_groundedness
            improvements.append(groundedness_improvement)

        # F1 score (higher is better)
        if without.detection_metrics.f1_score > 0:
            f1_improvement = (with_.detection_metrics.f1_score - without.detection_metrics.f1_score) \
                / without.detection_metrics.f1_score
            improvements.append(f1_improvement)

        # Average improvement
        overall_impact = float(np.mean(improvements)) if improvements else 0.0
        return overall_impact

    def get_contribution_summary(self) -> pd.DataFrame:
        """Get component contribution as dataframe."""
        contributions = self.calculate_component_contribution()

        # Sort by impact
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        data = {
            'Component': [c[0] for c in sorted_contribs],
            'Impact Score': [f"{c[1]:.2%}" for c in sorted_contribs],
        }

        return pd.DataFrame(data)

    def get_ablation_results(self) -> List[AblationResult]:
        """Get all ablation results as list."""
        results = []
        for config_key, result_dict in self.results.items():
            config = result_dict['config']
            metrics = result_dict['calculator'].calculate_all_metrics()
            results.append(AblationResult(config=config, metrics=metrics))
        return results

    def reset(self):
        """Reset all configurations."""
        for result_dict in self.results.values():
            result_dict['calculator'].reset()
        logger.info("Ablation study reset")
