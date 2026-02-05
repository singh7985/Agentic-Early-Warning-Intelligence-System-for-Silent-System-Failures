"""
System Evaluator: High-level orchestrator for comprehensive evaluation

Purpose:
- Coordinate all evaluation components
- Run comparative evaluations
- Generate evaluation reports
- Provide summary metrics
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.evaluation.metrics import MetricsCalculator, CompleteMetrics
from src.evaluation.comparison import SystemComparison, ComparisonResult
from src.evaluation.ablation import AblationStudy, AblationResult
from src.evaluation.failure_analysis import FailureAnalyzer, FailureAnalysis

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    comparison_result: ComparisonResult
    ablation_result: List[AblationResult]
    failure_analysis: FailureAnalysis
    summary_metrics: Dict[str, Any]
    recommendations: List[str]


class SystemEvaluator:
    """
    High-level evaluator that coordinates all evaluation components.
    
    Workflow:
    1. Initialize with evaluation data
    2. Run system comparison (3 variants)
    3. Run ablation study (7 configurations)
    4. Analyze failure cases
    5. Generate comprehensive report
    """

    def __init__(self):
        """Initialize system evaluator."""
        self.comparison = SystemComparison()
        self.ablation = AblationStudy()
        self.failure_analyzer = FailureAnalyzer()
        self.report: Optional[EvaluationReport] = None
        logger.info("SystemEvaluator initialized")

    def add_evaluation_result(
        self,
        engine_id: int,
        cycle: int,
        # ML baseline results
        ml_rul: float,
        ml_confidence: float = 1.0,
        # ML + RAG results
        rag_rul: Optional[float] = None,
        rag_explanation: Optional[str] = None,
        rag_citations: int = 0,
        rag_confidence: Optional[float] = None,
        # Full system results (ML + RAG + Agents)
        agents_rul: Optional[float] = None,
        agents_explanation: Optional[str] = None,
        agents_citations: int = 0,
        agents_patterns: int = 0,
        agents_confidence: Optional[float] = None,
        # Ground truth
        actual_rul: float = None,
        failure_cycle: Optional[int] = None,
    ):
        """
        Add an evaluation result for all systems.
        
        Args:
            engine_id: Engine ID
            cycle: Current cycle
            ml_rul: ML baseline RUL prediction
            ml_confidence: ML confidence
            rag_rul: ML + RAG RUL prediction
            rag_explanation: RAG explanation with context
            rag_citations: Number of citations in RAG explanation
            rag_confidence: ML + RAG confidence
            agents_rul: Full system RUL prediction
            agents_explanation: Full system explanation
            agents_citations: Number of citations
            agents_patterns: Number of sensor patterns matched
            agents_confidence: Full system confidence
            actual_rul: Actual RUL (ground truth)
            failure_cycle: Actual failure cycle
        """
        # Add ML baseline
        self.comparison.add_ml_only_result(ml_rul, actual_rul)

        # Add ML + RAG
        if rag_rul is not None:
            self.comparison.add_ml_rag_result(
                rag_rul,
                actual_rul,
                rag_explanation or "",
                rag_citations,
            )

        # Add full system
        if agents_rul is not None:
            self.comparison.add_ml_rag_agents_result(
                agents_rul,
                actual_rul,
                agents_explanation or "",
                agents_citations,
                agents_patterns,
            )

        # Add failure event to all systems
        if actual_rul is not None and failure_cycle is not None:
            self.comparison.add_failure_event(engine_id, failure_cycle)
            self.ablation.add_failure(engine_id, failure_cycle, actual_rul)

    def add_warning(
        self,
        system: str,  # 'ml', 'rag', 'agents'
        engine_id: int,
        warning_cycle: int,
        confidence: float = 1.0,
        correct: bool = True,
    ):
        """Add warning event."""
        self.comparison.add_warning(system, engine_id, warning_cycle, confidence, correct)
        self.ablation.add_warning(system, engine_id, warning_cycle, confidence, correct)

    def add_failure_case(
        self,
        engine_id: int,
        cycle: int,
        predicted_rul: float,
        actual_rul: float,
        confidence: float = 1.0,
        warning_cycle: Optional[int] = None,
        diagnosis: str = "unknown",
        root_cause: str = "unknown",
        severity: str = "medium",
    ):
        """Add failure case for analysis."""
        self.failure_analyzer.add_case(
            engine_id=engine_id,
            cycle=cycle,
            predicted_rul=predicted_rul,
            actual_rul=actual_rul,
            confidence=confidence,
            warning_cycle=warning_cycle,
            diagnosis=diagnosis,
            root_cause=root_cause,
            severity=severity,
        )

    def evaluate(self) -> EvaluationReport:
        """Run comprehensive evaluation."""
        logger.info("Starting comprehensive system evaluation...")

        # Run comparison
        logger.info("Running system comparison...")
        comparison_result = self.comparison.compare()

        # Run ablation
        logger.info("Running ablation study...")
        self.ablation.compute_ablation()
        ablation_results = self.ablation.get_ablation_results()

        # Run failure analysis
        logger.info("Running failure analysis...")
        failure_analysis = self.failure_analyzer.analyze()

        # Generate summary metrics
        logger.info("Generating summary metrics...")
        summary_metrics = self._generate_summary(
            comparison_result,
            ablation_results,
            failure_analysis,
        )

        # Generate recommendations
        logger.info("Generating recommendations...")
        recommendations = self._generate_recommendations(
            comparison_result,
            ablation_results,
            failure_analysis,
        )

        # Create report
        self.report = EvaluationReport(
            comparison_result=comparison_result,
            ablation_result=ablation_results,
            failure_analysis=failure_analysis,
            summary_metrics=summary_metrics,
            recommendations=recommendations,
        )

        logger.info("Evaluation complete")
        return self.report

    def _generate_summary(
        self,
        comparison_result: ComparisonResult,
        ablation_results: List[AblationResult],
        failure_analysis: FailureAnalysis,
    ) -> Dict[str, Any]:
        """Generate summary metrics."""
        summary = {
            'Total Engines': failure_analysis.total_cases,
            'Failure Cases': len(failure_analysis.failure_cases),
            'False Negatives': failure_analysis.failure_types.get('false_negative', 0),
            'False Positives': failure_analysis.failure_types.get('false_positive', 0),
            'False Negative Rate': f"{failure_analysis.false_negative_rate*100:.1f}%",
            'False Positive Rate': f"{failure_analysis.false_positive_rate*100:.1f}%",
            'Avg RUL Error': f"{failure_analysis.avg_rul_error:.1f} cycles",
            'Avg Warning Lead Time': f"{failure_analysis.avg_lead_time:.0f} cycles",
        }

        # Add comparison improvements
        if comparison_result.ml_to_rag_improvement:
            summary['ML to ML+RAG Improvement'] = \
                f"{comparison_result.ml_to_rag_improvement['overall']*100:.1f}%"

        if comparison_result.rag_to_agents_improvement:
            summary['ML+RAG to Full System Improvement'] = \
                f"{comparison_result.rag_to_agents_improvement['overall']*100:.1f}%"

        return summary

    def _generate_recommendations(
        self,
        comparison_result: ComparisonResult,
        ablation_results: List[AblationResult],
        failure_analysis: FailureAnalysis,
    ) -> List[str]:
        """Generate recommendations based on evaluation."""
        recommendations = []

        # Recommendation 1: System value
        if comparison_result.ml_to_rag_improvement:
            rag_improvement = comparison_result.ml_to_rag_improvement.get('overall', 0)
            if rag_improvement > 0.1:
                recommendations.append(
                    f"RAG adds {rag_improvement*100:.1f}% improvement over ML baseline. "
                    "Definitely include RAG for better context-aware predictions."
                )

        if comparison_result.rag_to_agents_improvement:
            agent_improvement = comparison_result.rag_to_agents_improvement.get('overall', 0)
            if agent_improvement > 0.1:
                recommendations.append(
                    f"Agentic system adds {agent_improvement*100:.1f}% improvement over ML+RAG. "
                    "Agents significantly enhance decision quality."
                )

        # Recommendation 2: False negative handling
        if failure_analysis.false_negative_rate > 0.1:
            recommendations.append(
                f"High false negative rate ({failure_analysis.false_negative_rate*100:.1f}%). "
                "Consider adjusting monitoring agent sensitivity or improving anomaly detection."
            )

        # Recommendation 3: RUL accuracy
        if failure_analysis.avg_rul_error > 30:
            recommendations.append(
                f"RUL prediction error is high ({failure_analysis.avg_rul_error:.1f} cycles). "
                "Improve ML model training or add more relevant features to RAG."
            )

        # Recommendation 4: Warning timing
        if failure_analysis.avg_lead_time < 20:
            recommendations.append(
                f"Warning lead time is short ({failure_analysis.avg_lead_time:.0f} cycles). "
                "Increase monitoring sensitivity to provide more advance notice."
            )

        # Recommendation 5: Root cause
        if failure_analysis.root_causes:
            primary_cause = max(
                failure_analysis.root_causes,
                key=failure_analysis.root_causes.get
            )
            recommendations.append(
                f"Primary failure cause: {primary_cause}. "
                "Focus improvement efforts on detecting/predicting this failure mode."
            )

        return recommendations

    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table."""
        return self.comparison.get_comparison_table()

    def get_ablation_table(self) -> pd.DataFrame:
        """Get ablation study table."""
        return self.ablation.get_ablation_table()

    def get_component_contribution(self) -> pd.DataFrame:
        """Get component contribution ranking."""
        contributions = self.ablation.get_contribution_summary()
        
        data = {
            'Component': list(contributions.keys()),
            'Impact': [f"{v*100:+.1f}%" for v in contributions.values()],
        }
        
        return pd.DataFrame(data)

    def get_failure_summary(self) -> pd.DataFrame:
        """Get failure summary."""
        return self.failure_analyzer.get_failure_summary()

    def get_root_causes(self) -> pd.DataFrame:
        """Get root causes analysis."""
        return self.failure_analyzer.get_root_causes()

    def get_detailed_failures(self) -> pd.DataFrame:
        """Get detailed failure cases."""
        return self.failure_analyzer.get_detailed_failures()

    def get_lessons_learned(self) -> Dict[str, str]:
        """Get lessons learned."""
        lessons = self.failure_analyzer.get_lessons_learned()
        
        # Add comparison lessons
        if self.report:
            comparison_result = self.report.comparison_result
            
            if comparison_result.ml_to_rag_improvement:
                lessons['RAG Impact'] = \
                    f"RAG improves overall system by {comparison_result.ml_to_rag_improvement.get('overall', 0)*100:.1f}%"
            
            if comparison_result.rag_to_agents_improvement:
                lessons['Agent Impact'] = \
                    f"Agents improve system by {comparison_result.rag_to_agents_improvement.get('overall', 0)*100:.1f}%"

        return lessons

    def print_report(self):
        """Print evaluation report to console."""
        if not self.report:
            logger.warning("No report available. Run evaluate() first.")
            return

        print("\n" + "="*80)
        print("SYSTEM EVALUATION REPORT")
        print("="*80 + "\n")

        # Summary
        print("SUMMARY METRICS")
        print("-" * 80)
        for key, value in self.report.summary_metrics.items():
            print(f"{key:.<40} {value}")

        # Comparison
        print("\n\nSYSTEM COMPARISON (ML vs ML+RAG vs Full System)")
        print("-" * 80)
        print(self.get_comparison_table().to_string())

        # Ablation
        print("\n\nABLATION STUDY (Component Contribution)")
        print("-" * 80)
        print(self.get_component_contribution().to_string())

        # Failure Analysis
        print("\n\nFAILURE ANALYSIS")
        print("-" * 80)
        print(self.get_failure_summary().to_string())

        print("\n\nROOT CAUSE ANALYSIS")
        print("-" * 80)
        print(self.get_root_causes().to_string())

        # Lessons Learned
        print("\n\nLESSONS LEARNED")
        print("-" * 80)
        for lesson, detail in self.report.summary_metrics.items():
            print(f"{lesson}: {detail}")

        # Recommendations
        print("\n\nRECOMMENDATIONS")
        print("-" * 80)
        for i, rec in enumerate(self.report.recommendations, 1):
            print(f"{i}. {rec}")

        print("\n" + "="*80 + "\n")

    def reset(self):
        """Reset evaluator."""
        self.comparison.reset()
        self.ablation.reset()
        self.failure_analyzer.reset()
        self.report = None
        logger.info("SystemEvaluator reset")
