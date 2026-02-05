"""
Failure Case Analysis: Analyze when and why the system fails

Purpose:
- Identify failure patterns
- Root cause analysis
- Error categorization
- Lessons learned
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of system failures"""
    FALSE_NEGATIVE = "false_negative"  # Missed a failure
    FALSE_POSITIVE = "false_positive"  # Incorrectly warned
    HIGH_RUL_ERROR = "high_rul_error"  # RUL prediction far off
    LATE_WARNING = "late_warning"  # Warned too close to failure
    LOW_CONFIDENCE = "low_confidence"  # Low confidence decision
    SENSOR_FAILURE = "sensor_failure"  # Sensor anomaly unrelated to RUL
    WRONG_DIAGNOSIS = "wrong_diagnosis"  # Correct warning, wrong cause


@dataclass
class FailureCase:
    """Single failure case for analysis"""
    engine_id: int
    cycle: int
    failure_type: str
    predicted_rul: float
    actual_rul: float
    rul_error: float
    confidence: float
    warning_cycle: Optional[int]
    lead_time: Optional[float]
    diagnosis: str
    root_cause: str
    severity: str  # 'critical', 'high', 'medium', 'low'


@dataclass
class FailureAnalysis:
    """Analysis of failure cases"""
    total_cases: int
    failure_cases: List[FailureCase]
    failure_types: Dict[str, int]
    root_causes: Dict[str, int]
    severity_distribution: Dict[str, int]
    avg_rul_error: float
    avg_lead_time: float
    high_error_rate: float
    late_warning_rate: float
    false_positive_rate: float
    false_negative_rate: float


class FailureAnalyzer:
    """
    Analyzes failure cases to understand when and why the system fails.
    
    Features:
    - Categorize failures by type
    - Root cause analysis
    - Error distribution analysis
    - Identify patterns in failures
    - Generate lessons learned
    """

    def __init__(self):
        """Initialize failure analyzer."""
        self.cases = []
        logger.info("FailureAnalyzer initialized")

    def add_case(
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
        """Add a case for analysis."""
        rul_error = abs(predicted_rul - actual_rul)
        
        # Determine failure type
        failure_type = self._determine_failure_type(
            predicted_rul,
            actual_rul,
            warning_cycle,
            confidence,
            rul_error,
        )

        # Calculate lead time
        lead_time = None
        if warning_cycle is not None:
            lead_time = cycle - warning_cycle

        case = FailureCase(
            engine_id=engine_id,
            cycle=cycle,
            failure_type=failure_type,
            predicted_rul=predicted_rul,
            actual_rul=actual_rul,
            rul_error=rul_error,
            confidence=confidence,
            warning_cycle=warning_cycle,
            lead_time=lead_time,
            diagnosis=diagnosis,
            root_cause=root_cause,
            severity=severity,
        )

        self.cases.append(case)

    def _determine_failure_type(
        self,
        predicted_rul: float,
        actual_rul: float,
        warning_cycle: Optional[int],
        confidence: float,
        rul_error: float,
    ) -> str:
        """Determine type of failure."""
        # False negative: No warning but failure occurred
        if warning_cycle is None:
            return FailureType.FALSE_NEGATIVE.value

        # High RUL error
        if rul_error > 50:
            return FailureType.HIGH_RUL_ERROR.value

        # Late warning: Warned but too late (< 5 cycles notice)
        if warning_cycle is not None:
            lead_time = actual_rul - (actual_rul - warning_cycle)
            if lead_time < 5:
                return FailureType.LATE_WARNING.value

        # Low confidence: Correct warning but low confidence
        if warning_cycle is not None and confidence < 0.5:
            return FailureType.LOW_CONFIDENCE.value

        # If we got here, it was a successful early warning
        return "successful_warning"

    def analyze(self) -> FailureAnalysis:
        """Analyze all failure cases."""
        if not self.cases:
            logger.warning("No cases to analyze")
            return FailureAnalysis(
                total_cases=0,
                failure_cases=[],
                failure_types={},
                root_causes={},
                severity_distribution={},
                avg_rul_error=0,
                avg_lead_time=0,
                high_error_rate=0,
                late_warning_rate=0,
                false_positive_rate=0,
                false_negative_rate=0,
            )

        logger.info(f"Analyzing {len(self.cases)} failure cases...")

        # Count failure types
        failure_types = {}
        for case in self.cases:
            failure_types[case.failure_type] = failure_types.get(case.failure_type, 0) + 1

        # Count root causes
        root_causes = {}
        for case in self.cases:
            root_causes[case.root_cause] = root_causes.get(case.root_cause, 0) + 1

        # Count severity
        severity_dist = {}
        for case in self.cases:
            severity_dist[case.severity] = severity_dist.get(case.severity, 0) + 1

        # Calculate metrics
        rul_errors = [c.rul_error for c in self.cases]
        avg_rul_error = float(np.mean(rul_errors))

        lead_times = [c.lead_time for c in self.cases if c.lead_time is not None]
        avg_lead_time = float(np.mean(lead_times)) if lead_times else 0

        high_error_rate = len([c for c in self.cases if c.rul_error > 50]) / len(self.cases)
        late_warning_rate = len([c for c in self.cases if c.failure_type == FailureType.LATE_WARNING.value]) / len(self.cases)
        false_negative_rate = len([c for c in self.cases if c.failure_type == FailureType.FALSE_NEGATIVE.value]) / len(self.cases)

        # Count false positives (warnings that didn't lead to failure)
        # This requires tracking warnings vs actual failures
        # For now, approximate as low-severity cases with no failure
        false_positive_rate = 0

        analysis = FailureAnalysis(
            total_cases=len(self.cases),
            failure_cases=self.cases,
            failure_types=failure_types,
            root_causes=root_causes,
            severity_distribution=severity_dist,
            avg_rul_error=avg_rul_error,
            avg_lead_time=avg_lead_time,
            high_error_rate=high_error_rate,
            late_warning_rate=late_warning_rate,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
        )

        logger.info(f"Analysis complete. {len(self.cases)} cases analyzed.")
        return analysis

    def get_failure_summary(self) -> pd.DataFrame:
        """Get failure summary as table."""
        analysis = self.analyze()

        data = {
            'Metric': [
                'Total Cases',
                'False Negatives',
                'False Positives',
                'High RUL Error',
                'Late Warnings',
                'Low Confidence',
                'Avg RUL Error',
                'Avg Lead Time',
            ],
            'Count': [
                analysis.total_cases,
                analysis.failure_types.get(FailureType.FALSE_NEGATIVE.value, 0),
                analysis.failure_types.get(FailureType.FALSE_POSITIVE.value, 0),
                analysis.failure_types.get(FailureType.HIGH_RUL_ERROR.value, 0),
                analysis.failure_types.get(FailureType.LATE_WARNING.value, 0),
                analysis.failure_types.get(FailureType.LOW_CONFIDENCE.value, 0),
                f"{analysis.avg_rul_error:.1f}",
                f"{analysis.avg_lead_time:.0f}",
            ],
            'Percentage': [
                '100%',
                f"{analysis.false_negative_rate*100:.1f}%",
                f"{analysis.false_positive_rate*100:.1f}%",
                f"{analysis.high_error_rate*100:.1f}%",
                f"{analysis.late_warning_rate*100:.1f}%",
                '-',
                '-',
                '-',
            ],
        }

        return pd.DataFrame(data)

    def get_root_causes(self) -> pd.DataFrame:
        """Get root causes analysis."""
        analysis = self.analyze()

        causes = sorted(
            analysis.root_causes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        data = {
            'Root Cause': [c[0] for c in causes],
            'Count': [c[1] for c in causes],
            'Percentage': [f"{c[1]/analysis.total_cases*100:.1f}%" for c in causes],
        }

        return pd.DataFrame(data)

    def get_detailed_failures(self) -> pd.DataFrame:
        """Get detailed failure cases."""
        analysis = self.analyze()

        data = []
        for case in analysis.failure_cases:
            data.append({
                'Engine': case.engine_id,
                'Cycle': case.cycle,
                'Type': case.failure_type,
                'RUL Error': f"{case.rul_error:.1f}",
                'Lead Time': f"{case.lead_time:.0f}" if case.lead_time else 'N/A',
                'Confidence': f"{case.confidence:.2f}",
                'Root Cause': case.root_cause,
                'Severity': case.severity,
            })

        return pd.DataFrame(data)

    def get_lessons_learned(self) -> Dict[str, str]:
        """Generate lessons learned from failure analysis."""
        analysis = self.analyze()

        lessons = {}

        # Lesson 1: Most common failure type
        if analysis.failure_types:
            most_common = max(analysis.failure_types, key=analysis.failure_types.get)
            count = analysis.failure_types[most_common]
            lessons['Most Common Failure'] = \
                f"{most_common}: {count} cases ({count/analysis.total_cases*100:.1f}%)"

        # Lesson 2: Primary root cause
        if analysis.root_causes:
            primary_cause = max(analysis.root_causes, key=analysis.root_causes.get)
            count = analysis.root_causes[primary_cause]
            lessons['Primary Root Cause'] = \
                f"{primary_cause}: {count} cases ({count/analysis.total_cases*100:.1f}%)"

        # Lesson 3: RUL prediction accuracy
        lessons['RUL Accuracy'] = \
            f"Average error: {analysis.avg_rul_error:.1f} cycles"

        # Lesson 4: Early warning effectiveness
        lessons['Warning Lead Time'] = \
            f"Average advance notice: {analysis.avg_lead_time:.0f} cycles"

        # Lesson 5: High-error cases
        if analysis.high_error_rate > 0:
            lessons['High Error Cases'] = \
                f"{analysis.high_error_rate*100:.1f}% of cases had RUL error > 50 cycles"

        # Lesson 6: Late warnings
        if analysis.late_warning_rate > 0:
            lessons['Late Warnings'] = \
                f"{analysis.late_warning_rate*100:.1f}% of warnings came too late"

        # Lesson 7: Missed failures
        if analysis.false_negative_rate > 0:
            lessons['Missed Failures'] = \
                f"{analysis.false_negative_rate*100:.1f}% of failures had no warning"

        return lessons

    def reset(self):
        """Reset analyzer."""
        self.cases = []
        logger.info("FailureAnalyzer reset")
