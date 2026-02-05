"""
Action Agent: Suggests interventions and manages escalations

Responsibilities:
- Generate maintenance recommendations
- Suggest predictive maintenance actions
- Escalate high-risk situations
- Track intervention outcomes
- Manage action confidence thresholding
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ActionPriority(Enum):
    """Action priority levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ActionType(Enum):
    """Types of recommended actions"""
    CONTINUE_MONITORING = "continue_monitoring"
    SCHEDULE_INSPECTION = "schedule_inspection"
    PERFORM_MAINTENANCE = "perform_maintenance"
    REPLACE_COMPONENT = "replace_component"
    ESCALATE_HUMAN = "escalate_human"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class Recommendation:
    """Single maintenance recommendation"""
    action_type: str  # ActionType value
    description: str
    priority: str  # ActionPriority name
    target_timeframe: str  # e.g., "within 10 cycles"
    components: List[str]  # Affected components
    estimated_downtime_hours: float
    cost_estimate: str
    justification: str
    confidence: float


@dataclass
class ActionPlan:
    """Complete action plan from agent"""
    engine_id: int
    cycle: int
    recommendations: List[Recommendation]
    primary_action: str
    should_escalate: bool
    escalation_reason: Optional[str]
    overall_confidence: float
    risk_mitigation_score: float
    timestamp: str


class ActionAgent:
    """
    Agent responsible for generating maintenance recommendations
    and managing action escalations.
    
    Features:
    - Priority-based action generation
    - Confidence thresholding with abstention
    - Escalation logic
    - Component-specific recommendations
    - Timeframe estimation
    - Cost analysis
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        escalation_threshold: float = 0.8,
        action_mappings: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize Action Agent.
        
        Args:
            confidence_threshold: Minimum confidence for actions
            escalation_threshold: Threshold for escalation
            action_mappings: Risk level to action type mappings
        """
        self.confidence_threshold = confidence_threshold
        self.escalation_threshold = escalation_threshold
        
        self.action_mappings = action_mappings or {
            'critical': [ActionType.EMERGENCY_SHUTDOWN.value, ActionType.ESCALATE_HUMAN.value],
            'high': [ActionType.ESCALATE_HUMAN.value, ActionType.REPLACE_COMPONENT.value],
            'medium': [ActionType.PERFORM_MAINTENANCE.value, ActionType.SCHEDULE_INSPECTION.value],
            'low': [ActionType.SCHEDULE_INSPECTION.value, ActionType.CONTINUE_MONITORING.value],
        }
        
        # Action history
        self.action_history = []
        
        logger.info(
            f"ActionAgent initialized. "
            f"confidence_threshold={confidence_threshold}, "
            f"escalation_threshold={escalation_threshold}"
        )

    def recommend_actions(
        self,
        reasoning_result: Any,
        monitoring_report: Any,
    ) -> ActionPlan:
        """
        Generate recommended actions based on reasoning result.
        
        Args:
            reasoning_result: ReasoningResult from ReasoningAgent
            monitoring_report: MonitoringReport from MonitoringAgent
        
        Returns:
            ActionPlan with recommendations
        """
        try:
            engine_id = reasoning_result.engine_id
            cycle = reasoning_result.cycle
            risk_explanation = reasoning_result.risk_explanation

            # Check for abstention
            if risk_explanation.abstention or not reasoning_result.is_confident:
                return self._generate_abstention_plan(engine_id, cycle)

            # Determine risk level
            risk_score = risk_explanation.risk_score
            risk_level = self._categorize_risk(risk_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                risk_level,
                risk_explanation,
                monitoring_report,
            )

            # Select primary action
            primary_action = self._select_primary_action(recommendations)

            # Determine escalation
            should_escalate = (
                risk_score >= self.escalation_threshold
                or any(r.priority == 'CRITICAL' for r in recommendations)
            )

            escalation_reason = None
            if should_escalate:
                escalation_reason = self._generate_escalation_reason(
                    risk_explanation,
                    recommendations,
                )

            # Calculate overall confidence
            overall_confidence = min(
                reasoning_result.reasoning_confidence,
                risk_explanation.confidence,
            )

            # Risk mitigation score
            risk_mitigation_score = self._calculate_mitigation_score(
                recommendations,
                risk_score,
            )

            plan = ActionPlan(
                engine_id=engine_id,
                cycle=cycle,
                recommendations=recommendations,
                primary_action=primary_action,
                should_escalate=should_escalate,
                escalation_reason=escalation_reason,
                overall_confidence=overall_confidence,
                risk_mitigation_score=risk_mitigation_score,
                timestamp=datetime.now().isoformat(),
            )

            self.action_history.append(plan)
            return plan

        except Exception as e:
            logger.error(f"Error in action recommendation: {e}")
            return self._generate_error_plan(
                reasoning_result.engine_id,
                reasoning_result.cycle,
            )

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level from score."""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def _generate_recommendations(
        self,
        risk_level: str,
        risk_explanation: Any,
        monitoring_report: Any,
    ) -> List[Recommendation]:
        """Generate maintenance recommendations based on risk level."""
        recommendations = []

        # Get action types for this risk level
        action_types = self.action_mappings.get(risk_level, [])

        # Generate recommendations for each action type
        for action_type in action_types:
            if action_type == ActionType.CONTINUE_MONITORING.value:
                rec = Recommendation(
                    action_type=action_type,
                    description="Continue normal monitoring and data collection",
                    priority=ActionPriority.NONE.name,
                    target_timeframe="next 10 cycles",
                    components=[],
                    estimated_downtime_hours=0.0,
                    cost_estimate="$0",
                    justification="System operating normally with no critical concerns",
                    confidence=0.9,
                )

            elif action_type == ActionType.SCHEDULE_INSPECTION.value:
                affected = monitoring_report.anomaly.affected_sensors or ['multiple sensors']
                rec = Recommendation(
                    action_type=action_type,
                    description=f"Schedule inspection of {', '.join(affected[:2])}",
                    priority=ActionPriority.MEDIUM.name,
                    target_timeframe="within next 20 cycles",
                    components=affected,
                    estimated_downtime_hours=2.0,
                    cost_estimate="$500-1000",
                    justification=f"Anomalies detected in {', '.join(affected)}, investigation needed",
                    confidence=0.75,
                )

            elif action_type == ActionType.PERFORM_MAINTENANCE.value:
                rec = Recommendation(
                    action_type=action_type,
                    description="Perform preventive maintenance per schedule",
                    priority=ActionPriority.HIGH.name,
                    target_timeframe=f"within {max(10, int(monitoring_report.prediction.predicted_rul / 2))} cycles",
                    components=monitoring_report.anomaly.affected_sensors or ['engine'],
                    estimated_downtime_hours=8.0,
                    cost_estimate="$2000-5000",
                    justification="Degradation patterns detected, maintenance recommended",
                    confidence=0.7,
                )

            elif action_type == ActionType.REPLACE_COMPONENT.value:
                failing_parts = self._identify_failing_components(risk_explanation)
                rec = Recommendation(
                    action_type=action_type,
                    description=f"Replace {', '.join(failing_parts) if failing_parts else 'failing components'}",
                    priority=ActionPriority.HIGH.name,
                    target_timeframe=f"within {max(5, int(monitoring_report.prediction.predicted_rul / 3))} cycles",
                    components=failing_parts or ['turbofan assembly'],
                    estimated_downtime_hours=24.0,
                    cost_estimate="$10000-20000",
                    justification=f"Critical degradation in {', '.join(failing_parts) if failing_parts else 'key components'}",
                    confidence=0.65,
                )

            elif action_type == ActionType.ESCALATE_HUMAN.value:
                rec = Recommendation(
                    action_type=action_type,
                    description="Escalate to human expert for decision",
                    priority=ActionPriority.CRITICAL.name,
                    target_timeframe="immediate",
                    components=risk_explanation.failure_patterns,
                    estimated_downtime_hours=0.0,
                    cost_estimate="Expert consultation",
                    justification="Risk level requires expert human judgment",
                    confidence=0.85,
                )

            elif action_type == ActionType.EMERGENCY_SHUTDOWN.value:
                rec = Recommendation(
                    action_type=action_type,
                    description="EMERGENCY: Immediate shutdown required",
                    priority=ActionPriority.CRITICAL.name,
                    target_timeframe="IMMEDIATE",
                    components=['entire engine'],
                    estimated_downtime_hours=float('inf'),
                    cost_estimate="Emergency repair/replacement",
                    justification="Imminent catastrophic failure detected",
                    confidence=0.9,
                )

            else:
                continue

            recommendations.append(rec)

        return recommendations

    def _select_primary_action(self, recommendations: List[Recommendation]) -> str:
        """Select primary action from list."""
        if not recommendations:
            return ActionType.CONTINUE_MONITORING.value

        # Prioritize by priority enum
        priority_order = {
            ActionPriority.CRITICAL.name: 0,
            ActionPriority.HIGH.name: 1,
            ActionPriority.MEDIUM.name: 2,
            ActionPriority.LOW.name: 3,
            ActionPriority.NONE.name: 4,
        }

        sorted_recs = sorted(
            recommendations,
            key=lambda r: priority_order.get(r.priority, 999)
        )

        return sorted_recs[0].action_type if sorted_recs else ActionType.CONTINUE_MONITORING.value

    def _identify_failing_components(self, risk_explanation: Any) -> List[str]:
        """Identify which components are likely failing."""
        components = []

        # Use failure patterns from reasoning
        if risk_explanation.failure_patterns:
            pattern = risk_explanation.failure_patterns[0]
            if 'bearing' in pattern.lower():
                components.append('bearing assembly')
            if 'blade' in pattern.lower():
                components.append('compressor blades')
            if 'seal' in pattern.lower():
                components.append('seal system')

        # Use affected sensors as clue to components
        for factor, importance in risk_explanation.key_factors:
            if 'temperature' in factor.lower():
                components.append('combustor')
            if 'vibration' in factor.lower():
                components.append('bearing assembly')
            if 'pressure' in factor.lower():
                components.append('compressor')

        return list(set(components))[:3]  # Return unique, top 3

    def _generate_escalation_reason(
        self,
        risk_explanation: Any,
        recommendations: List[Recommendation],
    ) -> str:
        """Generate reason for escalation."""
        reasons = []

        if risk_explanation.risk_score >= 0.9:
            reasons.append("Critical risk score detected")

        critical_recs = [r for r in recommendations if r.priority == 'CRITICAL']
        if critical_recs:
            reasons.append(f"Critical actions required: {', '.join([r.description for r in critical_recs])}")

        if risk_explanation.abstention:
            reasons.append("Insufficient confidence for autonomous action")

        if not reasons:
            reasons.append("Risk exceeds escalation threshold")

        return "; ".join(reasons)

    def _calculate_mitigation_score(
        self,
        recommendations: List[Recommendation],
        current_risk: float,
    ) -> float:
        """Calculate how much risk is mitigated by recommendations."""
        if not recommendations:
            return 0.0

        # Estimate risk reduction from primary action
        primary = recommendations[0]

        mitigation_by_action = {
            ActionType.CONTINUE_MONITORING.value: 0.1,
            ActionType.SCHEDULE_INSPECTION.value: 0.3,
            ActionType.PERFORM_MAINTENANCE.value: 0.6,
            ActionType.REPLACE_COMPONENT.value: 0.85,
            ActionType.ESCALATE_HUMAN.value: 0.5,  # Expert will decide
            ActionType.EMERGENCY_SHUTDOWN.value: 1.0,
        }

        base_mitigation = mitigation_by_action.get(primary.action_type, 0.2)

        # Scale by confidence
        mitigation = base_mitigation * primary.confidence

        # Maximum risk that remains
        residual_risk = current_risk * (1.0 - mitigation)

        # Score is inverse: higher is better (more risk mitigated)
        mitigation_score = 1.0 - np.clip(residual_risk, 0, 1)

        return float(mitigation_score)

    def _generate_abstention_plan(self, engine_id: int, cycle: int) -> ActionPlan:
        """Generate plan when unable to make confident decision."""
        rec = Recommendation(
            action_type=ActionType.ESCALATE_HUMAN.value,
            description="Unable to make confident decision, escalating to human expert",
            priority=ActionPriority.HIGH.name,
            target_timeframe="within next cycle",
            components=[],
            estimated_downtime_hours=0.0,
            cost_estimate="Expert consultation",
            justification="Insufficient data confidence for autonomous action",
            confidence=0.5,
        )

        return ActionPlan(
            engine_id=engine_id,
            cycle=cycle,
            recommendations=[rec],
            primary_action=rec.action_type,
            should_escalate=True,
            escalation_reason="Autonomous decision confidence below threshold",
            overall_confidence=0.0,
            risk_mitigation_score=0.0,
            timestamp=datetime.now().isoformat(),
        )

    def _generate_error_plan(self, engine_id: int, cycle: int) -> ActionPlan:
        """Generate plan when error occurs."""
        rec = Recommendation(
            action_type=ActionType.ESCALATE_HUMAN.value,
            description="Error occurred in action recommendation system, escalating",
            priority=ActionPriority.HIGH.name,
            target_timeframe="immediate",
            components=[],
            estimated_downtime_hours=0.0,
            cost_estimate="Expert consultation",
            justification="System error occurred during recommendation generation",
            confidence=0.0,
        )

        return ActionPlan(
            engine_id=engine_id,
            cycle=cycle,
            recommendations=[rec],
            primary_action=rec.action_type,
            should_escalate=True,
            escalation_reason="System error in action recommendation",
            overall_confidence=0.0,
            risk_mitigation_score=0.0,
            timestamp=datetime.now().isoformat(),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get action statistics."""
        if not self.action_history:
            return {
                'n_actions': 0,
                'escalation_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_mitigation': 0.0,
                'action_distribution': {},
            }

        escalated = len([a for a in self.action_history if a.should_escalate])
        
        # Count action types
        action_counts = {}
        for plan in self.action_history:
            for rec in plan.recommendations:
                action_counts[rec.action_type] = action_counts.get(rec.action_type, 0) + 1

        return {
            'n_actions': len(self.action_history),
            'escalation_rate': escalated / len(self.action_history) if self.action_history else 0.0,
            'avg_confidence': np.mean([a.overall_confidence for a in self.action_history]),
            'avg_mitigation': np.mean([a.risk_mitigation_score for a in self.action_history]),
            'action_distribution': action_counts,
        }

    def clear_history(self):
        """Clear action history."""
        self.action_history = []
        logger.info("Action history cleared")
