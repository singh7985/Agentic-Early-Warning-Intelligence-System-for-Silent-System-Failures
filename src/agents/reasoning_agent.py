"""
Reasoning Agent: Explains risk using retrieved evidence

Responsibilities:
- Synthesize monitoring signals and historical context
- Generate risk explanations with citations
- Score confidence levels
- Determine failure patterns and root causes
- Identify related risk factors
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskExplanation:
    """Risk explanation with evidence"""
    primary_risk: str
    risk_score: float  # 0-1
    confidence: float  # 0-1
    evidence: List[Dict[str, Any]]
    failure_patterns: List[str]
    related_risks: List[str]
    key_factors: List[Tuple[str, float]]  # (factor, importance)
    citations: List[str]
    reasoning: str
    timestamp: str
    abstention: bool  # True if confidence too low


@dataclass
class ReasoningResult:
    """Complete reasoning result from agent"""
    engine_id: int
    cycle: int
    risk_explanation: RiskExplanation
    is_confident: bool
    reasoning_confidence: float
    should_escalate: bool
    timestamp: str


class ReasoningAgent:
    """
    Agent responsible for synthesizing monitoring signals and
    generating risk explanations with historical context.
    
    Features:
    - Multi-signal evidence synthesis
    - Pattern matching with historical data
    - Confidence-based reasoning
    - Citation tracking
    - Risk scoring
    - Abstention when uncertain
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        evidence_weight: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Reasoning Agent.
        
        Args:
            confidence_threshold: Minimum confidence for non-abstention
            evidence_weight: Weights for different evidence types
                {'prediction': 0.3, 'anomaly': 0.3, 'pattern': 0.4}
        """
        self.confidence_threshold = confidence_threshold
        
        self.evidence_weight = evidence_weight or {
            'prediction': 0.3,
            'anomaly': 0.3,
            'retrieval': 0.4,
        }
        
        # Reasoning history
        self.reasoning_history = []
        
        logger.info(
            f"ReasoningAgent initialized. "
            f"confidence_threshold={confidence_threshold}"
        )

    def reason(
        self,
        monitoring_report: Any,
        retrieval_result: Any,
        sensor_deviations: Optional[Dict[str, float]] = None,
    ) -> ReasoningResult:
        """
        Generate risk explanation from monitoring and retrieval signals.
        
        Args:
            monitoring_report: MonitoringReport from MonitoringAgent
            retrieval_result: RetrievalResult from RetrievalAgent
            sensor_deviations: Dict of sensor deviations
        
        Returns:
            ReasoningResult with risk explanation
        """
        try:
            engine_id = monitoring_report.engine_id
            cycle = monitoring_report.cycle

            # Extract signals
            prediction_rul = monitoring_report.prediction.predicted_rul
            prediction_confidence = monitoring_report.prediction.confidence
            anomaly_score = monitoring_report.anomaly.anomaly_score
            anomaly_detected = monitoring_report.anomaly.is_anomaly
            drift_detected = monitoring_report.drift.is_drift

            # Build evidence list
            evidence = []
            
            # Add prediction evidence
            if prediction_confidence > 0.3:
                evidence.append({
                    'type': 'prediction',
                    'value': prediction_rul,
                    'confidence': prediction_confidence,
                    'description': f'ML prediction suggests {prediction_rul:.0f} cycles RUL',
                })

            # Add anomaly evidence
            if anomaly_detected:
                evidence.append({
                    'type': 'anomaly',
                    'value': anomaly_score,
                    'confidence': 1.0 - anomaly_score,
                    'description': f'Anomaly detected with score {anomaly_score:.3f}',
                    'sensors': monitoring_report.anomaly.affected_sensors,
                })

            # Add drift evidence
            if drift_detected:
                evidence.append({
                    'type': 'drift',
                    'value': monitoring_report.drift.drift_score,
                    'confidence': 1.0 - monitoring_report.drift.drift_score,
                    'description': f'Data drift detected',
                    'drifted_features': monitoring_report.drift.drifted_features,
                })

            # Add retrieval evidence (historical patterns)
            retrieved_incidents = retrieval_result.results
            if retrieved_incidents:
                top_incident = retrieved_incidents[0]
                evidence.append({
                    'type': 'retrieval',
                    'value': top_incident.similarity_score,
                    'confidence': retrieval_result.retrieval_confidence,
                    'description': f'Similar failure pattern found: {top_incident.failure_type}',
                    'failure_type': top_incident.failure_type,
                    'source_engine': top_incident.engine_id,
                    'citation': top_incident.citation.get('citation_string', ''),
                })

            # Synthesize evidence into risk explanation
            risk_explanation = self._synthesize_evidence(
                evidence,
                retrieved_incidents,
                prediction_rul,
                sensor_deviations,
            )

            # Calculate overall confidence
            reasoning_confidence = self._calculate_confidence(evidence)

            # Determine if should escalate
            should_escalate = (
                reasoning_confidence >= self.confidence_threshold
                and risk_explanation.risk_score >= 0.7
            )

            # Set abstention if confidence too low
            if reasoning_confidence < self.confidence_threshold:
                risk_explanation.abstention = True

            result = ReasoningResult(
                engine_id=engine_id,
                cycle=cycle,
                risk_explanation=risk_explanation,
                is_confident=reasoning_confidence >= self.confidence_threshold,
                reasoning_confidence=reasoning_confidence,
                should_escalate=should_escalate,
                timestamp=datetime.now().isoformat(),
            )

            self.reasoning_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            # Return abstention
            return ReasoningResult(
                engine_id=monitoring_report.engine_id if monitoring_report else 0,
                cycle=monitoring_report.cycle if monitoring_report else 0,
                risk_explanation=RiskExplanation(
                    primary_risk='Unable to assess',
                    risk_score=0.5,
                    confidence=0.0,
                    evidence=[],
                    failure_patterns=[],
                    related_risks=[],
                    key_factors=[],
                    citations=[],
                    reasoning=f'Error: {str(e)}',
                    timestamp=datetime.now().isoformat(),
                    abstention=True,
                ),
                is_confident=False,
                reasoning_confidence=0.0,
                should_escalate=False,
                timestamp=datetime.now().isoformat(),
            )

    def _synthesize_evidence(
        self,
        evidence: List[Dict[str, Any]],
        retrieved_incidents: List[Any],
        predicted_rul: float,
        sensor_deviations: Optional[Dict[str, float]],
    ) -> RiskExplanation:
        """Synthesize evidence into risk explanation."""
        
        # Calculate weighted risk score
        risk_scores = []
        weights = []
        
        for ev in evidence:
            ev_type = ev['type']
            weight = self.evidence_weight.get(ev_type, 0.2)
            
            if ev_type == 'prediction':
                # RUL-based risk: higher risk if RUL is low
                rul_risk = max(0, 1.0 - (ev['value'] / 150.0))
                risk_scores.append(rul_risk)
                weights.append(weight)
                
            elif ev_type == 'anomaly':
                # Anomaly score is already 0-1 risk
                risk_scores.append(ev['value'])
                weights.append(weight)
                
            elif ev_type == 'drift':
                # Drift is concerning
                risk_scores.append(ev['value'] * 0.8)  # Slightly less critical
                weights.append(weight)
                
            elif ev_type == 'retrieval':
                # Similar failures increase risk
                if 'degradation' in ev['description'].lower():
                    risk_scores.append(ev['value'] * 0.9)
                else:
                    risk_scores.append(ev['value'] * 0.6)
                weights.append(weight)
        
        if risk_scores and weights:
            total_weight = sum(weights)
            weighted_risk = sum(s * w for s, w in zip(risk_scores, weights)) / total_weight
            risk_score = float(np.clip(weighted_risk, 0, 1))
        else:
            risk_score = 0.3

        # Determine primary risk
        if predicted_rul < 30 and len(evidence) > 1:
            primary_risk = 'CRITICAL: Imminent failure expected'
        elif predicted_rul < 50:
            primary_risk = 'HIGH: Failure within normal maintenance window'
        elif any(e['type'] == 'anomaly' for e in evidence):
            primary_risk = 'MEDIUM: Abnormal sensor patterns detected'
        elif any(e['type'] == 'drift' for e in evidence):
            primary_risk = 'MEDIUM: Distribution shift detected'
        elif retrieved_incidents:
            primary_risk = f'MEDIUM: Similar to {retrieved_incidents[0].failure_type}'
        else:
            primary_risk = 'LOW: Normal operation'

        # Identify failure patterns from historical context
        failure_patterns = self._identify_patterns(retrieved_incidents)

        # Related risks
        related_risks = self._identify_related_risks(evidence, sensor_deviations)

        # Key factors by importance
        key_factors = self._rank_factors(evidence, sensor_deviations)

        # Build citations
        citations = [e.get('citation', '') for e in evidence if e.get('citation')]

        # Build reasoning narrative
        reasoning = self._build_narrative(evidence, retrieved_incidents, predicted_rul)

        return RiskExplanation(
            primary_risk=primary_risk,
            risk_score=risk_score,
            confidence=self._calculate_confidence(evidence),
            evidence=evidence,
            failure_patterns=failure_patterns,
            related_risks=related_risks,
            key_factors=key_factors,
            citations=citations,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            abstention=False,
        )

    def _identify_patterns(self, retrieved_incidents: List[Any]) -> List[str]:
        """Identify failure patterns from retrieved incidents."""
        patterns = []
        
        if not retrieved_incidents:
            return patterns

        # Count failure types
        failure_types = {}
        for incident in retrieved_incidents:
            ftype = incident.failure_type
            failure_types[ftype] = failure_types.get(ftype, 0) + 1

        # Return top patterns
        for ftype, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
            if count >= 1:
                patterns.append(f'{ftype} ({count} similar incidents)')

        return patterns[:3]  # Top 3 patterns

    def _identify_related_risks(
        self,
        evidence: List[Dict[str, Any]],
        sensor_deviations: Optional[Dict[str, float]],
    ) -> List[str]:
        """Identify related risk factors."""
        risks = []

        # Check for cascading failures
        anomaly_events = [e for e in evidence if e['type'] == 'anomaly']
        if len(anomaly_events) > 1:
            risks.append('Multiple sensor anomalies (potential cascading failure)')

        # Check for sensor-specific patterns
        if sensor_deviations:
            high_dev = {k: v for k, v in sensor_deviations.items() if abs(v) > 0.4}
            if len(high_dev) > 2:
                risks.append(f'Multiple sensors out of spec ({len(high_dev)} affected)')

        # Check for drift + anomaly combination
        drift_events = [e for e in evidence if e['type'] == 'drift']
        if drift_events and anomaly_events:
            risks.append('Distribution shift coinciding with anomalies')

        return risks

    def _rank_factors(
        self,
        evidence: List[Dict[str, Any]],
        sensor_deviations: Optional[Dict[str, float]],
    ) -> List[Tuple[str, float]]:
        """Rank contributing factors by importance."""
        factors = []

        for ev in evidence:
            ev_type = ev['type']
            confidence = ev.get('confidence', 0.5)
            
            if ev_type == 'prediction':
                factors.append(('RUL Prediction', confidence))
            elif ev_type == 'anomaly':
                sensors = ', '.join(ev.get('sensors', ['unknown']))
                factors.append((f'Anomalies in {sensors}', confidence))
            elif ev_type == 'drift':
                factors.append(('Feature Drift', confidence))
            elif ev_type == 'retrieval':
                factors.append((f"Historical: {ev.get('description', 'Similar pattern')}", confidence))

        # Add sensor deviations
        if sensor_deviations:
            sorted_sensors = sorted(
                sensor_deviations.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for sensor, deviation in sorted_sensors[:3]:
                factors.append((f'{sensor} deviation: {deviation:.2f}', abs(deviation) / 1.0))

        # Sort by importance
        factors.sort(key=lambda x: x[1], reverse=True)
        return factors[:5]  # Top 5 factors

    def _calculate_confidence(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate overall reasoning confidence."""
        if not evidence:
            return 0.0

        # Average confidence of all evidence
        confidences = [e.get('confidence', 0.5) for e in evidence]
        base_confidence = np.mean(confidences)

        # Bonus for multiple evidence types
        evidence_types = len(set(e['type'] for e in evidence))
        type_bonus = min(0.2, (evidence_types - 1) * 0.1)

        overall_confidence = min(1.0, base_confidence + type_bonus)
        return float(overall_confidence)

    def _build_narrative(
        self,
        evidence: List[Dict[str, Any]],
        retrieved_incidents: List[Any],
        predicted_rul: float,
    ) -> str:
        """Build narrative explanation of risk."""
        narrative_parts = []

        # Start with prediction
        pred_events = [e for e in evidence if e['type'] == 'prediction']
        if pred_events:
            pred = pred_events[0]
            narrative_parts.append(
                f"ML models predict {pred['value']:.0f} cycles of remaining useful life "
                f"(confidence: {pred['confidence']:.1%})."
            )

        # Add anomalies
        anom_events = [e for e in evidence if e['type'] == 'anomaly']
        if anom_events:
            sensors = anom_events[0].get('sensors', [])
            narrative_parts.append(
                f"Abnormal sensor readings detected in {', '.join(sensors or ['multiple sensors'])}."
            )

        # Add drift
        drift_events = [e for e in evidence if e['type'] == 'drift']
        if drift_events:
            narrative_parts.append(
                "Feature distributions have shifted from baseline, indicating changed operating conditions."
            )

        # Add historical context
        if retrieved_incidents:
            top = retrieved_incidents[0]
            narrative_parts.append(
                f"Historical data shows {len(retrieved_incidents)} similar failure patterns. "
                f"Most similar was {top.failure_type} "
                f"(similarity: {top.similarity_score:.1%}). {top.citation.get('citation_string', '')}"
            )

        # Add summary
        if predicted_rul < 50:
            narrative_parts.append(
                "RECOMMENDATION: Prioritize inspection and maintenance within next "
                f"{min(predicted_rul, 30):.0f} cycles."
            )

        return " ".join(narrative_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning statistics."""
        if not self.reasoning_history:
            return {
                'n_reasonings': 0,
                'avg_confidence': 0.0,
                'escalation_rate': 0.0,
                'abstention_rate': 0.0,
            }

        confident = len([r for r in self.reasoning_history if r.is_confident])
        escalated = len([r for r in self.reasoning_history if r.should_escalate])
        abstained = len([
            r for r in self.reasoning_history
            if r.risk_explanation.abstention
        ])

        return {
            'n_reasonings': len(self.reasoning_history),
            'avg_confidence': np.mean([
                r.reasoning_confidence for r in self.reasoning_history
            ]),
            'avg_risk_score': np.mean([
                r.risk_explanation.risk_score for r in self.reasoning_history
            ]),
            'escalation_rate': escalated / len(self.reasoning_history) if self.reasoning_history else 0.0,
            'abstention_rate': abstained / len(self.reasoning_history) if self.reasoning_history else 0.0,
            'confident_decisions': confident,
        }

    def clear_history(self):
        """Clear reasoning history."""
        self.reasoning_history = []
        logger.info("Reasoning history cleared")
