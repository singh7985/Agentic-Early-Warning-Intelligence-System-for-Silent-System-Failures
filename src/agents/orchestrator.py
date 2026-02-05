"""
Agent Orchestrator: Coordinates multi-agent workflow using LangGraph

Implements collaborative agent orchestration:
- Monitoring Agent: Detects anomalies and drifts
- Retrieval Agent: Retrieves historical context
- Reasoning Agent: Synthesizes evidence and explains risk
- Action Agent: Recommends interventions

Flow:
1. Monitoring Agent analyzes sensor data
2. Retrieval Agent queries historical patterns
3. Reasoning Agent synthesizes evidence
4. Action Agent generates recommendations
5. Confidence thresholding and escalation logic
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status for agent workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    ESCALATED = "escalated"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str  # 'monitoring', 'retrieval', 'reasoning', 'action'
    receiver: str  # recipient or 'orchestrator'
    timestamp: str
    data: Dict[str, Any]
    confidence: float
    status: str  # ExecutionStatus value


@dataclass
class AgentResult:
    """Complete result from agent orchestration"""
    engine_id: int
    cycle: int
    workflow_status: str  # ExecutionStatus value
    
    # Agent results
    monitoring_report: Optional[Any] = None
    retrieval_result: Optional[Any] = None
    reasoning_result: Optional[Any] = None
    action_plan: Optional[Any] = None
    
    # Overall metrics
    overall_confidence: float = 0.0
    overall_risk_score: float = 0.0
    should_escalate: bool = False
    escalation_reason: Optional[str] = None
    
    # Execution
    workflow_id: Optional[str] = None
    execution_time_ms: float = 0.0
    messages: List[AgentMessage] = None
    
    # Timestamp
    timestamp: str = ""
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class AgentOrchestrator:
    """
    Orchestrates multi-agent workflow for collaborative decision-making.
    
    Features:
    - Sequential agent pipeline
    - Confidence thresholding
    - Escalation logic
    - Message passing
    - Error handling
    - Workflow history
    """

    def __init__(
        self,
        monitoring_agent: Any,
        retrieval_agent: Any,
        reasoning_agent: Any,
        action_agent: Any,
        confidence_threshold: float = 0.6,
        escalation_threshold: float = 0.8,
    ):
        """
        Initialize Agent Orchestrator.
        
        Args:
            monitoring_agent: MonitoringAgent instance
            retrieval_agent: RetrievalAgent instance
            reasoning_agent: ReasoningAgent instance
            action_agent: ActionAgent instance
            confidence_threshold: Minimum confidence for decisions
            escalation_threshold: Threshold for escalation
        """
        self.monitoring_agent = monitoring_agent
        self.retrieval_agent = retrieval_agent
        self.reasoning_agent = reasoning_agent
        self.action_agent = action_agent
        
        self.confidence_threshold = confidence_threshold
        self.escalation_threshold = escalation_threshold
        
        # Workflow tracking
        self.workflow_history = []
        self.message_log = []
        
        logger.info(
            "AgentOrchestrator initialized with 4 agents. "
            f"Confidence threshold: {confidence_threshold}, "
            f"Escalation threshold: {escalation_threshold}"
        )

    def execute(
        self,
        sensor_data: np.ndarray,
        engine_id: int,
        cycle: int,
        sensor_names: Optional[List[str]] = None,
        reference_data: Optional[np.ndarray] = None,
    ) -> AgentResult:
        """
        Execute complete agent workflow.
        
        Args:
            sensor_data: Current sensor readings
            engine_id: Engine identifier
            cycle: Current cycle number
            sensor_names: Names of sensors
            reference_data: Reference data for drift detection
        
        Returns:
            AgentResult with complete workflow execution
        """
        import time
        start_time = time.time()
        workflow_id = f"{engine_id}_{cycle}_{int(time.time() * 1000)}"
        
        try:
            messages = []
            
            # Step 1: Monitoring Agent
            logger.info(f"[{workflow_id}] Starting Monitoring Agent")
            monitoring_report = self.monitoring_agent.generate_report(
                sensor_data=sensor_data,
                engine_id=engine_id,
                cycle=cycle,
                sensor_names=sensor_names,
                reference_data=reference_data,
            )
            
            self._add_message(
                messages,
                sender='monitoring',
                receiver='orchestrator',
                data={'report': asdict(monitoring_report) if hasattr(monitoring_report, '__dataclass_fields__') else monitoring_report},
                confidence=monitoring_report.overall_confidence,
            )
            
            logger.info(
                f"[{workflow_id}] Monitoring complete. "
                f"Alert: {monitoring_report.alert_flag}, "
                f"Confidence: {monitoring_report.overall_confidence:.2%}"
            )
            
            # Step 2: Retrieval Agent
            logger.info(f"[{workflow_id}] Starting Retrieval Agent")
            
            # Extract sensor deviations from monitoring report
            sensor_deviations = self._extract_sensor_deviations(
                sensor_data,
                sensor_names,
            )
            
            retrieval_result = self.retrieval_agent.search_by_sensor_pattern(
                sensor_deviations=sensor_deviations,
                rul=monitoring_report.prediction.predicted_rul,
                anomaly_score=monitoring_report.anomaly.anomaly_score,
            )
            
            self._add_message(
                messages,
                sender='retrieval',
                receiver='orchestrator',
                data={
                    'query': retrieval_result.query,
                    'n_results': retrieval_result.total_results,
                    'mean_score': retrieval_result.mean_score,
                },
                confidence=retrieval_result.retrieval_confidence,
            )
            
            logger.info(
                f"[{workflow_id}] Retrieval complete. "
                f"Found {retrieval_result.total_results} similar incidents. "
                f"Confidence: {retrieval_result.retrieval_confidence:.2%}"
            )
            
            # Step 3: Reasoning Agent
            logger.info(f"[{workflow_id}] Starting Reasoning Agent")
            reasoning_result = self.reasoning_agent.reason(
                monitoring_report=monitoring_report,
                retrieval_result=retrieval_result,
                sensor_deviations=sensor_deviations,
            )
            
            self._add_message(
                messages,
                sender='reasoning',
                receiver='orchestrator',
                data={
                    'primary_risk': reasoning_result.risk_explanation.primary_risk,
                    'risk_score': reasoning_result.risk_explanation.risk_score,
                    'abstention': reasoning_result.risk_explanation.abstention,
                },
                confidence=reasoning_result.reasoning_confidence,
            )
            
            logger.info(
                f"[{workflow_id}] Reasoning complete. "
                f"Risk: {reasoning_result.risk_explanation.primary_risk}, "
                f"Score: {reasoning_result.risk_explanation.risk_score:.2%}, "
                f"Confidence: {reasoning_result.reasoning_confidence:.2%}"
            )
            
            # Check if should continue to action agent
            if reasoning_result.risk_explanation.abstention:
                logger.warning(
                    f"[{workflow_id}] Reasoning abstained due to low confidence. "
                    f"Escalating..."
                )
                
                status = ExecutionStatus.ABSTAINED
                action_plan = None
                should_escalate = True
                escalation_reason = "Reasoning agent abstained: insufficient confidence"
                
            else:
                # Step 4: Action Agent
                logger.info(f"[{workflow_id}] Starting Action Agent")
                action_plan = self.action_agent.recommend_actions(
                    reasoning_result=reasoning_result,
                    monitoring_report=monitoring_report,
                )
                
                self._add_message(
                    messages,
                    sender='action',
                    receiver='orchestrator',
                    data={
                        'primary_action': action_plan.primary_action,
                        'n_recommendations': len(action_plan.recommendations),
                        'should_escalate': action_plan.should_escalate,
                    },
                    confidence=action_plan.overall_confidence,
                )
                
                logger.info(
                    f"[{workflow_id}] Action Agent complete. "
                    f"Primary action: {action_plan.primary_action}, "
                    f"Escalate: {action_plan.should_escalate}, "
                    f"Confidence: {action_plan.overall_confidence:.2%}"
                )
                
                status = ExecutionStatus.ESCALATED if action_plan.should_escalate else ExecutionStatus.COMPLETED
                should_escalate = action_plan.should_escalate
                escalation_reason = action_plan.escalation_reason
            
            # Calculate overall metrics
            overall_confidence = min(
                monitoring_report.overall_confidence,
                retrieval_result.retrieval_confidence,
                reasoning_result.reasoning_confidence,
            )
            
            overall_risk_score = reasoning_result.risk_explanation.risk_score
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            result = AgentResult(
                engine_id=engine_id,
                cycle=cycle,
                workflow_status=status.value,
                monitoring_report=monitoring_report,
                retrieval_result=retrieval_result,
                reasoning_result=reasoning_result,
                action_plan=action_plan,
                overall_confidence=overall_confidence,
                overall_risk_score=overall_risk_score,
                should_escalate=should_escalate,
                escalation_reason=escalation_reason,
                workflow_id=workflow_id,
                execution_time_ms=execution_time,
                messages=messages,
                timestamp=datetime.now().isoformat(),
            )
            
            self.workflow_history.append(result)
            self.message_log.extend(messages)
            
            logger.info(
                f"[{workflow_id}] Workflow complete. "
                f"Status: {status.value}, "
                f"Risk: {overall_risk_score:.2%}, "
                f"Time: {execution_time:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{workflow_id}] Error in agent orchestration: {e}")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = AgentResult(
                engine_id=engine_id,
                cycle=cycle,
                workflow_status=ExecutionStatus.ERROR.value,
                monitoring_report=None,
                retrieval_result=None,
                reasoning_result=None,
                action_plan=None,
                overall_confidence=0.0,
                overall_risk_score=0.5,  # Default to medium risk
                should_escalate=True,
                escalation_reason=f"System error: {str(e)}",
                workflow_id=workflow_id,
                execution_time_ms=execution_time,
                messages=[],
                timestamp=datetime.now().isoformat(),
            )
            
            self.workflow_history.append(result)
            return result

    def _extract_sensor_deviations(
        self,
        sensor_data: np.ndarray,
        sensor_names: Optional[List[str]],
    ) -> Dict[str, float]:
        """Extract sensor deviations for retrieval query."""
        if len(sensor_data.shape) == 1:
            values = sensor_data
        else:
            values = sensor_data.flatten()

        # Normalize to [-1, 1] range based on assumed sensor bounds
        normalized = np.clip((values - 0.5) * 2, -1, 1)

        # Create deviation dict
        if sensor_names:
            deviations = {
                name: float(norm)
                for name, norm in zip(sensor_names, normalized[:len(sensor_names)])
            }
        else:
            deviations = {
                f'sensor_{i}': float(norm)
                for i, norm in enumerate(normalized[:10])
            }

        return deviations

    def _add_message(
        self,
        messages: List[AgentMessage],
        sender: str,
        receiver: str,
        data: Dict[str, Any],
        confidence: float,
    ):
        """Add message to message log."""
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            timestamp=datetime.now().isoformat(),
            data=data,
            confidence=confidence,
            status=ExecutionStatus.COMPLETED.value,
        )
        messages.append(message)

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        if not self.workflow_history:
            return {
                'n_workflows': 0,
                'avg_execution_time_ms': 0.0,
                'escalation_rate': 0.0,
                'abstention_rate': 0.0,
                'error_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_risk_score': 0.0,
            }

        escalated = len([w for w in self.workflow_history if w.should_escalate])
        abstained = len([w for w in self.workflow_history if w.workflow_status == 'abstained'])
        errors = len([w for w in self.workflow_history if w.workflow_status == 'error'])

        return {
            'n_workflows': len(self.workflow_history),
            'avg_execution_time_ms': np.mean([w.execution_time_ms for w in self.workflow_history]),
            'escalation_rate': escalated / len(self.workflow_history) if self.workflow_history else 0.0,
            'abstention_rate': abstained / len(self.workflow_history) if self.workflow_history else 0.0,
            'error_rate': errors / len(self.workflow_history) if self.workflow_history else 0.0,
            'avg_confidence': np.mean([w.overall_confidence for w in self.workflow_history]),
            'avg_risk_score': np.mean([w.overall_risk_score for w in self.workflow_history]),
            'status_distribution': self._count_statuses(),
        }

    def _count_statuses(self) -> Dict[str, int]:
        """Count workflows by status."""
        counts = {}
        for workflow in self.workflow_history:
            status = workflow.workflow_status
            counts[status] = counts.get(status, 0) + 1
        return counts

    def reset(self):
        """Reset all agents and history."""
        self.monitoring_agent.reset_history()
        self.retrieval_agent.clear_history()
        self.reasoning_agent.clear_history()
        self.action_agent.clear_history()
        self.workflow_history = []
        self.message_log = []
        logger.info("Agent orchestrator reset")
