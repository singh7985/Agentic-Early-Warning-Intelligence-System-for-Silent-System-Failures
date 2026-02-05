"""
Performance Logging: Track token usage, latency, and system metrics

Purpose:
- Log token usage for RAG queries
- Log API call latencies
- Track system performance metrics
- Monitor resource consumption
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage metrics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    @property
    def cost_usd(self) -> float:
        """Estimate cost in USD (GPT-3.5 pricing)."""
        # GPT-3.5: $0.0005 per 1K prompt tokens, $0.0015 per 1K completion tokens
        prompt_cost = (self.prompt_tokens / 1000) * 0.0005
        completion_cost = (self.completion_tokens / 1000) * 0.0015
        return prompt_cost + completion_cost


@dataclass
class LatencyMetrics:
    """Latency measurement"""
    component: str  # 'ml_prediction', 'rag_retrieval', 'agent_reasoning', etc.
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics"""
    timestamp: datetime
    avg_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    predictions_count: int
    errors_count: int
    avg_confidence: float


class PerformanceLogger:
    """
    Log and track performance metrics including token usage and latency.
    
    Features:
    - Token usage tracking
    - Latency measurement
    - Cost estimation
    - Performance aggregation
    - Historical tracking
    """

    def __init__(self):
        """Initialize performance logger."""
        self.token_usage: List[TokenUsage] = []
        self.latencies: List[LatencyMetrics] = []
        self.predictions: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.confidences: List[float] = []
        logger.info("PerformanceLogger initialized")

    def log_token_usage(self, prompt_tokens: int, completion_tokens: int,
                       model: str = "unknown"):
        """Log token usage."""
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        self.token_usage.append(usage)
        logger.debug(f"Logged tokens: {usage.total_tokens} (cost: ${usage.cost_usd:.4f})")

    def log_latency(self, component: str, latency_ms: float, metadata: Dict = None):
        """Log latency for a component."""
        metric = LatencyMetrics(
            component=component,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self.latencies.append(metric)
        logger.debug(f"Logged latency: {component}={latency_ms:.2f}ms")

    def start_timer(self, component: str) -> float:
        """Start timing a component."""
        return time.time()

    def end_timer(self, component: str, start_time: float, metadata: Dict = None):
        """End timing and log latency."""
        latency_ms = (time.time() - start_time) * 1000
        self.log_latency(component, latency_ms, metadata)

    def log_prediction(self, engine_id: int, cycle: int, predicted_rul: float,
                      actual_rul: Optional[float] = None, confidence: float = 1.0,
                      ml_latency_ms: float = 0, rag_latency_ms: float = 0,
                      agent_latency_ms: float = 0):
        """Log a prediction."""
        self.predictions.append({
            'engine_id': engine_id,
            'cycle': cycle,
            'predicted_rul': predicted_rul,
            'actual_rul': actual_rul,
            'confidence': confidence,
            'ml_latency_ms': ml_latency_ms,
            'rag_latency_ms': rag_latency_ms,
            'agent_latency_ms': agent_latency_ms,
            'total_latency_ms': ml_latency_ms + rag_latency_ms + agent_latency_ms,
            'timestamp': datetime.now().isoformat(),
        })
        self.confidences.append(confidence)
        logger.debug(f"Logged prediction: engine={engine_id}, RUL={predicted_rul:.1f}, confidence={confidence:.2f}")

    def log_error(self, component: str, error_type: str, error_message: str,
                 engine_id: Optional[int] = None, cycle: Optional[int] = None):
        """Log error."""
        self.errors.append({
            'component': component,
            'error_type': error_type,
            'error_message': error_message,
            'engine_id': engine_id,
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
        })
        logger.error(f"Logged error: {component}/{error_type}: {error_message}")

    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get token usage summary."""
        if not self.token_usage:
            return {
                'total_tokens': 0,
                'total_prompt_tokens': 0,
                'total_completion_tokens': 0,
                'avg_tokens_per_query': 0,
                'estimated_total_cost_usd': 0,
                'cost_breakdown': {
                    'prompt_cost': 0,
                    'completion_cost': 0,
                }
            }

        total_tokens = sum(t.total_tokens for t in self.token_usage)
        total_prompt_tokens = sum(t.prompt_tokens for t in self.token_usage)
        total_completion_tokens = sum(t.completion_tokens for t in self.token_usage)
        total_cost = sum(t.cost_usd for t in self.token_usage)

        return {
            'total_tokens': total_tokens,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'avg_tokens_per_query': total_tokens / len(self.token_usage),
            'estimated_total_cost_usd': total_cost,
            'cost_breakdown': {
                'prompt_cost': (total_prompt_tokens / 1000) * 0.0005,
                'completion_cost': (total_completion_tokens / 1000) * 0.0015,
            }
        }

    def get_latency_summary(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get latency summary."""
        latencies = [l for l in self.latencies if component is None or l.component == component]
        
        if not latencies:
            return {
                'component': component or 'all',
                'count': 0,
                'avg_ms': 0,
                'min_ms': 0,
                'max_ms': 0,
                'p50_ms': 0,
                'p95_ms': 0,
                'p99_ms': 0,
            }

        latency_values = [l.latency_ms for l in latencies]
        latency_values.sort()

        return {
            'component': component or 'all',
            'count': len(latencies),
            'avg_ms': sum(latency_values) / len(latency_values),
            'min_ms': min(latency_values),
            'max_ms': max(latency_values),
            'p50_ms': latency_values[len(latency_values) // 2],
            'p95_ms': latency_values[int(len(latency_values) * 0.95)],
            'p99_ms': latency_values[int(len(latency_values) * 0.99)],
        }

    def get_latency_by_component(self) -> Dict[str, Dict[str, Any]]:
        """Get latency summary for each component."""
        components = set(l.component for l in self.latencies)
        return {
            comp: self.get_latency_summary(comp)
            for comp in sorted(components)
        }

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get prediction summary."""
        if not self.predictions:
            return {
                'total_predictions': 0,
                'avg_confidence': 0,
                'min_confidence': 0,
                'max_confidence': 0,
                'avg_latency_ms': 0,
            }

        confidences = self.confidences
        latencies = [p['total_latency_ms'] for p in self.predictions]

        return {
            'total_predictions': len(self.predictions),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'avg_latency_ms': sum(latencies) / len(latencies),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5),
            'high_latency_count': sum(1 for l in latencies if l > 1000),  # > 1 second
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        error_types = {}
        for error in self.errors:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        components = {}
        for error in self.errors:
            component = error['component']
            components[component] = components.get(component, 0) + 1

        return {
            'total_errors': len(self.errors),
            'error_types': error_types,
            'error_by_component': components,
            'error_rate': len(self.errors) / max(len(self.predictions), 1),
        }

    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get overall performance snapshot."""
        latencies = [l.latency_ms for l in self.latencies] if self.latencies else [0]
        token_summary = self.get_token_usage_summary()
        pred_summary = self.get_prediction_summary()

        return PerformanceSnapshot(
            timestamp=datetime.now(),
            avg_latency_ms=sum(latencies) / len(latencies),
            max_latency_ms=max(latencies),
            min_latency_ms=min(latencies),
            total_tokens=token_summary['total_tokens'],
            estimated_cost_usd=token_summary['estimated_total_cost_usd'],
            predictions_count=pred_summary['total_predictions'],
            errors_count=len(self.errors),
            avg_confidence=pred_summary['avg_confidence'],
        )

    def export_logs(self, filepath: str):
        """Export all logs to JSON file."""
        logs = {
            'token_usage': [
                {
                    'prompt': t.prompt_tokens,
                    'completion': t.completion_tokens,
                    'total': t.total_tokens,
                    'cost': t.cost_usd,
                }
                for t in self.token_usage
            ],
            'latencies': [
                {
                    'component': l.component,
                    'latency_ms': l.latency_ms,
                    'timestamp': l.timestamp.isoformat(),
                }
                for l in self.latencies
            ],
            'predictions': self.predictions,
            'errors': self.errors,
            'summary': {
                'token_usage': self.get_token_usage_summary(),
                'latency': self.get_latency_by_component(),
                'predictions': self.get_prediction_summary(),
                'errors': self.get_error_summary(),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2, default=str)

        logger.info(f"Exported logs to {filepath}")

    def reset(self):
        """Reset all logs."""
        self.token_usage = []
        self.latencies = []
        self.predictions = []
        self.errors = []
        self.confidences = []
        logger.info("PerformanceLogger reset")
