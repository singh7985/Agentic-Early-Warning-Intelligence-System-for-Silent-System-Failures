"""
Alerting System: Monitor model performance and trigger alerts

Purpose:
- Monitor model confidence degradation
- Alert on performance drops
- Track alert history
- Generate alert reports
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    CONFIDENCE_LOW = "confidence_low"
    CONFIDENCE_DEGRADING = "confidence_degrading"
    PERFORMANCE_DROP = "performance_drop"
    ERROR_RATE_HIGH = "error_rate_high"
    LATENCY_HIGH = "latency_high"
    DRIFT_DETECTED = "drift_detected"
    MODEL_STALE = "model_stale"
    RESOURCE_LIMITED = "resource_limited"


@dataclass
class Alert:
    """Single alert"""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class AlertThresholds:
    """Thresholds for triggering alerts"""
    min_confidence: float = 0.5  # Alert if average confidence drops below this
    confidence_degradation_rate: float = 0.05  # Alert if confidence drops > 5% per interval
    error_rate_threshold: float = 0.1  # Alert if error rate > 10%
    latency_threshold_ms: float = 2000  # Alert if latency > 2 seconds
    max_allowed_stale_days: int = 30  # Alert if model older than 30 days
    enabled_alert_types: List[AlertType] = field(
        default_factory=lambda: [
            AlertType.CONFIDENCE_LOW,
            AlertType.CONFIDENCE_DEGRADING,
            AlertType.PERFORMANCE_DROP,
            AlertType.ERROR_RATE_HIGH,
            AlertType.LATENCY_HIGH,
        ]
    )


class AlertingSystem:
    """
    Monitor model performance and generate alerts.
    
    Features:
    - Confidence degradation monitoring
    - Error rate tracking
    - Latency monitoring
    - Alert history
    - Custom alert handlers
    """

    def __init__(self, thresholds: Optional[AlertThresholds] = None):
        """
        Initialize alerting system.
        
        Args:
            thresholds: Custom alert thresholds
        """
        self.thresholds = thresholds or AlertThresholds()
        self.alerts: List[Alert] = []
        self.confidence_history: List[float] = []
        self.error_history: List[float] = []
        self.latency_history: List[float] = []
        self.alert_handlers: List[Callable] = []
        self.suppress_duplicate_alerts = True
        self.duplicate_suppression_window = timedelta(minutes=5)
        logger.info("AlertingSystem initialized")

    def add_alert_handler(self, handler: Callable):
        """Register a handler for alerts."""
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")

    def remove_alert_handler(self, handler: Callable):
        """Remove alert handler."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
            logger.info(f"Removed alert handler: {handler.__name__}")

    def trigger_alert(self, alert: Alert):
        """Trigger an alert and notify handlers."""
        # Check for duplicate suppression
        if self.suppress_duplicate_alerts:
            similar_recent = self._find_similar_recent_alert(alert)
            if similar_recent:
                logger.debug(f"Suppressed duplicate alert: {alert.alert_type.value}")
                return

        # Add to history
        self.alerts.append(alert)
        logger.warning(f"Alert triggered: {alert.alert_type.value} ({alert.severity.value})")

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__name__}: {e}")

    def _find_similar_recent_alert(self, alert: Alert) -> Optional[Alert]:
        """Find similar alert within suppression window."""
        cutoff_time = datetime.now() - self.duplicate_suppression_window

        for prev_alert in reversed(self.alerts):
            if prev_alert.timestamp < cutoff_time:
                break
            if (prev_alert.alert_type == alert.alert_type and
                prev_alert.severity == alert.severity):
                return prev_alert

        return None

    def check_confidence_degradation(self, new_confidence: float):
        """Check for confidence degradation."""
        self.confidence_history.append(new_confidence)

        # Keep only last 100 samples
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]

        # Check if confidence is too low
        if new_confidence < self.thresholds.min_confidence:
            alert = Alert(
                alert_type=AlertType.CONFIDENCE_LOW,
                severity=AlertSeverity.WARNING,
                message=f"Model confidence is low: {new_confidence:.2f} "
                       f"(threshold: {self.thresholds.min_confidence})",
                details={'confidence': new_confidence},
            )
            self.trigger_alert(alert)

        # Check for degrading confidence
        if len(self.confidence_history) >= 10:
            recent_avg = statistics.mean(self.confidence_history[-10:])
            older_avg = statistics.mean(self.confidence_history[-30:-10]) if len(self.confidence_history) >= 30 else recent_avg

            degradation = (older_avg - recent_avg) / (older_avg + 1e-6)

            if degradation > self.thresholds.confidence_degradation_rate:
                alert = Alert(
                    alert_type=AlertType.CONFIDENCE_DEGRADING,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Model confidence degrading: {degradation:.2%} drop "
                           f"(recent: {recent_avg:.3f}, older: {older_avg:.3f})",
                    details={
                        'recent_avg': recent_avg,
                        'older_avg': older_avg,
                        'degradation_rate': degradation,
                    },
                )
                self.trigger_alert(alert)

    def check_error_rate(self, error_count: int, total_count: int):
        """Check error rate."""
        error_rate = error_count / max(total_count, 1)
        self.error_history.append(error_rate)

        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]

        if error_rate > self.thresholds.error_rate_threshold:
            alert = Alert(
                alert_type=AlertType.ERROR_RATE_HIGH,
                severity=AlertSeverity.CRITICAL,
                message=f"Error rate is high: {error_rate:.2%} "
                       f"({error_count}/{total_count} errors) "
                       f"(threshold: {self.thresholds.error_rate_threshold:.2%})",
                details={
                    'error_count': error_count,
                    'total_count': total_count,
                    'error_rate': error_rate,
                },
            )
            self.trigger_alert(alert)

    def check_latency(self, latency_ms: float):
        """Check latency."""
        self.latency_history.append(latency_ms)

        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]

        if latency_ms > self.thresholds.latency_threshold_ms:
            alert = Alert(
                alert_type=AlertType.LATENCY_HIGH,
                severity=AlertSeverity.WARNING,
                message=f"High latency detected: {latency_ms:.0f}ms "
                       f"(threshold: {self.thresholds.latency_threshold_ms:.0f}ms)",
                details={'latency_ms': latency_ms},
            )
            self.trigger_alert(alert)

    def check_drift_detection(self, drift_severity: str, affected_features: int):
        """Check drift detection."""
        if drift_severity in ['medium', 'high']:
            alert = Alert(
                alert_type=AlertType.DRIFT_DETECTED,
                severity=AlertSeverity.CRITICAL if drift_severity == 'high' else AlertSeverity.WARNING,
                message=f"Data drift detected ({drift_severity}): "
                       f"{affected_features} features affected",
                details={
                    'drift_severity': drift_severity,
                    'affected_features': affected_features,
                },
            )
            self.trigger_alert(alert)

    def check_model_staleness(self, model_age_days: int):
        """Check if model is too old."""
        if model_age_days > self.thresholds.max_allowed_stale_days:
            alert = Alert(
                alert_type=AlertType.MODEL_STALE,
                severity=AlertSeverity.WARNING,
                message=f"Model is stale: {model_age_days} days old "
                       f"(max allowed: {self.thresholds.max_allowed_stale_days} days)",
                details={'model_age_days': model_age_days},
            )
            self.trigger_alert(alert)

    def acknowledge_alert(self, alert_index: int, acknowledged_by: str = "system"):
        """Acknowledge an alert."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].acknowledged = True
            self.alerts[alert_index].acknowledged_by = acknowledged_by
            self.alerts[alert_index].acknowledged_at = datetime.now()
            logger.info(f"Alert acknowledged: {self.alerts[alert_index].alert_type.value}")

    def get_active_alerts(self) -> List[Alert]:
        """Get active (unacknowledged) alerts."""
        return [a for a in self.alerts if not a.acknowledged]

    def get_critical_alerts(self) -> List[Alert]:
        """Get critical alerts."""
        return [a for a in self.get_active_alerts() 
                if a.severity == AlertSeverity.CRITICAL]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts."""
        active = self.get_active_alerts()
        critical = self.get_critical_alerts()

        alert_type_counts = {}
        for alert in self.alerts:
            alert_type = alert.alert_type.value
            alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1

        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active),
            'critical_alerts': len(critical),
            'alert_types': alert_type_counts,
            'latest_alert': self.alerts[-1].timestamp.isoformat() if self.alerts else None,
            'avg_confidence': statistics.mean(self.confidence_history) if self.confidence_history else 0,
            'avg_error_rate': statistics.mean(self.error_history) if self.error_history else 0,
            'avg_latency_ms': statistics.mean(self.latency_history) if self.latency_history else 0,
        }

    def get_alert_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert report for recent period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

        severity_counts = {}
        for alert in recent_alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'by_severity': severity_counts,
            'by_type': {
                alert_type.value: sum(1 for a in recent_alerts if a.alert_type == alert_type)
                for alert_type in AlertType
            },
            'alerts': [
                {
                    'type': a.alert_type.value,
                    'severity': a.severity.value,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat(),
                    'acknowledged': a.acknowledged,
                }
                for a in recent_alerts
            ]
        }

    def reset(self):
        """Reset alerting system."""
        self.alerts = []
        self.confidence_history = []
        self.error_history = []
        self.latency_history = []
        logger.info("AlertingSystem reset")


class EmailAlertHandler:
    """Send alerts via email."""

    def __init__(self, email_addresses: List[str], smtp_server: str = "localhost"):
        """Initialize email handler."""
        self.email_addresses = email_addresses
        self.smtp_server = smtp_server
        logger.info(f"EmailAlertHandler initialized for {email_addresses}")

    def __call__(self, alert: Alert):
        """Handle alert by sending email."""
        # In production, use actual email library
        logger.info(f"[EMAIL] To: {self.email_addresses}")
        logger.info(f"[EMAIL] Subject: {alert.alert_type.value} - {alert.severity.value}")
        logger.info(f"[EMAIL] Body: {alert.message}")


class SlackAlertHandler:
    """Send alerts to Slack."""

    def __init__(self, webhook_url: str):
        """Initialize Slack handler."""
        self.webhook_url = webhook_url
        logger.info("SlackAlertHandler initialized")

    def __call__(self, alert: Alert):
        """Handle alert by sending to Slack."""
        # In production, use actual Slack API
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
        }
        
        emoji = severity_emoji.get(alert.severity, "")
        logger.info(f"[SLACK] {emoji} {alert.alert_type.value}: {alert.message}")


class LogAlertHandler:
    """Log alerts to file/console."""

    def __call__(self, alert: Alert):
        """Handle alert by logging."""
        logger.critical(f"ALERT: {alert.alert_type.value} - {alert.message}")
