"""
ICEBURG Monitoring and Alerting System

Provides real-time alerting for:
- Agent failures and errors
- Performance degradation
- Resource exhaustion
- Circuit breaker states
- Security violations
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

from .observability_dashboard import get_dashboard, ObservabilityDashboard

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    AGENT_FAILURE = "agent_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    SECURITY_VIOLATION = "security_violation"
    HIGH_ERROR_RATE = "high_error_rate"
    SLOW_PERFORMANCE = "slow_performance"
    SYSTEM_HEALTH = "system_health"


@dataclass
class Alert:
    """Single alert"""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


class AlertingSystem:
    """
    Monitoring and alerting system for ICEBURG.
    
    Features:
    - Real-time alert generation
    - Alert aggregation and deduplication
    - Alert routing and notification
    - Alert acknowledgment and resolution
    - Alert history and trends
    """
    
    def __init__(self, dashboard: Optional[ObservabilityDashboard] = None):
        self.dashboard = dashboard or get_dashboard()
        self.alerts: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Alert thresholds
        self.thresholds = {
            "error_rate": 0.5,  # 50% error rate triggers alert
            "execution_time": 30.0,  # 30 seconds average execution time
            "memory_usage": 0.9,  # 90% memory usage
            "cpu_usage": 0.9,  # 90% CPU usage
            "circuit_breaker_failures": 5,  # 5 failures trigger circuit breaker
        }
        
        # Alert aggregation
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self.last_alert_time: Dict[str, datetime] = {}
        
        logger.info("Alerting System initialized")
    
    def register_alert_handler(self, handler: Callable[[Alert], None]):
        """Register an alert handler"""
        self.alert_handlers.append(handler)
    
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Create and process an alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            agent_id: Optional agent ID
            metadata: Optional metadata
            
        Returns:
            Created alert
        """
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        # Check for duplicate alerts (same type, agent, within 5 minutes)
        alert_key = f"{alert_type.value}_{agent_id or 'global'}"
        if alert_key in self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time[alert_key]
            if time_since_last < timedelta(minutes=5):
                # Duplicate alert, increment count
                self.alert_counts[alert_key] += 1
                return alert
        
        # New alert
        self.last_alert_time[alert_key] = datetime.now()
        self.alert_counts[alert_key] = 1
        
        # Add to active alerts
        self.active_alerts[alert_key] = alert
        
        # Add to alerts queue
        self.alerts.append(alert)
        
        # Add to history
        self.alert_history.append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT [{severity.value.upper()}] {alert_type.value}: {message}")
        
        return alert
    
    def check_agent_performance(self):
        """Check agent performance and create alerts if needed"""
        agent_performance = self.dashboard.get_agent_performance()
        
        for agent_id, metrics in agent_performance.items():
            # Check error rate
            if metrics.get("error_rate", 0) > self.thresholds["error_rate"]:
                if metrics.get("total_executions", 0) > 10:
                    self.create_alert(
                        alert_type=AlertType.HIGH_ERROR_RATE,
                        severity=AlertSeverity.ERROR,
                        message=f"Agent {agent_id} has high error rate: {metrics['error_rate']:.1%}",
                        agent_id=agent_id,
                        metadata={"error_rate": metrics["error_rate"], "total_executions": metrics["total_executions"]}
                    )
            
            # Check execution time
            avg_time = metrics.get("average_execution_time", 0)
            if avg_time > self.thresholds["execution_time"]:
                if metrics.get("total_executions", 0) > 5:
                    self.create_alert(
                        alert_type=AlertType.SLOW_PERFORMANCE,
                        severity=AlertSeverity.WARNING,
                        message=f"Agent {agent_id} is slow: {avg_time:.1f}s average execution time",
                        agent_id=agent_id,
                        metadata={"average_execution_time": avg_time, "total_executions": metrics["total_executions"]}
                    )
            
            # Check circuit breaker state
            circuit_state = metrics.get("circuit_breaker_state", "CLOSED")
            if circuit_state == "OPEN":
                self.create_alert(
                    alert_type=AlertType.CIRCUIT_BREAKER_OPEN,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Circuit breaker OPEN for agent {agent_id}",
                    agent_id=agent_id,
                    metadata={"circuit_breaker_state": circuit_state}
                )
    
    def check_resource_utilization(self):
        """Check resource utilization and create alerts if needed"""
        resource_status = self.dashboard.resource_stats
        
        if not resource_status:
            return
        
        # Check memory usage
        allocated = resource_status.get("allocated", {})
        system = resource_status.get("system", {})
        
        used_memory_mb = allocated.get("used_memory_mb", 0)
        total_memory_mb = system.get("total_memory_mb", 1)
        memory_usage = used_memory_mb / max(1, total_memory_mb)
        
        if memory_usage > self.thresholds["memory_usage"]:
            self.create_alert(
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                severity=AlertSeverity.CRITICAL,
                message=f"Memory usage critical: {memory_usage:.1%} ({used_memory_mb:.1f}MB / {total_memory_mb:.1f}MB)",
                metadata={
                    "resource": "memory",
                    "usage": memory_usage,
                    "used_mb": used_memory_mb,
                    "total_mb": total_memory_mb
                }
            )
        
        # Check CPU usage
        used_cpu_cores = allocated.get("used_cpu_cores", 0)
        total_cpu_cores = system.get("total_cpu_cores", 1)
        cpu_usage = used_cpu_cores / max(1, total_cpu_cores)
        
        if cpu_usage > self.thresholds["cpu_usage"]:
            self.create_alert(
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                severity=AlertSeverity.WARNING,
                message=f"CPU usage high: {cpu_usage:.1%} ({used_cpu_cores:.1f} / {total_cpu_cores:.1f} cores)",
                metadata={
                    "resource": "cpu",
                    "usage": cpu_usage,
                    "used_cores": used_cpu_cores,
                    "total_cores": total_cpu_cores
                }
            )
    
    def check_system_health(self):
        """Check overall system health and create alerts"""
        health_status = self.dashboard.get_health_status()
        
        if health_status["status"] == "critical":
            self.create_alert(
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.CRITICAL,
                message="System health is CRITICAL",
                metadata={"alerts": health_status.get("alerts", [])}
            )
        elif health_status["status"] == "degraded":
            self.create_alert(
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.WARNING,
                message="System health is DEGRADED",
                metadata={"alerts": health_status.get("alerts", [])}
            )
    
    def record_agent_failure(self, agent_id: str, error: str):
        """Record agent failure and create alert"""
        self.create_alert(
            alert_type=AlertType.AGENT_FAILURE,
            severity=AlertSeverity.ERROR,
            message=f"Agent {agent_id} failed: {error}",
            agent_id=agent_id,
            metadata={"error": error}
        )
    
    def record_security_violation(self, violation_type: str, details: Dict[str, Any]):
        """Record security violation and create alert"""
        self.create_alert(
            alert_type=AlertType.SECURITY_VIOLATION,
            severity=AlertSeverity.CRITICAL,
            message=f"Security violation detected: {violation_type}",
            metadata={"violation_type": violation_type, **details}
        )
    
    def run_health_checks(self):
        """Run all health checks and generate alerts"""
        self.check_agent_performance()
        self.check_resource_utilization()
        self.check_system_health()
    
    def acknowledge_alert(self, alert_key: str):
        """Acknowledge an alert"""
        if alert_key in self.active_alerts:
            self.active_alerts[alert_key].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_key}")
    
    def resolve_alert(self, alert_key: str):
        """Resolve an alert"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_key]
            logger.info(f"Alert resolved: {alert_key}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
        
        total_by_type = defaultdict(int)
        for alert in self.alert_history:
            total_by_type[alert.alert_type.value] += 1
        
        return {
            "active_alerts": len(self.active_alerts),
            "active_by_severity": dict(active_by_severity),
            "total_alerts": len(self.alert_history),
            "total_by_type": dict(total_by_type),
            "recent_alerts": [
                {
                    "type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "agent_id": alert.agent_id
                }
                for alert in list(self.alerts)[-20:]  # Last 20 alerts
            ]
        }


# Global alerting system instance
_alerting_system: Optional[AlertingSystem] = None


def get_alerting_system() -> AlertingSystem:
    """Get or create global alerting system"""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    return _alerting_system
