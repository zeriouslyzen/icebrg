"""
Prometheus Integration for ICEBURG Monitoring
Implements metrics collection, auto-scaling, and performance monitoring.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, 
        CollectorRegistry, generate_latest, 
        CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None
    Summary = None
    CollectorRegistry = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None
    start_http_server = None

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str
    threshold: float
    duration: float = 60.0  # seconds
    severity: str = "warning"
    action: str = "log"
    enabled: bool = True


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    name: str
    metric: str
    threshold: float
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_instances: int = 1
    max_instances: int = 10
    cooldown_period: float = 300.0  # seconds
    enabled: bool = True


class ICEBURGMetrics:
    """
    ICEBURG-specific metrics collection and monitoring.
    
    Features:
    - Custom metrics for ICEBURG operations
    - Performance tracking
    - Error monitoring
    - Resource utilization
    - Auto-scaling triggers
    """
    
    def __init__(self, 
                 registry: Optional[CollectorRegistry] = None,
                 enable_http_server: bool = True,
                 http_port: int = 8000):
        """
        Initialize ICEBURG metrics.
        
        Args:
            registry: Prometheus registry (uses default if None)
            enable_http_server: Whether to start HTTP server
            http_port: HTTP server port
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available, using mock metrics")
            self._init_mock_metrics()
            return
        
        self.registry = registry or CollectorRegistry()
        self.enable_http_server = enable_http_server
        self.http_port = http_port
        
        # Initialize metrics
        self._init_metrics()
        
        # Start HTTP server if enabled
        if self.enable_http_server:
            self._start_http_server()
    
    def _init_mock_metrics(self):
        """Initialize mock metrics when Prometheus is not available."""
        self.metrics = {
            "requests_total": {"value": 0, "type": "counter"},
            "request_duration_seconds": {"value": 0.0, "type": "histogram"},
            "active_connections": {"value": 0, "type": "gauge"},
            "cache_hit_ratio": {"value": 0.0, "type": "gauge"},
            "memory_usage_bytes": {"value": 0, "type": "gauge"},
            "cpu_usage_percent": {"value": 0.0, "type": "gauge"},
            "error_rate": {"value": 0.0, "type": "gauge"},
            "agent_execution_time": {"value": 0.0, "type": "histogram"},
            "civilization_simulation_steps": {"value": 0, "type": "counter"},
            "emergence_detection_count": {"value": 0, "type": "counter"}
        }
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.requests_total = Counter(
            'iceburg_requests_total',
            'Total number of ICEBURG requests',
            ['mode', 'status'],
            registry=self.registry
        )
        
        self.request_duration_seconds = Histogram(
            'iceburg_request_duration_seconds',
            'Request duration in seconds',
            ['mode'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Connection metrics
        self.active_connections = Gauge(
            'iceburg_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hit_ratio = Gauge(
            'iceburg_cache_hit_ratio',
            'Cache hit ratio',
            registry=self.registry
        )
        
        # Resource metrics
        self.memory_usage_bytes = Gauge(
            'iceburg_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'iceburg_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Error metrics
        self.error_rate = Gauge(
            'iceburg_error_rate',
            'Error rate percentage',
            registry=self.registry
        )
        
        # Agent metrics
        self.agent_execution_time = Histogram(
            'iceburg_agent_execution_time_seconds',
            'Agent execution time in seconds',
            ['agent_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Civilization metrics
        self.civilization_simulation_steps = Counter(
            'iceburg_civilization_simulation_steps_total',
            'Total civilization simulation steps',
            ['simulation_id'],
            registry=self.registry
        )
        
        # Emergence detection metrics
        self.emergence_detection_count = Counter(
            'iceburg_emergence_detection_count_total',
            'Total emergence detections',
            ['emergence_type'],
            registry=self.registry
        )
        
        # Load balancer metrics
        self.load_balancer_requests = Counter(
            'iceburg_load_balancer_requests_total',
            'Load balancer requests',
            ['worker_id', 'status'],
            registry=self.registry
        )
        
        self.worker_load = Gauge(
            'iceburg_worker_load',
            'Worker load percentage',
            ['worker_id'],
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'iceburg_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['worker_id'],
            registry=self.registry
        )
        
        # Redis metrics
        self.redis_operations = Counter(
            'iceburg_redis_operations_total',
            'Redis operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.redis_connection_pool_size = Gauge(
            'iceburg_redis_connection_pool_size',
            'Redis connection pool size',
            registry=self.registry
        )
    
    def _start_http_server(self):
        """Start Prometheus HTTP server."""
        try:
            start_http_server(self.http_port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {self.http_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def record_request(self, mode: str, status: str, duration: float):
        """Record a request metric."""
        if PROMETHEUS_AVAILABLE:
            self.requests_total.labels(mode=mode, status=status).inc()
            self.request_duration_seconds.labels(mode=mode).observe(duration)
        else:
            self.metrics["requests_total"]["value"] += 1
            self.metrics["request_duration_seconds"]["value"] = duration
    
    def record_connection(self, active: int):
        """Record active connections."""
        if PROMETHEUS_AVAILABLE:
            self.active_connections.set(active)
        else:
            self.metrics["active_connections"]["value"] = active
    
    def record_cache_hit_ratio(self, ratio: float):
        """Record cache hit ratio."""
        if PROMETHEUS_AVAILABLE:
            self.cache_hit_ratio.set(ratio)
        else:
            self.metrics["cache_hit_ratio"]["value"] = ratio
    
    def record_resource_usage(self, memory_bytes: int, cpu_percent: float):
        """Record resource usage."""
        if PROMETHEUS_AVAILABLE:
            self.memory_usage_bytes.set(memory_bytes)
            self.cpu_usage_percent.set(cpu_percent)
        else:
            self.metrics["memory_usage_bytes"]["value"] = memory_bytes
            self.metrics["cpu_usage_percent"]["value"] = cpu_percent
    
    def record_error_rate(self, rate: float):
        """Record error rate."""
        if PROMETHEUS_AVAILABLE:
            self.error_rate.set(rate)
        else:
            self.metrics["error_rate"]["value"] = rate
    
    def record_agent_execution(self, agent_type: str, duration: float):
        """Record agent execution time."""
        if PROMETHEUS_AVAILABLE:
            self.agent_execution_time.labels(agent_type=agent_type).observe(duration)
        else:
            self.metrics["agent_execution_time"]["value"] = duration
    
    def record_civilization_step(self, simulation_id: str):
        """Record civilization simulation step."""
        if PROMETHEUS_AVAILABLE:
            self.civilization_simulation_steps.labels(simulation_id=simulation_id).inc()
        else:
            self.metrics["civilization_simulation_steps"]["value"] += 1
    
    def record_emergence_detection(self, emergence_type: str):
        """Record emergence detection."""
        if PROMETHEUS_AVAILABLE:
            self.emergence_detection_count.labels(emergence_type=emergence_type).inc()
        else:
            self.metrics["emergence_detection_count"]["value"] += 1
    
    def record_load_balancer_request(self, worker_id: str, status: str):
        """Record load balancer request."""
        if PROMETHEUS_AVAILABLE:
            self.load_balancer_requests.labels(worker_id=worker_id, status=status).inc()
    
    def record_worker_load(self, worker_id: str, load: float):
        """Record worker load."""
        if PROMETHEUS_AVAILABLE:
            self.worker_load.labels(worker_id=worker_id).set(load)
    
    def record_circuit_breaker_state(self, worker_id: str, state: str):
        """Record circuit breaker state."""
        if PROMETHEUS_AVAILABLE:
            state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
            self.circuit_breaker_state.labels(worker_id=worker_id).set(state_value)
    
    def record_redis_operation(self, operation: str, status: str):
        """Record Redis operation."""
        if PROMETHEUS_AVAILABLE:
            self.redis_operations.labels(operation=operation, status=status).inc()
    
    def record_redis_connection_pool_size(self, size: int):
        """Record Redis connection pool size."""
        if PROMETHEUS_AVAILABLE:
            self.redis_connection_pool_size.set(size)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            # Return mock metrics as JSON
            return json.dumps(self.metrics, indent=2)


class AutoScaler:
    """
    Auto-scaling system based on Prometheus metrics.
    
    Features:
    - Metric-based scaling decisions
    - Cooldown periods
    - Multiple scaling policies
    - Scaling actions
    """
    
    def __init__(self, 
                 metrics: ICEBURGMetrics,
                 scaling_policies: List[ScalingPolicy] = None):
        """
        Initialize auto-scaler.
        
        Args:
            metrics: ICEBURG metrics instance
            scaling_policies: List of scaling policies
        """
        self.metrics = metrics
        self.scaling_policies = scaling_policies or []
        
        # Scaling state
        self.last_scale_time = 0.0
        self.current_instances = 1
        self.scaling_history = []
        
        # Scaling actions
        self.scaling_actions = {
            "scale_up": self._scale_up,
            "scale_down": self._scale_down,
            "no_action": lambda: None
        }
    
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add a scaling policy."""
        self.scaling_policies.append(policy)
        logger.info(f"Added scaling policy: {policy.name}")
    
    def remove_scaling_policy(self, policy_name: str):
        """Remove a scaling policy."""
        self.scaling_policies = [p for p in self.scaling_policies if p.name != policy_name]
        logger.info(f"Removed scaling policy: {policy_name}")
    
    async def evaluate_scaling(self) -> str:
        """
        Evaluate scaling based on current metrics.
        
        Returns:
            Scaling action to take
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_time < 60.0:  # 1 minute cooldown
            return "no_action"
        
        # Evaluate each policy
        for policy in self.scaling_policies:
            if not policy.enabled:
                continue
            
            action = await self._evaluate_policy(policy)
            if action != "no_action":
                return action
        
        return "no_action"
    
    async def _evaluate_policy(self, policy: ScalingPolicy) -> str:
        """Evaluate a specific scaling policy."""
        # Get current metric value
        metric_value = await self._get_metric_value(policy.metric)
        
        if metric_value is None:
            return "no_action"
        
        # Check scaling thresholds
        if metric_value >= policy.scale_up_threshold:
            if self.current_instances < policy.max_instances:
                return "scale_up"
        elif metric_value <= policy.scale_down_threshold:
            if self.current_instances > policy.min_instances:
                return "scale_down"
        
        return "no_action"
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        # This would typically query Prometheus for the metric value
        # For now, return a mock value
        mock_metrics = {
            "cpu_usage_percent": 45.0,
            "memory_usage_bytes": 1024 * 1024 * 1024,  # 1GB
            "active_connections": 10,
            "request_duration_seconds": 2.5,
            "error_rate": 0.05
        }
        
        return mock_metrics.get(metric_name, 0.0)
    
    async def _scale_up(self):
        """Scale up the system."""
        if self.current_instances < 10:  # Max instances
            self.current_instances += 1
            self.last_scale_time = time.time()
            
            # Record scaling action
            self.scaling_history.append({
                "action": "scale_up",
                "timestamp": time.time(),
                "new_instances": self.current_instances
            })
            
            logger.info(f"Scaled up to {self.current_instances} instances")
            
            # Trigger actual scaling (e.g., start new worker processes)
            await self._trigger_scaling_action("scale_up")
    
    async def _scale_down(self):
        """Scale down the system."""
        if self.current_instances > 1:  # Min instances
            self.current_instances -= 1
            self.last_scale_time = time.time()
            
            # Record scaling action
            self.scaling_history.append({
                "action": "scale_down",
                "timestamp": time.time(),
                "new_instances": self.current_instances
            })
            
            logger.info(f"Scaled down to {self.current_instances} instances")
            
            # Trigger actual scaling (e.g., stop worker processes)
            await self._trigger_scaling_action("scale_down")
    
    async def _trigger_scaling_action(self, action: str):
        """Trigger actual scaling action."""
        # This would integrate with container orchestration (Kubernetes, Docker Swarm)
        # or cloud provider APIs (AWS Auto Scaling, Azure Scale Sets, GCP Managed Instance Groups)
        
        if action == "scale_up":
            # Start new worker instances
            logger.info("Triggering scale up action")
        elif action == "scale_down":
            # Stop worker instances
            logger.info("Triggering scale down action")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "current_instances": self.current_instances,
            "last_scale_time": self.last_scale_time,
            "scaling_policies": len(self.scaling_policies),
            "scaling_history": self.scaling_history[-10:],  # Last 10 actions
            "cooldown_remaining": max(0, 60.0 - (time.time() - self.last_scale_time))
        }


class AlertManager:
    """
    Alert management system for ICEBURG monitoring.
    
    Features:
    - Alert rules and conditions
    - Alert actions (log, notify, scale)
    - Alert history and tracking
    """
    
    def __init__(self, metrics: ICEBURGMetrics):
        """
        Initialize alert manager.
        
        Args:
            metrics: ICEBURG metrics instance
        """
        self.metrics = metrics
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Alert actions
        self.alert_actions = {
            "log": self._log_alert,
            "notify": self._notify_alert,
            "scale": self._scale_alert
        }
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    async def check_alerts(self):
        """Check all alert rules."""
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            await self._check_alert_rule(rule)
    
    async def _check_alert_rule(self, rule: AlertRule):
        """Check a specific alert rule."""
        # Get current metric value
        metric_value = await self._get_metric_value(rule.condition)
        
        if metric_value is None:
            return
        
        # Check if threshold is exceeded
        if metric_value > rule.threshold:
            # Check if alert is already active
            if rule.name in self.active_alerts:
                return
            
            # Trigger alert
            await self._trigger_alert(rule, metric_value)
        else:
            # Clear alert if it was active
            if rule.name in self.active_alerts:
                await self._clear_alert(rule)
    
    async def _get_metric_value(self, condition: str) -> Optional[float]:
        """Get metric value for alert condition."""
        # Parse condition (e.g., "cpu_usage_percent > 80")
        # For now, return mock values
        mock_metrics = {
            "cpu_usage_percent": 45.0,
            "memory_usage_bytes": 1024 * 1024 * 1024,
            "active_connections": 10,
            "error_rate": 0.05
        }
        
        # Simple condition parsing
        if "cpu_usage_percent" in condition:
            return mock_metrics["cpu_usage_percent"]
        elif "memory_usage_bytes" in condition:
            return mock_metrics["memory_usage_bytes"]
        elif "active_connections" in condition:
            return mock_metrics["active_connections"]
        elif "error_rate" in condition:
            return mock_metrics["error_rate"]
        
        return None
    
    async def _trigger_alert(self, rule: AlertRule, metric_value: float):
        """Trigger an alert."""
        alert_data = {
            "rule_name": rule.name,
            "severity": rule.severity,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "current_value": metric_value,
            "timestamp": time.time(),
            "action": rule.action
        }
        
        self.active_alerts[rule.name] = alert_data
        self.alert_history.append(alert_data)
        
        # Execute alert action
        if rule.action in self.alert_actions:
            await self.alert_actions[rule.action](alert_data)
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.condition} > {rule.threshold}")
    
    async def _clear_alert(self, rule: AlertRule):
        """Clear an alert."""
        if rule.name in self.active_alerts:
            del self.active_alerts[rule.name]
            logger.info(f"Alert cleared: {rule.name}")
    
    async def _log_alert(self, alert_data: Dict[str, Any]):
        """Log alert action."""
        logger.warning(f"ALERT: {alert_data['rule_name']} - {alert_data['condition']} = {alert_data['current_value']}")
    
    async def _notify_alert(self, alert_data: Dict[str, Any]):
        """Notify alert action."""
        # This would integrate with notification systems (Slack, email, PagerDuty)
        logger.warning(f"NOTIFICATION: {alert_data['rule_name']} - {alert_data['condition']} = {alert_data['current_value']}")
    
    async def _scale_alert(self, alert_data: Dict[str, Any]):
        """Scale alert action."""
        # This would trigger auto-scaling
        logger.warning(f"SCALE: {alert_data['rule_name']} - {alert_data['condition']} = {alert_data['current_value']}")
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status."""
        return {
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.alert_rules),
            "alert_history": self.alert_history[-10:],  # Last 10 alerts
            "active_alert_details": list(self.active_alerts.values())
        }


class PrometheusMonitor:
    """
    Main monitoring system for ICEBURG.
    
    Features:
    - Metrics collection
    - Auto-scaling
    - Alert management
    - Performance monitoring
    """
    
    def __init__(self, 
                 enable_http_server: bool = True,
                 http_port: int = 8000):
        """
        Initialize Prometheus monitor.
        
        Args:
            enable_http_server: Whether to start HTTP server
            http_port: HTTP server port
        """
        self.metrics = ICEBURGMetrics(
            enable_http_server=enable_http_server,
            http_port=http_port
        )
        
        self.auto_scaler = AutoScaler(self.metrics)
        self.alert_manager = AlertManager(self.metrics)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Default scaling policies
        self._setup_default_policies()
        self._setup_default_alerts()
    
    def _setup_default_policies(self):
        """Setup default scaling policies."""
        policies = [
            ScalingPolicy(
                name="cpu_scaling",
                metric="cpu_usage_percent",
                threshold=70.0,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                min_instances=1,
                max_instances=10
            ),
            ScalingPolicy(
                name="memory_scaling",
                metric="memory_usage_bytes",
                threshold=1024 * 1024 * 1024 * 8,  # 8GB
                scale_up_threshold=1024 * 1024 * 1024 * 9,  # 9GB
                scale_down_threshold=1024 * 1024 * 1024 * 4,  # 4GB
                min_instances=1,
                max_instances=10
            )
        ]
        
        for policy in policies:
            self.auto_scaler.add_scaling_policy(policy)
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        alerts = [
            AlertRule(
                name="high_cpu_usage",
                condition="cpu_usage_percent > 90",
                threshold=90.0,
                severity="critical",
                action="notify"
            ),
            AlertRule(
                name="high_memory_usage",
                condition="memory_usage_bytes > 1024*1024*1024*9",
                threshold=1024 * 1024 * 1024 * 9,  # 9GB
                severity="warning",
                action="log"
            ),
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 0.1",
                threshold=0.1,
                severity="critical",
                action="scale"
            )
        ]
        
        for alert in alerts:
            self.alert_manager.add_alert_rule(alert)
    
    async def start_monitoring(self):
        """Start monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Prometheus monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring system."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Prometheus monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check alerts
                await self.alert_manager.check_alerts()
                
                # Evaluate scaling
                scaling_action = await self.auto_scaler.evaluate_scaling()
                if scaling_action != "no_action":
                    logger.info(f"Auto-scaling action: {scaling_action}")
                
                # Update metrics
                await self._update_system_metrics()
                
                # Wait before next check
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _update_system_metrics(self):
        """Update system metrics."""
        # This would collect actual system metrics
        # For now, use mock values
        
        # Mock system metrics
        memory_usage = 1024 * 1024 * 1024 * 2  # 2GB
        cpu_usage = 45.0
        active_connections = 10
        error_rate = 0.05
        
        # Record metrics
        self.metrics.record_resource_usage(memory_usage, cpu_usage)
        self.metrics.record_connection(active_connections)
        self.metrics.record_error_rate(error_rate)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status."""
        return {
            "monitoring_active": self.monitoring_active,
            "metrics": self.metrics.get_metrics(),
            "scaling_status": self.auto_scaler.get_scaling_status(),
            "alert_status": self.alert_manager.get_alert_status()
        }
    
    async def cleanup(self):
        """Cleanup monitoring resources."""
        await self.stop_monitoring()
        logger.info("Prometheus monitoring cleanup completed")


# Convenience functions
async def create_prometheus_monitor(enable_http_server: bool = True, 
                                 http_port: int = 8000) -> PrometheusMonitor:
    """Create a new Prometheus monitor."""
    return PrometheusMonitor(enable_http_server=enable_http_server, http_port=http_port)


async def start_iceburg_monitoring(monitor: PrometheusMonitor = None) -> PrometheusMonitor:
    """Start ICEBURG monitoring."""
    if monitor is None:
        monitor = await create_prometheus_monitor()
    
    await monitor.start_monitoring()
    return monitor


async def get_iceburg_metrics(monitor: PrometheusMonitor) -> str:
    """Get ICEBURG metrics in Prometheus format."""
    return monitor.metrics.get_metrics()
