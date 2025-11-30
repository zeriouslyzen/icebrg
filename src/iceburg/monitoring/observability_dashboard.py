"""
ICEBURG Real-Time Observability Dashboard

Provides real-time monitoring and visualization of:
- Agent performance metrics
- Resource utilization
- Circuit breaker states
- Load balancer statistics
- Error rates and patterns
- Execution timelines
"""

import time
import json
import logging
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

from ..agents.capability_registry import get_registry
from ..infrastructure.retry_manager import RetryManager
from ..infrastructure.dynamic_resource_allocator import get_resource_allocator
from ..distributed.load_balancer import IntelligentLoadBalancer
from ..optimization.performance_optimizer import get_performance_optimizer

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ERROR = "error"
    CIRCUIT_BREAKER = "circuit_breaker"
    LOAD_BALANCER = "load_balancer"
    AGENT = "agent"


@dataclass
class Metric:
    """Single metric data point"""
    metric_type: MetricType
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent"""
    agent_id: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    total_execution_time: float = 0.0
    error_rate: float = 0.0
    circuit_breaker_state: str = "CLOSED"
    last_execution: Optional[datetime] = None
    recent_execution_times: deque = field(default_factory=lambda: deque(maxlen=100))


class ObservabilityDashboard:
    """
    Real-time observability dashboard for agent performance.
    
    Features:
    - Real-time metrics collection
    - Agent performance tracking
    - Resource utilization monitoring
    - Circuit breaker state tracking
    - Load balancer statistics
    - Error pattern analysis
    - Performance trend analysis
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics: deque = deque(maxlen=history_size)
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.circuit_breaker_states: Dict[str, str] = {}
        self.load_balancer_stats: Dict[str, Any] = {}
        self.resource_stats: Dict[str, Any] = {}
        
        # Component references
        self.registry = get_registry()
        self.resource_allocator = get_resource_allocator()
        self.performance_optimizer = get_performance_optimizer()
        
        # Initialize agent metrics
        self._initialize_agent_metrics()
        
        logger.info("Observability Dashboard initialized")
    
    def _initialize_agent_metrics(self):
        """Initialize metrics for all agents"""
        all_agents = self.registry.get_all_agents()
        for agent_id in all_agents.keys():
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(agent_id=agent_id)
    
    def record_agent_execution(
        self,
        agent_id: str,
        success: bool,
        execution_time: float,
        error: Optional[str] = None
    ):
        """
        Record agent execution metrics.
        
        Args:
            agent_id: Agent identifier
            success: Whether execution was successful
            execution_time: Execution time in seconds
            error: Error message if failed
        """
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(agent_id=agent_id)
        
        metrics = self.agent_metrics[agent_id]
        
        # Update metrics
        metrics.total_executions += 1
        if success:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
            if error:
                self.error_patterns[error] += 1
        
        # Update execution time statistics
        metrics.recent_execution_times.append(execution_time)
        metrics.total_execution_time += execution_time
        metrics.average_execution_time = metrics.total_execution_time / metrics.total_executions
        metrics.min_execution_time = min(metrics.min_execution_time, execution_time)
        metrics.max_execution_time = max(metrics.max_execution_time, execution_time)
        metrics.last_execution = datetime.now()
        
        # Update error rate
        if metrics.total_executions > 0:
            metrics.error_rate = metrics.failed_executions / metrics.total_executions
        
        # Record metric
        metric = Metric(
            metric_type=MetricType.AGENT,
            name=f"agent.{agent_id}.execution",
            value=execution_time,
            tags={"agent_id": agent_id, "success": str(success)},
            metadata={"error": error} if error else {}
        )
        self.metrics.append(metric)
    
    def record_circuit_breaker_state(self, agent_id: str, state: str):
        """Record circuit breaker state"""
        self.circuit_breaker_states[agent_id] = state
        
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id].circuit_breaker_state = state
        
        metric = Metric(
            metric_type=MetricType.CIRCUIT_BREAKER,
            name=f"circuit_breaker.{agent_id}.state",
            value=1.0 if state == "OPEN" else 0.0,
            tags={"agent_id": agent_id, "state": state}
        )
        self.metrics.append(metric)
    
    def record_resource_utilization(self):
        """Record current resource utilization"""
        status = self.resource_allocator.get_resource_status()
        self.resource_stats = status
        
        # Record metrics
        metric = Metric(
            metric_type=MetricType.RESOURCE,
            name="resource.memory.used",
            value=status["allocated"]["used_memory_mb"],
            tags={"resource": "memory"},
            metadata=status
        )
        self.metrics.append(metric)
        
        metric = Metric(
            metric_type=MetricType.RESOURCE,
            name="resource.cpu.used",
            value=status["allocated"]["used_cpu_cores"],
            tags={"resource": "cpu"},
            metadata=status
        )
        self.metrics.append(metric)
    
    def record_load_balancer_stats(self, stats: Dict[str, Any]):
        """Record load balancer statistics"""
        self.load_balancer_stats = stats
        
        metric = Metric(
            metric_type=MetricType.LOAD_BALANCER,
            name="load_balancer.success_rate",
            value=stats.get("success_rate", 0.0),
            tags={},
            metadata=stats
        )
        self.metrics.append(metric)
    
    def get_agent_performance(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Args:
            agent_id: Optional agent ID to filter by
            
        Returns:
            Agent performance metrics
        """
        if agent_id:
            if agent_id in self.agent_metrics:
                return asdict(self.agent_metrics[agent_id])
            return {}
        
        return {
            agent_id: asdict(metrics)
            for agent_id, metrics in self.agent_metrics.items()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data for visualization.
        
        Returns:
            Complete dashboard data
        """
        # Update resource stats
        self.record_resource_utilization()
        
        # Get recent metrics (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics
            if m.timestamp > cutoff_time
        ]
        
        # Calculate aggregate statistics
        total_executions = sum(m.total_executions for m in self.agent_metrics.values())
        total_successful = sum(m.successful_executions for m in self.agent_metrics.values())
        total_failed = sum(m.failed_executions for m in self.agent_metrics.values())
        overall_success_rate = total_successful / max(1, total_executions)
        
        # Get top error patterns
        top_errors = sorted(
            self.error_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Get circuit breaker states
        open_circuits = [
            agent_id for agent_id, state in self.circuit_breaker_states.items()
            if state == "OPEN"
        ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overview": {
                "total_executions": total_executions,
                "successful_executions": total_successful,
                "failed_executions": total_failed,
                "success_rate": overall_success_rate,
                "active_agents": len([m for m in self.agent_metrics.values() if m.total_executions > 0]),
                "open_circuit_breakers": len(open_circuits)
            },
            "agent_performance": self.get_agent_performance(),
            "resource_utilization": self.resource_stats,
            "load_balancer": self.load_balancer_stats,
            "circuit_breakers": {
                agent_id: state
                for agent_id, state in self.circuit_breaker_states.items()
            },
            "error_patterns": dict(top_errors),
            "recent_metrics": [
                {
                    "type": m.metric_type.value,
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags
                }
                for m in recent_metrics[-100:]  # Last 100 metrics
            ],
            "performance_trends": self._calculate_performance_trends()
        }
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        # Group metrics by time windows (last hour, 5-minute windows)
        now = datetime.now()
        windows = []
        for i in range(12):  # 12 windows of 5 minutes each
            window_start = now - timedelta(minutes=(i+1)*5)
            window_end = now - timedelta(minutes=i*5)
            windows.append((window_start, window_end))
        
        trends = {}
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.total_executions == 0:
                continue
            
            # Calculate average execution time per window
            window_averages = []
            for window_start, window_end in windows:
                # Filter recent execution times within window
                window_times = [
                    t for t in metrics.recent_execution_times
                    if window_start <= metrics.last_execution <= window_end
                ] if metrics.last_execution else []
                
                if window_times:
                    window_averages.append(statistics.mean(window_times))
                else:
                    window_averages.append(0.0)
            
            trends[agent_id] = {
                "execution_times": window_averages,
                "trend": "improving" if window_averages[0] < window_averages[-1] else "degrading"
            }
        
        return trends
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Health status with alerts
        """
        alerts = []
        
        # Check for high error rates
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.error_rate > 0.5 and metrics.total_executions > 10:
                alerts.append({
                    "severity": "high",
                    "type": "high_error_rate",
                    "agent_id": agent_id,
                    "error_rate": metrics.error_rate,
                    "message": f"Agent {agent_id} has high error rate: {metrics.error_rate:.1%}"
                })
        
        # Check for open circuit breakers
        for agent_id, state in self.circuit_breaker_states.items():
            if state == "OPEN":
                alerts.append({
                    "severity": "critical",
                    "type": "circuit_breaker_open",
                    "agent_id": agent_id,
                    "message": f"Circuit breaker OPEN for agent {agent_id}"
                })
        
        # Check for resource exhaustion
        resource_status = self.resource_allocator.get_resource_status()
        memory_usage = resource_status["allocated"]["used_memory_mb"] / max(1, resource_status["system"]["total_memory_mb"])
        if memory_usage > 0.9:
            alerts.append({
                "severity": "high",
                "type": "resource_exhaustion",
                "resource": "memory",
                "usage": memory_usage,
                "message": f"Memory usage critical: {memory_usage:.1%}"
            })
        
        # Check for slow performance
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.average_execution_time > 30.0 and metrics.total_executions > 5:
                alerts.append({
                    "severity": "medium",
                    "type": "slow_performance",
                    "agent_id": agent_id,
                    "average_time": metrics.average_execution_time,
                    "message": f"Agent {agent_id} is slow: {metrics.average_execution_time:.1f}s average"
                })
        
        return {
            "status": "healthy" if len(alerts) == 0 else "degraded" if len([a for a in alerts if a["severity"] == "critical"]) == 0 else "critical",
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format (json, csv)
            
        Returns:
            Exported metrics as string
        """
        if format == "json":
            return json.dumps(self.get_dashboard_data(), indent=2, default=str)
        elif format == "csv":
            # Simple CSV export
            lines = ["metric_type,name,value,timestamp"]
            for metric in self.metrics:
                lines.append(f"{metric.metric_type.value},{metric.name},{metric.value},{metric.timestamp.isoformat()}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global dashboard instance
_dashboard: Optional[ObservabilityDashboard] = None


def get_dashboard() -> ObservabilityDashboard:
    """Get or create global observability dashboard"""
    global _dashboard
    if _dashboard is None:
        _dashboard = ObservabilityDashboard()
    return _dashboard

