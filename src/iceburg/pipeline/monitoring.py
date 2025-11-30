"""
ICEBURG Pipeline Monitoring

Comprehensive monitoring system for the financial analysis pipeline,
including performance metrics, health checks, and alerting.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from ..config import IceburgConfig

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    enable_health_checks: bool = True
    enable_performance_metrics: bool = True
    enable_alerting: bool = True
    health_check_interval: int = 60  # 1 minute
    performance_metrics_interval: int = 30  # 30 seconds
    alert_thresholds: Dict[str, float] = None
    max_alert_history: int = 1000
    enable_dashboard: bool = True
    dashboard_port: int = 8080


class PipelineMonitor:
    """
    Comprehensive monitoring system for the financial analysis pipeline.
    
    Provides real-time monitoring, health checks, performance metrics,
    and alerting capabilities.
    """
    
    def __init__(self, config: IceburgConfig, monitoring_config: MonitoringConfig = None):
        """Initialize pipeline monitor."""
        self.config = config
        self.monitoring_config = monitoring_config or MonitoringConfig()
        
        # Monitoring state
        self.monitoring_active = False
        self.health_status = {}
        self.performance_metrics = {}
        self.alert_history = []
        self.dashboard_data = {}
        
        # Initialize alert thresholds
        if self.monitoring_config.alert_thresholds is None:
            self.monitoring_config.alert_thresholds = {
                "cpu_usage": 80.0,
                "memory_usage": 80.0,
                "response_time": 5.0,
                "error_rate": 5.0,
                "queue_size": 100
            }
        
        # Initialize monitoring tasks
        self.monitoring_tasks = []
    
    async def start_monitoring(self):
        """Start monitoring system."""
        try:
            logger.info("Starting pipeline monitoring...")
            
            # Start health checks
            if self.monitoring_config.enable_health_checks:
                health_task = asyncio.create_task(self._health_check_loop())
                self.monitoring_tasks.append(health_task)
            
            # Start performance metrics collection
            if self.monitoring_config.enable_performance_metrics:
                metrics_task = asyncio.create_task(self._performance_metrics_loop())
                self.monitoring_tasks.append(metrics_task)
            
            # Start dashboard
            if self.monitoring_config.enable_dashboard:
                dashboard_task = asyncio.create_task(self._start_dashboard())
                self.monitoring_tasks.append(dashboard_task)
            
            # Activate monitoring
            self.monitoring_active = True
            
            logger.info("Pipeline monitoring started successfully")
        
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop monitoring system."""
        try:
            logger.info("Stopping pipeline monitoring...")
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Clear monitoring tasks
            self.monitoring_tasks.clear()
            
            # Deactivate monitoring
            self.monitoring_active = False
            
            logger.info("Pipeline monitoring stopped successfully")
        
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
            raise
    
    async def _health_check_loop(self):
        """Main health check loop."""
        while self.monitoring_active:
            try:
                # Perform health checks
                health_status = await self._perform_health_checks()
                
                # Update health status
                self.health_status.update(health_status)
                
                # Check for alerts
                await self._check_health_alerts(health_status)
                
                # Wait for next health check
                await asyncio.sleep(self.monitoring_config.health_check_interval)
            
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.monitoring_config.health_check_interval)
    
    async def _performance_metrics_loop(self):
        """Main performance metrics loop."""
        while self.monitoring_active:
            try:
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                
                # Update performance metrics
                self.performance_metrics.update(metrics)
                
                # Check for alerts
                await self._check_performance_alerts(metrics)
                
                # Wait for next metrics collection
                await asyncio.sleep(self.monitoring_config.performance_metrics_interval)
            
            except Exception as e:
                logger.error(f"Error in performance metrics loop: {e}")
                await asyncio.sleep(self.monitoring_config.performance_metrics_interval)
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive health checks."""
        try:
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy",
                "components": {}
            }
            
            # Check system resources
            system_health = await self._check_system_resources()
            health_status["components"]["system"] = system_health
            
            # Check pipeline components
            pipeline_health = await self._check_pipeline_components()
            health_status["components"]["pipeline"] = pipeline_health
            
            # Check integrations
            integration_health = await self._check_integrations()
            health_status["components"]["integrations"] = integration_health
            
            # Determine overall status
            overall_status = "healthy"
            for component, status in health_status["components"].items():
                if status.get("status") == "unhealthy":
                    overall_status = "unhealthy"
                    break
                elif status.get("status") == "degraded":
                    overall_status = "degraded"
            
            health_status["overall_status"] = overall_status
            
            return health_status
        
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # Mock system resource checks
            system_health = {
                "status": "healthy",
                "cpu_usage": 45.0,
                "memory_usage": 60.0,
                "disk_usage": 30.0,
                "network_latency": 10.0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check thresholds
            if system_health["cpu_usage"] > self.monitoring_config.alert_thresholds["cpu_usage"]:
                system_health["status"] = "degraded"
                system_health["alerts"] = ["High CPU usage"]
            
            if system_health["memory_usage"] > self.monitoring_config.alert_thresholds["memory_usage"]:
                system_health["status"] = "degraded"
                system_health["alerts"] = system_health.get("alerts", []) + ["High memory usage"]
            
            return system_health
        
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_pipeline_components(self) -> Dict[str, Any]:
        """Check pipeline component health."""
        try:
            # Mock pipeline component checks
            pipeline_health = {
                "status": "healthy",
                "data_pipeline": "active",
                "feature_engineering": "active",
                "quantum_rl": "active",
                "financial_ai": "active",
                "elite_trading": "active",
                "timestamp": datetime.now().isoformat()
            }
            
            # Check component status
            unhealthy_components = []
            for component, status in pipeline_health.items():
                if component != "status" and component != "timestamp" and status != "active":
                    unhealthy_components.append(component)
            
            if unhealthy_components:
                pipeline_health["status"] = "degraded"
                pipeline_health["alerts"] = [f"Unhealthy components: {', '.join(unhealthy_components)}"]
            
            return pipeline_health
        
        except Exception as e:
            logger.error(f"Error checking pipeline components: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_integrations(self) -> Dict[str, Any]:
        """Check integration health."""
        try:
            # Mock integration checks
            integration_health = {
                "status": "healthy",
                "quantum_rl_integration": "active",
                "financial_ai_integration": "active",
                "elite_trading_integration": "active",
                "timestamp": datetime.now().isoformat()
            }
            
            # Check integration status
            unhealthy_integrations = []
            for integration, status in integration_health.items():
                if integration != "status" and integration != "timestamp" and status != "active":
                    unhealthy_integrations.append(integration)
            
            if unhealthy_integrations:
                integration_health["status"] = "degraded"
                integration_health["alerts"] = [f"Unhealthy integrations: {', '.join(unhealthy_integrations)}"]
            
            return integration_health
        
        except Exception as e:
            logger.error(f"Error checking integrations: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "response_time": 1.5,  # seconds
                "throughput": 100.0,  # requests per minute
                "error_rate": 0.5,  # percentage
                "queue_size": 10,
                "memory_usage": 60.0,  # percentage
                "cpu_usage": 45.0,  # percentage
                "active_connections": 25,
                "cache_hit_rate": 85.0,  # percentage
                "quantum_advantage": 0.15,
                "financial_confidence": 0.8,
                "elite_trading_performance": 0.9
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _check_health_alerts(self, health_status: Dict[str, Any]):
        """Check for health-related alerts."""
        try:
            if health_status.get("overall_status") == "unhealthy":
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    "Pipeline Health Critical",
                    "Pipeline is in unhealthy state",
                    health_status
                )
            elif health_status.get("overall_status") == "degraded":
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Pipeline Health Degraded",
                    "Pipeline is in degraded state",
                    health_status
                )
        
        except Exception as e:
            logger.error(f"Error checking health alerts: {e}")
    
    async def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance-related alerts."""
        try:
            # Check response time
            if metrics.get("response_time", 0) > self.monitoring_config.alert_thresholds["response_time"]:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "High Response Time",
                    f"Response time is {metrics.get('response_time')}s",
                    metrics
                )
            
            # Check error rate
            if metrics.get("error_rate", 0) > self.monitoring_config.alert_thresholds["error_rate"]:
                await self._create_alert(
                    AlertLevel.ERROR,
                    "High Error Rate",
                    f"Error rate is {metrics.get('error_rate')}%",
                    metrics
                )
            
            # Check queue size
            if metrics.get("queue_size", 0) > self.monitoring_config.alert_thresholds["queue_size"]:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Large Queue Size",
                    f"Queue size is {metrics.get('queue_size')}",
                    metrics
                )
        
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    async def _create_alert(self, level: AlertLevel, title: str, message: str, data: Dict[str, Any]):
        """Create an alert."""
        try:
            alert = {
                "level": level.value,
                "title": title,
                "message": message,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to alert history
            self.alert_history.append(alert)
            
            # Keep only recent alerts
            if len(self.alert_history) > self.monitoring_config.max_alert_history:
                self.alert_history.pop(0)
            
            # Log alert
            if level == AlertLevel.CRITICAL:
                logger.critical(f"ALERT: {title} - {message}")
            elif level == AlertLevel.ERROR:
                logger.error(f"ALERT: {title} - {message}")
            elif level == AlertLevel.WARNING:
                logger.warning(f"ALERT: {title} - {message}")
            else:
                logger.info(f"ALERT: {title} - {message}")
        
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def _start_dashboard(self):
        """Start monitoring dashboard."""
        try:
            # Mock dashboard startup
            logger.info(f"Starting monitoring dashboard on port {self.monitoring_config.dashboard_port}")
            
            # Dashboard would be implemented with a web framework like FastAPI
            # For now, we'll just log that it's started
            while self.monitoring_active:
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status and metrics."""
        return {
            "monitoring_active": self.monitoring_active,
            "health_status": self.health_status,
            "performance_metrics": self.performance_metrics,
            "alert_count": len(self.alert_history),
            "recent_alerts": self.alert_history[-10:] if self.alert_history else []
        }
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            if not self.performance_metrics:
                return {"summary": "No performance data available"}
            
            # Calculate summary statistics
            response_times = [m.get("response_time", 0) for m in self.performance_metrics.values() if isinstance(m, dict)]
            error_rates = [m.get("error_rate", 0) for m in self.performance_metrics.values() if isinstance(m, dict)]
            throughputs = [m.get("throughput", 0) for m in self.performance_metrics.values() if isinstance(m, dict)]
            
            summary = {
                "avg_response_time": np.mean(response_times) if response_times else 0,
                "max_response_time": np.max(response_times) if response_times else 0,
                "avg_error_rate": np.mean(error_rates) if error_rates else 0,
                "max_error_rate": np.max(error_rates) if error_rates else 0,
                "avg_throughput": np.mean(throughputs) if throughputs else 0,
                "max_throughput": np.max(throughputs) if throughputs else 0,
                "total_metrics": len(self.performance_metrics)
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    async def clear_alert_history(self):
        """Clear alert history."""
        try:
            self.alert_history.clear()
            logger.info("Alert history cleared")
        
        except Exception as e:
            logger.error(f"Error clearing alert history: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Test pipeline monitoring
    config = IceburgConfig()
    monitoring_config = MonitoringConfig(
        enable_health_checks=True,
        enable_performance_metrics=True,
        enable_alerting=True,
        enable_dashboard=True
    )
    
    # Create monitor
    monitor = PipelineMonitor(config, monitoring_config)
    
    # Test monitoring
    import asyncio
    
    async def test_monitoring():
        # Start monitoring
        await monitor.start_monitoring()
        
        # Wait for some monitoring data
        await asyncio.sleep(5)
        
        # Get monitoring status
        status = monitor.get_monitoring_status()
        print(f"Monitoring status: {status}")
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        print(f"Performance summary: {summary}")
        
        # Get alert history
        alerts = monitor.get_alert_history()
        print(f"Alert history: {alerts}")
        
        # Stop monitoring
        await monitor.stop_monitoring()
    
    # Run test
    asyncio.run(test_monitoring())
