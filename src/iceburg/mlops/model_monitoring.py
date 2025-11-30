"""
Model Monitoring
Monitors model performance and metrics
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from collections import defaultdict
import time


class ModelMonitoring:
    """Monitors model performance"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alerts: List[Dict[str, Any]] = []
    
    def record_metric(
        self,
        model_name: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record a metric"""
        metric = {
            "model_name": model_name,
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.metrics[model_name].append(metric)
        return True
    
    def get_metrics(
        self,
        model_name: str,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics for a model"""
        if model_name not in self.metrics:
            return []
        
        metrics = self.metrics[model_name]
        
        if metric_name:
            metrics = [m for m in metrics if m["metric_name"] == metric_name]
        
        if start_time:
            metrics = [m for m in metrics if datetime.fromisoformat(m["timestamp"]) >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if datetime.fromisoformat(m["timestamp"]) <= end_time]
        
        return metrics
    
    def get_latest_metric(self, model_name: str, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get latest metric value"""
        metrics = self.get_metrics(model_name, metric_name)
        if not metrics:
            return None
        return sorted(metrics, key=lambda x: x["timestamp"])[-1]
    
    def get_average_metric(self, model_name: str, metric_name: str, window: int = 10) -> Optional[float]:
        """Get average metric over window"""
        metrics = self.get_metrics(model_name, metric_name)
        if not metrics:
            return None
        
        recent_metrics = sorted(metrics, key=lambda x: x["timestamp"])[-window:]
        values = [m["value"] for m in recent_metrics]
        return sum(values) / len(values) if values else None
    
    def check_threshold(
        self,
        model_name: str,
        metric_name: str,
        threshold: float,
        operator: str = ">"
    ) -> bool:
        """Check if metric exceeds threshold"""
        latest = self.get_latest_metric(model_name, metric_name)
        if not latest:
            return False
        
        value = latest["value"]
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        return False
    
    def create_alert(
        self,
        model_name: str,
        alert_type: str,
        message: str,
        severity: str = "warning"
    ) -> bool:
        """Create an alert"""
        alert = {
            "model_name": model_name,
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        self.alerts.append(alert)
        return True
    
    def get_alerts(
        self,
        model_name: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get alerts"""
        alerts = self.alerts
        
        if model_name:
            alerts = [a for a in alerts if a["model_name"] == model_name]
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        return alerts
    
    def clear_alerts(self, model_name: Optional[str] = None) -> int:
        """Clear alerts"""
        if model_name:
            count = len([a for a in self.alerts if a["model_name"] == model_name])
            self.alerts = [a for a in self.alerts if a["model_name"] != model_name]
            return count
        else:
            count = len(self.alerts)
            self.alerts = []
            return count

