"""
Elite Financial AI Metrics Integration for ICEBURG

This module integrates performance metrics from Elite Financial AI operations
with ICEBURG's system metrics tracking, enabling comprehensive monitoring
of quantum circuits, RL training, and financial predictions.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
import time
import uuid
import asyncio
from dataclasses import dataclass, asdict

from ..database.elite_financial_schema import EliteFinancialSchema
from ..integration.elite_database_integration import EliteDatabaseIntegration
from ..integration.elite_memory_integration import EliteMemoryIntegration

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data class."""
    metric_id: str
    metric_name: str
    metric_value: float
    metric_type: str
    timestamp: datetime
    context: Dict[str, Any]
    tags: List[str]
    source: str
    unit: str = ""


@dataclass
class SystemMetrics:
    """System metrics data class."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    gpu_usage: Optional[float] = None
    quantum_device_status: Optional[str] = None


class EliteMetricsIntegration:
    """
    Elite Financial AI metrics integration with ICEBURG.
    
    Provides comprehensive performance metrics tracking for Elite Financial AI,
    including quantum circuits, RL training, and financial predictions
    with ICEBURG system integration.
    """
    
    def __init__(self, db_path: str = "iceburg_unified.db", 
                 memory_dir: str = "data/memory", vector_dir: str = "data/vector_store"):
        """
        Initialize Elite Financial AI metrics integration.
        
        Args:
            db_path: Path to ICEBURG unified database
            memory_dir: Directory for memory storage
            vector_dir: Directory for vector storage
        """
        self.db_path = db_path
        self.memory_dir = memory_dir
        self.vector_dir = vector_dir
        self.database_integration = EliteDatabaseIntegration(db_path)
        self.memory_integration = EliteMemoryIntegration(memory_dir, vector_dir)
        self.metrics_buffer = []
        self.system_metrics_buffer = []
        self.metrics_stats = {}
        self._setup_metrics_tables()
    
    def _setup_metrics_tables(self):
        """Setup metrics tables in database."""
        try:
            cursor = self.database_integration.connection.cursor()
            
            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elite_performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    source TEXT NOT NULL,
                    unit TEXT
                )
            """)
            
            # Create system metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elite_system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    disk_usage REAL NOT NULL,
                    network_usage REAL NOT NULL,
                    gpu_usage REAL,
                    quantum_device_status TEXT
                )
            """)
            
            # Create metrics aggregation table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elite_metrics_aggregation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    aggregation_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    time_window TEXT NOT NULL
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metric_name ON elite_performance_metrics(metric_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON elite_performance_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON elite_system_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_aggregation_metric ON elite_metrics_aggregation(metric_name)")
            
            self.database_integration.connection.commit()
            logger.info("Elite Financial AI metrics tables created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up metrics tables: {e}")
            raise
    
    def track_quantum_circuit_metrics(self, circuit_id: str, execution_time: float,
                                    n_qubits: int, n_layers: int, device: str,
                                    success: bool, result: List[float],
                                    context: Dict[str, Any] = None) -> str:
        """
        Track quantum circuit performance metrics.
        
        Args:
            circuit_id: Quantum circuit identifier
            execution_time: Circuit execution time
            n_qubits: Number of qubits
            n_layers: Number of layers
            device: Quantum device used
            success: Whether execution was successful
            result: Circuit execution result
            context: Additional context
            
        Returns:
            Metric ID
        """
        try:
            metric_id = f"quantum_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now()
            
            # Create context
            if context is None:
                context = {}
            context.update({
                "circuit_id": circuit_id,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "device": device,
                "success": success
            })
            
            # Track execution time
            execution_metric = PerformanceMetric(
                metric_id=f"{metric_id}_execution_time",
                metric_name="quantum_execution_time",
                metric_value=execution_time,
                metric_type="performance",
                timestamp=timestamp,
                context=context,
                tags=["quantum", "execution_time", "performance"],
                source="quantum_circuit",
                unit="seconds"
            )
            self._store_metric(execution_metric)
            
            # Track success rate
            success_metric = PerformanceMetric(
                metric_id=f"{metric_id}_success",
                metric_name="quantum_success_rate",
                metric_value=1.0 if success else 0.0,
                metric_type="reliability",
                timestamp=timestamp,
                context=context,
                tags=["quantum", "success_rate", "reliability"],
                source="quantum_circuit",
                unit="ratio"
            )
            self._store_metric(success_metric)
            
            # Track circuit complexity
            complexity_metric = PerformanceMetric(
                metric_id=f"{metric_id}_complexity",
                metric_name="quantum_circuit_complexity",
                metric_value=n_qubits * n_layers,
                metric_type="complexity",
                timestamp=timestamp,
                context=context,
                tags=["quantum", "complexity", "circuit"],
                source="quantum_circuit",
                unit="operations"
            )
            self._store_metric(complexity_metric)
            
            # Track result magnitude
            if result:
                result_magnitude = np.linalg.norm(result)
                magnitude_metric = PerformanceMetric(
                    metric_id=f"{metric_id}_magnitude",
                    metric_name="quantum_result_magnitude",
                    metric_value=result_magnitude,
                    metric_type="output",
                    timestamp=timestamp,
                    context=context,
                    tags=["quantum", "result", "magnitude"],
                    source="quantum_circuit",
                    unit="magnitude"
                )
                self._store_metric(magnitude_metric)
            
            logger.debug(f"Tracked quantum circuit metrics: {circuit_id}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error tracking quantum circuit metrics: {e}")
            return f"error_{uuid.uuid4().hex[:8]}"
    
    def track_rl_training_metrics(self, episode_id: str, reward: float, steps: int,
                                 algorithm: str, convergence_metric: float,
                                 breakthrough_detected: bool, training_time: float,
                                 context: Dict[str, Any] = None) -> str:
        """
        Track RL training performance metrics.
        
        Args:
            episode_id: RL episode identifier
            reward: Episode reward
            steps: Number of steps
            algorithm: RL algorithm used
            convergence_metric: Convergence metric
            breakthrough_detected: Whether breakthrough was detected
            training_time: Training time
            context: Additional context
            
        Returns:
            Metric ID
        """
        try:
            metric_id = f"rl_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now()
            
            # Create context
            if context is None:
                context = {}
            context.update({
                "episode_id": episode_id,
                "algorithm": algorithm,
                "breakthrough_detected": breakthrough_detected
            })
            
            # Track reward
            reward_metric = PerformanceMetric(
                metric_id=f"{metric_id}_reward",
                metric_name="rl_episode_reward",
                metric_value=reward,
                metric_type="performance",
                timestamp=timestamp,
                context=context,
                tags=["rl", "reward", "performance"],
                source="rl_training",
                unit="reward"
            )
            self._store_metric(reward_metric)
            
            # Track steps
            steps_metric = PerformanceMetric(
                metric_id=f"{metric_id}_steps",
                metric_name="rl_episode_steps",
                metric_value=float(steps),
                metric_type="efficiency",
                timestamp=timestamp,
                context=context,
                tags=["rl", "steps", "efficiency"],
                source="rl_training",
                unit="steps"
            )
            self._store_metric(steps_metric)
            
            # Track convergence
            if convergence_metric is not None:
                convergence_metric_obj = PerformanceMetric(
                    metric_id=f"{metric_id}_convergence",
                    metric_name="rl_convergence",
                    metric_value=convergence_metric,
                    metric_type="convergence",
                    timestamp=timestamp,
                    context=context,
                    tags=["rl", "convergence", "training"],
                    source="rl_training",
                    unit="ratio"
                )
                self._store_metric(convergence_metric_obj)
            
            # Track training time
            training_time_metric = PerformanceMetric(
                metric_id=f"{metric_id}_training_time",
                metric_name="rl_training_time",
                metric_value=training_time,
                metric_type="performance",
                timestamp=timestamp,
                context=context,
                tags=["rl", "training_time", "performance"],
                source="rl_training",
                unit="seconds"
            )
            self._store_metric(training_time_metric)
            
            # Track breakthrough
            breakthrough_metric = PerformanceMetric(
                metric_id=f"{metric_id}_breakthrough",
                metric_name="rl_breakthrough_detected",
                metric_value=1.0 if breakthrough_detected else 0.0,
                metric_type="breakthrough",
                timestamp=timestamp,
                context=context,
                tags=["rl", "breakthrough", "discovery"],
                source="rl_training",
                unit="boolean"
            )
            self._store_metric(breakthrough_metric)
            
            logger.debug(f"Tracked RL training metrics: {episode_id}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error tracking RL training metrics: {e}")
            return f"error_{uuid.uuid4().hex[:8]}"
    
    def track_financial_prediction_metrics(self, prediction_id: str, symbol: str,
                                         prediction_type: str, value: float,
                                         confidence: float, accuracy: float,
                                         model_type: str, prediction_time: float,
                                         context: Dict[str, Any] = None) -> str:
        """
        Track financial prediction performance metrics.
        
        Args:
            prediction_id: Financial prediction identifier
            symbol: Financial symbol
            prediction_type: Type of prediction
            value: Predicted value
            confidence: Prediction confidence
            accuracy: Prediction accuracy
            model_type: Model type used
            prediction_time: Prediction time
            context: Additional context
            
        Returns:
            Metric ID
        """
        try:
            metric_id = f"financial_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now()
            
            # Create context
            if context is None:
                context = {}
            context.update({
                "prediction_id": prediction_id,
                "symbol": symbol,
                "prediction_type": prediction_type,
                "model_type": model_type
            })
            
            # Track confidence
            confidence_metric = PerformanceMetric(
                metric_id=f"{metric_id}_confidence",
                metric_name="financial_prediction_confidence",
                metric_value=confidence,
                metric_type="reliability",
                timestamp=timestamp,
                context=context,
                tags=["financial", "confidence", "reliability"],
                source="financial_prediction",
                unit="ratio"
            )
            self._store_metric(confidence_metric)
            
            # Track accuracy
            if accuracy is not None:
                accuracy_metric = PerformanceMetric(
                    metric_id=f"{metric_id}_accuracy",
                    metric_name="financial_prediction_accuracy",
                    metric_value=accuracy,
                    metric_type="performance",
                    timestamp=timestamp,
                    context=context,
                    tags=["financial", "accuracy", "performance"],
                    source="financial_prediction",
                    unit="ratio"
                )
                self._store_metric(accuracy_metric)
            
            # Track prediction time
            prediction_time_metric = PerformanceMetric(
                metric_id=f"{metric_id}_prediction_time",
                metric_name="financial_prediction_time",
                metric_value=prediction_time,
                metric_type="performance",
                timestamp=timestamp,
                context=context,
                tags=["financial", "prediction_time", "performance"],
                source="financial_prediction",
                unit="seconds"
            )
            self._store_metric(prediction_time_metric)
            
            # Track value magnitude
            value_magnitude = abs(value)
            magnitude_metric = PerformanceMetric(
                metric_id=f"{metric_id}_value_magnitude",
                metric_name="financial_prediction_value_magnitude",
                metric_value=value_magnitude,
                metric_type="output",
                timestamp=timestamp,
                context=context,
                tags=["financial", "value", "magnitude"],
                source="financial_prediction",
                unit="value"
            )
            self._store_metric(magnitude_metric)
            
            logger.debug(f"Tracked financial prediction metrics: {prediction_id}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error tracking financial prediction metrics: {e}")
            return f"error_{uuid.uuid4().hex[:8]}"
    
    def track_system_metrics(self, cpu_usage: float, memory_usage: float,
                           disk_usage: float, network_usage: float,
                           gpu_usage: Optional[float] = None,
                           quantum_device_status: Optional[str] = None) -> str:
        """
        Track system performance metrics.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            disk_usage: Disk usage percentage
            network_usage: Network usage percentage
            gpu_usage: GPU usage percentage
            quantum_device_status: Quantum device status
            
        Returns:
            Metric ID
        """
        try:
            metric_id = f"system_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now()
            
            # Create system metrics
            system_metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_usage=network_usage,
                gpu_usage=gpu_usage,
                quantum_device_status=quantum_device_status
            )
            
            # Store system metrics
            self._store_system_metrics(system_metrics)
            
            # Track individual metrics
            metrics = [
                ("cpu_usage", cpu_usage, "system"),
                ("memory_usage", memory_usage, "system"),
                ("disk_usage", disk_usage, "system"),
                ("network_usage", network_usage, "system")
            ]
            
            if gpu_usage is not None:
                metrics.append(("gpu_usage", gpu_usage, "system"))
            
            for metric_name, metric_value, metric_type in metrics:
                metric = PerformanceMetric(
                    metric_id=f"{metric_id}_{metric_name}",
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metric_type=metric_type,
                    timestamp=timestamp,
                    context={"system_metrics": True},
                    tags=["system", metric_name, "performance"],
                    source="system_monitoring",
                    unit="percentage"
                )
                self._store_metric(metric)
            
            logger.debug(f"Tracked system metrics: {metric_id}")
            return metric_id
            
        except Exception as e:
            logger.error(f"Error tracking system metrics: {e}")
            return f"error_{uuid.uuid4().hex[:8]}"
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store performance metric in database."""
        try:
            cursor = self.database_integration.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO elite_performance_metrics
                (metric_id, metric_name, metric_value, metric_type, timestamp,
                 context, tags, source, unit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_id,
                metric.metric_name,
                metric.metric_value,
                metric.metric_type,
                metric.timestamp.isoformat(),
                json.dumps(metric.context),
                json.dumps(metric.tags),
                metric.source,
                metric.unit
            ))
            self.database_integration.connection.commit()
            
            # Add to buffer for aggregation
            self.metrics_buffer.append(metric)
            
        except Exception as e:
            logger.error(f"Error storing metric: {e}")
    
    def _store_system_metrics(self, system_metrics: SystemMetrics):
        """Store system metrics in database."""
        try:
            cursor = self.database_integration.connection.cursor()
            cursor.execute("""
                INSERT INTO elite_system_metrics
                (timestamp, cpu_usage, memory_usage, disk_usage, network_usage,
                 gpu_usage, quantum_device_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                system_metrics.timestamp.isoformat(),
                system_metrics.cpu_usage,
                system_metrics.memory_usage,
                system_metrics.disk_usage,
                system_metrics.network_usage,
                system_metrics.gpu_usage,
                system_metrics.quantum_device_status
            ))
            self.database_integration.connection.commit()
            
            # Add to buffer
            self.system_metrics_buffer.append(system_metrics)
            
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
    
    def get_metrics_summary(self, time_window: timedelta = timedelta(hours=24),
                           metric_names: List[str] = None) -> Dict[str, Any]:
        """
        Get metrics summary for specified time window.
        
        Args:
            time_window: Time window for summary
            metric_names: Specific metric names to include
            
        Returns:
            Metrics summary
        """
        try:
            start_time = datetime.now() - time_window
            
            cursor = self.database_integration.connection.cursor()
            
            # Build query
            if metric_names:
                placeholders = ','.join(['?' for _ in metric_names])
                query = f"""
                    SELECT metric_name, AVG(metric_value) as avg_value,
                           MIN(metric_value) as min_value, MAX(metric_value) as max_value,
                           COUNT(*) as count
                    FROM elite_performance_metrics
                    WHERE timestamp >= ? AND metric_name IN ({placeholders})
                    GROUP BY metric_name
                """
                params = [start_time.isoformat()] + metric_names
            else:
                query = """
                    SELECT metric_name, AVG(metric_value) as avg_value,
                           MIN(metric_value) as min_value, MAX(metric_value) as max_value,
                           COUNT(*) as count
                    FROM elite_performance_metrics
                    WHERE timestamp >= ?
                    GROUP BY metric_name
                """
                params = [start_time.isoformat()]
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Format results
            summary = {}
            for row in rows:
                metric_name = row[0]
                summary[metric_name] = {
                    "average": row[1],
                    "minimum": row[2],
                    "maximum": row[3],
                    "count": row[4]
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}
    
    def get_system_metrics_summary(self, time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """
        Get system metrics summary.
        
        Args:
            time_window: Time window for summary
            
        Returns:
            System metrics summary
        """
        try:
            start_time = datetime.now() - time_window
            
            cursor = self.database_integration.connection.cursor()
            cursor.execute("""
                SELECT AVG(cpu_usage) as avg_cpu, AVG(memory_usage) as avg_memory,
                       AVG(disk_usage) as avg_disk, AVG(network_usage) as avg_network,
                       AVG(gpu_usage) as avg_gpu, COUNT(*) as count
                FROM elite_system_metrics
                WHERE timestamp >= ?
            """, (start_time.isoformat(),))
            
            row = cursor.fetchone()
            if row:
                summary = {
                    "cpu_usage": {
                        "average": row[0],
                        "count": row[5]
                    },
                    "memory_usage": {
                        "average": row[1],
                        "count": row[5]
                    },
                    "disk_usage": {
                        "average": row[2],
                        "count": row[5]
                    },
                    "network_usage": {
                        "average": row[3],
                        "count": row[5]
                    },
                    "gpu_usage": {
                        "average": row[4],
                        "count": row[5]
                    }
                }
                return summary
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting system metrics summary: {e}")
            return {}
    
    def aggregate_metrics(self, aggregation_type: str = "hourly",
                         time_window: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """
        Aggregate metrics for specified time window.
        
        Args:
            aggregation_type: Type of aggregation (hourly, daily, weekly)
            time_window: Time window for aggregation
            
        Returns:
            Aggregated metrics
        """
        try:
            start_time = datetime.now() - time_window
            
            cursor = self.database_integration.connection.cursor()
            
            # Get metrics for aggregation
            cursor.execute("""
                SELECT metric_name, metric_value, timestamp
                FROM elite_performance_metrics
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, (start_time.isoformat(),))
            
            rows = cursor.fetchall()
            
            # Group by metric name
            metrics_by_name = {}
            for row in rows:
                metric_name = row[0]
                if metric_name not in metrics_by_name:
                    metrics_by_name[metric_name] = []
                metrics_by_name[metric_name].append({
                    "value": row[1],
                    "timestamp": datetime.fromisoformat(row[2])
                })
            
            # Aggregate metrics
            aggregated = {}
            for metric_name, values in metrics_by_name.items():
                if aggregation_type == "hourly":
                    aggregated[metric_name] = self._aggregate_hourly(values)
                elif aggregation_type == "daily":
                    aggregated[metric_name] = self._aggregate_daily(values)
                elif aggregation_type == "weekly":
                    aggregated[metric_name] = self._aggregate_weekly(values)
                else:
                    aggregated[metric_name] = self._aggregate_hourly(values)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            return {}
    
    def _aggregate_hourly(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics by hour."""
        try:
            # Group by hour
            hourly_groups = {}
            for value in values:
                hour_key = value["timestamp"].replace(minute=0, second=0, microsecond=0)
                if hour_key not in hourly_groups:
                    hourly_groups[hour_key] = []
                hourly_groups[hour_key].append(value["value"])
            
            # Calculate hourly aggregates
            hourly_aggregates = {}
            for hour, hour_values in hourly_groups.items():
                hourly_aggregates[hour.isoformat()] = {
                    "average": np.mean(hour_values),
                    "minimum": np.min(hour_values),
                    "maximum": np.max(hour_values),
                    "count": len(hour_values)
                }
            
            return hourly_aggregates
            
        except Exception as e:
            logger.error(f"Error aggregating hourly metrics: {e}")
            return {}
    
    def _aggregate_daily(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics by day."""
        try:
            # Group by day
            daily_groups = {}
            for value in values:
                day_key = value["timestamp"].replace(hour=0, minute=0, second=0, microsecond=0)
                if day_key not in daily_groups:
                    daily_groups[day_key] = []
                daily_groups[day_key].append(value["value"])
            
            # Calculate daily aggregates
            daily_aggregates = {}
            for day, day_values in daily_groups.items():
                daily_aggregates[day.isoformat()] = {
                    "average": np.mean(day_values),
                    "minimum": np.min(day_values),
                    "maximum": np.max(day_values),
                    "count": len(day_values)
                }
            
            return daily_aggregates
            
        except Exception as e:
            logger.error(f"Error aggregating daily metrics: {e}")
            return {}
    
    def _aggregate_weekly(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics by week."""
        try:
            # Group by week
            weekly_groups = {}
            for value in values:
                week_key = value["timestamp"].replace(hour=0, minute=0, second=0, microsecond=0)
                # Get start of week (Monday)
                days_since_monday = week_key.weekday()
                week_start = week_key - timedelta(days=days_since_monday)
                
                if week_start not in weekly_groups:
                    weekly_groups[week_start] = []
                weekly_groups[week_start].append(value["value"])
            
            # Calculate weekly aggregates
            weekly_aggregates = {}
            for week, week_values in weekly_groups.items():
                weekly_aggregates[week.isoformat()] = {
                    "average": np.mean(week_values),
                    "minimum": np.min(week_values),
                    "maximum": np.max(week_values),
                    "count": len(week_values)
                }
            
            return weekly_aggregates
            
        except Exception as e:
            logger.error(f"Error aggregating weekly metrics: {e}")
            return {}
    
    def get_metrics_stats(self) -> Dict[str, Any]:
        """Get metrics statistics."""
        return self.metrics_stats.copy()
    
    def close(self):
        """Close metrics integration."""
        if self.database_integration:
            self.database_integration.close()
        if self.memory_integration:
            self.memory_integration.close()


# Example usage and testing
if __name__ == "__main__":
    # Test Elite Financial AI metrics integration
    metrics = EliteMetricsIntegration()
    
    # Test quantum circuit metrics tracking
    quantum_metric_id = metrics.track_quantum_circuit_metrics(
        circuit_id="test_circuit_001",
        execution_time=0.123,
        n_qubits=4,
        n_layers=2,
        device="default.qubit",
        success=True,
        result=[0.5, -0.3, 0.8, -0.1]
    )
    # Tracked quantum circuit metrics
    
    # Test RL training metrics tracking
    rl_metric_id = metrics.track_rl_training_metrics(
        episode_id="test_episode_001",
        reward=150.5,
        steps=1000,
        algorithm="PPO",
        convergence_metric=0.85,
        breakthrough_detected=False,
        training_time=0.5
    )
    # Tracked RL training metrics
    
    # Test financial prediction metrics tracking
    financial_metric_id = metrics.track_financial_prediction_metrics(
        prediction_id="test_prediction_001",
        symbol="AAPL",
        prediction_type="price",
        value=150.25,
        confidence=0.85,
        accuracy=0.87,
        model_type="quantum_rl",
        prediction_time=0.1
    )
    # Tracked financial prediction metrics
    
    # Test system metrics tracking
    system_metric_id = metrics.track_system_metrics(
        cpu_usage=45.2,
        memory_usage=67.8,
        disk_usage=23.1,
        network_usage=12.5,
        gpu_usage=34.6,
        quantum_device_status="active"
    )
    # Tracked system metrics
    
    # Test metrics summary
    summary = metrics.get_metrics_summary()
    
    # Test system metrics summary
    system_summary = metrics.get_system_metrics_summary()
    
    # Test metrics aggregation
    aggregated = metrics.aggregate_metrics(aggregation_type="hourly")
    
    # Test metrics statistics
    stats = metrics.get_metrics_stats()
    
    # Close metrics integration
    metrics.close()
