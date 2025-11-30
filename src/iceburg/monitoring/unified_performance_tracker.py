"""
Unified Performance Tracking System for ICEBURG

Consolidates performance metrics from all ICEBURG systems into a single
unified tracking system with real-time monitoring, historical analysis,
and baseline comparison capabilities.
"""

import asyncio
import sqlite3
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Fallback: use basic statistics without pandas
    import statistics as stat_module
from collections import defaultdict, deque
import threading
import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    query_id: str
    response_time: float
    accuracy: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_rps: float
    cache_hit_rate: float
    error_rate: float
    agent_count: int
    parallel_execution: bool
    query_complexity: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    timestamp: float
    response_time: float
    accuracy: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_rps: float
    cache_hit_rate: float
    error_rate: float
    sample_size: int
    confidence_level: float


@dataclass
class PerformanceRegression:
    """Represents a performance regression."""
    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    severity: str  # "low", "medium", "high", "critical"
    timestamp: float
    query_id: Optional[str] = None


class UnifiedPerformanceTracker:
    """
    Unified performance tracking system for ICEBURG.
    
    Consolidates metrics from all systems and provides:
    - Real-time performance monitoring
    - Historical data analysis
    - Baseline establishment and comparison
    - Regression detection
    - Export capabilities for analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize unified performance tracker."""
        self.config = config or {}
        self.db_path = self.config.get("db_path", "data/performance_metrics.db")
        self.metrics_buffer = deque(maxlen=1000)  # In-memory buffer
        self.baseline: Optional[PerformanceBaseline] = None
        self.regressions: List[PerformanceRegression] = []
        
        # Performance thresholds
        self.thresholds = {
            "response_time": 30.0,  # seconds
            "memory_usage": 2048.0,  # MB
            "cpu_usage": 80.0,  # percent
            "error_rate": 5.0,  # percent
            "cache_hit_rate": 70.0,  # percent
        }
        
        # Initialize database
        self._init_database()
        
        # Buffer settings
        self.buffer_size = config.get("buffer_size", 1000) if config else 1000
        
        # Start background tasks
        self.tracking_active = False
        self.background_tasks = []
        
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        try:
            # Ensure data directory exists
            db_path = Path(self.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        query_id TEXT NOT NULL,
                        response_time REAL NOT NULL,
                        accuracy REAL NOT NULL,
                        memory_usage_mb REAL NOT NULL,
                        cpu_usage_percent REAL NOT NULL,
                        throughput_rps REAL NOT NULL,
                        cache_hit_rate REAL NOT NULL,
                        error_rate REAL NOT NULL,
                        agent_count INTEGER NOT NULL,
                        parallel_execution BOOLEAN NOT NULL,
                        query_complexity REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create baseline table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_baselines (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        response_time REAL NOT NULL,
                        accuracy REAL NOT NULL,
                        memory_usage_mb REAL NOT NULL,
                        cpu_usage_percent REAL NOT NULL,
                        throughput_rps REAL NOT NULL,
                        cache_hit_rate REAL NOT NULL,
                        error_rate REAL NOT NULL,
                        sample_size INTEGER NOT NULL,
                        confidence_level REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create regressions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_regressions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        baseline_value REAL NOT NULL,
                        current_value REAL NOT NULL,
                        regression_percent REAL NOT NULL,
                        severity TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        query_id TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON performance_metrics(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_query_id 
                    ON performance_metrics(query_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_success 
                    ON performance_metrics(success)
                """)
                
                conn.commit()
                logger.info("Performance tracking database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize performance database: {e}")
            raise
    
    async def start_tracking(self):
        """Start performance tracking."""
        if self.tracking_active:
            return
        
        self.tracking_active = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._buffer_flush_loop()),
            asyncio.create_task(self._regression_detection_loop()),
            asyncio.create_task(self._baseline_update_loop())
        ]
        
        logger.info("Performance tracking started")
    
    async def stop_tracking(self):
        """Stop performance tracking."""
        if not self.tracking_active:
            return
        
        self.tracking_active = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Flush remaining metrics
        await self._flush_metrics_buffer()
        
        self.background_tasks.clear()
        logger.info("Performance tracking stopped")
    
    def track_query_performance(self, 
                              query_id: str,
                              response_time: float,
                              accuracy: float,
                              resources: Dict[str, Any],
                              success: bool = True,
                              error_message: Optional[str] = None,
                              metadata: Dict[str, Any] = None) -> None:
        """Track performance metrics for a single query."""
        try:
            # Get system resources
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            # Calculate throughput (queries per second)
            throughput = 1.0 / response_time if response_time > 0 else 0.0
            
            # Extract metrics from resources
            memory_usage = resources.get("memory_usage_mb", memory_info.used / (1024 * 1024))
            cache_hit_rate = resources.get("cache_hit_rate", 0.0)
            error_rate = 0.0 if success else 100.0
            agent_count = resources.get("agent_count", 1)
            parallel_execution = resources.get("parallel_execution", False)
            query_complexity = resources.get("query_complexity", 0.5)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                query_id=query_id,
                response_time=response_time,
                accuracy=accuracy,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_percent,
                throughput_rps=throughput,
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate,
                agent_count=agent_count,
                parallel_execution=parallel_execution,
                query_complexity=query_complexity,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.metrics_buffer.append(metrics)
            
            # Flush immediately if buffer is full
            if len(self.metrics_buffer) >= self.buffer_size:
                # Schedule flush in background
                asyncio.create_task(self._flush_metrics_buffer())
            
            # Also flush immediately for testing purposes
            asyncio.create_task(self._flush_metrics_buffer())
            
            # Check for immediate regressions
            if self.baseline:
                self._check_immediate_regression(metrics)
            
        except Exception as e:
            logger.error(f"Error tracking query performance: {e}")
    
    async def _buffer_flush_loop(self):
        """Background loop to flush metrics buffer to database."""
        while self.tracking_active:
            try:
                await asyncio.sleep(2)  # Flush every 2 seconds for better responsiveness
                await self._flush_metrics_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in buffer flush loop: {e}")
    
    async def _flush_metrics_buffer(self):
        """Flush metrics buffer to database."""
        if not self.metrics_buffer:
            return
        
        try:
            # Ensure database is initialized
            self._init_database()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare batch insert
                metrics_data = []
                buffer_copy = list(self.metrics_buffer)  # Copy to avoid modification during iteration
                
                for metrics in buffer_copy:
                    metrics_data.append((
                        metrics.timestamp,
                        metrics.query_id,
                        metrics.response_time,
                        metrics.accuracy,
                        metrics.memory_usage_mb,
                        metrics.cpu_usage_percent,
                        metrics.throughput_rps,
                        metrics.cache_hit_rate,
                        metrics.error_rate,
                        metrics.agent_count,
                        metrics.parallel_execution,
                        metrics.query_complexity,
                        metrics.success,
                        metrics.error_message,
                        json.dumps(metrics.metadata) if metrics.metadata else "{}"
                    ))
                
                # Batch insert
                cursor.executemany("""
                    INSERT INTO performance_metrics (
                        timestamp, query_id, response_time, accuracy,
                        memory_usage_mb, cpu_usage_percent, throughput_rps,
                        cache_hit_rate, error_rate, agent_count,
                        parallel_execution, query_complexity, success,
                        error_message, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, metrics_data)
                
                conn.commit()
                
                # Clear buffer
                self.metrics_buffer.clear()
                
                logger.debug(f"Flushed {len(metrics_data)} metrics to database")
                
        except Exception as e:
            logger.error(f"Error flushing metrics buffer: {e}")
    
    async def _regression_detection_loop(self):
        """Background loop to detect performance regressions."""
        while self.tracking_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._detect_regressions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in regression detection loop: {e}")
    
    async def _detect_regressions(self):
        """Detect performance regressions compared to baseline."""
        if not self.baseline:
            return
        
        try:
            # Get recent metrics (last hour)
            recent_metrics = await self._get_recent_metrics(hours=1)
            if len(recent_metrics) < 10:  # Need minimum sample size
                return
            
            # Calculate current averages
            current_avg = self._calculate_averages(recent_metrics)
            
            # Check for regressions
            regressions = []
            
            for metric_name in ["response_time", "accuracy", "memory_usage_mb", 
                              "cpu_usage_percent", "throughput_rps", "cache_hit_rate", "error_rate"]:
                baseline_value = getattr(self.baseline, metric_name)
                current_value = current_avg.get(metric_name, 0)
                
                if baseline_value > 0:
                    regression_percent = ((current_value - baseline_value) / baseline_value) * 100
                    
                    # Determine severity
                    severity = "low"
                    if abs(regression_percent) > 50:
                        severity = "critical"
                    elif abs(regression_percent) > 25:
                        severity = "high"
                    elif abs(regression_percent) > 10:
                        severity = "medium"
                    
                    # Only flag as regression if it's a negative change
                    if regression_percent > 10:  # 10% worse
                        regression = PerformanceRegression(
                            metric_name=metric_name,
                            baseline_value=baseline_value,
                            current_value=current_value,
                            regression_percent=regression_percent,
                            severity=severity,
                            timestamp=time.time()
                        )
                        regressions.append(regression)
            
            # Store regressions
            if regressions:
                await self._store_regressions(regressions)
                self.regressions.extend(regressions)
                
                # Log critical regressions
                for regression in regressions:
                    if regression.severity == "critical":
                        logger.critical(f"CRITICAL PERFORMANCE REGRESSION: {regression.metric_name} "
                                      f"increased by {regression.regression_percent:.1f}%")
                    elif regression.severity == "high":
                        logger.warning(f"High performance regression: {regression.metric_name} "
                                     f"increased by {regression.regression_percent:.1f}%")
        
        except Exception as e:
            logger.error(f"Error detecting regressions: {e}")
    
    async def _baseline_update_loop(self):
        """Background loop to update performance baseline."""
        while self.tracking_active:
            try:
                await asyncio.sleep(3600)  # Update every hour
                await self._update_baseline()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in baseline update loop: {e}")
    
    async def _update_baseline(self):
        """Update performance baseline from recent metrics."""
        try:
            # Get metrics from last 24 hours
            recent_metrics = await self._get_recent_metrics(hours=24)
            if len(recent_metrics) < 50:  # Need sufficient data
                return
            
            # Calculate averages
            averages = self._calculate_averages(recent_metrics)
            
            # Create new baseline
            new_baseline = PerformanceBaseline(
                timestamp=time.time(),
                response_time=averages["response_time"],
                accuracy=averages["accuracy"],
                memory_usage_mb=averages["memory_usage_mb"],
                cpu_usage_percent=averages["cpu_usage_percent"],
                throughput_rps=averages["throughput_rps"],
                cache_hit_rate=averages["cache_hit_rate"],
                error_rate=averages["error_rate"],
                sample_size=len(recent_metrics),
                confidence_level=0.95
            )
            
            # Store baseline
            await self._store_baseline(new_baseline)
            self.baseline = new_baseline
            
            logger.info(f"Updated performance baseline with {len(recent_metrics)} samples")
            
        except Exception as e:
            logger.error(f"Error updating baseline: {e}")
    
    def _get_recent_metrics(self, hours: int = 1) -> List[PerformanceMetrics]:
        """Get recent metrics from database."""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, query_id, response_time, accuracy,
                           memory_usage_mb, cpu_usage_percent, throughput_rps,
                           cache_hit_rate, error_rate, agent_count,
                           parallel_execution, query_complexity, success,
                           error_message, metadata
                    FROM performance_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_time,))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append(PerformanceMetrics(
                        timestamp=row[0],
                        query_id=row[1],
                        response_time=row[2],
                        accuracy=row[3],
                        memory_usage_mb=row[4],
                        cpu_usage_percent=row[5],
                        throughput_rps=row[6],
                        cache_hit_rate=row[7],
                        error_rate=row[8],
                        agent_count=row[9],
                        parallel_execution=bool(row[10]),
                        query_complexity=row[11],
                        success=bool(row[12]),
                        error_message=row[13],
                        metadata=json.loads(row[14]) if row[14] else {}
                    ))
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error getting recent metrics: {e}")
            return []
    
    def _calculate_averages(self, metrics: List[PerformanceMetrics]) -> Dict[str, float]:
        """Calculate average metrics from a list of PerformanceMetrics."""
        if not metrics:
            return {}
        
        if PANDAS_AVAILABLE:
            return {
                "response_time": np.mean([m.response_time for m in metrics]),
                "accuracy": np.mean([m.accuracy for m in metrics]),
                "memory_usage_mb": np.mean([m.memory_usage_mb for m in metrics]),
                "cpu_usage_percent": np.mean([m.cpu_usage_percent for m in metrics]),
                "throughput_rps": np.mean([m.throughput_rps for m in metrics]),
                "cache_hit_rate": np.mean([m.cache_hit_rate for m in metrics]),
                "error_rate": np.mean([m.error_rate for m in metrics])
            }
        else:
            # Fallback: use basic statistics
            import statistics
            return {
                "response_time": statistics.mean([m.response_time for m in metrics]),
                "accuracy": statistics.mean([m.accuracy for m in metrics]),
                "memory_usage_mb": statistics.mean([m.memory_usage_mb for m in metrics]),
                "cpu_usage_percent": statistics.mean([m.cpu_usage_percent for m in metrics]),
                "throughput_rps": statistics.mean([m.throughput_rps for m in metrics]),
                "cache_hit_rate": statistics.mean([m.cache_hit_rate for m in metrics]),
                "error_rate": statistics.mean([m.error_rate for m in metrics])
            }
    
    def _check_immediate_regression(self, metrics: PerformanceMetrics):
        """Check for immediate regression in a single metric."""
        if not self.baseline:
            return
        
        # Check response time threshold
        if metrics.response_time > self.thresholds["response_time"]:
            logger.warning(f"High response time detected: {metrics.response_time:.2f}s "
                          f"(threshold: {self.thresholds['response_time']}s)")
        
        # Check memory usage threshold
        if metrics.memory_usage_mb > self.thresholds["memory_usage"]:
            logger.warning(f"High memory usage detected: {metrics.memory_usage_mb:.2f}MB "
                          f"(threshold: {self.thresholds['memory_usage']}MB)")
        
        # Check error rate threshold
        if metrics.error_rate > self.thresholds["error_rate"]:
            logger.warning(f"High error rate detected: {metrics.error_rate:.1f}% "
                          f"(threshold: {self.thresholds['error_rate']}%)")
    
    async def _store_baseline(self, baseline: PerformanceBaseline):
        """Store performance baseline in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_baselines (
                        timestamp, response_time, accuracy, memory_usage_mb,
                        cpu_usage_percent, throughput_rps, cache_hit_rate,
                        error_rate, sample_size, confidence_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    baseline.timestamp, baseline.response_time, baseline.accuracy,
                    baseline.memory_usage_mb, baseline.cpu_usage_percent,
                    baseline.throughput_rps, baseline.cache_hit_rate,
                    baseline.error_rate, baseline.sample_size, baseline.confidence_level
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing baseline: {e}")
    
    async def _store_regressions(self, regressions: List[PerformanceRegression]):
        """Store performance regressions in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                regression_data = []
                for regression in regressions:
                    regression_data.append((
                        regression.metric_name,
                        regression.baseline_value,
                        regression.current_value,
                        regression.regression_percent,
                        regression.severity,
                        regression.timestamp,
                        regression.query_id
                    ))
                
                cursor.executemany("""
                    INSERT INTO performance_regressions (
                        metric_name, baseline_value, current_value,
                        regression_percent, severity, timestamp, query_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, regression_data)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing regressions: {e}")
    
    def get_performance_baseline(self) -> Optional[PerformanceBaseline]:
        """Get current performance baseline."""
        return self.baseline
    
    def compare_to_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compare current metrics to baseline."""
        if not self.baseline:
            return {}
        
        comparison = {}
        for metric_name, current_value in current_metrics.items():
            baseline_value = getattr(self.baseline, metric_name, 0)
            if baseline_value > 0:
                comparison[metric_name] = ((current_value - baseline_value) / baseline_value) * 100
        
        return comparison
    
    def identify_performance_regressions(self) -> List[PerformanceRegression]:
        """Get identified performance regressions."""
        return self.regressions.copy()
    
    def export_metrics_for_analysis(self, hours: int = 24):
        """Export metrics as pandas DataFrame for analysis (or list of dicts if pandas unavailable)."""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM performance_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_time,))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                if PANDAS_AVAILABLE:
                    import pandas as pd
                    return pd.DataFrame(rows, columns=columns)
                else:
                    # Return as list of dicts if pandas unavailable
                    return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return [] if not PANDAS_AVAILABLE else None
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        try:
            recent_metrics = self._get_recent_metrics(hours)
            if not recent_metrics:
                return {"error": "No metrics available"}
            
            averages = self._calculate_averages(recent_metrics)
            
            # Calculate additional statistics
            response_times = [m.response_time for m in recent_metrics]
            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics) * 100
            
            if PANDAS_AVAILABLE:
                response_time_stats = {
                    "min": min(response_times),
                    "max": max(response_times),
                    "median": np.median(response_times),
                    "std": np.std(response_times)
                }
            else:
                import statistics
                response_time_stats = {
                    "min": min(response_times),
                    "max": max(response_times),
                    "median": statistics.median(response_times),
                    "std": statistics.stdev(response_times) if len(response_times) > 1 else 0.0
                }
            
            summary = {
                "time_period_hours": hours,
                "total_queries": len(recent_metrics),
                "success_rate": success_rate,
                "averages": averages,
                "response_time_stats": response_time_stats,
                "baseline_comparison": self.compare_to_baseline(averages),
                "regressions_count": len(self.regressions),
                "recent_regressions": [r for r in self.regressions if r.timestamp > time.time() - 3600]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Clean up old metrics to prevent database bloat."""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old metrics
                cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_time,))
                metrics_deleted = cursor.rowcount
                
                # Delete old regressions
                cursor.execute("DELETE FROM performance_regressions WHERE timestamp < ?", (cutoff_time,))
                regressions_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up {metrics_deleted} old metrics and {regressions_deleted} old regressions")
                
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")


# Global instance for easy access
_global_tracker: Optional[UnifiedPerformanceTracker] = None


def get_global_tracker() -> UnifiedPerformanceTracker:
    """Get global performance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = UnifiedPerformanceTracker()
    return _global_tracker


def track_query_performance(query_id: str, response_time: float, accuracy: float, 
                          resources: Dict[str, Any], success: bool = True, 
                          error_message: Optional[str] = None, 
                          metadata: Dict[str, Any] = None):
    """Convenience function to track query performance."""
    tracker = get_global_tracker()
    tracker.track_query_performance(query_id, response_time, accuracy, resources, 
                                  success, error_message, metadata)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_performance_tracker():
        # Create tracker
        tracker = UnifiedPerformanceTracker()
        
        # Start tracking
        await tracker.start_tracking()
        
        # Simulate some queries
        for i in range(10):
            tracker.track_query_performance(
                query_id=f"test_query_{i}",
                response_time=1.0 + (i * 0.1),
                accuracy=0.8 + (i * 0.01),
                resources={
                    "memory_usage_mb": 100 + i * 10,
                    "cache_hit_rate": 0.7 + (i * 0.02),
                    "agent_count": 3,
                    "parallel_execution": i % 2 == 0,
                    "query_complexity": 0.5 + (i * 0.05)
                },
                success=i < 8,  # 80% success rate
                metadata={"test": True, "iteration": i}
            )
        
        # Wait for buffer flush
        await asyncio.sleep(15)
        
        # Get summary
        summary = tracker.get_performance_summary()
        print("Performance Summary:")
        print(json.dumps(summary, indent=2))
        
        # Stop tracking
        await tracker.stop_tracking()
    
    # Run test
    asyncio.run(test_performance_tracker())
