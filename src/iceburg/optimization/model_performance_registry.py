"""
Model Performance Tracking Registry
Comprehensive tracking and analysis of AI model performance across ICEBURG

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from ..config import IceburgConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceRecord:
    """Record of model performance for a specific task"""
    record_id: str
    model_name: str
    model_version: str
    task_type: str
    execution_time: float
    success: bool
    quality_score: float
    resource_usage: Dict[str, float]
    input_size: int
    output_size: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ModelCapabilityProfile:
    """Profile of model capabilities and performance"""
    model_name: str
    model_version: str
    capability_scores: Dict[str, float]
    performance_metrics: Dict[str, float]
    usage_statistics: Dict[str, int]
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceBenchmark:
    """Benchmark for model performance comparison"""
    benchmark_id: str
    benchmark_name: str
    task_type: str
    test_cases: List[Dict[str, Any]]
    baseline_metrics: Dict[str, float]
    model_results: Dict[str, Dict[str, float]]
    created_at: float = field(default_factory=time.time)

class ModelPerformanceRegistry:
    """
    Comprehensive registry for tracking and analyzing AI model performance
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/model_performance")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.records_file = self.data_dir / "performance_records.json"
        self.profiles_file = self.data_dir / "model_profiles.json"
        self.benchmarks_file = self.data_dir / "performance_benchmarks.json"
        
        # Data structures
        self.performance_records: List[ModelPerformanceRecord] = []
        self.model_profiles: Dict[str, ModelCapabilityProfile] = {}
        self.performance_benchmarks: Dict[str, PerformanceBenchmark] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.model_statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Threading for concurrent operations
        self.registry_lock = threading.Lock()
        self.analysis_executor = ThreadPoolExecutor(max_workers=2)
        
        # Load existing data
        self._load_data()
        self._initialize_default_benchmarks()
        
        logger.info("📊 Model Performance Registry initialized")
    
    def record_model_performance(
        self,
        model_name: str,
        model_version: str,
        task_type: str,
        execution_time: float,
        success: bool,
        quality_score: float,
        resource_usage: Dict[str, float],
        input_size: int,
        output_size: int,
        error_message: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Record model performance for analysis"""
        
        if metadata is None:
            metadata = {}
        
        record_id = f"{model_name}_{model_version}_{task_type}_{int(time.time())}"
        
        record = ModelPerformanceRecord(
            record_id=record_id,
            model_name=model_name,
            model_version=model_version,
            task_type=task_type,
            execution_time=execution_time,
            success=success,
            quality_score=quality_score,
            resource_usage=resource_usage,
            input_size=input_size,
            output_size=output_size,
            error_message=error_message,
            metadata=metadata
        )
        
        with self.registry_lock:
            self.performance_records.append(record)
            
            # Keep only last 10000 records
            if len(self.performance_records) > 10000:
                self.performance_records = self.performance_records[-10000:]
        
        # Update model profile asynchronously
        self.analysis_executor.submit(self._update_model_profile, record)
        
        # Update performance history
        self._update_performance_history(record)
        
        # Save data
        self._save_data()
        
        logger.info(f"📝 Recorded performance for {model_name} v{model_version}: {task_type} - {execution_time:.2f}s, success: {success}")
        
        return record_id
    
    def _update_model_profile(self, record: ModelPerformanceRecord) -> None:
        """Update model capability profile based on performance record"""
        
        model_key = f"{record.model_name}_{record.model_version}"
        
        if model_key not in self.model_profiles:
            # Create new profile
            profile = ModelCapabilityProfile(
                model_name=record.model_name,
                model_version=record.model_version,
                capability_scores={},
                performance_metrics={},
                usage_statistics={}
            )
            self.model_profiles[model_key] = profile
        
        profile = self.model_profiles[model_key]
        
        # Update capability scores
        if record.task_type not in profile.capability_scores:
            profile.capability_scores[record.task_type] = []
        
        profile.capability_scores[record.task_type].append(record.quality_score)
        
        # Keep only last 100 scores per task type
        if len(profile.capability_scores[record.task_type]) > 100:
            profile.capability_scores[record.task_type] = profile.capability_scores[record.task_type][-100:]
        
        # Update performance metrics
        if "execution_time" not in profile.performance_metrics:
            profile.performance_metrics["execution_time"] = []
        if "success_rate" not in profile.performance_metrics:
            profile.performance_metrics["success_rate"] = []
        if "resource_efficiency" not in profile.performance_metrics:
            profile.performance_metrics["resource_efficiency"] = []
        
        profile.performance_metrics["execution_time"].append(record.execution_time)
        profile.performance_metrics["success_rate"].append(1.0 if record.success else 0.0)
        
        # Calculate resource efficiency
        resource_efficiency = self._calculate_resource_efficiency(record.resource_usage, record.execution_time)
        profile.performance_metrics["resource_efficiency"].append(resource_efficiency)
        
        # Keep only last 1000 metrics
        for metric_name in profile.performance_metrics:
            if len(profile.performance_metrics[metric_name]) > 1000:
                profile.performance_metrics[metric_name] = profile.performance_metrics[metric_name][-1000:]
        
        # Update usage statistics
        if record.task_type not in profile.usage_statistics:
            profile.usage_statistics[record.task_type] = 0
        profile.usage_statistics[record.task_type] += 1
        
        profile.last_updated = time.time()
    
    def _calculate_resource_efficiency(self, resource_usage: Dict[str, float], execution_time: float) -> float:
        """Calculate resource efficiency score"""
        
        # Simple efficiency calculation based on resource usage and execution time
        cpu_usage = resource_usage.get("cpu", 0.0)
        memory_usage = resource_usage.get("memory", 0.0)
        
        # Efficiency = (1 - resource_usage) / execution_time
        resource_penalty = (cpu_usage + memory_usage) / 2.0
        efficiency = (1.0 - resource_penalty) / max(0.1, execution_time)
        
        return min(1.0, max(0.0, efficiency))
    
    def _update_performance_history(self, record: ModelPerformanceRecord) -> None:
        """Update performance history for analysis"""
        
        # Update overall performance history
        self.performance_history["execution_time"].append(record.execution_time)
        self.performance_history["quality_score"].append(record.quality_score)
        self.performance_history["success_rate"].append(1.0 if record.success else 0.0)
        
        # Update model-specific statistics
        model_key = f"{record.model_name}_{record.model_version}"
        if model_key not in self.model_statistics:
            self.model_statistics[model_key] = {
                "total_executions": 0,
                "successful_executions": 0,
                "avg_execution_time": 0.0,
                "avg_quality_score": 0.0,
                "task_types": set()
            }
        
        stats = self.model_statistics[model_key]
        stats["total_executions"] += 1
        if record.success:
            stats["successful_executions"] += 1
        
        # Update running averages
        stats["avg_execution_time"] = (stats["avg_execution_time"] * (stats["total_executions"] - 1) + record.execution_time) / stats["total_executions"]
        stats["avg_quality_score"] = (stats["avg_quality_score"] * (stats["total_executions"] - 1) + record.quality_score) / stats["total_executions"]
        stats["task_types"].add(record.task_type)
    
    def get_model_performance_summary(self, model_name: str, model_version: str = None) -> Dict[str, Any]:
        """Get performance summary for a specific model"""
        
        model_key = f"{model_name}_{model_version}" if model_version else model_name
        
        # Find matching profiles
        matching_profiles = []
        for profile_key, profile in self.model_profiles.items():
            if profile.model_name == model_name and (model_version is None or profile.model_version == model_version):
                matching_profiles.append(profile)
        
        if not matching_profiles:
            return {"status": "no_data", "message": f"No performance data found for {model_name}"}
        
        # Aggregate performance data
        aggregated_metrics = {
            "execution_time": [],
            "success_rate": [],
            "resource_efficiency": []
        }
        
        capability_scores = {}
        usage_statistics = {}
        
        for profile in matching_profiles:
            # Aggregate performance metrics
            for metric_name, values in profile.performance_metrics.items():
                if metric_name in aggregated_metrics:
                    aggregated_metrics[metric_name].extend(values)
            
            # Aggregate capability scores
            for task_type, scores in profile.capability_scores.items():
                if task_type not in capability_scores:
                    capability_scores[task_type] = []
                capability_scores[task_type].extend(scores)
            
            # Aggregate usage statistics
            for task_type, count in profile.usage_statistics.items():
                if task_type not in usage_statistics:
                    usage_statistics[task_type] = 0
                usage_statistics[task_type] += count
        
        # Calculate summary statistics
        summary = {
            "model_name": model_name,
            "model_version": model_version,
            "total_profiles": len(matching_profiles),
            "performance_metrics": {
                "avg_execution_time": np.mean(aggregated_metrics["execution_time"]) if aggregated_metrics["execution_time"] else 0.0,
                "avg_success_rate": np.mean(aggregated_metrics["success_rate"]) if aggregated_metrics["success_rate"] else 0.0,
                "avg_resource_efficiency": np.mean(aggregated_metrics["resource_efficiency"]) if aggregated_metrics["resource_efficiency"] else 0.0,
                "total_executions": sum(len(values) for values in aggregated_metrics.values())
            },
            "capability_scores": {
                task_type: {
                    "avg_score": np.mean(scores),
                    "max_score": np.max(scores),
                    "min_score": np.min(scores),
                    "sample_count": len(scores)
                }
                for task_type, scores in capability_scores.items()
            },
            "usage_statistics": usage_statistics,
            "recommendations": self._generate_model_recommendations(model_name, aggregated_metrics, capability_scores)
        }
        
        return summary
    
    def _generate_model_recommendations(
        self,
        model_name: str,
        aggregated_metrics: Dict[str, List[float]],
        capability_scores: Dict[str, List[float]]
    ) -> List[str]:
        """Generate recommendations for model optimization"""
        
        recommendations = []
        
        # Performance-based recommendations
        if aggregated_metrics["execution_time"]:
            avg_execution_time = np.mean(aggregated_metrics["execution_time"])
            if avg_execution_time > 5.0:
                recommendations.append("High execution time detected - consider model optimization or hardware upgrade")
        
        if aggregated_metrics["success_rate"]:
            avg_success_rate = np.mean(aggregated_metrics["success_rate"])
            if avg_success_rate < 0.8:
                recommendations.append("Low success rate detected - consider model retraining or parameter tuning")
        
        if aggregated_metrics["resource_efficiency"]:
            avg_resource_efficiency = np.mean(aggregated_metrics["resource_efficiency"])
            if avg_resource_efficiency < 0.5:
                recommendations.append("Low resource efficiency detected - consider resource optimization")
        
        # Capability-based recommendations
        for task_type, scores in capability_scores.items():
            avg_score = np.mean(scores)
            if avg_score < 0.7:
                recommendations.append(f"Low performance in {task_type} tasks - consider specialized training")
        
        return recommendations
    
    def create_performance_benchmark(
        self,
        benchmark_name: str,
        task_type: str,
        test_cases: List[Dict[str, Any]],
        baseline_metrics: Dict[str, float]
    ) -> str:
        """Create a performance benchmark for model comparison"""
        
        benchmark_id = f"benchmark_{benchmark_name}_{int(time.time())}"
        
        benchmark = PerformanceBenchmark(
            benchmark_id=benchmark_id,
            benchmark_name=benchmark_name,
            task_type=task_type,
            test_cases=test_cases,
            baseline_metrics=baseline_metrics,
            model_results={}
        )
        
        self.performance_benchmarks[benchmark_id] = benchmark
        
        self._save_data()
        
        logger.info(f"📊 Created performance benchmark: {benchmark_name} with {len(test_cases)} test cases")
        
        return benchmark_id
    
    def run_benchmark(
        self,
        benchmark_id: str,
        model_name: str,
        model_version: str,
        test_function: callable
    ) -> Dict[str, Any]:
        """Run a benchmark test for a specific model"""
        
        if benchmark_id not in self.performance_benchmarks:
            return {"error": "Benchmark not found"}
        
        benchmark = self.performance_benchmarks[benchmark_id]
        model_key = f"{model_name}_{model_version}"
        
        # Run benchmark tests
        test_results = []
        total_execution_time = 0.0
        total_quality_score = 0.0
        successful_tests = 0
        
        for i, test_case in enumerate(benchmark.test_cases):
            start_time = time.time()
            
            try:
                # Run test case
                result = test_function(test_case, model_name, model_version)
                
                execution_time = time.time() - start_time
                total_execution_time += execution_time
                
                if result.get("success", False):
                    successful_tests += 1
                    quality_score = result.get("quality_score", 0.0)
                    total_quality_score += quality_score
                
                test_results.append({
                    "test_case_id": i,
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    "quality_score": result.get("quality_score", 0.0),
                    "result": result
                })
                
            except Exception as e:
                test_results.append({
                    "test_case_id": i,
                    "execution_time": time.time() - start_time,
                    "success": False,
                    "quality_score": 0.0,
                    "error": str(e)
                })
        
        # Calculate benchmark metrics
        benchmark_metrics = {
            "avg_execution_time": total_execution_time / len(benchmark.test_cases),
            "success_rate": successful_tests / len(benchmark.test_cases),
            "avg_quality_score": total_quality_score / max(1, successful_tests),
            "total_tests": len(benchmark.test_cases),
            "successful_tests": successful_tests
        }
        
        # Store results
        benchmark.model_results[model_key] = benchmark_metrics
        
        # Record performance
        self.record_model_performance(
            model_name=model_name,
            model_version=model_version,
            task_type=benchmark.task_type,
            execution_time=benchmark_metrics["avg_execution_time"],
            success=benchmark_metrics["success_rate"] > 0.8,
            quality_score=benchmark_metrics["avg_quality_score"],
            resource_usage={"cpu": 0.8, "memory": 0.6},  # Simulated
            input_size=sum(len(str(tc)) for tc in benchmark.test_cases),
            output_size=sum(len(str(tr["result"])) for tr in test_results),
            metadata={"benchmark_id": benchmark_id, "test_results": test_results}
        )
        
        self._save_data()
        
        logger.info(f"🏃 Ran benchmark {benchmark_name} for {model_name}: {benchmark_metrics['success_rate']:.2%} success rate")
        
        return {
            "benchmark_id": benchmark_id,
            "model_name": model_name,
            "model_version": model_version,
            "benchmark_metrics": benchmark_metrics,
            "test_results": test_results
        }
    
    def compare_models(
        self,
        model_names: List[str],
        task_type: str = None,
        metric: str = "quality_score"
    ) -> Dict[str, Any]:
        """Compare performance of multiple models"""
        
        comparison_results = {}
        
        for model_name in model_names:
            # Get performance summary
            summary = self.get_model_performance_summary(model_name)
            
            if summary.get("status") == "no_data":
                comparison_results[model_name] = {"error": "No performance data available"}
                continue
            
            # Extract relevant metrics
            if task_type and task_type in summary.get("capability_scores", {}):
                model_metric = summary["capability_scores"][task_type]["avg_score"]
            else:
                model_metric = summary["performance_metrics"].get(metric, 0.0)
            
            comparison_results[model_name] = {
                "metric_value": model_metric,
                "performance_summary": summary
            }
        
        # Rank models by metric
        ranked_models = sorted(
            comparison_results.items(),
            key=lambda x: x[1].get("metric_value", 0.0),
            reverse=True
        )
        
        return {
            "comparison_metric": metric,
            "task_type": task_type,
            "model_rankings": ranked_models,
            "best_model": ranked_models[0][0] if ranked_models else None,
            "comparison_results": comparison_results
        }
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the performance registry"""
        
        # Calculate overall statistics
        total_records = len(self.performance_records)
        total_models = len(set(f"{r.model_name}_{r.model_version}" for r in self.performance_records))
        total_benchmarks = len(self.performance_benchmarks)
        
        # Performance trends
        recent_records = [r for r in self.performance_records if time.time() - r.timestamp < 86400]  # Last 24 hours
        
        if recent_records:
            recent_avg_execution_time = np.mean([r.execution_time for r in recent_records])
            recent_success_rate = np.mean([r.success for r in recent_records])
            recent_avg_quality = np.mean([r.quality_score for r in recent_records])
        else:
            recent_avg_execution_time = 0.0
            recent_success_rate = 0.0
            recent_avg_quality = 0.0
        
        # Top performing models
        model_performance = {}
        for record in self.performance_records:
            model_key = f"{record.model_name}_{record.model_version}"
            if model_key not in model_performance:
                model_performance[model_key] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "total_quality": 0.0,
                    "total_time": 0.0
                }
            
            perf = model_performance[model_key]
            perf["total_executions"] += 1
            if record.success:
                perf["successful_executions"] += 1
            perf["total_quality"] += record.quality_score
            perf["total_time"] += record.execution_time
        
        # Calculate model scores
        model_scores = {}
        for model_key, perf in model_performance.items():
            if perf["total_executions"] > 0:
                success_rate = perf["successful_executions"] / perf["total_executions"]
                avg_quality = perf["total_quality"] / perf["total_executions"]
                avg_time = perf["total_time"] / perf["total_executions"]
                
                # Composite score (higher is better)
                score = (success_rate * 0.4 + avg_quality * 0.4 + (1.0 / max(0.1, avg_time)) * 0.2)
                model_scores[model_key] = score
        
        # Top 5 models
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "registry_statistics": {
                "total_records": total_records,
                "total_models": total_models,
                "total_benchmarks": total_benchmarks,
                "active_models": len(self.model_profiles)
            },
            "recent_performance": {
                "avg_execution_time": recent_avg_execution_time,
                "success_rate": recent_success_rate,
                "avg_quality_score": recent_avg_quality,
                "records_last_24h": len(recent_records)
            },
            "top_performing_models": [
                {
                    "model": model_key,
                    "score": score,
                    "executions": model_performance[model_key]["total_executions"]
                }
                for model_key, score in top_models
            ],
            "benchmarks": [
                {
                    "benchmark_id": benchmark.benchmark_id,
                    "benchmark_name": benchmark.benchmark_name,
                    "task_type": benchmark.task_type,
                    "test_cases": len(benchmark.test_cases),
                    "models_tested": len(benchmark.model_results)
                }
                for benchmark in self.performance_benchmarks.values()
            ]
        }
    
    def _initialize_default_benchmarks(self) -> None:
        """Initialize default performance benchmarks"""
        
        if not self.performance_benchmarks:
            # Create default benchmarks
            default_benchmarks = [
                {
                    "benchmark_name": "text_generation_quality",
                    "task_type": "text_generation",
                    "test_cases": [
                        {"prompt": "Explain quantum computing", "expected_length": 200},
                        {"prompt": "Write a creative story", "expected_length": 300},
                        {"prompt": "Summarize a complex topic", "expected_length": 150}
                    ],
                    "baseline_metrics": {"quality_score": 0.7, "execution_time": 2.0}
                },
                {
                    "benchmark_name": "reasoning_accuracy",
                    "task_type": "reasoning",
                    "test_cases": [
                        {"problem": "Solve: 2x + 5 = 15", "expected_answer": "x = 5"},
                        {"problem": "What is the capital of France?", "expected_answer": "Paris"},
                        {"problem": "Explain the water cycle", "expected_answer": "evaporation, condensation, precipitation"}
                    ],
                    "baseline_metrics": {"accuracy": 0.8, "execution_time": 1.5}
                }
            ]
            
            for benchmark_data in default_benchmarks:
                benchmark_id = self.create_performance_benchmark(
                    benchmark_data["benchmark_name"],
                    benchmark_data["task_type"],
                    benchmark_data["test_cases"],
                    benchmark_data["baseline_metrics"]
                )
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        try:
            # Load performance records
            if self.records_file.exists():
                with open(self.records_file, 'r') as f:
                    data = json.load(f)
                    self.performance_records = [
                        ModelPerformanceRecord(**record_data)
                        for record_data in data
                    ]
            
            # Load model profiles
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r') as f:
                    data = json.load(f)
                    self.model_profiles = {
                        profile_key: ModelCapabilityProfile(**profile_data)
                        for profile_key, profile_data in data.items()
                    }
            
            # Load performance benchmarks
            if self.benchmarks_file.exists():
                with open(self.benchmarks_file, 'r') as f:
                    data = json.load(f)
                    self.performance_benchmarks = {
                        benchmark_id: PerformanceBenchmark(**benchmark_data)
                        for benchmark_id, benchmark_data in data.items()
                    }
            
            logger.info(f"📁 Loaded performance registry data: {len(self.performance_records)} records, {len(self.model_profiles)} profiles")
            
        except Exception as e:
            logger.warning(f"Failed to load performance registry data: {e}")
    
    def _save_data(self) -> None:
        """Save data to storage files"""
        try:
            # Save performance records
            records_data = [
                {
                    "record_id": record.record_id,
                    "model_name": record.model_name,
                    "model_version": record.model_version,
                    "task_type": record.task_type,
                    "execution_time": record.execution_time,
                    "success": record.success,
                    "quality_score": record.quality_score,
                    "resource_usage": record.resource_usage,
                    "input_size": record.input_size,
                    "output_size": record.output_size,
                    "error_message": record.error_message,
                    "metadata": record.metadata,
                    "timestamp": record.timestamp
                }
                for record in self.performance_records
            ]
            
            with open(self.records_file, 'w') as f:
                json.dump(records_data, f, indent=2)
            
            # Save model profiles
            profiles_data = {
                profile_key: {
                    "model_name": profile.model_name,
                    "model_version": profile.model_version,
                    "capability_scores": profile.capability_scores,
                    "performance_metrics": profile.performance_metrics,
                    "usage_statistics": profile.usage_statistics,
                    "last_updated": profile.last_updated,
                    "metadata": profile.metadata
                }
                for profile_key, profile in self.model_profiles.items()
            }
            
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            # Save performance benchmarks
            benchmarks_data = {
                benchmark_id: {
                    "benchmark_id": benchmark.benchmark_id,
                    "benchmark_name": benchmark.benchmark_name,
                    "task_type": benchmark.task_type,
                    "test_cases": benchmark.test_cases,
                    "baseline_metrics": benchmark.baseline_metrics,
                    "model_results": benchmark.model_results,
                    "created_at": benchmark.created_at
                }
                for benchmark_id, benchmark in self.performance_benchmarks.items()
            }
            
            with open(self.benchmarks_file, 'w') as f:
                json.dump(benchmarks_data, f, indent=2)
            
            logger.debug("💾 Saved performance registry data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save performance registry data: {e}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'analysis_executor'):
            self.analysis_executor.shutdown(wait=True)


# Helper functions for integration
def create_model_performance_registry(cfg: IceburgConfig) -> ModelPerformanceRegistry:
    """Create model performance registry instance"""
    return ModelPerformanceRegistry(cfg)

def record_model_performance(
    registry: ModelPerformanceRegistry,
    model_name: str,
    model_version: str,
    task_type: str,
    execution_time: float,
    success: bool,
    quality_score: float,
    resource_usage: Dict[str, float],
    input_size: int,
    output_size: int,
    error_message: str = None,
    metadata: Dict[str, Any] = None
) -> str:
    """Record model performance in registry"""
    return registry.record_model_performance(
        model_name, model_version, task_type, execution_time, success,
        quality_score, resource_usage, input_size, output_size, error_message, metadata
    )

def get_model_performance_summary(
    registry: ModelPerformanceRegistry,
    model_name: str,
    model_version: str = None
) -> Dict[str, Any]:
    """Get performance summary for a model"""
    return registry.get_model_performance_summary(model_name, model_version)
