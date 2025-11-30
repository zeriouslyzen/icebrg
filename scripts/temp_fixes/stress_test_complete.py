#!/usr/bin/env python3
"""
Complete Stress Test Suite for ICEBURG Autonomous Self-Evolution System

Comprehensive testing of all autonomous components under various stress conditions.
"""

import asyncio
import time
import random
import sys
import os
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil
import gc
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from iceburg.monitoring.unified_performance_tracker import (
    UnifiedPerformanceTracker, 
    get_global_tracker
)
from iceburg.evolution.specification_generator import (
    SpecificationGenerator,
    PerformanceAnalysis
)
from iceburg.autonomous.research_orchestrator import (
    AutonomousResearchOrchestrator,
    ResearchQuery
)
from iceburg.evolution.evolution_pipeline import (
    EvolutionPipeline,
    EvolutionStage
)
from benchmarks.self_evolution_benchmarks import SelfEvolutionBenchmark


class CompleteStressTestSuite:
    """Comprehensive stress testing suite for autonomous system."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
        self.start_time = time.time()
        self.test_data = []
        
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def log_result(self, test_name, success, duration, details="", metrics=None):
        """Log test result with metrics."""
        self.results[test_name] = {
            "success": success,
            "duration": duration,
            "details": details,
            "metrics": metrics or {},
            "timestamp": time.time()
        }
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {duration:.2f}s {details}")
        if metrics:
            for key, value in metrics.items():
                print(f"    📊 {key}: {value}")
    
    async def test_performance_tracker_stress(self):
        """Test performance tracker under extreme load."""
        print("\n🔥 Testing Performance Tracker Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            config = {"db_path": os.path.join(self.temp_dir, "stress_metrics.db")}
            tracker = UnifiedPerformanceTracker(config)
            await tracker.start_tracking()
            
            # Test 1: High frequency tracking
            query_count = 2000
            tracking_start = time.time()
            
            for i in range(query_count):
                tracker.track_query_performance(
                    query_id=f"stress_test_{i}",
                    response_time=random.uniform(0.1, 5.0),
                    accuracy=random.uniform(0.6, 1.0),
                    resources={
                        "memory_usage_mb": random.uniform(50, 500),
                        "cache_hit_rate": random.uniform(0.3, 0.9),
                        "agent_count": random.randint(1, 10),
                        "parallel_execution": random.choice([True, False]),
                        "query_complexity": random.uniform(0.1, 1.0)
                    },
                    success=random.random() > 0.05,  # 95% success rate
                    metadata={"stress_test": True, "iteration": i}
                )
                
                if i % 200 == 0:
                    await asyncio.sleep(0.001)
            
            tracking_time = time.time() - tracking_start
            metrics["tracking_rate"] = query_count / tracking_time
            
            # Test 2: Concurrent tracking
            concurrent_tasks = []
            for batch in range(5):
                task = asyncio.create_task(self._concurrent_tracking_batch(tracker, batch))
                concurrent_tasks.append(task)
            
            await asyncio.gather(*concurrent_tasks)
            
            # Wait for buffer flush
            await asyncio.sleep(3)
            
            # Test 3: Data retrieval under load
            retrieval_start = time.time()
            summary = await tracker.get_performance_summary()
            retrieval_time = time.time() - retrieval_start
            
            stored_queries = summary.get("total_queries", 0)
            metrics["stored_queries"] = stored_queries
            metrics["retrieval_time"] = retrieval_time
            metrics["buffer_size"] = len(tracker.metrics_buffer)
            
            if stored_queries >= query_count * 0.8:  # Allow 20% loss
                details = f"Stored {stored_queries}/{query_count} queries, {tracking_time:.2f}s tracking"
            else:
                success = False
                details = f"Only stored {stored_queries}/{query_count} queries"
            
            await tracker.stop_tracking()
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Performance Tracker Stress", success, duration, details, metrics)
        return success
    
    async def _concurrent_tracking_batch(self, tracker, batch_id):
        """Concurrent tracking batch for stress testing."""
        for i in range(100):
            tracker.track_query_performance(
                query_id=f"concurrent_{batch_id}_{i}",
                response_time=random.uniform(0.1, 2.0),
                accuracy=random.uniform(0.7, 1.0),
                resources={
                    "memory_usage_mb": random.uniform(100, 300),
                    "cache_hit_rate": random.uniform(0.5, 0.9),
                    "agent_count": random.randint(1, 5),
                    "parallel_execution": True,
                    "query_complexity": random.uniform(0.3, 0.8)
                },
                success=True,
                metadata={"concurrent_test": True, "batch": batch_id}
            )
    
    async def test_specification_generator_stress(self):
        """Test specification generator under stress."""
        print("\n🔥 Testing Specification Generator Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            generator = SpecificationGenerator()
            
            # Test 1: Multiple concurrent analyses
            analysis_tasks = []
            for i in range(10):
                task = asyncio.create_task(generator.analyze_system_performance())
                analysis_tasks.append(task)
            
            analyses = await asyncio.gather(*analysis_tasks)
            metrics["concurrent_analyses"] = len(analyses)
            
            # Test 2: Generate multiple specifications
            spec_count = 0
            for analysis in analyses:
                opportunities = generator.identify_optimization_opportunities(analysis)
                for opportunity in opportunities[:3]:  # Top 3 opportunities
                    spec = generator.generate_improvement_spec(opportunity)
                    safety = generator.validate_spec_safety(spec)
                    impact = generator.estimate_improvement_impact(spec)
                    spec_count += 1
            
            metrics["specifications_generated"] = spec_count
            
            # Test 3: Memory usage during generation
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)
            
            # Generate many specifications rapidly
            for i in range(50):
                analysis = await generator.analyze_system_performance()
                opportunities = generator.identify_optimization_opportunities(analysis)
                for opportunity in opportunities[:2]:
                    spec = generator.generate_improvement_spec(opportunity)
            
            memory_after = process.memory_info().rss / (1024 * 1024)
            memory_increase = memory_after - memory_before
            metrics["memory_increase_mb"] = memory_increase
            
            if spec_count >= 20 and memory_increase < 100:
                details = f"Generated {spec_count} specs, {memory_increase:.1f}MB memory"
            else:
                success = False
                details = f"Only {spec_count} specs, {memory_increase:.1f}MB memory"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Specification Generator Stress", success, duration, details, metrics)
        return success
    
    async def test_benchmark_suite_stress(self):
        """Test benchmark suite under stress."""
        print("\n🔥 Testing Benchmark Suite Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            config = {"results_dir": os.path.join(self.temp_dir, "benchmark_stress")}
            
            # Test 1: Concurrent benchmarks
            benchmark_tasks = []
            for i in range(8):
                benchmark = SelfEvolutionBenchmark(config)
                task = asyncio.create_task(benchmark.benchmark_current_version())
                benchmark_tasks.append(task)
            
            results = await asyncio.gather(*benchmark_tasks, return_exceptions=True)
            successful_benchmarks = sum(1 for r in results if not isinstance(r, Exception))
            metrics["concurrent_benchmarks"] = successful_benchmarks
            
            # Test 2: Safety validation stress
            validation_count = 0
            for i in range(100):
                spec = {
                    "name": f"stress_spec_{i}",
                    "max_memory_usage": random.uniform(500, 2000),
                    "max_cpu_usage": random.uniform(30, 80),
                    "min_accuracy": random.uniform(0.6, 0.9),
                    "expected_improvement": random.uniform(1.0, 2.0)
                }
                
                benchmark = SelfEvolutionBenchmark(config)
                safety_report = benchmark.validate_safety_constraints(spec)
                validation_count += 1
            
            metrics["safety_validations"] = validation_count
            
            # Test 3: Version comparison stress
            comparison_count = 0
            for i in range(20):
                from benchmarks.self_evolution_benchmarks import BenchmarkResults
                
                baseline = BenchmarkResults(
                    test_name=f"baseline_{i}",
                    timestamp=time.time(),
                    duration=10.0,
                    success=True,
                    metrics={
                        "response_time": random.uniform(1.0, 5.0),
                        "accuracy": random.uniform(0.7, 0.9),
                        "memory_usage_mb": random.uniform(100, 500)
                    }
                )
                
                improved = BenchmarkResults(
                    test_name=f"improved_{i}",
                    timestamp=time.time(),
                    duration=10.0,
                    success=True,
                    metrics={
                        "response_time": baseline.metrics["response_time"] * random.uniform(0.7, 1.0),
                        "accuracy": min(1.0, baseline.metrics["accuracy"] * random.uniform(1.0, 1.2)),
                        "memory_usage_mb": baseline.metrics["memory_usage_mb"] * random.uniform(0.8, 1.0)
                    }
                )
                
                benchmark = SelfEvolutionBenchmark(config)
                comparison = benchmark.compare_versions(baseline, improved)
                comparison_count += 1
            
            metrics["version_comparisons"] = comparison_count
            
            if successful_benchmarks >= 6 and validation_count == 100 and comparison_count == 20:
                details = f"{successful_benchmarks} benchmarks, {validation_count} validations, {comparison_count} comparisons"
            else:
                success = False
                details = f"Only {successful_benchmarks} benchmarks, {validation_count} validations, {comparison_count} comparisons"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Benchmark Suite Stress", success, duration, details, metrics)
        return success
    
    async def test_evolution_pipeline_stress(self):
        """Test evolution pipeline under stress."""
        print("\n🔥 Testing Evolution Pipeline Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            config = {
                "max_concurrent_jobs": 10,
                "timeout_per_stage": 5,  # Short timeout for stress testing
                "auto_approve_threshold": 0.8
            }
            
            pipeline = EvolutionPipeline(config)
            
            # Test 1: Multiple concurrent evolution jobs
            job_tasks = []
            for i in range(20):
                task = asyncio.create_task(pipeline.evolve_system(f"stress_test_{i}"))
                job_tasks.append(task)
            
            job_ids = await asyncio.gather(*job_tasks, return_exceptions=True)
            successful_jobs = sum(1 for job_id in job_ids if isinstance(job_id, str))
            failed_jobs = sum(1 for job_id in job_ids if isinstance(job_id, Exception))
            metrics["successful_jobs"] = successful_jobs
            metrics["failed_jobs"] = failed_jobs
            
            # Test 2: Job status monitoring under load
            status_checks = 0
            for job_id in job_ids:
                if isinstance(job_id, str):
                    status = pipeline.get_job_status(job_id)
                    if status:
                        status_checks += 1
            
            metrics["status_checks"] = status_checks
            
            # Test 3: Pipeline status under load
            pipeline_status = pipeline.get_pipeline_status()
            metrics["total_jobs"] = pipeline_status["total_jobs"]
            metrics["active_jobs"] = pipeline_status["active_jobs"]
            metrics["success_rate"] = pipeline_status["success_rate"]
            
            if successful_jobs >= 10 and status_checks >= 10:
                details = f"{successful_jobs} successful jobs, {status_checks} status checks"
            else:
                success = False
                details = f"Only {successful_jobs} successful jobs, {status_checks} status checks"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Evolution Pipeline Stress", success, duration, details, metrics)
        return success
    
    async def test_memory_stress(self):
        """Test memory usage under extreme stress."""
        print("\n🔥 Testing Memory Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            # Create many components
            trackers = []
            generators = []
            benchmarks = []
            
            for i in range(50):
                # Create trackers
                config = {"db_path": os.path.join(self.temp_dir, f"memory_test_{i}.db")}
                tracker = UnifiedPerformanceTracker(config)
                await tracker.start_tracking()
                trackers.append(tracker)
                
                # Create generators
                generator = SpecificationGenerator()
                generators.append(generator)
                
                # Create benchmarks
                benchmark = SelfEvolutionBenchmark({"results_dir": os.path.join(self.temp_dir, f"benchmark_{i}")})
                benchmarks.append(benchmark)
            
            # Generate lots of data
            for i in range(1000):
                for j, tracker in enumerate(trackers[:10]):  # Use first 10 trackers
                    tracker.track_query_performance(
                        query_id=f"memory_test_{i}_{j}",
                        response_time=random.uniform(0.1, 2.0),
                        accuracy=random.uniform(0.7, 1.0),
                        resources={
                            "memory_usage_mb": random.uniform(100, 200),
                            "cache_hit_rate": random.uniform(0.5, 0.9),
                            "agent_count": random.randint(1, 5),
                            "parallel_execution": True,
                            "query_complexity": random.uniform(0.3, 0.8)
                        },
                        success=True,
                        metadata={"memory_test": True}
                    )
                
                # Generate specifications
                if i % 20 == 0:
                    for generator in generators[:5]:  # Use first 5 generators
                        analysis = await generator.analyze_system_performance()
                        opportunities = generator.identify_optimization_opportunities(analysis)
                        for opportunity in opportunities[:2]:
                            spec = generator.generate_improvement_spec(opportunity)
            
            # Force garbage collection
            gc.collect()
            
            # Check final memory
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_increase = final_memory - initial_memory
            metrics["memory_increase_mb"] = memory_increase
            metrics["components_created"] = len(trackers) + len(generators) + len(benchmarks)
            
            # Stop all trackers
            for tracker in trackers:
                await tracker.stop_tracking()
            
            if memory_increase < 1000:  # Less than 1GB increase
                details = f"Memory increase: {memory_increase:.1f}MB with {metrics['components_created']} components"
            else:
                success = False
                details = f"Memory increase too high: {memory_increase:.1f}MB"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Memory Stress", success, duration, details, metrics)
        return success
    
    async def test_concurrent_operations(self):
        """Test concurrent operations across all components."""
        print("\n🔥 Testing Concurrent Operations...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            # Create components
            tracker = UnifiedPerformanceTracker({"db_path": os.path.join(self.temp_dir, "concurrent.db")})
            await tracker.start_tracking()
            
            generator = SpecificationGenerator()
            orchestrator = AutonomousResearchOrchestrator({
                "max_concurrent_queries": 5,
                "research_cycle_interval": 1
            })
            pipeline = EvolutionPipeline({"max_concurrent_jobs": 3})
            
            # Run concurrent operations
            tasks = []
            
            # Task 1: Performance tracking
            for i in range(5):
                task = asyncio.create_task(self._concurrent_tracking_task(tracker, i))
                tasks.append(task)
            
            # Task 2: Specification generation
            for i in range(3):
                task = asyncio.create_task(self._concurrent_generation_task(generator, i))
                tasks.append(task)
            
            # Task 3: Research orchestration
            for i in range(2):
                task = asyncio.create_task(self._concurrent_research_task(orchestrator, i))
                tasks.append(task)
            
            # Task 4: Evolution pipeline
            for i in range(3):
                task = asyncio.create_task(self._concurrent_evolution_task(pipeline, i))
                tasks.append(task)
            
            # Wait for all tasks with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                results = [Exception("Task timeout")] * len(tasks)
            
            # Count successful tasks
            successful_tasks = sum(1 for result in results if not isinstance(result, Exception) and result is not None)
            failed_tasks = sum(1 for result in results if isinstance(result, Exception) or result is None)
            
            metrics["successful_tasks"] = successful_tasks
            metrics["failed_tasks"] = failed_tasks
            metrics["total_tasks"] = len(tasks)
            
            await tracker.stop_tracking()
            
            if successful_tasks >= len(tasks) * 0.8:  # 80% success rate
                details = f"{successful_tasks}/{len(tasks)} tasks successful"
            else:
                success = False
                details = f"Only {successful_tasks}/{len(tasks)} tasks successful"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Concurrent Operations", success, duration, details, metrics)
        return success
    
    async def _concurrent_tracking_task(self, tracker, task_id):
        """Concurrent tracking task."""
        for i in range(50):
            tracker.track_query_performance(
                query_id=f"concurrent_tracking_{task_id}_{i}",
                response_time=random.uniform(0.1, 1.0),
                accuracy=random.uniform(0.8, 1.0),
                resources={
                    "memory_usage_mb": random.uniform(50, 150),
                    "cache_hit_rate": random.uniform(0.6, 0.9),
                    "agent_count": random.randint(1, 3),
                    "parallel_execution": True,
                    "query_complexity": random.uniform(0.2, 0.6)
                },
                success=True,
                metadata={"concurrent_task": task_id}
            )
            await asyncio.sleep(0.001)
    
    async def _concurrent_generation_task(self, generator, task_id):
        """Concurrent generation task with retry logic."""
        for i in range(10):
            try:
                analysis = await generator.analyze_system_performance()
                opportunities = generator.identify_optimization_opportunities(analysis)
                for opportunity in opportunities[:2]:
                    try:
                        spec = generator.generate_improvement_spec(opportunity)
                        safety = generator.validate_spec_safety(spec)
                    except Exception as e:
                        # Log but continue with other opportunities
                        print(f"Generation task {task_id} opportunity {i} failed: {e}")
                        continue
            except Exception as e:
                # Log but continue with next iteration
                print(f"Generation task {task_id} iteration {i} failed: {e}")
                continue
            await asyncio.sleep(0.01)
    
    async def _concurrent_research_task(self, orchestrator, task_id):
        """Concurrent research task with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                queries = await orchestrator._generate_curiosity_queries()
                if not queries:
                    return 0
                
                results = await orchestrator._execute_research_queries(queries)
                patterns = await orchestrator._detect_emergence_patterns(results)
                return len(patterns)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Research task {task_id} failed after {max_retries} attempts: {e}")
                    return 0
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
        return 0
    
    async def _concurrent_evolution_task(self, pipeline, task_id):
        """Concurrent evolution task with retry logic."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                job_id = await pipeline.evolve_system(f"concurrent_evolution_{task_id}")
                return job_id
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Evolution task {task_id} failed after {max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
        return None
    
    async def test_error_recovery_stress(self):
        """Test error recovery under stress."""
        print("\n🔥 Testing Error Recovery Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            # Test 1: Invalid configurations
            invalid_configs = [
                {"db_path": "/invalid/path/that/does/not/exist.db"},
                {"max_concurrent_queries": -1},
                {"research_cycle_interval": 0},
                {"emergence_threshold": 2.0},
                {"breakthrough_threshold": -0.5},
                {"max_concurrent_jobs": "invalid"},
                {"timeout_per_stage": -1},
                {"auto_approve_threshold": 1.5}
            ]
            
            recovery_count = 0
            for i, invalid_config in enumerate(invalid_configs):
                try:
                    if i % 3 == 0:
                        tracker = UnifiedPerformanceTracker(invalid_config)
                        await tracker.start_tracking()
                        await tracker.stop_tracking()
                    elif i % 3 == 1:
                        orchestrator = AutonomousResearchOrchestrator(invalid_config)
                    else:
                        pipeline = EvolutionPipeline(invalid_config)
                    recovery_count += 1
                except Exception:
                    pass  # Expected to fail
            
            # Test 2: Recovery with valid config after invalid ones
            valid_config = {"db_path": os.path.join(self.temp_dir, "recovery_test.db")}
            tracker = UnifiedPerformanceTracker(valid_config)
            await tracker.start_tracking()
            
            # Should work normally
            tracker.track_query_performance(
                query_id="recovery_test",
                response_time=1.0,
                accuracy=0.9,
                resources={
                    "memory_usage_mb": 100,
                    "cache_hit_rate": 0.8,
                    "agent_count": 3,
                    "parallel_execution": True,
                    "query_complexity": 0.5
                },
                success=True
            )
            
            await tracker.stop_tracking()
            
            metrics["recovery_count"] = recovery_count
            metrics["invalid_configs"] = len(invalid_configs)
            
            if recovery_count >= len(invalid_configs) * 0.5:  # At least 50% should handle gracefully
                details = f"Recovered from {recovery_count}/{len(invalid_configs)} invalid configs"
            else:
                success = False
                details = f"Only recovered from {recovery_count}/{len(invalid_configs)} invalid configs"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Error Recovery Stress", success, duration, details, metrics)
        return success
    
    async def test_system_resources_stress(self):
        """Test system resource usage under stress."""
        print("\n🔥 Testing System Resources Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            process = psutil.Process()
            
            # Get initial resources
            initial_memory = process.memory_info().rss / (1024 * 1024)
            initial_cpu = process.cpu_percent()
            
            # Run intensive operations
            config = {"db_path": os.path.join(self.temp_dir, "resource_test.db")}
            tracker = UnifiedPerformanceTracker(config)
            await tracker.start_tracking()
            
            generator = SpecificationGenerator()
            
            # Generate lots of data and specifications
            for i in range(3000):
                # Track performance
                tracker.track_query_performance(
                    query_id=f"resource_test_{i}",
                    response_time=random.uniform(0.1, 2.0),
                    accuracy=random.uniform(0.7, 1.0),
                    resources={
                        "memory_usage_mb": random.uniform(50, 200),
                        "cache_hit_rate": random.uniform(0.6, 0.9),
                        "agent_count": random.randint(1, 6),
                        "parallel_execution": True,
                        "query_complexity": random.uniform(0.3, 0.8)
                    },
                    success=True,
                    metadata={"resource_test": True}
                )
                
                # Generate specifications
                if i % 30 == 0:
                    analysis = await generator.analyze_system_performance()
                    opportunities = generator.identify_optimization_opportunities(analysis)
                    for opportunity in opportunities[:2]:
                        spec = generator.generate_improvement_spec(opportunity)
                
                if i % 300 == 0:
                    await asyncio.sleep(0.01)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check final resources
            final_memory = process.memory_info().rss / (1024 * 1024)
            final_cpu = process.cpu_percent()
            
            memory_increase = final_memory - initial_memory
            cpu_usage = final_cpu
            
            metrics["memory_increase_mb"] = memory_increase
            metrics["cpu_usage_percent"] = cpu_usage
            metrics["operations_performed"] = 3000
            
            await tracker.stop_tracking()
            
            # Check if resources are reasonable
            if memory_increase < 500 and cpu_usage < 90:
                details = f"Memory: +{memory_increase:.1f}MB, CPU: {cpu_usage:.1f}%"
            else:
                success = False
                details = f"High resource usage: Memory +{memory_increase:.1f}MB, CPU {cpu_usage:.1f}%"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("System Resources Stress", success, duration, details, metrics)
        return success
    
    def print_comprehensive_summary(self):
        """Print comprehensive stress test summary."""
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*80)
        print("🔥 ICEBURG AUTONOMOUS SYSTEM COMPREHENSIVE STRESS TEST RESULTS")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        print()
        
        # Performance metrics summary
        print("📊 PERFORMANCE METRICS SUMMARY:")
        print("-" * 40)
        
        all_metrics = {}
        for test_name, result in self.results.items():
            if result["metrics"]:
                for key, value in result["metrics"].items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        for key, values in all_metrics.items():
            if values:
                if isinstance(values[0], (int, float)):
                    avg_val = sum(values) / len(values)
                    max_val = max(values)
                    min_val = min(values)
                    print(f"  {key}: avg={avg_val:.2f}, max={max_val:.2f}, min={min_val:.2f}")
                else:
                    print(f"  {key}: {len(values)} occurrences")
        
        print()
        
        if failed_tests > 0:
            print("❌ FAILED TESTS:")
            for test_name, result in self.results.items():
                if not result["success"]:
                    print(f"  - {test_name}: {result['details']}")
            print()
        
        print("✅ PASSED TESTS:")
        for test_name, result in self.results.items():
            if result["success"]:
                print(f"  - {test_name}: {result['duration']:.2f}s")
                if result["metrics"]:
                    for key, value in result["metrics"].items():
                        print(f"    📊 {key}: {value}")
        
        print("\n" + "="*80)
        
        if failed_tests == 0:
            print("🎉 ALL STRESS TESTS PASSED! System is robust and ready for production!")
            print("✅ ICEBURG Autonomous Self-Evolution System is fully stress-tested!")
        elif failed_tests <= 2:
            print("⚠️  Minor issues detected. System is mostly ready for production.")
        else:
            print(f"❌ {failed_tests} tests failed. System needs attention before production.")
        
        print("="*80)
        
        # Save detailed results
        results_file = os.path.join(self.temp_dir, "stress_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")


async def main():
    """Run all comprehensive stress tests."""
    print("🚀 Starting ICEBURG Autonomous System Comprehensive Stress Tests")
    print("="*80)
    
    suite = CompleteStressTestSuite()
    
    try:
        # Run all stress tests
        await suite.test_performance_tracker_stress()
        await suite.test_specification_generator_stress()
        await suite.test_benchmark_suite_stress()
        await suite.test_evolution_pipeline_stress()
        await suite.test_memory_stress()
        await suite.test_concurrent_operations()
        await suite.test_error_recovery_stress()
        await suite.test_system_resources_stress()
        
        # Print comprehensive summary
        suite.print_comprehensive_summary()
        
    finally:
        # Cleanup
        suite.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
