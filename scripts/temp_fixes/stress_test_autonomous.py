#!/usr/bin/env python3
"""
Stress Test Suite for ICEBURG Autonomous Self-Evolution System

Tests the system under high load, concurrent operations, and edge cases.
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


class StressTestSuite:
    """Comprehensive stress testing suite."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
        self.start_time = time.time()
        
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def log_result(self, test_name, success, duration, details=""):
        """Log test result."""
        self.results[test_name] = {
            "success": success,
            "duration": duration,
            "details": details,
            "timestamp": time.time()
        }
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {duration:.2f}s {details}")
    
    async def test_high_frequency_performance_tracking(self):
        """Test performance tracker under high frequency load."""
        print("\n🔥 Testing High Frequency Performance Tracking...")
        
        start_time = time.time()
        success = True
        details = ""
        
        try:
            config = {"db_path": os.path.join(self.temp_dir, "stress_metrics.db")}
            tracker = UnifiedPerformanceTracker(config)
            await tracker.start_tracking()
            
            # Generate 1000 queries rapidly
            query_count = 1000
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
                
                # Add some delay to simulate real usage
                if i % 100 == 0:
                    await asyncio.sleep(0.001)
            
            # Wait for buffer flush
            await asyncio.sleep(2)
            
            # Verify data was stored
            summary = await tracker.get_performance_summary()
            stored_queries = summary.get("total_queries", 0)
            
            if stored_queries >= query_count * 0.9:  # Allow 10% loss
                details = f"Stored {stored_queries}/{query_count} queries"
            else:
                success = False
                details = f"Only stored {stored_queries}/{query_count} queries"
            
            await tracker.stop_tracking()
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("High Frequency Performance Tracking", success, duration, details)
        return success
    
    async def test_concurrent_research_cycles(self):
        """Test multiple concurrent research cycles."""
        print("\n🔥 Testing Concurrent Research Cycles...")
        
        start_time = time.time()
        success = True
        details = ""
        
        try:
            config = {
                "max_concurrent_queries": 10,
                "research_cycle_interval": 1,  # 1 second for stress testing
                "emergence_threshold": 0.6,
                "breakthrough_threshold": 0.7
            }
            
            # Create multiple orchestrators
            orchestrators = []
            for i in range(5):
                orchestrator = AutonomousResearchOrchestrator(config)
                orchestrators.append(orchestrator)
            
            # Start all orchestrators concurrently
            tasks = []
            for i, orchestrator in enumerate(orchestrators):
                task = asyncio.create_task(self._run_orchestrator_stress(orchestrator, i))
                tasks.append(task)
            
            # Run for 10 seconds
            await asyncio.sleep(10)
            
            # Stop all orchestrators
            for orchestrator in orchestrators:
                await orchestrator.stop_autonomous_research()
            
            # Wait for tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            successful_cycles = sum(1 for result in results if isinstance(result, int) and result > 0)
            total_cycles = sum(result for result in results if isinstance(result, int))
            
            if successful_cycles >= 3:  # At least 3 orchestrators should succeed
                details = f"{successful_cycles}/5 orchestrators completed {total_cycles} cycles"
            else:
                success = False
                details = f"Only {successful_cycles}/5 orchestrators succeeded"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Concurrent Research Cycles", success, duration, details)
        return success
    
    async def _run_orchestrator_stress(self, orchestrator, orchestrator_id):
        """Run orchestrator stress test."""
        try:
            await orchestrator.start_autonomous_research()
            cycle_count = 0
            
            # Run for 8 seconds (orchestrator will run for 10 total)
            for _ in range(8):
                await asyncio.sleep(1)
                cycle_count = orchestrator.cycle_count
            
            return cycle_count
        except Exception as e:
            print(f"Orchestrator {orchestrator_id} failed: {e}")
            return 0
    
    async def test_evolution_pipeline_stress(self):
        """Test evolution pipeline under stress."""
        print("\n🔥 Testing Evolution Pipeline Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        
        try:
            config = {
                "max_concurrent_jobs": 5,
                "timeout_per_stage": 10,  # Short timeout for stress testing
                "auto_approve_threshold": 0.8
            }
            
            pipeline = EvolutionPipeline(config)
            
            # Start multiple evolution jobs concurrently
            tasks = []
            for i in range(10):
                task = asyncio.create_task(pipeline.evolve_system(f"stress_test_{i}"))
                tasks.append(task)
            
            # Wait for all jobs to complete or timeout
            job_ids = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            successful_jobs = sum(1 for job_id in job_ids if isinstance(job_id, str))
            failed_jobs = sum(1 for job_id in job_ids if isinstance(job_id, Exception))
            
            if successful_jobs >= 5:  # At least half should succeed
                details = f"{successful_jobs} successful, {failed_jobs} failed"
            else:
                success = False
                details = f"Only {successful_jobs} successful, {failed_jobs} failed"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Evolution Pipeline Stress", success, duration, details)
        return success
    
    async def test_memory_stress(self):
        """Test memory usage under stress."""
        print("\n🔥 Testing Memory Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        
        try:
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Create multiple trackers and generators
            trackers = []
            generators = []
            
            for i in range(20):
                config = {"db_path": os.path.join(self.temp_dir, f"memory_test_{i}.db")}
                tracker = UnifiedPerformanceTracker(config)
                await tracker.start_tracking()
                trackers.append(tracker)
                
                generator = SpecificationGenerator()
                generators.append(generator)
            
            # Generate lots of data
            for i in range(1000):
                for j, tracker in enumerate(trackers):
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
                if i % 10 == 0:
                    for generator in generators:
                        analysis = await generator.analyze_system_performance()
                        opportunities = generator.identify_optimization_opportunities(analysis)
            
            # Force garbage collection
            gc.collect()
            
            # Check final memory
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            # Stop all trackers
            for tracker in trackers:
                await tracker.stop_tracking()
            
            if memory_increase < 500:  # Less than 500MB increase
                details = f"Memory increase: {memory_increase:.1f}MB"
            else:
                success = False
                details = f"Memory increase too high: {memory_increase:.1f}MB"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Memory Stress", success, duration, details)
        return success
    
    async def test_database_stress(self):
        """Test database performance under stress."""
        print("\n🔥 Testing Database Stress...")
        
        start_time = time.time()
        success = True
        details = ""
        
        try:
            config = {"db_path": os.path.join(self.temp_dir, "db_stress.db")}
            tracker = UnifiedPerformanceTracker(config)
            await tracker.start_tracking()
            
            # Insert large amounts of data rapidly
            batch_size = 100
            total_inserts = 5000
            
            for batch in range(0, total_inserts, batch_size):
                for i in range(batch, min(batch + batch_size, total_inserts)):
                    tracker.track_query_performance(
                        query_id=f"db_stress_{i}",
                        response_time=random.uniform(0.1, 3.0),
                        accuracy=random.uniform(0.6, 1.0),
                        resources={
                            "memory_usage_mb": random.uniform(50, 300),
                            "cache_hit_rate": random.uniform(0.4, 0.95),
                            "agent_count": random.randint(1, 8),
                            "parallel_execution": random.choice([True, False]),
                            "query_complexity": random.uniform(0.2, 1.0)
                        },
                        success=random.random() > 0.02,  # 98% success rate
                        metadata={"db_stress": True, "batch": batch}
                    )
                
                # Force buffer flush every batch
                await asyncio.sleep(0.01)
            
            # Wait for final flush
            await asyncio.sleep(2)
            
            # Verify data integrity
            summary = await tracker.get_performance_summary()
            stored_queries = summary.get("total_queries", 0)
            
            if stored_queries >= total_inserts * 0.95:  # Allow 5% loss
                details = f"Stored {stored_queries}/{total_inserts} queries"
            else:
                success = False
                details = f"Only stored {stored_queries}/{total_inserts} queries"
            
            await tracker.stop_tracking()
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Database Stress", success, duration, details)
        return success
    
    async def test_error_recovery(self):
        """Test system recovery from errors."""
        print("\n🔥 Testing Error Recovery...")
        
        start_time = time.time()
        success = True
        details = ""
        
        try:
            # Test with invalid configurations
            invalid_configs = [
                {"db_path": "/invalid/path/that/does/not/exist.db"},
                {"max_concurrent_queries": -1},
                {"research_cycle_interval": 0},
                {"emergence_threshold": 2.0},  # Invalid range
                {"breakthrough_threshold": -0.5}  # Invalid range
            ]
            
            recovery_count = 0
            
            for i, invalid_config in enumerate(invalid_configs):
                try:
                    # Try to create components with invalid configs
                    if i % 2 == 0:
                        tracker = UnifiedPerformanceTracker(invalid_config)
                        await tracker.start_tracking()
                        await tracker.stop_tracking()
                    else:
                        orchestrator = AutonomousResearchOrchestrator(invalid_config)
                        # Should handle invalid config gracefully
                    
                    recovery_count += 1
                except Exception:
                    # Expected to fail, but system should recover
                    pass
            
            # Test with valid config after invalid ones
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
            
            if recovery_count >= 2:  # At least some should handle gracefully
                details = f"Recovered from {recovery_count}/5 invalid configs"
            else:
                success = False
                details = f"Only recovered from {recovery_count}/5 invalid configs"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Error Recovery", success, duration, details)
        return success
    
    async def test_concurrent_benchmarks(self):
        """Test concurrent benchmark execution."""
        print("\n🔥 Testing Concurrent Benchmarks...")
        
        start_time = time.time()
        success = True
        details = ""
        
        try:
            config = {"results_dir": os.path.join(self.temp_dir, "benchmark_stress")}
            
            # Run multiple benchmarks concurrently
            tasks = []
            for i in range(5):
                benchmark = SelfEvolutionBenchmark(config)
                task = asyncio.create_task(benchmark.benchmark_current_version())
                tasks.append(task)
            
            # Wait for all benchmarks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            successful_benchmarks = sum(1 for result in results if not isinstance(result, Exception))
            failed_benchmarks = sum(1 for result in results if isinstance(result, Exception))
            
            if successful_benchmarks >= 3:  # At least 3 should succeed
                details = f"{successful_benchmarks} successful, {failed_benchmarks} failed"
            else:
                success = False
                details = f"Only {successful_benchmarks} successful, {failed_benchmarks} failed"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("Concurrent Benchmarks", success, duration, details)
        return success
    
    async def test_system_resources(self):
        """Test system resource usage."""
        print("\n🔥 Testing System Resources...")
        
        start_time = time.time()
        success = True
        details = ""
        
        try:
            # Monitor system resources during stress test
            process = psutil.Process()
            
            # Get initial resources
            initial_memory = process.memory_info().rss / (1024 * 1024)
            initial_cpu = process.cpu_percent()
            
            # Run intensive operations
            config = {"db_path": os.path.join(self.temp_dir, "resource_test.db")}
            tracker = UnifiedPerformanceTracker(config)
            await tracker.start_tracking()
            
            # Generate lots of data
            for i in range(2000):
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
                
                if i % 200 == 0:
                    await asyncio.sleep(0.01)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check final resources
            final_memory = process.memory_info().rss / (1024 * 1024)
            final_cpu = process.cpu_percent()
            
            memory_increase = final_memory - initial_memory
            
            await tracker.stop_tracking()
            
            # Check if resources are reasonable
            if memory_increase < 300 and final_cpu < 80:
                details = f"Memory: +{memory_increase:.1f}MB, CPU: {final_cpu:.1f}%"
            else:
                success = False
                details = f"High resource usage: Memory +{memory_increase:.1f}MB, CPU {final_cpu:.1f}%"
            
        except Exception as e:
            success = False
            details = f"Exception: {e}"
        
        duration = time.time() - start_time
        self.log_result("System Resources", success, duration, details)
        return success
    
    def print_summary(self):
        """Print stress test summary."""
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("🔥 ICEBURG AUTONOMOUS SYSTEM STRESS TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Time: {total_time:.2f}s")
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
        
        print("\n" + "="*60)
        
        if failed_tests == 0:
            print("🎉 ALL STRESS TESTS PASSED! System is robust and ready for production!")
        else:
            print(f"⚠️  {failed_tests} tests failed. System needs attention before production.")
        
        print("="*60)


async def main():
    """Run all stress tests."""
    print("🚀 Starting ICEBURG Autonomous System Stress Tests")
    print("="*60)
    
    suite = StressTestSuite()
    
    try:
        # Run all stress tests
        await suite.test_high_frequency_performance_tracking()
        await suite.test_concurrent_research_cycles()
        await suite.test_evolution_pipeline_stress()
        await suite.test_memory_stress()
        await suite.test_database_stress()
        await suite.test_error_recovery()
        await suite.test_concurrent_benchmarks()
        await suite.test_system_resources()
        
        # Print summary
        suite.print_summary()
        
    finally:
        # Cleanup
        suite.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
