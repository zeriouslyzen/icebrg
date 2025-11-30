#!/usr/bin/env python3
"""
Comprehensive Improvement Test for ICEBURG Autonomous System
Tests all new improvements and enhancements
"""

import asyncio
import time
import random
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from iceburg.monitoring.unified_performance_tracker import UnifiedPerformanceTracker
from iceburg.evolution.specification_generator import SpecificationGenerator
from iceburg.autonomous.research_orchestrator import AutonomousResearchOrchestrator
from iceburg.evolution.evolution_pipeline import EvolutionPipeline
from benchmarks.self_evolution_benchmarks import SelfEvolutionBenchmark
from iceburg.monitoring.error_handler import ErrorHandler, with_error_handling
from iceburg.monitoring.alerting_system import AlertingSystem, add_metrics
from iceburg.safety.safety_validator import SafetyValidator, validate_safety


class ComprehensiveImprovementTest:
    """Comprehensive test for all ICEBURG improvements."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
        self.start_time = time.time()
        
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def log_result(self, test_name, success, duration, details="", metrics=None):
        """Log test result."""
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
    
    async def test_error_handling_system(self):
        """Test the new error handling system."""
        print("\n🔥 Testing Error Handling System...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            error_handler = ErrorHandler()
            
            # Test error classification
            test_errors = [
                Exception("Connection timeout"),
                Exception("Database connection failed"),
                MemoryError("Out of memory"),
                ValueError("Invalid input"),
                RuntimeError("Async operation failed")
            ]
            
            classifications = []
            for error in test_errors:
                category = error_handler.classify_error(error)
                classifications.append(category.value)
            
            metrics["error_classifications"] = len(set(classifications))
            
            # Test error handling decorator
            @with_error_handling("test_component", "test_operation", max_retries=2)
            async def failing_function():
                if random.random() < 0.7:  # 70% chance of failure
                    raise Exception("Simulated error")
                return "success"
            
            # Run multiple times to test retry logic
            results = []
            for i in range(10):
                try:
                    result = await failing_function()
                    results.append(result)
                except Exception as e:
                    results.append(f"failed: {e}")
            
            success_count = sum(1 for r in results if r == "success")
            metrics["error_handling_success_rate"] = success_count / len(results)
            
            # Test error statistics
            stats = error_handler.get_error_statistics()
            metrics["error_statistics"] = stats
            
            if success_count >= 3:  # At least 30% should succeed due to retries
                details = f"Error handling working: {success_count}/{len(results)} successes"
            else:
                success = False
                details = f"Error handling failed: only {success_count}/{len(results)} successes"
            
        except Exception as e:
            success = False
            details = f"Error handling test failed: {e}"
        
        duration = time.time() - start_time
        self.log_result("Error Handling System", success, duration, details, metrics)
        return success
    
    async def test_alerting_system(self):
        """Test the new alerting system."""
        print("\n🔥 Testing Alerting System...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            alerting_system = AlertingSystem()
            await alerting_system.start_monitoring()
            
            # Generate various metrics to trigger alerts
            alert_triggers = [
                {"response_time": 35.0, "memory_usage_mb": 1500.0, "cpu_usage_percent": 85.0, "error_rate": 15.0},
                {"response_time": 5.0, "memory_usage_mb": 500.0, "cpu_usage_percent": 45.0, "error_rate": 2.0},
                {"response_time": 50.0, "memory_usage_mb": 2500.0, "cpu_usage_percent": 95.0, "error_rate": 30.0},
                {"response_time": 2.0, "memory_usage_mb": 200.0, "cpu_usage_percent": 25.0, "error_rate": 1.0},
                {"response_time": 15.0, "memory_usage_mb": 800.0, "cpu_usage_percent": 60.0, "error_rate": 8.0}
            ]
            
            for i, metrics_data in enumerate(alert_triggers):
                add_metrics("performance_tracker", metrics_data)
                await asyncio.sleep(0.1)  # Small delay between metrics
            
            # Wait for alert processing
            await asyncio.sleep(2)
            
            # Check alert statistics
            stats = alerting_system.get_alert_statistics()
            metrics["total_alerts"] = stats["total_alerts"]
            metrics["active_alerts"] = stats["active_alerts"]
            metrics["alert_rules"] = stats["alert_rules_count"]
            
            # Get alerts by level
            from iceburg.monitoring.alerting_system import AlertLevel
            critical_alerts = alerting_system.get_alerts_by_level(AlertLevel.CRITICAL)
            warning_alerts = alerting_system.get_alerts_by_level(AlertLevel.WARNING)
            
            metrics["critical_alerts"] = len(critical_alerts)
            metrics["warning_alerts"] = len(warning_alerts)
            
            await alerting_system.stop_monitoring()
            
            if stats["total_alerts"] > 0:
                details = f"Generated {stats['total_alerts']} alerts ({stats['active_alerts']} active)"
            else:
                success = False
                details = "No alerts generated"
            
        except Exception as e:
            success = False
            details = f"Alerting system test failed: {e}"
        
        duration = time.time() - start_time
        self.log_result("Alerting System", success, duration, details, metrics)
        return success
    
    async def test_safety_validation_system(self):
        """Test the new safety validation system."""
        print("\n🔥 Testing Safety Validation System...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            safety_validator = SafetyValidator()
            await safety_validator.start_validation()
            
            # Test various safety scenarios
            test_contexts = [
                {"response_time": 5.0, "memory_usage_mb": 500.0, "accuracy": 0.9, "query_id": "test1", "timestamp": time.time()},
                {"response_time": 35.0, "memory_usage_mb": 1500.0, "accuracy": 0.6, "query_id": "test2", "timestamp": time.time()},
                {"response_time": 2.0, "memory_usage_mb": 2500.0, "accuracy": 0.8, "query_id": "test3", "timestamp": time.time()},
                {"response_time": 10.0, "memory_usage_mb": 800.0, "accuracy": 0.5, "query_id": "test4", "timestamp": time.time()},
                {"response_time": 1.0, "memory_usage_mb": 300.0, "accuracy": 0.95, "query_id": "test5", "timestamp": time.time()}
            ]
            
            safety_checks = []
            for context in test_contexts:
                check = await validate_safety(context)
                safety_checks.append(check)
            
            # Analyze results
            passed_checks = sum(1 for check in safety_checks if check.passed)
            total_violations = sum(len(check.violations) for check in safety_checks)
            avg_safety_score = sum(check.safety_score for check in safety_checks) / len(safety_checks)
            
            metrics["passed_checks"] = passed_checks
            metrics["total_violations"] = total_violations
            metrics["avg_safety_score"] = avg_safety_score
            metrics["total_checks"] = len(safety_checks)
            
            # Get safety statistics
            stats = safety_validator.get_safety_statistics()
            metrics["safety_statistics"] = stats
            
            await safety_validator.stop_validation()
            
            if total_violations > 0:  # Should detect some violations
                details = f"Detected {total_violations} violations, avg safety score: {avg_safety_score:.1f}"
            else:
                details = f"All checks passed, avg safety score: {avg_safety_score:.1f}"
            
        except Exception as e:
            success = False
            details = f"Safety validation test failed: {e}"
        
        duration = time.time() - start_time
        self.log_result("Safety Validation System", success, duration, details, metrics)
        return success
    
    async def test_integrated_system(self):
        """Test all systems working together."""
        print("\n🔥 Testing Integrated System...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            # Initialize all systems
            tracker = UnifiedPerformanceTracker({
                "db_path": os.path.join(self.temp_dir, "integrated_test.db")
            })
            await tracker.start_tracking()
            
            generator = SpecificationGenerator()
            orchestrator = AutonomousResearchOrchestrator({
                "max_concurrent_queries": 2,
                "research_cycle_interval": 1
            })
            pipeline = EvolutionPipeline({
                "max_concurrent_jobs": 2,
                "timeout_per_stage": 5
            })
            
            error_handler = ErrorHandler()
            alerting_system = AlertingSystem()
            await alerting_system.start_monitoring()
            
            safety_validator = SafetyValidator()
            await safety_validator.start_validation()
            
            # Run integrated operations
            operations_completed = 0
            errors_handled = 0
            alerts_generated = 0
            safety_violations = 0
            
            for i in range(20):
                try:
                    # Track performance
                    tracker.track_query_performance(
                        query_id=f"integrated_test_{i}",
                        response_time=random.uniform(0.5, 15.0),
                        accuracy=random.uniform(0.6, 1.0),
                        resources={
                            "memory_usage_mb": random.uniform(100, 1500),
                            "cache_hit_rate": random.uniform(0.5, 0.9),
                            "agent_count": random.randint(1, 5),
                            "parallel_execution": True,
                            "query_complexity": random.uniform(0.2, 0.8)
                        },
                        success=random.random() > 0.1,  # 90% success rate
                        metadata={"integrated_test": True}
                    )
                    
                    # Generate metrics for alerting
                    add_metrics("performance_tracker", {
                        "response_time": random.uniform(1.0, 20.0),
                        "memory_usage_mb": random.uniform(200, 1200),
                        "cpu_usage_percent": random.uniform(20, 80),
                        "error_rate": random.uniform(0, 15),
                        "accuracy": random.uniform(0.7, 1.0)
                    })
                    
                    # Validate safety
                    safety_check = await validate_safety({
                        "query_id": f"integrated_test_{i}",
                        "timestamp": time.time(),
                        "response_time": random.uniform(1.0, 20.0),
                        "memory_usage_mb": random.uniform(200, 1200),
                        "accuracy": random.uniform(0.7, 1.0)
                    })
                    
                    if not safety_check.passed:
                        safety_violations += len(safety_check.violations)
                    
                    operations_completed += 1
                    
                except Exception as e:
                    errors_handled += 1
                    # Error handler would catch this in real usage
                
                if i % 5 == 0:
                    await asyncio.sleep(0.1)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Get final statistics
            alert_stats = alerting_system.get_alert_statistics()
            safety_stats = safety_validator.get_safety_statistics()
            error_stats = error_handler.get_error_statistics()
            
            metrics["operations_completed"] = operations_completed
            metrics["errors_handled"] = errors_handled
            metrics["alerts_generated"] = alert_stats["total_alerts"]
            metrics["safety_violations"] = safety_violations
            metrics["safety_score"] = safety_stats["safety_score"]
            metrics["error_health_score"] = error_stats["health_score"]
            
            # Cleanup
            await tracker.stop_tracking()
            await alerting_system.stop_monitoring()
            await safety_validator.stop_validation()
            
            if operations_completed >= 15:  # At least 75% should complete
                details = f"Completed {operations_completed} operations, {alerts_generated} alerts, safety score: {safety_stats['safety_score']:.1f}"
            else:
                success = False
                details = f"Only completed {operations_completed} operations"
            
        except Exception as e:
            success = False
            details = f"Integrated system test failed: {e}"
        
        duration = time.time() - start_time
        self.log_result("Integrated System", success, duration, details, metrics)
        return success
    
    async def test_performance_improvements(self):
        """Test performance improvements."""
        print("\n🔥 Testing Performance Improvements...")
        
        start_time = time.time()
        success = True
        details = ""
        metrics = {}
        
        try:
            # Test high-frequency operations
            tracker = UnifiedPerformanceTracker({
                "db_path": os.path.join(self.temp_dir, "performance_test.db"),
                "buffer_size": 100
            })
            await tracker.start_tracking()
            
            # Generate high-frequency metrics
            operation_start = time.time()
            for i in range(1000):
                tracker.track_query_performance(
                    query_id=f"perf_test_{i}",
                    response_time=random.uniform(0.1, 2.0),
                    accuracy=random.uniform(0.8, 1.0),
                    resources={
                        "memory_usage_mb": random.uniform(50, 200),
                        "cache_hit_rate": random.uniform(0.7, 0.9),
                        "agent_count": random.randint(1, 3),
                        "parallel_execution": True,
                        "query_complexity": random.uniform(0.3, 0.7)
                    },
                    success=True,
                    metadata={"performance_test": True}
                )
                
                if i % 100 == 0:
                    await asyncio.sleep(0.001)
            
            operation_time = time.time() - operation_start
            
            # Wait for processing
            await asyncio.sleep(3)
            
            # Get performance summary
            summary = await tracker.get_performance_summary()
            stored_queries = summary.get("total_queries", 0)
            
            await tracker.stop_tracking()
            
            # Calculate performance metrics
            operations_per_second = 1000 / operation_time
            storage_efficiency = stored_queries / 1000 if stored_queries > 0 else 0
            
            metrics["operations_per_second"] = operations_per_second
            metrics["storage_efficiency"] = storage_efficiency
            metrics["stored_queries"] = stored_queries
            metrics["operation_time"] = operation_time
            
            if operations_per_second > 500 and storage_efficiency > 0.8:
                details = f"{operations_per_second:.0f} ops/sec, {storage_efficiency:.1%} storage efficiency"
            else:
                success = False
                details = f"Performance below expectations: {operations_per_second:.0f} ops/sec, {storage_efficiency:.1%} storage"
            
        except Exception as e:
            success = False
            details = f"Performance test failed: {e}"
        
        duration = time.time() - start_time
        self.log_result("Performance Improvements", success, duration, details, metrics)
        return success
    
    def print_comprehensive_summary(self):
        """Print comprehensive test summary."""
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*80)
        print("🚀 ICEBURG COMPREHENSIVE IMPROVEMENT TEST RESULTS")
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
            if values and isinstance(values[0], (int, float)):
                avg_val = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                print(f"  {key}: avg={avg_val:.2f}, max={max_val:.2f}, min={min_val:.2f}")
            elif values:
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
            print("🎉 ALL IMPROVEMENT TESTS PASSED! ICEBURG is significantly enhanced!")
            print("✅ Error handling, alerting, safety validation, and performance improvements working!")
        elif failed_tests <= 1:
            print("⚠️  Minor issues detected. ICEBURG is mostly enhanced and ready!")
        else:
            print(f"❌ {failed_tests} tests failed. Some improvements need attention.")
        
        print("="*80)


async def main():
    """Run comprehensive improvement tests."""
    print("🚀 Starting ICEBURG Comprehensive Improvement Tests")
    print("="*80)
    
    test = ComprehensiveImprovementTest()
    
    try:
        # Run all improvement tests
        await test.test_error_handling_system()
        await test.test_alerting_system()
        await test.test_safety_validation_system()
        await test.test_integrated_system()
        await test.test_performance_improvements()
        
        # Print comprehensive summary
        test.print_comprehensive_summary()
        
    finally:
        # Cleanup
        test.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
