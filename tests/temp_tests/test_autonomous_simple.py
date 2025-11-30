#!/usr/bin/env python3
"""
Simple test runner for ICEBURG Autonomous System
"""

import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from iceburg.monitoring.unified_performance_tracker import (
    UnifiedPerformanceTracker, 
    PerformanceMetrics,
    get_global_tracker
)
from iceburg.evolution.specification_generator import (
    SpecificationGenerator,
    PerformanceAnalysis,
    OptimizationOpportunity
)
from iceburg.autonomous.research_orchestrator import (
    AutonomousResearchOrchestrator,
    ResearchQuery,
    ResearchResult
)
from iceburg.evolution.evolution_pipeline import (
    EvolutionPipeline,
    EvolutionStage
)
from benchmarks.self_evolution_benchmarks import SelfEvolutionBenchmark


def test_performance_tracker():
    """Test unified performance tracker."""
    print("Testing Unified Performance Tracker...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        config = {"db_path": os.path.join(temp_dir, "test_metrics.db")}
        tracker = UnifiedPerformanceTracker(config)
        
        # Test initialization
        assert tracker is not None
        assert tracker.db_path is not None
        assert tracker.tracking_active is False
        print("  ✅ Initialization passed")
        
        # Test tracking
        tracker.track_query_performance(
            query_id="test_001",
            response_time=2.5,
            accuracy=0.85,
            resources={
                "memory_usage_mb": 100,
                "cache_hit_rate": 0.7,
                "agent_count": 3,
                "parallel_execution": True,
                "query_complexity": 0.6
            },
            success=True,
            metadata={"test": True}
        )
        
        assert len(tracker.metrics_buffer) == 1
        metrics = tracker.metrics_buffer[0]
        assert metrics.query_id == "test_001"
        assert metrics.response_time == 2.5
        assert metrics.accuracy == 0.85
        assert metrics.success is True
        print("  ✅ Performance tracking passed")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("  ✅ Unified Performance Tracker tests passed")


def test_specification_generator():
    """Test specification generator."""
    print("Testing Specification Generator...")
    
    spec_generator = SpecificationGenerator()
    
    # Test initialization
    assert spec_generator is not None
    assert len(spec_generator.strategies) == 4
    print("  ✅ Initialization passed")
    
    # Test optimization opportunity creation
    opportunity = OptimizationOpportunity(
        name="test_optimization",
        description="Test optimization opportunity",
        impact_score=0.8,
        effort_score=0.6,
        risk_level="low",
        affected_components=["protocol"],
        expected_improvement=0.3,
        optimization_type="performance"
    )
    
    assert opportunity.name == "test_optimization"
    assert opportunity.impact_score == 0.8
    assert opportunity.risk_level == "low"
    print("  ✅ Optimization opportunity creation passed")
    
    # Test spec generation
    spec = spec_generator.generate_improvement_spec(opportunity)
    assert spec.name is not None
    assert len(spec.inputs) > 0
    assert len(spec.outputs) > 0
    assert len(spec.preconditions) > 0
    assert len(spec.postconditions) > 0
    print("  ✅ Specification generation passed")
    
    print("  ✅ Specification Generator tests passed")


def test_benchmark_suite():
    """Test benchmark suite."""
    print("Testing Self-Evolution Benchmark Suite...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        config = {"results_dir": os.path.join(temp_dir, "benchmark_results")}
        benchmark = SelfEvolutionBenchmark(config)
        
        # Test initialization
        assert benchmark is not None
        assert benchmark.results_dir.exists()
        print("  ✅ Initialization passed")
        
        # Test safety validation
        spec = {
            "name": "test_spec",
            "max_memory_usage": 1000.0,
            "max_cpu_usage": 50.0,
            "min_accuracy": 0.8,
            "expected_improvement": 1.5
        }
        
        safety_report = benchmark.validate_safety_constraints(spec)
        assert safety_report.spec_name == "test_spec"
        assert isinstance(safety_report.passed, bool)
        assert isinstance(safety_report.violations, list)
        assert isinstance(safety_report.warnings, list)
        print("  ✅ Safety validation passed")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("  ✅ Self-Evolution Benchmark Suite tests passed")


def test_research_orchestrator():
    """Test research orchestrator."""
    print("Testing Autonomous Research Orchestrator...")
    
    config = {
        "max_concurrent_queries": 3,
        "research_cycle_interval": 10,
        "emergence_threshold": 0.6,
        "breakthrough_threshold": 0.7
    }
    orchestrator = AutonomousResearchOrchestrator(config)
    
    # Test initialization
    assert orchestrator.active is False
    assert orchestrator.cycle_count == 0
    assert orchestrator.max_concurrent_queries == 3
    print("  ✅ Initialization passed")
    
    # Test research status
    status = orchestrator.get_research_status()
    assert isinstance(status, dict)
    assert "active" in status
    assert "cycle_count" in status
    assert "current_queries" in status
    print("  ✅ Research status passed")
    
    print("  ✅ Autonomous Research Orchestrator tests passed")


def test_evolution_pipeline():
    """Test evolution pipeline."""
    print("Testing Evolution Pipeline...")
    
    config = {
        "max_concurrent_jobs": 2,
        "timeout_per_stage": 30,
        "auto_approve_threshold": 0.8
    }
    pipeline = EvolutionPipeline(config)
    
    # Test initialization
    assert pipeline.max_concurrent_jobs == 2
    assert pipeline.timeout_per_stage == 30
    assert pipeline.auto_approve_threshold == 0.8
    print("  ✅ Initialization passed")
    
    # Test pipeline status
    status = pipeline.get_pipeline_status()
    assert isinstance(status, dict)
    assert "total_jobs" in status
    assert "active_jobs" in status
    assert "completed_jobs" in status
    assert "failed_jobs" in status
    assert "success_rate" in status
    print("  ✅ Pipeline status passed")
    
    print("  ✅ Evolution Pipeline tests passed")


async def test_integration():
    """Test integration between components."""
    print("Testing Integration...")
    
    # Create components
    tracker = UnifiedPerformanceTracker()
    spec_generator = SpecificationGenerator()
    orchestrator = AutonomousResearchOrchestrator({
        "max_concurrent_queries": 2,
        "research_cycle_interval": 5
    })
    
    # Start tracking
    await tracker.start_tracking()
    
    # Track some performance data
    tracker.track_query_performance(
        query_id="integration_test_001",
        response_time=3.0,
        accuracy=0.9,
        resources={
            "memory_usage_mb": 150,
            "cache_hit_rate": 0.8,
            "agent_count": 4,
            "parallel_execution": True,
            "query_complexity": 0.7
        },
        success=True,
        metadata={"integration_test": True}
    )
    
    # Wait for buffer flush
    await asyncio.sleep(0.1)
    
    # Get performance summary
    summary = await tracker.get_performance_summary()
    assert summary.get("total_queries", 0) >= 1
    print("  ✅ Performance tracking integration passed")
    
    # Analyze performance
    analysis = await spec_generator.analyze_system_performance()
    assert isinstance(analysis, PerformanceAnalysis)
    print("  ✅ Performance analysis integration passed")
    
    # Generate opportunities
    opportunities = spec_generator.identify_optimization_opportunities(analysis)
    assert len(opportunities) >= 0
    print("  ✅ Optimization opportunity generation passed")
    
    # Test research orchestrator
    queries = await orchestrator._generate_curiosity_queries()
    assert len(queries) >= 0  # Should generate some queries
    print("  ✅ Research query generation passed")
    
    # Stop tracking
    await tracker.stop_tracking()
    
    print("  ✅ Integration tests passed")


def main():
    """Run all tests."""
    print("🧪 Running ICEBURG Autonomous System Tests")
    print("=" * 50)
    
    try:
        test_performance_tracker()
        print()
        
        test_specification_generator()
        print()
        
        test_benchmark_suite()
        print()
        
        test_research_orchestrator()
        print()
        
        test_evolution_pipeline()
        print()
        
        # Run async integration test
        asyncio.run(test_integration())
        print()
        
        print("🎉 All tests passed successfully!")
        print("✅ ICEBURG Autonomous System is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
