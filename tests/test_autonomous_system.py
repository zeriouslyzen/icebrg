"""
Comprehensive tests for ICEBURG Autonomous Self-Evolution System

Tests all components of the autonomous system including:
- Performance tracking
- Specification generation
- Benchmark execution
- Evolution pipeline
- Autonomous research orchestrator
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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


class TestUnifiedPerformanceTracker:
    """Test unified performance tracking system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def tracker(self, temp_dir):
        """Create performance tracker with temp directory."""
        config = {"db_path": os.path.join(temp_dir, "test_metrics.db")}
        return UnifiedPerformanceTracker(config)
    
    @pytest.mark.asyncio
    async def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker is not None
        assert tracker.db_path is not None
        assert tracker.tracking_active is False
    
    @pytest.mark.asyncio
    async def test_tracking_lifecycle(self, tracker):
        """Test starting and stopping tracking."""
        await tracker.start_tracking()
        assert tracker.tracking_active is True
        
        await tracker.stop_tracking()
        assert tracker.tracking_active is False
    
    def test_track_query_performance(self, tracker):
        """Test tracking query performance."""
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
    
    @pytest.mark.asyncio
    async def test_performance_summary(self, tracker):
        """Test getting performance summary."""
        # Add some test data
        tracker.track_query_performance(
            query_id="test_001",
            response_time=2.0,
            accuracy=0.8,
            resources={"memory_usage_mb": 100, "cache_hit_rate": 0.7, "agent_count": 3, "parallel_execution": True, "query_complexity": 0.5},
            success=True
        )
        
        # Start tracking to enable background tasks
        await tracker.start_tracking()
        await asyncio.sleep(0.1)  # Let buffer flush
        
        summary = await tracker.get_performance_summary()
        assert "total_queries" in summary
        assert summary["total_queries"] >= 0


class TestSpecificationGenerator:
    """Test specification generation system."""
    
    @pytest.fixture
    def spec_generator(self):
        """Create specification generator."""
        return SpecificationGenerator()
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self, spec_generator):
        """Test performance analysis."""
        analysis = await spec_generator.analyze_system_performance()
        
        assert isinstance(analysis, PerformanceAnalysis)
        assert analysis.timestamp > 0
        assert isinstance(analysis.bottlenecks, list)
        assert isinstance(analysis.slow_components, list)
        assert isinstance(analysis.memory_hogs, list)
        assert isinstance(analysis.accuracy_issues, list)
        assert isinstance(analysis.recommendations, list)
    
    def test_identify_optimization_opportunities(self, spec_generator):
        """Test identifying optimization opportunities."""
        # Create mock analysis
        analysis = PerformanceAnalysis(
            timestamp=1234567890,
            bottlenecks=["response_time", "memory_usage"],
            slow_components=["protocol", "agents"],
            memory_hogs=["high_memory_usage"],
            accuracy_issues=["low_accuracy"],
            recommendations=["optimize_memory", "improve_accuracy"]
        )
        
        opportunities = spec_generator.identify_optimization_opportunities(analysis)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        for opportunity in opportunities:
            assert isinstance(opportunity, OptimizationOpportunity)
            assert opportunity.name
            assert opportunity.description
            assert 0 <= opportunity.impact_score <= 1
            assert 0 <= opportunity.effort_score <= 1
            assert opportunity.risk_level in ["low", "medium", "high"]
    
    def test_generate_improvement_spec(self, spec_generator):
        """Test generating improvement specifications."""
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
        
        spec = spec_generator.generate_improvement_spec(opportunity)
        
        assert spec.name
        assert spec.inputs
        assert spec.outputs
        assert spec.preconditions
        assert spec.postconditions
        assert spec.implementation
        assert spec.optimization_targets
        assert spec.safety_constraints
    
    def test_validate_spec_safety(self, spec_generator):
        """Test safety validation."""
        from iceburg.evolution.specification_generator import TaskSpec
        
        spec = TaskSpec(
            name="test_spec",
            inputs=[{"name": "input", "type": "string"}],
            outputs=[{"name": "output", "type": "string"}],
            preconditions=["input is not empty"],
            postconditions=["output is valid"],
            implementation={"memory_usage": 0.1},  # 10% memory increase
            optimization_targets=["performance"]
        )
        
        validation = spec_generator.validate_spec_safety(spec)
        
        assert validation.spec_name == "test_spec"
        assert isinstance(validation.passed, bool)
        assert isinstance(validation.violations, list)
        assert isinstance(validation.warnings, list)
        assert validation.risk_assessment in ["low", "medium", "high", "critical"]


class TestSelfEvolutionBenchmark:
    """Test self-evolution benchmark suite."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def benchmark(self, temp_dir):
        """Create benchmark suite with temp directory."""
        config = {"results_dir": os.path.join(temp_dir, "benchmark_results")}
        return SelfEvolutionBenchmark(config)
    
    @pytest.mark.asyncio
    async def test_benchmark_current_version(self, benchmark):
        """Test benchmarking current version."""
        result = await benchmark.benchmark_current_version()
        
        assert result.test_name == "current_version_comprehensive"
        assert result.timestamp > 0
        assert result.duration > 0
        assert isinstance(result.success, bool)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.errors, list)
    
    def test_validate_safety_constraints(self, benchmark):
        """Test safety constraint validation."""
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
        assert safety_report.risk_level in ["low", "medium", "high", "critical"]
    
    def test_compare_versions(self, benchmark):
        """Test version comparison."""
        from benchmarks.self_evolution_benchmarks import BenchmarkResults
        
        baseline = BenchmarkResults(
            test_name="baseline",
            timestamp=1234567890,
            duration=10.0,
            success=True,
            metrics={"response_time": 5.0, "accuracy": 0.8}
        )
        
        improved = BenchmarkResults(
            test_name="improved",
            timestamp=1234567890,
            duration=10.0,
            success=True,
            metrics={"response_time": 4.0, "accuracy": 0.85}
        )
        
        comparison = benchmark.compare_versions(baseline, improved)
        
        assert comparison.baseline == baseline
        assert comparison.improved == improved
        assert isinstance(comparison.improvements, dict)
        assert isinstance(comparison.regressions, dict)
        assert isinstance(comparison.overall_improvement, float)
        assert comparison.recommendation


class TestAutonomousResearchOrchestrator:
    """Test autonomous research orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create research orchestrator."""
        config = {
            "max_concurrent_queries": 3,
            "research_cycle_interval": 10,  # 10 seconds for testing
            "emergence_threshold": 0.6,
            "breakthrough_threshold": 0.7
        }
        return AutonomousResearchOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.active is False
        assert orchestrator.cycle_count == 0
        assert orchestrator.max_concurrent_queries == 3
        assert orchestrator.research_cycle_interval == 10
    
    @pytest.mark.asyncio
    async def test_generate_curiosity_queries(self, orchestrator):
        """Test generating curiosity queries."""
        queries = await orchestrator._generate_curiosity_queries()
        
        assert isinstance(queries, list)
        assert len(queries) <= orchestrator.max_concurrent_queries
        
        for query in queries:
            assert isinstance(query, ResearchQuery)
            assert query.query_id
            assert query.query_text
            assert 0 <= query.complexity <= 1
            assert query.domain
            assert 1 <= query.priority <= 10
            assert query.generated_at > 0
    
    @pytest.mark.asyncio
    async def test_execute_research_queries(self, orchestrator):
        """Test executing research queries."""
        queries = [
            ResearchQuery(
                query_id="test_001",
                query_text="What is artificial intelligence?",
                complexity=0.5,
                domain="artificial_intelligence",
                priority=5,
                generated_at=1234567890
            )
        ]
        
        results = await orchestrator._execute_research_queries(queries)
        
        assert isinstance(results, list)
        assert len(results) == len(queries)
        
        for result in results:
            assert isinstance(result, ResearchResult)
            assert result.query_id
            assert result.result_text
            assert isinstance(result.success, bool)
            assert result.execution_time > 0
            assert 0 <= result.quality_score <= 1
    
    def test_get_research_status(self, orchestrator):
        """Test getting research status."""
        status = orchestrator.get_research_status()
        
        assert isinstance(status, dict)
        assert "active" in status
        assert "cycle_count" in status
        assert "current_queries" in status
        assert "results_history" in status
        assert "emergence_patterns" in status
        assert "improvement_queue" in status
        assert "active_agents" in status


class TestEvolutionPipeline:
    """Test evolution pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create evolution pipeline."""
        config = {
            "max_concurrent_jobs": 2,
            "timeout_per_stage": 30,  # 30 seconds for testing
            "auto_approve_threshold": 0.8
        }
        return EvolutionPipeline(config)
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.max_concurrent_jobs == 2
        assert pipeline.timeout_per_stage == 30
        assert pipeline.auto_approve_threshold == 0.8
        assert len(pipeline.jobs) == 0
        assert len(pipeline.active_jobs) == 0
    
    @pytest.mark.asyncio
    async def test_evolve_system(self, pipeline):
        """Test system evolution."""
        job_id = await pipeline.evolve_system("test_trigger")
        
        assert job_id is not None
        assert job_id in pipeline.jobs
        assert job_id in pipeline.active_jobs
        
        job = pipeline.jobs[job_id]
        assert job.job_id == job_id
        assert job.stage in [EvolutionStage.ANALYSIS, EvolutionStage.FAILED]
        assert job.status in ["pending", "running", "failed"]
        assert job.created_at > 0
    
    def test_get_job_status(self, pipeline):
        """Test getting job status."""
        # Create a test job
        from iceburg.evolution.evolution_pipeline import EvolutionJob, EvolutionStage
        import time
        
        job = EvolutionJob(
            job_id="test_job",
            stage=EvolutionStage.ANALYSIS,
            created_at=time.time()
        )
        pipeline.jobs["test_job"] = job
        
        status = pipeline.get_job_status("test_job")
        
        assert status is not None
        assert status["job_id"] == "test_job"
        assert status["stage"] == "analysis"
        assert status["status"] == "pending"
        assert status["created_at"] > 0
    
    def test_get_pipeline_status(self, pipeline):
        """Test getting pipeline status."""
        status = pipeline.get_pipeline_status()
        
        assert isinstance(status, dict)
        assert "total_jobs" in status
        assert "active_jobs" in status
        assert "completed_jobs" in status
        assert "failed_jobs" in status
        assert "success_rate" in status
        
        assert status["total_jobs"] >= 0
        assert status["active_jobs"] >= 0
        assert status["completed_jobs"] >= 0
        assert status["failed_jobs"] >= 0
        assert 0 <= status["success_rate"] <= 1


class TestIntegration:
    """Integration tests for the autonomous system."""
    
    @pytest.mark.asyncio
    async def test_full_autonomous_cycle(self):
        """Test a complete autonomous research cycle."""
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
        
        # Analyze performance
        analysis = await spec_generator.analyze_system_performance()
        assert isinstance(analysis, PerformanceAnalysis)
        
        # Generate opportunities
        opportunities = spec_generator.identify_optimization_opportunities(analysis)
        assert len(opportunities) >= 0
        
        # Test research orchestrator
        queries = await orchestrator._generate_curiosity_queries()
        assert len(queries) <= 2
        
        # Stop tracking
        await tracker.stop_tracking()
        
        print("✅ Full autonomous cycle integration test passed!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
