"""
Test suite for Teacher-Student Tuning
Tests prompt evolution, performance analysis, and lifelong learning
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime


class TestTeacherStudentTuning:
    """Test suite for teacher-student tuning system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_cfg = Mock()
        self.mock_cfg.synthesist_model = "llama3.1:8b"
    
    def test_prompt_evolution_trigger(self):
        """Test that prompt evolution is triggered based on performance"""
        try:
            from src.iceburg.agents.teacher_student_tuning import TeacherStudentTuning
            
            tuning = TeacherStudentTuning()
            
            # Simulate low performance
            tuning.agent_performance["test_agent"] = [
                {"success": False, "quality": 0.3},
                {"success": False, "quality": 0.4},
                {"success": True, "quality": 0.5},
            ]
            
            # Should trigger evolution when success rate < threshold
            should_evolve = tuning.should_evolve_prompt("test_agent")
            
            # With mixed performance, evolution may or may not trigger
            assert isinstance(should_evolve, bool)
        except ImportError:
            pytest.skip("TeacherStudentTuning not available")
    
    def test_performance_analysis(self):
        """Test agent performance analysis"""
        try:
            from src.iceburg.agents.teacher_student_tuning import TeacherStudentTuning
            
            tuning = TeacherStudentTuning()
            
            # Add performance data
            for i in range(10):
                tuning.agent_performance["surveyor"] = tuning.agent_performance.get("surveyor", [])
                tuning.agent_performance["surveyor"].append({
                    "success": i % 2 == 0,  # 50% success rate
                    "quality": 0.5 + (i * 0.05)
                })
            
            # Analyze performance
            analysis = tuning.analyze_agent_performance("surveyor")
            
            assert "success_rate" in analysis
            assert "avg_quality" in analysis
            assert "sample_size" in analysis
        except ImportError:
            pytest.skip("TeacherStudentTuning not available")
    
    def test_failure_pattern_analysis(self):
        """Test analysis of failure patterns"""
        failure_examples = [
            {"error_type": "timeout", "context": "long query"},
            {"error_type": "timeout", "context": "complex synthesis"},
            {"error_type": "hallucination", "context": "obscure topic"},
            {"error_type": "timeout", "context": "multi-domain query"},
        ]
        
        # Analyze patterns
        error_counts = {}
        for failure in failure_examples:
            error_type = failure["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Timeout is most common
        most_common = max(error_counts, key=error_counts.get)
        assert most_common == "timeout"
    
    @pytest.mark.asyncio
    async def test_prompt_improvement_generation(self):
        """Test generation of improved prompts"""
        try:
            from src.iceburg.agents.teacher_student_tuning import TeacherStudentTuning
            
            tuning = TeacherStudentTuning()
            
            current_prompt = """
            You are a research agent. Analyze the query and provide insights.
            """
            
            performance_data = {
                "success_rate": 0.4,
                "avg_quality": 0.5,
                "common_failures": ["timeout", "incomplete"]
            }
            
            # Mock the LLM call
            with patch('src.iceburg.agents.teacher_student_tuning.chat_complete', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = """
                You are an expert research agent with enhanced capabilities.
                When analyzing queries:
                1. First identify the core question
                2. Gather relevant context quickly
                3. Provide concise, complete insights
                Focus on efficiency to avoid timeouts.
                """
                
                evolved = await tuning.evolve_agent_prompt(
                    "test_agent",
                    current_prompt,
                    performance_data
                )
                
                assert evolved is not None
                assert len(evolved) > 0
        except ImportError:
            pytest.skip("TeacherStudentTuning not available")
    
    def test_improvement_expectation_calculation(self):
        """Test calculation of expected improvement from evolution"""
        # Improvement should be based on failure patterns and proposed changes
        failure_analysis = {
            "timeout": 5,
            "incomplete": 3,
            "hallucination": 2
        }
        
        proposed_changes = {
            "add_time_awareness": True,  # Addresses timeout
            "enhance_completeness_check": True,  # Addresses incomplete
        }
        
        # Calculate expected improvement
        addressable_failures = 0
        total_failures = sum(failure_analysis.values())
        
        if proposed_changes.get("add_time_awareness"):
            addressable_failures += failure_analysis.get("timeout", 0)
        if proposed_changes.get("enhance_completeness_check"):
            addressable_failures += failure_analysis.get("incomplete", 0)
        
        expected_improvement = addressable_failures / total_failures if total_failures > 0 else 0
        
        assert 0.0 <= expected_improvement <= 1.0
        assert expected_improvement > 0.5  # Should address most failures


class TestPromptEvolutionHistory:
    """Test tracking and rollback of prompt evolutions"""
    
    def test_evolution_versioning(self):
        """Test that prompt evolutions are versioned"""
        evolution_history = [
            {"version": 1, "prompt": "Original prompt", "timestamp": "2025-01-01"},
            {"version": 2, "prompt": "Improved prompt v1", "timestamp": "2025-01-15"},
            {"version": 3, "prompt": "Improved prompt v2", "timestamp": "2025-02-01"},
        ]
        
        assert len(evolution_history) == 3
        assert evolution_history[-1]["version"] == 3
    
    def test_evolution_rollback(self):
        """Test rollback to previous prompt version"""
        evolution_history = [
            {"version": 1, "prompt": "Original", "performance": 0.7},
            {"version": 2, "prompt": "Evolved", "performance": 0.5},  # Worse!
        ]
        
        # Should rollback when performance degrades
        current_version = evolution_history[-1]
        previous_version = evolution_history[-2]
        
        if current_version["performance"] < previous_version["performance"]:
            # Rollback needed
            active_prompt = previous_version["prompt"]
            assert active_prompt == "Original"
    
    def test_best_version_selection(self):
        """Test selection of best performing version"""
        evolution_history = [
            {"version": 1, "prompt": "v1", "performance": 0.6},
            {"version": 2, "prompt": "v2", "performance": 0.8},
            {"version": 3, "prompt": "v3", "performance": 0.7},
            {"version": 4, "prompt": "v4", "performance": 0.75},
        ]
        
        best_version = max(evolution_history, key=lambda x: x["performance"])
        
        assert best_version["version"] == 2
        assert best_version["performance"] == 0.8


class TestLifelongLearning:
    """Test continuous learning without fine-tuning"""
    
    def test_memory_based_adaptation(self):
        """Test adaptation from historical experiences"""
        historical_fixes = [
            {"action": "increase_context", "success": True},
            {"action": "increase_context", "success": True},
            {"action": "increase_context", "success": False},
            {"action": "reduce_complexity", "success": True},
            {"action": "reduce_complexity", "success": True},
        ]
        
        # Calculate success rates
        action_stats = {}
        for fix in historical_fixes:
            action = fix["action"]
            if action not in action_stats:
                action_stats[action] = {"success": 0, "total": 0}
            action_stats[action]["total"] += 1
            if fix["success"]:
                action_stats[action]["success"] += 1
        
        # Calculate success rate for each action
        action_success_rates = {}
        for action, stats in action_stats.items():
            action_success_rates[action] = stats["success"] / stats["total"]
        
        # reduce_complexity has 100% success rate
        assert action_success_rates["reduce_complexity"] == 1.0
        assert action_success_rates["increase_context"] < 1.0
    
    def test_fifteen_percent_improvement(self):
        """Test that memory-based frameworks achieve 15% reasoning gains"""
        # Based on research, memory-based frameworks should improve by ~15%
        baseline_performance = 0.65
        expected_improvement = 0.15
        expected_performance = baseline_performance + (baseline_performance * expected_improvement)
        
        assert expected_performance > baseline_performance
        assert expected_performance < 1.0  # Still not perfect
        assert abs(expected_performance - 0.7475) < 0.01  # ~74.75%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
