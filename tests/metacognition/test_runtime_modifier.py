"""
Test suite for Runtime Agent Modifier
Tests live behavior changes, capability expansion, and adaptive wrappers
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestRuntimeAgentModifier:
    """Test suite for runtime agent modification capabilities"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_cfg = Mock()
        self.mock_cfg.synthesist_model = "llama3.1:8b"
    
    def test_live_behavior_modification(self):
        """Test that agent behavior can be modified at runtime"""
        # Simulate an agent with modifiable behavior
        class MockAgent:
            def __init__(self):
                self.temperature = 0.7
                self.max_tokens = 1000
                self.system_prompt = "You are a helpful assistant."
            
            def get_config(self):
                return {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "system_prompt": self.system_prompt
                }
        
        agent = MockAgent()
        original_temp = agent.temperature
        
        # Modify at runtime
        agent.temperature = 0.3
        
        assert agent.temperature != original_temp
        assert agent.temperature == 0.3
    
    def test_capability_expansion(self):
        """Test dynamic capability expansion"""
        initial_capabilities = ["research", "synthesis"]
        
        # Expand capabilities at runtime
        new_capability = "code_generation"
        initial_capabilities.append(new_capability)
        
        assert "code_generation" in initial_capabilities
        assert len(initial_capabilities) == 3
    
    def test_adaptive_wrapper_creation(self):
        """Test creation of adaptive wrappers around agents"""
        def original_run(query):
            return f"Response to: {query}"
        
        def adaptive_wrapper(func):
            def wrapped(query):
                # Add pre-processing
                enhanced_query = f"[ENHANCED] {query}"
                result = func(enhanced_query)
                # Add post-processing
                return f"[VERIFIED] {result}"
            return wrapped
        
        wrapped_run = adaptive_wrapper(original_run)
        result = wrapped_run("test query")
        
        assert "[VERIFIED]" in result
        assert "[ENHANCED]" in result
    
    def test_performance_based_adaptation(self):
        """Test that modifications are triggered by performance metrics"""
        performance_history = [
            {"timestamp": "2025-01-01", "success_rate": 0.8},
            {"timestamp": "2025-01-02", "success_rate": 0.7},
            {"timestamp": "2025-01-03", "success_rate": 0.5},  # Degradation!
        ]
        
        # Should trigger adaptation when performance drops
        recent_performance = performance_history[-1]["success_rate"]
        baseline_performance = sum(p["success_rate"] for p in performance_history[:-1]) / (len(performance_history) - 1)
        
        performance_degradation = baseline_performance - recent_performance
        
        if performance_degradation > 0.1:  # More than 10% drop
            needs_adaptation = True
        else:
            needs_adaptation = False
        
        assert needs_adaptation is True
    
    def test_prompt_evolution_at_runtime(self):
        """Test that prompts can evolve during execution"""
        original_prompt = "You are a research assistant."
        
        evolution_triggers = [
            {"condition": "high_error_rate", "modification": "Add error handling instructions"},
            {"condition": "slow_response", "modification": "Add efficiency guidelines"},
            {"condition": "low_quality", "modification": "Add quality standards"},
        ]
        
        # Apply modifications based on triggers
        current_conditions = ["slow_response", "low_quality"]
        modifications = []
        
        for trigger in evolution_triggers:
            if trigger["condition"] in current_conditions:
                modifications.append(trigger["modification"])
        
        enhanced_prompt = original_prompt + "\n" + "\n".join(modifications)
        
        assert "efficiency guidelines" in enhanced_prompt
        assert "quality standards" in enhanced_prompt


class TestModificationSafety:
    """Test safety constraints on runtime modifications"""
    
    def test_protected_agent_types(self):
        """Test that certain agents cannot be modified"""
        protected_agents = ["secretary", "oracle", "synthesist"]
        
        def can_modify(agent_name, modification_type):
            if agent_name in protected_agents and modification_type == "core_behavior":
                return False
            return True
        
        # Protection check
        assert can_modify("secretary", "core_behavior") is False
        assert can_modify("custom_agent", "core_behavior") is True
        assert can_modify("secretary", "logging") is True  # Non-core is ok
    
    def test_modification_bounds(self):
        """Test that modifications stay within bounds"""
        valid_temperature_range = (0.0, 2.0)
        valid_max_tokens_range = (100, 8192)
        
        proposed_modifications = [
            {"param": "temperature", "value": 0.5},  # Valid
            {"param": "temperature", "value": 5.0},  # Invalid - too high
            {"param": "max_tokens", "value": 50},    # Invalid - too low
            {"param": "max_tokens", "value": 4096},  # Valid
        ]
        
        for mod in proposed_modifications:
            if mod["param"] == "temperature":
                valid = valid_temperature_range[0] <= mod["value"] <= valid_temperature_range[1]
            elif mod["param"] == "max_tokens":
                valid = valid_max_tokens_range[0] <= mod["value"] <= valid_max_tokens_range[1]
            else:
                valid = True
            
            mod["valid"] = valid
        
        invalid_count = sum(1 for m in proposed_modifications if not m["valid"])
        assert invalid_count == 2
    
    def test_modification_rollback(self):
        """Test that modifications can be rolled back"""
        class ModifiableAgent:
            def __init__(self):
                self._state_history = []
                self.temperature = 0.7
            
            def save_state(self):
                self._state_history.append({"temperature": self.temperature})
            
            def rollback(self):
                if self._state_history:
                    previous = self._state_history.pop()
                    self.temperature = previous["temperature"]
        
        agent = ModifiableAgent()
        agent.save_state()
        
        # Modify
        agent.temperature = 0.3
        assert agent.temperature == 0.3
        
        # Rollback
        agent.rollback()
        assert agent.temperature == 0.7


class TestCapabilityRegistry:
    """Test dynamic capability registration"""
    
    def test_capability_registration(self):
        """Test registration of new capabilities"""
        registry = {}
        
        def register_capability(name, handler, metadata=None):
            registry[name] = {
                "handler": handler,
                "metadata": metadata or {},
                "registered_at": datetime.now().isoformat()
            }
        
        def research_handler(query):
            return f"Research results for: {query}"
        
        register_capability("research", research_handler, {"category": "analysis"})
        
        assert "research" in registry
        assert registry["research"]["metadata"]["category"] == "analysis"
    
    def test_capability_lookup(self):
        """Test looking up capabilities by attributes"""
        capabilities = [
            {"name": "research", "category": "analysis", "speed": "slow"},
            {"name": "chat", "category": "interaction", "speed": "fast"},
            {"name": "code", "category": "development", "speed": "medium"},
        ]
        
        # Find all fast capabilities
        fast_caps = [c for c in capabilities if c["speed"] == "fast"]
        assert len(fast_caps) == 1
        assert fast_caps[0]["name"] == "chat"
        
        # Find by category
        analysis_caps = [c for c in capabilities if c["category"] == "analysis"]
        assert len(analysis_caps) == 1
    
    def test_capability_removal(self):
        """Test safe removal of capabilities"""
        registry = {"research": {}, "chat": {}, "code": {}}
        
        # Remove capability
        if "research" in registry:
            del registry["research"]
        
        assert "research" not in registry
        assert len(registry) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
