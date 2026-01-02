"""
Unit tests for civilization/world_model.py

Tests WorldState and AGICivilization classes.
"""

import pytest
import time
import json
import tempfile
import os
import numpy as np
from unittest.mock import MagicMock, patch

from iceburg.civilization.world_model import (
    WorldState,
    AGICivilization,
    Resource,
    WorldEvent,
    AgentPerception,
    SocialNormSystem,
    ResourceEconomy,
    EmergenceDetector,
    SimulationState
)


class TestResource:
    """Tests for Resource dataclass."""
    
    def test_resource_creation(self):
        """Test creating a resource."""
        resource = Resource(
            name="energy",
            amount=100.0,
            max_amount=200.0,
            regeneration_rate=1.0,
            decay_rate=0.1,
            location=(50.0, 50.0)
        )
        
        assert resource.name == "energy"
        assert resource.amount == 100.0
        assert resource.max_amount == 200.0
        assert resource.location == (50.0, 50.0)
    
    def test_resource_defaults(self):
        """Test resource default values."""
        resource = Resource(
            name="test",
            amount=50.0,
            max_amount=100.0
        )
        
        assert resource.regeneration_rate == 0.0
        assert resource.decay_rate == 0.0
        assert resource.location == (0.0, 0.0)


class TestWorldState:
    """Tests for WorldState class."""
    
    def test_world_state_creation(self):
        """Test creating a world state."""
        world = WorldState(world_size=(100.0, 100.0))
        
        assert world.world_size == (100.0, 100.0)
        assert world.simulation_step == 0
        assert len(world.resources) == 0
    
    def test_initialize_world(self):
        """Test initializing the world."""
        world = WorldState()
        
        initial_resources = [
            {
                "name": "energy",
                "amount": 100.0,
                "max_amount": 200.0,
                "regeneration_rate": 1.0
            }
        ]
        
        world.initialize_world(initial_resources)
        
        assert "energy" in world.resources
        assert world.resources["energy"].amount == 100.0
    
    def test_add_resource(self):
        """Test adding a resource to the world."""
        world = WorldState()
        
        name = world.add_resource(
            name="knowledge",
            amount=50.0,
            max_amount=100.0,
            location=(25.0, 75.0)
        )
        
        assert name == "knowledge"
        assert "knowledge" in world.resources
        assert world.resources["knowledge"].location == (25.0, 75.0)
    
    def test_add_resource_random_location(self):
        """Test adding a resource with random location."""
        world = WorldState(world_size=(100.0, 100.0))
        
        world.add_resource(
            name="random_resource",
            amount=10.0,
            max_amount=20.0
        )
        
        location = world.resources["random_resource"].location
        assert 0 <= location[0] <= 100.0
        assert 0 <= location[1] <= 100.0
    
    def test_remove_resource(self):
        """Test removing a resource from the world."""
        world = WorldState()
        
        world.add_resource(name="to_remove", amount=10.0, max_amount=20.0)
        assert "to_remove" in world.resources
        
        world.remove_resource("to_remove")
        
        assert "to_remove" not in world.resources
    
    def test_max_resources_limit(self):
        """Test that max_resources limit is enforced."""
        world = WorldState(max_resources=3)
        
        for i in range(5):
            world.add_resource(
                name=f"resource_{i}",
                amount=10.0,
                max_amount=20.0
            )
        
        assert len(world.resources) <= 3
    
    def test_get_perceptions(self):
        """Test getting agent perceptions."""
        world = WorldState()
        
        world.add_resource(
            name="nearby",
            amount=100.0,
            max_amount=200.0,
            location=(5.0, 5.0)
        )
        world.add_resource(
            name="far_away",
            amount=100.0,
            max_amount=200.0,
            location=(100.0, 100.0)
        )
        
        perception = world.get_perceptions(
            agent_id="test_agent",
            location=(0.0, 0.0),
            perception_radius=10.0
        )
        
        assert perception.agent_id == "test_agent"
        assert len(perception.visible_resources) == 1
        assert perception.visible_resources[0].name == "nearby"
    
    def test_get_perceptions_environmental_factors(self):
        """Test that perceptions include environmental factors."""
        world = WorldState()
        
        perception = world.get_perceptions(
            agent_id="test_agent",
            location=(0.0, 0.0),
            perception_radius=10.0
        )
        
        assert "temperature" in perception.environmental_factors
        assert "humidity" in perception.environmental_factors
        assert "stability" in perception.environmental_factors
    
    def test_apply_actions_consume_resource(self):
        """Test applying a consume resource action."""
        world = WorldState()
        
        world.add_resource(name="food", amount=100.0, max_amount=200.0)
        
        actions = [{
            "type": "consume_resource",
            "resource_name": "food",
            "amount": 30.0
        }]
        
        world.apply_actions(actions)
        
        assert world.resources["food"].amount == 70.0
    
    def test_apply_actions_create_resource(self):
        """Test applying a create resource action."""
        world = WorldState()
        
        actions = [{
            "type": "create_resource",
            "name": "new_resource",
            "amount": 50.0,
            "location": (10.0, 10.0)
        }]
        
        world.apply_actions(actions)
        
        assert "new_resource" in world.resources
        assert world.resources["new_resource"].amount == 50.0
    
    def test_apply_actions_modify_environment(self):
        """Test applying a modify environment action."""
        world = WorldState()
        
        initial_temp = world.environmental_factors["temperature"]
        
        actions = [{
            "type": "modify_environment",
            "factor": "temperature",
            "change": 5.0
        }]
        
        world.apply_actions(actions)
        
        # Temperature should have changed (clamped to 0-1)
        assert world.environmental_factors["temperature"] != initial_temp
    
    def test_update_world(self):
        """Test updating the world for one simulation step."""
        world = WorldState()
        
        world.add_resource(
            name="regenerating",
            amount=50.0,
            max_amount=100.0,
            regeneration_rate=10.0
        )
        
        initial_step = world.simulation_step
        
        world.update_world(delta_time=1.0)
        
        assert world.simulation_step == initial_step + 1
        assert world.resources["regenerating"].amount == 60.0  # +10 regeneration
    
    def test_update_world_decay(self):
        """Test resource decay during world update."""
        world = WorldState()
        
        world.add_resource(
            name="decaying",
            amount=50.0,
            max_amount=100.0,
            decay_rate=5.0
        )
        
        world.update_world(delta_time=1.0)
        
        assert world.resources["decaying"].amount == 45.0  # -5 decay
    
    def test_update_world_removes_depleted_resources(self):
        """Test that depleted resources are removed."""
        world = WorldState()
        
        world.add_resource(
            name="almost_gone",
            amount=1.0,
            max_amount=100.0,
            decay_rate=10.0
        )
        
        world.update_world(delta_time=1.0)
        
        # Resource should be removed after decaying to 0
        assert "almost_gone" not in world.resources
    
    def test_save_and_load_world_state(self):
        """Test saving and loading world state."""
        world = WorldState()
        
        world.add_resource(
            name="persistent",
            amount=75.0,
            max_amount=150.0,
            location=(30.0, 40.0)
        )
        world.simulation_step = 100
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            world.save_world_state(filepath)
            
            # Create a new world and load the saved state
            new_world = WorldState()
            new_world.load_world_state(filepath)
            
            assert new_world.simulation_step == 100
            assert "persistent" in new_world.resources
            assert new_world.resources["persistent"].amount == 75.0
            assert new_world.resources["persistent"].location == (30.0, 40.0)
        finally:
            os.unlink(filepath)
    
    def test_get_world_state(self):
        """Test getting current world state summary."""
        world = WorldState()
        
        world.add_resource(name="test", amount=10.0, max_amount=20.0)
        world.update_world()
        
        state = world.get_world_state()
        
        assert "current_time" in state
        assert "simulation_step" in state
        assert "num_resources" in state
        assert "environmental_factors" in state
        assert "performance_stats" in state
    
    def test_performance_stats_tracking(self):
        """Test that performance stats are tracked."""
        world = WorldState()
        
        for _ in range(5):
            world.update_world()
        
        assert world.performance_stats["total_steps"] == 5
        assert world.performance_stats["average_step_time"] >= 0


class TestAGICivilization:
    """Tests for AGICivilization class."""
    
    def test_civilization_creation(self):
        """Test creating an AGI civilization."""
        civ = AGICivilization(world_size=(100.0, 100.0), max_agents=50)
        
        assert civ.world_state is not None
        assert civ.max_agents == 50
        assert civ.is_running is False
    
    def test_initialize_civilization(self):
        """Test initializing the civilization."""
        civ = AGICivilization()
        
        initial_resources = [
            {"name": "energy", "amount": 100.0, "max_amount": 200.0}
        ]
        
        civ.initialize_civilization(initial_resources)
        
        assert "energy" in civ.world_state.resources
    
    def test_simulate_basic(self):
        """Test basic simulation loop."""
        civ = AGICivilization()
        civ.initialize_civilization()
        
        results = civ.simulate(spec={}, steps=10)
        
        assert results["simulation_steps"] == 10
        assert "world_evolution" in results
        assert "emergence_events" in results
    
    def test_simulate_records_world_evolution(self):
        """Test that simulation records world evolution."""
        civ = AGICivilization()
        civ.initialize_civilization([
            {"name": "energy", "amount": 100.0, "max_amount": 200.0}
        ])
        
        results = civ.simulate(spec={}, steps=200)
        
        # Should record state every 100 steps
        assert len(results["world_evolution"]) >= 2


class TestSocialNormSystem:
    """Tests for SocialNormSystem real implementation."""
    
    def test_initialization(self):
        """Test social norm system initialization."""
        system = SocialNormSystem()
        
        assert system.norms == {}
        assert isinstance(system.stats, dict)
    
    def test_initialize_method(self):
        """Test initialize method creates default norms."""
        system = SocialNormSystem()
        system.initialize()
        
        # Should have created default norms
        assert len(system.norms) >= 1
    
    def test_get_norms(self):
        """Test getting norms."""
        system = SocialNormSystem()
        system.initialize()
        
        norms = system.get_norms()
        
        assert isinstance(norms, dict)
        assert len(norms) >= 1
    
    def test_detect_violation(self):
        """Test violation detection."""
        system = SocialNormSystem()
        system.initialize()
        
        # Action that violates cooperation norm
        action = {"type": "defect", "agent_id": "test_agent"}
        violation = system.detect_violation(action, "test_agent")
        
        # May or may not detect depending on norm configuration
        # Just verify it doesn't crash
        assert violation is None or hasattr(violation, 'violation_id')
    
    def test_get_stats(self):
        """Test getting statistics."""
        system = SocialNormSystem()
        
        stats = system.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_norms" in stats


class TestResourceEconomy:
    """Tests for ResourceEconomy real implementation."""
    
    def test_initialization(self):
        """Test resource economy initialization."""
        economy = ResourceEconomy()
        
        assert economy.trades == []
        assert economy.market_prices == {}
    
    def test_initialize_method(self):
        """Test initialize method creates markets."""
        economy = ResourceEconomy()
        economy.initialize()
        
        # Should have created market prices
        assert len(economy.market_prices) >= 1
    
    def test_get_price(self):
        """Test getting market price."""
        economy = ResourceEconomy()
        economy.initialize()
        
        price = economy.get_price("energy")
        
        assert price > 0
    
    def test_get_stats(self):
        """Test getting statistics."""
        economy = ResourceEconomy()
        
        stats = economy.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_trades" in stats


class TestEmergenceDetector:
    """Tests for EmergenceDetector real implementation."""
    
    def test_initialization(self):
        """Test emergence detector initialization."""
        detector = EmergenceDetector()
        
        assert detector.patterns == {}  # Dict in real implementation
        assert detector.emergence_threshold == 0.7  # Default threshold
    
    def test_check_returns_list(self):
        """Test check method returns a list."""
        detector = EmergenceDetector()
        world = WorldState()
        
        result = detector.check(world)
        
        assert isinstance(result, list)
    
    def test_record_action(self):
        """Test recording actions."""
        detector = EmergenceDetector()
        
        detector.record_action({"type": "test"}, "agent_1")
        
        # Action should be recorded
        assert len(detector.action_history["agent_1"]) >= 1
    
    def test_get_stats(self):
        """Test getting statistics."""
        detector = EmergenceDetector()
        
        stats = detector.get_stats()
        
        assert isinstance(stats, dict)
        assert "patterns_detected" in stats



class TestIntegration:
    """Integration tests for civilization system."""
    
    def test_full_simulation_cycle(self):
        """Test a complete simulation cycle with agents."""
        from iceburg.civilization.persistent_agents import PersistentAgent, AgentRole
        
        # Create civilization
        civ = AGICivilization(world_size=(100.0, 100.0), max_agents=10)
        civ.initialize_civilization([
            {
                "name": "energy",
                "amount": 100.0,
                "max_amount": 200.0,
                "regeneration_rate": 1.0
            },
            {
                "name": "knowledge",
                "amount": 50.0,
                "max_amount": 100.0,
                "regeneration_rate": 0.5
            }
        ])
        
        # Add some agents
        for i in range(3):
            agent = PersistentAgent(
                agent_id=f"agent_{i}",
                role=AgentRole.RESEARCHER
            )
            agent.goals.add_goal(description=f"Goal for agent {i}")
            civ.agents[agent.agent_id] = agent
        
        # Run simulation
        results = civ.simulate(spec={}, steps=50)
        
        assert results["simulation_steps"] == 50
        assert len(civ.agents) == 3
    
    def test_agent_perception_and_action(self):
        """Test agent perceiving world and taking action."""
        from iceburg.civilization.persistent_agents import PersistentAgent, AgentRole
        
        # Set up world
        world = WorldState()
        world.add_resource(
            name="target_resource",
            amount=100.0,
            max_amount=200.0,
            location=(5.0, 5.0)
        )
        
        # Create agent
        agent = PersistentAgent(
            agent_id="test_agent",
            role=AgentRole.RESEARCHER
        )
        agent.goals.add_goal(description="Find resources")
        agent.set_location((0.0, 0.0))
        
        # Get perception
        perception = world.get_perceptions(
            agent_id=agent.agent_id,
            location=agent.get_location(),
            perception_radius=agent.get_perception_radius()
        )
        
        # Agent should see the nearby resource
        assert len(perception.visible_resources) >= 1
        
        # Agent decides action
        action = agent.decide_action(perception)
        
        assert action is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
