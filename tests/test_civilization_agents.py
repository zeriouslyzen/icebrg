"""
Unit tests for civilization/persistent_agents.py

Tests GoalHierarchy, AgentMemory, and PersistentAgent classes.
"""

import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch

from iceburg.civilization.persistent_agents import (
    GoalHierarchy,
    GoalPriority,
    Goal,
    AgentMemory,
    Memory,
    PersistentAgent,
    AgentRole,
    AgentIdentity
)


class TestGoalHierarchy:
    """Tests for GoalHierarchy class."""
    
    def test_add_goal(self):
        """Test adding a goal to the hierarchy."""
        hierarchy = GoalHierarchy(max_goals=10)
        
        goal_id = hierarchy.add_goal(
            description="Research quantum computing",
            priority=GoalPriority.HIGH
        )
        
        assert goal_id is not None
        assert goal_id in hierarchy.goals
        assert hierarchy.goals[goal_id].description == "Research quantum computing"
        assert hierarchy.goals[goal_id].priority == GoalPriority.HIGH
        assert hierarchy.goals[goal_id].progress == 0.0
    
    def test_add_goal_with_dependencies(self):
        """Test adding a goal with dependencies."""
        hierarchy = GoalHierarchy()
        
        # Add prerequisite goal
        prereq_id = hierarchy.add_goal(
            description="Learn basic physics",
            priority=GoalPriority.MEDIUM
        )
        
        # Add goal that depends on it
        main_id = hierarchy.add_goal(
            description="Research quantum mechanics",
            priority=GoalPriority.HIGH,
            dependencies=[prereq_id]
        )
        
        assert prereq_id in hierarchy.goal_dependencies[main_id]
    
    def test_add_goal_with_deadline(self):
        """Test adding a goal with a deadline."""
        hierarchy = GoalHierarchy()
        deadline = time.time() + 3600  # 1 hour from now
        
        goal_id = hierarchy.add_goal(
            description="Complete report",
            priority=GoalPriority.CRITICAL,
            deadline=deadline
        )
        
        assert hierarchy.goals[goal_id].deadline == deadline
    
    def test_remove_goal(self):
        """Test removing a goal."""
        hierarchy = GoalHierarchy()
        
        goal_id = hierarchy.add_goal(description="Test goal")
        assert goal_id in hierarchy.goals
        
        result = hierarchy.remove_goal(goal_id)
        
        assert result is True
        assert goal_id not in hierarchy.goals
    
    def test_remove_nonexistent_goal(self):
        """Test removing a goal that doesn't exist."""
        hierarchy = GoalHierarchy()
        
        result = hierarchy.remove_goal("nonexistent_goal")
        
        assert result is False
    
    def test_update_goal_progress(self):
        """Test updating goal progress."""
        hierarchy = GoalHierarchy()
        
        goal_id = hierarchy.add_goal(description="In progress goal")
        
        result = hierarchy.update_goal_progress(goal_id, 0.5)
        
        assert result is True
        assert hierarchy.goals[goal_id].progress == 0.5
    
    def test_update_goal_progress_completion(self):
        """Test that reaching 1.0 progress completes the goal."""
        hierarchy = GoalHierarchy()
        
        goal_id = hierarchy.add_goal(description="Almost done")
        initial_completed = hierarchy.goal_stats["completed_goals"]
        
        hierarchy.update_goal_progress(goal_id, 1.0)
        
        assert hierarchy.goal_stats["completed_goals"] == initial_completed + 1
    
    def test_update_goal_progress_clamping(self):
        """Test that progress is clamped between 0 and 1."""
        hierarchy = GoalHierarchy()
        
        goal_id = hierarchy.add_goal(description="Test clamping")
        
        # Try to set above 1.0
        hierarchy.update_goal_progress(goal_id, 1.5)
        assert hierarchy.goals[goal_id].progress == 1.0
        
        # Add new goal for negative test
        goal_id2 = hierarchy.add_goal(description="Test clamping 2")
        hierarchy.update_goal_progress(goal_id2, -0.5)
        assert hierarchy.goals[goal_id2].progress == 0.0
    
    def test_get_ready_goals(self):
        """Test getting goals that are ready to work on."""
        hierarchy = GoalHierarchy()
        
        # Add a goal with no dependencies (ready)
        ready_id = hierarchy.add_goal(
            description="Ready goal",
            priority=GoalPriority.HIGH
        )
        
        # Add a goal with uncompleted dependency (not ready)
        prereq_id = hierarchy.add_goal(
            description="Prerequisite",
            priority=GoalPriority.LOW
        )
        blocked_id = hierarchy.add_goal(
            description="Blocked goal",
            priority=GoalPriority.CRITICAL,
            dependencies=[prereq_id]
        )
        
        ready_goals = hierarchy.get_ready_goals()
        ready_ids = [g.goal_id for g in ready_goals]
        
        assert ready_id in ready_ids
        assert prereq_id in ready_ids
        assert blocked_id not in ready_ids  # Blocked by dependency
    
    def test_get_ready_goals_priority_order(self):
        """Test that ready goals are sorted by priority."""
        hierarchy = GoalHierarchy()
        
        low_id = hierarchy.add_goal(description="Low", priority=GoalPriority.LOW)
        critical_id = hierarchy.add_goal(description="Critical", priority=GoalPriority.CRITICAL)
        medium_id = hierarchy.add_goal(description="Medium", priority=GoalPriority.MEDIUM)
        
        ready_goals = hierarchy.get_ready_goals()
        
        assert ready_goals[0].goal_id == critical_id
        assert ready_goals[-1].goal_id == low_id
    
    def test_max_goals_limit(self):
        """Test that max_goals limit is enforced."""
        max_goals = 3
        hierarchy = GoalHierarchy(max_goals=max_goals)
        
        for i in range(5):
            hierarchy.add_goal(
                description=f"Goal {i}",
                priority=GoalPriority.LOW if i < 3 else GoalPriority.HIGH
            )
        
        assert len(hierarchy.goals) <= max_goals
    
    def test_goal_stats(self):
        """Test goal statistics tracking."""
        hierarchy = GoalHierarchy()
        
        goal_id = hierarchy.add_goal(description="Stats test")
        
        assert hierarchy.goal_stats["total_goals"] >= 1
        assert hierarchy.goal_stats["active_goals"] >= 1
        
        hierarchy.update_goal_progress(goal_id, 1.0)
        
        assert hierarchy.goal_stats["completed_goals"] >= 1


class TestAgentMemory:
    """Tests for AgentMemory class."""
    
    def test_add_memory(self):
        """Test adding a memory."""
        memory = AgentMemory(max_memories=100, enable_persistence=False)
        
        memory_id = memory.add_memory(
            content="I learned about quantum computing today",
            memory_type="episodic",
            importance=0.8
        )
        
        assert memory_id is not None
        assert memory_id in memory.memories
        assert memory.memories[memory_id].content == "I learned about quantum computing today"
        assert memory.memories[memory_id].memory_type == "episodic"
        assert memory.memories[memory_id].importance == 0.8
    
    def test_add_memory_types(self):
        """Test adding different memory types."""
        memory = AgentMemory(enable_persistence=False)
        
        episodic_id = memory.add_memory(
            content="Experienced a test",
            memory_type="episodic"
        )
        semantic_id = memory.add_memory(
            content="Pi is approximately 3.14159",
            memory_type="semantic"
        )
        procedural_id = memory.add_memory(
            content="How to run tests",
            memory_type="procedural"
        )
        
        assert memory.memory_stats["episodic_memories"] >= 1
        assert memory.memory_stats["semantic_memories"] >= 1
        assert memory.memory_stats["procedural_memories"] >= 1
    
    def test_add_memory_with_goals(self):
        """Test adding a memory associated with goals."""
        memory = AgentMemory(enable_persistence=False)
        
        memory_id = memory.add_memory(
            content="Progress on goal",
            associated_goals=["goal_1", "goal_2"]
        )
        
        assert "goal_1" in memory.memories[memory_id].associated_goals
        assert "goal_2" in memory.memories[memory_id].associated_goals
    
    def test_remove_memory(self):
        """Test removing a memory."""
        memory = AgentMemory(enable_persistence=False)
        
        memory_id = memory.add_memory(content="To be removed")
        assert memory_id in memory.memories
        
        result = memory.remove_memory(memory_id)
        
        assert result is True
        assert memory_id not in memory.memories
        assert memory.memory_stats["forgotten_memories"] >= 1
    
    def test_remove_nonexistent_memory(self):
        """Test removing a memory that doesn't exist."""
        memory = AgentMemory(enable_persistence=False)
        
        result = memory.remove_memory("nonexistent_memory")
        
        assert result is False
    
    def test_get_memories_by_type(self):
        """Test retrieving memories by type."""
        memory = AgentMemory(enable_persistence=False)
        
        memory.add_memory(content="Episodic 1", memory_type="episodic")
        memory.add_memory(content="Semantic 1", memory_type="semantic")
        memory.add_memory(content="Episodic 2", memory_type="episodic")
        
        episodic_memories = memory.get_memories_by_type("episodic")
        
        assert len(episodic_memories) == 2
        assert all(m.memory_type == "episodic" for m in episodic_memories)
    
    def test_get_memories_by_importance(self):
        """Test retrieving memories above importance threshold."""
        memory = AgentMemory(enable_persistence=False)
        
        memory.add_memory(content="Low importance", importance=0.2)
        memory.add_memory(content="Medium importance", importance=0.5)
        memory.add_memory(content="High importance", importance=0.9)
        
        important_memories = memory.get_memories_by_importance(min_importance=0.7)
        
        assert len(important_memories) == 1
        assert important_memories[0].importance >= 0.7
    
    def test_get_memories_by_goals(self):
        """Test retrieving memories associated with specific goals."""
        memory = AgentMemory(enable_persistence=False)
        
        memory.add_memory(content="Goal 1 progress", associated_goals=["goal_1"])
        memory.add_memory(content="Goal 2 progress", associated_goals=["goal_2"])
        memory.add_memory(content="Both goals", associated_goals=["goal_1", "goal_2"])
        
        goal1_memories = memory.get_memories_by_goals(["goal_1"])
        
        assert len(goal1_memories) == 2
    
    def test_max_memories_limit(self):
        """Test that max_memories limit is enforced."""
        max_memories = 3
        memory = AgentMemory(max_memories=max_memories, enable_persistence=False)
        
        for i in range(5):
            memory.add_memory(content=f"Memory {i}", importance=0.1 * i)
        
        assert len(memory.memories) <= max_memories
    
    def test_memory_stats(self):
        """Test memory statistics tracking."""
        memory = AgentMemory(enable_persistence=False)
        
        memory.add_memory(content="Test memory")
        
        stats = memory.get_memory_stats()
        
        assert stats["total_memories"] >= 1


class TestPersistentAgent:
    """Tests for PersistentAgent class."""
    
    def test_agent_creation(self):
        """Test creating a persistent agent."""
        agent = PersistentAgent(
            agent_id="test_agent_1",
            role=AgentRole.RESEARCHER
        )
        
        assert agent.agent_id == "test_agent_1"
        assert agent.role == AgentRole.RESEARCHER
        assert agent.reputation == 0.5  # Default
        assert agent.energy == 1.0  # Default
    
    def test_agent_creation_with_personality(self):
        """Test creating an agent with custom personality traits."""
        personality = {
            "cooperation": 0.9,
            "aggression": 0.1,
            "curiosity": 0.8,
            "persistence": 0.7,
            "creativity": 0.6,
            "analytical": 0.9
        }
        
        agent = PersistentAgent(
            agent_id="test_agent_2",
            role=AgentRole.SPECIALIST,
            personality_traits=personality
        )
        
        assert agent.personality_traits["cooperation"] == 0.9
        assert agent.personality_traits["aggression"] == 0.1
    
    def test_agent_default_personality(self):
        """Test that default personality traits are assigned."""
        agent = PersistentAgent()
        
        assert "cooperation" in agent.personality_traits
        assert "curiosity" in agent.personality_traits
    
    def test_agent_decide_action_no_goals(self):
        """Test that decide_action returns None when no goals exist."""
        agent = PersistentAgent()
        
        action = agent.decide_action(perception={})
        
        assert action is None
    
    def test_agent_decide_action_with_goal(self):
        """Test that decide_action returns action when goals exist."""
        agent = PersistentAgent(personality_traits={"cooperation": 0.8, "aggression": 0.1, "curiosity": 0.5})
        agent.goals.add_goal(description="Test goal", priority=GoalPriority.HIGH)
        
        action = agent.decide_action(perception={})
        
        assert action is not None
        assert "type" in action
        assert "goal_id" in action
        assert "agent_id" in action
    
    def test_agent_update_from_action_result_success(self):
        """Test updating agent state after successful action."""
        agent = PersistentAgent()
        goal_id = agent.goals.add_goal(description="Test goal")
        
        action = {"type": "test", "goal_id": goal_id}
        result = {"success": True, "outcome": "completed", "performance": 0.8}
        
        initial_progress = agent.goals.goals[goal_id].progress
        agent.update_from_action_result(action, result)
        
        # Progress should increase
        assert agent.goals.goals[goal_id].progress > initial_progress
        # Mood should improve
        assert agent.mood >= 0.5
    
    def test_agent_update_from_action_result_failure(self):
        """Test updating agent state after failed action."""
        agent = PersistentAgent()
        goal_id = agent.goals.add_goal(description="Test goal")
        initial_mood = agent.mood
        
        action = {"type": "test", "goal_id": goal_id}
        result = {"success": False, "outcome": "failed"}
        
        agent.update_from_action_result(action, result)
        
        # Mood should decrease
        assert agent.mood < initial_mood
    
    def test_agent_location(self):
        """Test agent location management."""
        agent = PersistentAgent()
        
        assert agent.get_location() == (0.0, 0.0)  # Default
        
        agent.set_location((10.5, 20.3))
        
        assert agent.get_location() == (10.5, 20.3)
    
    def test_agent_perception_radius(self):
        """Test perception radius calculation."""
        low_curiosity_agent = PersistentAgent(
            personality_traits={"curiosity": 0.2, "cooperation": 0.5, "aggression": 0.3}
        )
        high_curiosity_agent = PersistentAgent(
            personality_traits={"curiosity": 0.9, "cooperation": 0.5, "aggression": 0.3}
        )
        
        assert high_curiosity_agent.get_perception_radius() > low_curiosity_agent.get_perception_radius()
    
    def test_agent_resources(self):
        """Test agent resource management."""
        agent = PersistentAgent()
        
        assert agent.get_resources() == {}
        assert agent.needs_resources() is True
        
        agent.resources = {"energy": 5.0}
        
        assert agent.get_resources() == {"energy": 5.0}
        assert agent.needs_resources() is False
    
    def test_agent_vote_cooperative(self):
        """Test agent voting behavior with high cooperation."""
        agent = PersistentAgent(
            personality_traits={"cooperation": 0.9, "aggression": 0.1, "curiosity": 0.5}
        )
        
        options = ["cooperate_with_team", "compete_alone"]
        
        # Run multiple times to check tendency
        votes = [agent.vote(options, {}) for _ in range(20)]
        cooperative_votes = votes.count("cooperate_with_team")
        
        # Should tend toward cooperative options
        assert cooperative_votes > 0
    
    def test_agent_vote_empty_options(self):
        """Test voting with no options."""
        agent = PersistentAgent()
        
        vote = agent.vote([], {})
        
        assert vote is None
    
    def test_agent_profile(self):
        """Test getting comprehensive agent profile."""
        agent = PersistentAgent(
            agent_id="profile_test",
            role=AgentRole.LEADER
        )
        agent.goals.add_goal(description="Test goal")
        agent.memory.add_memory(content="Test memory")
        
        profile = agent.get_agent_profile()
        
        assert profile["agent_id"] == "profile_test"
        assert profile["role"] == "leader"
        assert "personality_traits" in profile
        assert "goals" in profile
        assert "memory" in profile
    
    def test_agent_all_roles(self):
        """Test creating agents with all role types."""
        roles = [
            AgentRole.RESEARCHER,
            AgentRole.COORDINATOR,
            AgentRole.SPECIALIST,
            AgentRole.GENERALIST,
            AgentRole.LEADER,
            AgentRole.FOLLOWER
        ]
        
        for role in roles:
            agent = PersistentAgent(role=role)
            assert agent.role == role


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
