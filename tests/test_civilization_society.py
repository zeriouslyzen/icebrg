"""
Unit tests for civilization/agent_society.py

Tests SocialLearningSystem and MultiAgentSociety classes.
"""

import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch

from iceburg.civilization.agent_society import (
    SocialLearningSystem,
    MultiAgentSociety,
    CooperationStrategy,
    NormType,
    SocialInteraction,
    SocialNorm
)


class TestSocialLearningSystem:
    """Tests for SocialLearningSystem class."""
    
    def test_record_interaction(self):
        """Test recording a social interaction."""
        system = SocialLearningSystem()
        
        interaction_id = system.record_interaction(
            agent_a="agent_1",
            agent_b="agent_2",
            interaction_type="collaboration",
            outcome="cooperation",
            reward=1.0
        )
        
        assert interaction_id is not None
        assert system.learning_stats["total_interactions"] >= 1
    
    def test_record_interaction_updates_reputation(self):
        """Test that recording an interaction updates agent reputations."""
        system = SocialLearningSystem()
        
        initial_rep_a = system.agent_reputation["agent_1"]
        initial_rep_b = system.agent_reputation["agent_2"]
        
        system.record_interaction(
            agent_a="agent_1",
            agent_b="agent_2",
            interaction_type="collaboration",
            outcome="cooperation",
            reward=1.0
        )
        
        # Reputation should increase slightly for cooperation
        assert system.agent_reputation["agent_1"] >= initial_rep_a
        assert system.agent_reputation["agent_2"] >= initial_rep_b
    
    def test_record_interaction_tracks_cooperation_history(self):
        """Test that cooperation history is tracked."""
        system = SocialLearningSystem()
        
        system.record_interaction(
            agent_a="agent_1",
            agent_b="agent_2",
            interaction_type="test",
            outcome="cooperation",
            reward=1.0
        )
        
        assert len(system.agent_cooperation_history["agent_1"]) >= 1
        assert system.agent_cooperation_history["agent_1"][-1] is True
    
    def test_record_interaction_tracks_defection(self):
        """Test that defection is tracked in cooperation history."""
        system = SocialLearningSystem()
        
        system.record_interaction(
            agent_a="agent_1",
            agent_b="agent_2",
            interaction_type="test",
            outcome="defection",  # Not cooperation
            reward=0.0
        )
        
        assert system.agent_cooperation_history["agent_1"][-1] is False
    
    def test_get_peer_success_models(self):
        """Test getting top successful peers for imitation."""
        system = SocialLearningSystem()
        
        # Create some agents with different performance histories
        system.agent_performance["high_performer"] = [0.9, 0.95, 0.92]
        system.agent_reputation["high_performer"] = 0.8
        
        system.agent_performance["low_performer"] = [0.2, 0.3, 0.1]
        system.agent_reputation["low_performer"] = 0.3
        
        system.agent_performance["medium_performer"] = [0.5, 0.6, 0.55]
        system.agent_reputation["medium_performer"] = 0.5
        
        top_peers = system.get_peer_success_models("requesting_agent", top_k=2)
        
        assert len(top_peers) == 2
        assert top_peers[0][0] == "high_performer"
    
    def test_get_peer_success_models_excludes_self(self):
        """Test that an agent is excluded from its own peer list."""
        system = SocialLearningSystem()
        
        system.agent_performance["agent_1"] = [0.9, 0.95]
        system.agent_reputation["agent_1"] = 0.8
        
        top_peers = system.get_peer_success_models("agent_1", top_k=5)
        peer_ids = [p[0] for p in top_peers]
        
        assert "agent_1" not in peer_ids
    
    def test_should_imitate_peer_successful(self):
        """Test imitation decision for a successful peer."""
        system = SocialLearningSystem()
        system.imitation_rate = 1.0  # Always imitate for testing
        
        system.agent_performance["agent_1"] = [0.1, 0.2]
        system.agent_performance["successful_peer"] = [0.9, 0.95, 0.92]
        
        # Run multiple times due to randomness
        imitation_count = sum(
            system.should_imitate_peer("agent_1", "successful_peer")
            for _ in range(100)
        )
        
        # Should imitate at least sometimes
        assert imitation_count > 0
    
    def test_should_imitate_peer_no_data(self):
        """Test imitation decision when peer has no data."""
        system = SocialLearningSystem()
        
        result = system.should_imitate_peer("agent_1", "unknown_peer")
        
        assert result is False
    
    def test_get_majority_decision(self):
        """Test majority rule decision making."""
        system = SocialLearningSystem()
        system.majority_threshold = 0.5
        
        # Record several decision interactions
        current_time = time.time()
        for i in range(6):
            interaction = SocialInteraction(
                interaction_id=f"decision_{i}",
                timestamp=current_time,
                agent_a=f"agent_{i}",
                agent_b="system",
                interaction_type="decision",
                outcome="option_a" if i < 4 else "option_b",
                reward=1.0
            )
            system.interactions.append(interaction)
        
        decision = system.get_majority_decision("test_agent", ["option_a", "option_b"])
        
        assert decision == "option_a"
    
    def test_get_majority_decision_no_clear_majority(self):
        """Test majority decision when there's no clear winner."""
        system = SocialLearningSystem()
        system.majority_threshold = 0.9  # Very high threshold
        
        current_time = time.time()
        for i, outcome in enumerate(["option_a", "option_b", "option_a"]):
            interaction = SocialInteraction(
                interaction_id=f"decision_{i}",
                timestamp=current_time,
                agent_a=f"agent_{i}",
                agent_b="system",
                interaction_type="decision",
                outcome=outcome,
                reward=1.0
            )
            system.interactions.append(interaction)
        
        decision = system.get_majority_decision("test_agent", ["option_a", "option_b"])
        
        # 66% option_a is less than 90% threshold
        assert decision is None
    
    def test_should_punish_agent_defector(self):
        """Test punishment decision for a defecting agent."""
        system = SocialLearningSystem()
        system.punishment_rate = 1.0  # Always punish for testing
        
        # Set up defection history
        system.agent_cooperation_history["defector"] = [False] * 10
        
        punishment_count = sum(
            system.should_punish_agent("enforcer", "defector")
            for _ in range(100)
        )
        
        assert punishment_count > 0
    
    def test_should_punish_agent_cooperator(self):
        """Test punishment decision for a cooperating agent."""
        system = SocialLearningSystem()
        
        # Set up cooperation history
        system.agent_cooperation_history["cooperator"] = [True] * 10
        
        punishment_count = sum(
            system.should_punish_agent("enforcer", "cooperator")
            for _ in range(100)
        )
        
        # Cooperators should rarely be punished
        assert punishment_count < 50
    
    def test_should_reward_agent_cooperator(self):
        """Test reward decision for a cooperating agent."""
        system = SocialLearningSystem()
        system.reward_rate = 1.0  # High reward rate for testing
        
        system.agent_cooperation_history["cooperator"] = [True] * 10
        
        reward_count = sum(
            system.should_reward_agent("rewarder", "cooperator")
            for _ in range(100)
        )
        
        assert reward_count > 0
    
    def test_should_reward_agent_no_history(self):
        """Test reward decision for agent with no history."""
        system = SocialLearningSystem()
        
        result = system.should_reward_agent("rewarder", "unknown_agent")
        
        assert result is False
    
    def test_form_norm(self):
        """Test creating a social norm."""
        system = SocialLearningSystem()
        
        norm_id = system.form_norm(
            norm_type=NormType.COOPERATION,
            description="Always share resources",
            strength=0.7,
            enforcement_rate=0.2
        )
        
        assert norm_id is not None
        assert norm_id in system.norms
        assert system.norms[norm_id].description == "Always share resources"
        assert system.norms[norm_id].strength == 0.7
        assert system.learning_stats["norms_formed"] >= 1
    
    def test_enforce_norm_punishment(self):
        """Test enforcing a punishment norm."""
        system = SocialLearningSystem()
        
        norm_id = system.form_norm(
            norm_type=NormType.PUNISHMENT,
            description="Punish defectors",
            strength=0.8,
            enforcement_rate=100.0  # High rate for testing
        )
        
        system.agent_reputation["target"] = 0.5
        
        result = system.enforce_norm(norm_id, "enforcer", "target")
        
        assert result is True
        assert system.agent_reputation["target"] < 0.5  # Decreased
        assert system.learning_stats["punishments_distributed"] >= 1
    
    def test_enforce_norm_reward(self):
        """Test enforcing a reward norm."""
        system = SocialLearningSystem()
        
        norm_id = system.form_norm(
            norm_type=NormType.REWARD,
            description="Reward cooperators",
            strength=0.8,
            enforcement_rate=100.0
        )
        
        system.agent_reputation["target"] = 0.5
        
        result = system.enforce_norm(norm_id, "enforcer", "target")
        
        assert result is True
        assert system.agent_reputation["target"] > 0.5  # Increased
        assert system.learning_stats["rewards_distributed"] >= 1
    
    def test_enforce_nonexistent_norm(self):
        """Test enforcing a norm that doesn't exist."""
        system = SocialLearningSystem()
        
        result = system.enforce_norm("nonexistent_norm", "enforcer", "target")
        
        assert result is False
    
    def test_update_cooperation_rate(self):
        """Test updating overall cooperation rate."""
        system = SocialLearningSystem()
        
        system.agent_cooperation_history["agent_1"] = [True, True, True, False]
        system.agent_cooperation_history["agent_2"] = [True, False, True, True]
        
        system.update_cooperation_rate()
        
        # 6 out of 8 total = 0.75
        assert system.learning_stats["cooperation_rate"] == 0.75
    
    def test_get_social_learning_stats(self):
        """Test getting social learning statistics."""
        system = SocialLearningSystem()
        
        system.record_interaction(
            agent_a="agent_1",
            agent_b="agent_2",
            interaction_type="test",
            outcome="cooperation",
            reward=1.0
        )
        
        stats = system.get_social_learning_stats()
        
        assert "learning_stats" in stats
        assert "num_agents" in stats
        assert "num_norms" in stats
        assert "avg_reputation" in stats
    
    def test_get_agent_social_profile(self):
        """Test getting an agent's social profile."""
        system = SocialLearningSystem()
        
        system.agent_performance["agent_1"] = [0.8, 0.9]
        system.agent_reputation["agent_1"] = 0.7
        system.agent_cooperation_history["agent_1"] = [True, True, False]
        
        profile = system.get_agent_social_profile("agent_1")
        
        assert profile["agent_id"] == "agent_1"
        assert profile["reputation"] == 0.7
        assert profile["avg_performance"] == pytest.approx(0.85)
        assert profile["cooperation_rate"] == 2/3


class TestMultiAgentSociety:
    """Tests for MultiAgentSociety class."""
    
    def test_add_agent(self):
        """Test adding an agent to the society."""
        society = MultiAgentSociety(max_agents=10)
        
        mock_agent = MagicMock()
        mock_agent.agent_id = "test_agent"
        
        result = society.add_agent(mock_agent)
        
        assert result is True
        assert "test_agent" in society.agents
        assert society.society_stats["total_agents"] >= 1
    
    def test_add_agent_max_limit(self):
        """Test that max_agents limit is enforced."""
        society = MultiAgentSociety(max_agents=2)
        
        for i in range(3):
            mock_agent = MagicMock()
            mock_agent.agent_id = f"agent_{i}"
            society.add_agent(mock_agent)
        
        assert len(society.agents) <= 2
    
    def test_remove_agent(self):
        """Test removing an agent from the society."""
        society = MultiAgentSociety()
        
        mock_agent = MagicMock()
        mock_agent.agent_id = "test_agent"
        society.add_agent(mock_agent)
        
        result = society.remove_agent("test_agent")
        
        assert result is True
        assert "test_agent" not in society.agents
    
    def test_remove_nonexistent_agent(self):
        """Test removing an agent that doesn't exist."""
        society = MultiAgentSociety()
        
        result = society.remove_agent("nonexistent_agent")
        
        assert result is False
    
    def test_coordinate_agents_resource_sharing(self):
        """Test resource sharing coordination."""
        society = MultiAgentSociety()
        
        # Create mock agents with resources
        agent_with_resources = MagicMock()
        agent_with_resources.agent_id = "rich_agent"
        agent_with_resources.get_resources.return_value = {"gold": 100}
        
        agent_in_need = MagicMock()
        agent_in_need.agent_id = "poor_agent"
        agent_in_need.needs_resources.return_value = True
        
        society.add_agent(agent_with_resources)
        society.add_agent(agent_in_need)
        
        result = society.coordinate_agents("resource_sharing")
        
        assert result["coordination_type"] == "resource_sharing"
        assert "sharing_events" in result
    
    def test_coordinate_agents_collective_decision(self):
        """Test collective decision coordination."""
        society = MultiAgentSociety()
        
        # Create agents that can vote
        for i in range(3):
            mock_agent = MagicMock()
            mock_agent.agent_id = f"voter_{i}"
            mock_agent.vote.return_value = "option_a" if i < 2 else "option_b"
            society.add_agent(mock_agent)
        
        result = society.coordinate_agents(
            "collective_decision",
            context={"options": ["option_a", "option_b"]}
        )
        
        assert result["coordination_type"] == "collective_decision"
        assert result["decision"] == "option_a"  # Majority
    
    def test_coordinate_agents_unknown_task(self):
        """Test coordination with unknown task type."""
        society = MultiAgentSociety()
        
        result = society.coordinate_agents("unknown_task")
        
        assert "error" in result
    
    def test_update_society(self):
        """Test updating society state."""
        society = MultiAgentSociety()
        
        mock_agent = MagicMock()
        mock_agent.agent_id = "test_agent"
        society.add_agent(mock_agent)
        
        society.update_society()
        
        assert society.society_stats["active_agents"] == 1
    
    def test_get_society_stats(self):
        """Test getting society statistics."""
        society = MultiAgentSociety()
        
        mock_agent = MagicMock()
        mock_agent.agent_id = "test_agent"
        society.add_agent(mock_agent)
        
        stats = society.get_society_stats()
        
        assert "society_stats" in stats
        assert "social_learning" in stats
        assert "agent_profiles" in stats


class TestCooperationStrategy:
    """Tests for CooperationStrategy enum."""
    
    def test_all_strategies_exist(self):
        """Test that all cooperation strategies are defined."""
        strategies = [
            CooperationStrategy.CONSERVATIVE,
            CooperationStrategy.BALANCED,
            CooperationStrategy.EXPLORATORY,
            CooperationStrategy.AGGRESSIVE
        ]
        
        assert len(strategies) == 4
    
    def test_strategy_values(self):
        """Test cooperation strategy string values."""
        assert CooperationStrategy.CONSERVATIVE.value == "conservative"
        assert CooperationStrategy.BALANCED.value == "balanced"
        assert CooperationStrategy.EXPLORATORY.value == "exploratory"
        assert CooperationStrategy.AGGRESSIVE.value == "aggressive"


class TestNormType:
    """Tests for NormType enum."""
    
    def test_all_norm_types_exist(self):
        """Test that all norm types are defined."""
        norm_types = [
            NormType.COOPERATION,
            NormType.PUNISHMENT,
            NormType.REWARD,
            NormType.IMITATION,
            NormType.MAJORITY_RULE
        ]
        
        assert len(norm_types) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
