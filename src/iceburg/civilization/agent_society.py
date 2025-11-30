"""
Multi-Agent Social Dynamics for AGI Civilization
Implements social learning, norm formation, and cooperation mechanisms.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CooperationStrategy(Enum):
    """Agent cooperation strategies."""
    CONSERVATIVE = "conservative"  # GPT-like
    BALANCED = "balanced"  # Claude-like
    EXPLORATORY = "exploratory"  # DeepSeek-like
    AGGRESSIVE = "aggressive"  # Small LLM-like


class NormType(Enum):
    """Types of social norms."""
    COOPERATION = "cooperation"
    PUNISHMENT = "punishment"
    REWARD = "reward"
    IMITATION = "imitation"
    MAJORITY_RULE = "majority_rule"


@dataclass
class SocialInteraction:
    """Represents a social interaction between agents."""
    interaction_id: str
    timestamp: float
    agent_a: str
    agent_b: str
    interaction_type: str
    outcome: str
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialNorm:
    """Represents a social norm in the society."""
    norm_id: str
    norm_type: NormType
    description: str
    strength: float  # 0.0 to 1.0
    enforcement_rate: float  # How often it's enforced
    created_time: float
    last_enforced: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SocialLearningSystem:
    """
    Implements social learning mechanisms for multi-agent systems.
    
    Features:
    - Imitation of successful peers
    - Majority rule decision making
    - Punishment mechanisms
    - Reward distribution
    - Norm formation and enforcement
    """
    
    def __init__(self, max_interactions: int = 10000):
        """
        Initialize the social learning system.
        
        Args:
            max_interactions: Maximum number of interactions to track
        """
        self.max_interactions = max_interactions
        
        # Interaction tracking
        self.interactions: deque = deque(maxlen=max_interactions)
        self.interaction_counter = 0
        
        # Agent performance tracking
        self.agent_performance: Dict[str, List[float]] = defaultdict(list)
        self.agent_reputation: Dict[str, float] = defaultdict(lambda: 0.5)
        self.agent_cooperation_history: Dict[str, List[bool]] = defaultdict(list)
        
        # Social norms
        self.norms: Dict[str, SocialNorm] = {}
        self.norm_counter = 0
        
        # Learning parameters
        self.imitation_rate = 0.3  # Probability of imitating successful peers
        self.punishment_rate = 0.1  # Probability of punishing defectors
        self.reward_rate = 0.2  # Probability of rewarding cooperators
        self.majority_threshold = 0.6  # Threshold for majority rule decisions
        
        # Performance metrics
        self.learning_stats = {
            "total_interactions": 0,
            "successful_imitations": 0,
            "punishments_distributed": 0,
            "rewards_distributed": 0,
            "norms_formed": 0,
            "cooperation_rate": 0.0
        }
    
    def record_interaction(self, 
                          agent_a: str, 
                          agent_b: str, 
                          interaction_type: str,
                          outcome: str,
                          reward: float,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Record a social interaction.
        
        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            interaction_type: Type of interaction
            outcome: Outcome of interaction
            reward: Reward received
            metadata: Additional metadata
            
        Returns:
            Interaction ID
        """
        if metadata is None:
            metadata = {}
        
        interaction = SocialInteraction(
            interaction_id=f"interaction_{self.interaction_counter}",
            timestamp=time.time(),
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_type=interaction_type,
            outcome=outcome,
            reward=reward,
            metadata=metadata
        )
        
        self.interactions.append(interaction)
        self.interaction_counter += 1
        self.learning_stats["total_interactions"] += 1
        
        # Update agent performance
        self.agent_performance[agent_a].append(reward)
        self.agent_performance[agent_b].append(reward)
        
        # Update cooperation history
        cooperated = outcome == "cooperation"
        self.agent_cooperation_history[agent_a].append(cooperated)
        self.agent_cooperation_history[agent_b].append(cooperated)
        
        # Update reputation
        self._update_reputation(agent_a, reward, cooperated)
        self._update_reputation(agent_b, reward, cooperated)
        
        logger.debug(f"Recorded interaction: {agent_a} <-> {agent_b} ({outcome})")
        return interaction.interaction_id
    
    def _update_reputation(self, agent_id: str, reward: float, cooperated: bool):
        """Update agent reputation based on behavior."""
        # Simple reputation update based on cooperation and reward
        reputation_change = 0.01 if cooperated else -0.01
        reputation_change += reward * 0.001  # Reward influence
        
        self.agent_reputation[agent_id] = max(0.0, min(1.0, 
            self.agent_reputation[agent_id] + reputation_change))
    
    def get_peer_success_models(self, agent_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k successful peers for imitation.
        
        Args:
            agent_id: Agent requesting peer models
            top_k: Number of top peers to return
            
        Returns:
            List of (peer_id, success_score) tuples
        """
        # Calculate success scores for all agents
        success_scores = {}
        for peer_id, performance_history in self.agent_performance.items():
            if peer_id == agent_id:
                continue
            
            if performance_history:
                # Success score based on recent performance and reputation
                recent_performance = np.mean(performance_history[-10:]) if len(performance_history) >= 10 else np.mean(performance_history)
                reputation = self.agent_reputation[peer_id]
                success_scores[peer_id] = 0.7 * recent_performance + 0.3 * reputation
        
        # Return top-k peers
        sorted_peers = sorted(success_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_peers[:top_k]
    
    def should_imitate_peer(self, agent_id: str, peer_id: str) -> bool:
        """
        Determine if agent should imitate a specific peer.
        
        Args:
            agent_id: Agent considering imitation
            peer_id: Peer to potentially imitate
            
        Returns:
            True if should imitate
        """
        # Check if peer is successful enough
        peer_performance = self.agent_performance.get(peer_id, [])
        if not peer_performance:
            return False
        
        peer_success = np.mean(peer_performance[-5:]) if len(peer_performance) >= 5 else np.mean(peer_performance)
        agent_performance = self.agent_performance.get(agent_id, [])
        agent_success = np.mean(agent_performance[-5:]) if len(agent_performance) >= 5 else 0.5
        
        # Imitate if peer is significantly more successful
        success_gap = peer_success - agent_success
        imitation_probability = min(self.imitation_rate, success_gap * 0.5)
        
        return np.random.random() < imitation_probability
    
    def get_majority_decision(self, agent_id: str, decision_options: List[str]) -> Optional[str]:
        """
        Get majority rule decision from peer group.
        
        Args:
            agent_id: Agent requesting decision
            decision_options: Available decision options
            
        Returns:
            Majority decision or None if no clear majority
        """
        # Get recent decisions from peer interactions
        recent_interactions = [
            interaction for interaction in self.interactions
            if time.time() - interaction.timestamp < 300.0  # Last 5 minutes
        ]
        
        # Count decisions by option
        decision_counts = defaultdict(int)
        for interaction in recent_interactions:
            if interaction.interaction_type == "decision" and interaction.outcome in decision_options:
                decision_counts[interaction.outcome] += 1
        
        if not decision_counts:
            return None
        
        # Find majority decision
        total_decisions = sum(decision_counts.values())
        for option, count in decision_counts.items():
            if count / total_decisions >= self.majority_threshold:
                return option
        
        return None
    
    def should_punish_agent(self, agent_id: str, target_agent: str) -> bool:
        """
        Determine if agent should punish another agent.
        
        Args:
            agent_id: Agent considering punishment
            target_agent: Agent to potentially punish
            
        Returns:
            True if should punish
        """
        # Check target agent's cooperation history
        cooperation_history = self.agent_cooperation_history.get(target_agent, [])
        if not cooperation_history:
            return False
        
        # Calculate defection rate
        recent_cooperation = cooperation_history[-10:] if len(cooperation_history) >= 10 else cooperation_history
        defection_rate = 1.0 - sum(recent_cooperation) / len(recent_cooperation)
        
        # Punish if defection rate is high
        punishment_probability = min(self.punishment_rate, defection_rate * 0.5)
        return np.random.random() < punishment_probability
    
    def should_reward_agent(self, agent_id: str, target_agent: str) -> bool:
        """
        Determine if agent should reward another agent.
        
        Args:
            agent_id: Agent considering reward
            target_agent: Agent to potentially reward
            
        Returns:
            True if should reward
        """
        # Check target agent's cooperation history
        cooperation_history = self.agent_cooperation_history.get(target_agent, [])
        if not cooperation_history:
            return False
        
        # Calculate cooperation rate
        recent_cooperation = cooperation_history[-10:] if len(cooperation_history) >= 10 else cooperation_history
        cooperation_rate = sum(recent_cooperation) / len(recent_cooperation)
        
        # Reward if cooperation rate is high
        reward_probability = min(self.reward_rate, cooperation_rate * 0.3)
        return np.random.random() < reward_probability
    
    def form_norm(self, norm_type: NormType, description: str, 
                  strength: float = 0.5, enforcement_rate: float = 0.1) -> str:
        """
        Form a new social norm.
        
        Args:
            norm_type: Type of norm
            description: Description of the norm
            strength: Initial strength of the norm
            enforcement_rate: Rate of enforcement
            
        Returns:
            Norm ID
        """
        norm = SocialNorm(
            norm_id=f"norm_{self.norm_counter}",
            norm_type=norm_type,
            description=description,
            strength=strength,
            enforcement_rate=enforcement_rate,
            created_time=time.time(),
            last_enforced=0.0,
            metadata={}
        )
        
        self.norms[norm.norm_id] = norm
        self.norm_counter += 1
        self.learning_stats["norms_formed"] += 1
        
        logger.info(f"Formed new norm: {description} (strength: {strength})")
        return norm.norm_id
    
    def enforce_norm(self, norm_id: str, agent_id: str, target_agent: str) -> bool:
        """
        Enforce a social norm.
        
        Args:
            norm_id: ID of norm to enforce
            agent_id: Agent enforcing the norm
            target_agent: Agent being targeted
            
        Returns:
            True if norm was enforced
        """
        if norm_id not in self.norms:
            return False
        
        norm = self.norms[norm_id]
        current_time = time.time()
        
        # Check if norm should be enforced
        if current_time - norm.last_enforced < 1.0 / norm.enforcement_rate:
            return False
        
        # Enforce based on norm type
        if norm.norm_type == NormType.PUNISHMENT:
            # Apply punishment
            self.agent_reputation[target_agent] = max(0.0, 
                self.agent_reputation[target_agent] - 0.1)
            self.learning_stats["punishments_distributed"] += 1
            
        elif norm.norm_type == NormType.REWARD:
            # Apply reward
            self.agent_reputation[target_agent] = min(1.0, 
                self.agent_reputation[target_agent] + 0.1)
            self.learning_stats["rewards_distributed"] += 1
        
        # Update norm enforcement time
        norm.last_enforced = current_time
        
        logger.debug(f"Enforced norm {norm_id} by {agent_id} on {target_agent}")
        return True
    
    def update_cooperation_rate(self):
        """Update the overall cooperation rate in the society."""
        if not self.agent_cooperation_history:
            self.learning_stats["cooperation_rate"] = 0.0
            return
        
        total_cooperation = 0
        total_interactions = 0
        
        for cooperation_history in self.agent_cooperation_history.values():
            if cooperation_history:
                total_cooperation += sum(cooperation_history)
                total_interactions += len(cooperation_history)
        
        if total_interactions > 0:
            self.learning_stats["cooperation_rate"] = total_cooperation / total_interactions
    
    def get_social_learning_stats(self) -> Dict[str, Any]:
        """Get social learning statistics."""
        self.update_cooperation_rate()
        
        return {
            "learning_stats": self.learning_stats.copy(),
            "num_agents": len(self.agent_performance),
            "num_norms": len(self.norms),
            "avg_reputation": np.mean(list(self.agent_reputation.values())) if self.agent_reputation else 0.0,
            "recent_interactions": len([i for i in self.interactions if time.time() - i.timestamp < 3600.0])
        }
    
    def get_agent_social_profile(self, agent_id: str) -> Dict[str, Any]:
        """Get social profile for a specific agent."""
        performance_history = self.agent_performance.get(agent_id, [])
        cooperation_history = self.agent_cooperation_history.get(agent_id, [])
        
        return {
            "agent_id": agent_id,
            "reputation": self.agent_reputation.get(agent_id, 0.5),
            "avg_performance": np.mean(performance_history) if performance_history else 0.0,
            "cooperation_rate": sum(cooperation_history) / len(cooperation_history) if cooperation_history else 0.0,
            "total_interactions": len(performance_history),
            "recent_performance": performance_history[-10:] if len(performance_history) >= 10 else performance_history
        }


class MultiAgentSociety:
    """
    Main class for managing multi-agent social dynamics.
    
    Features:
    - Agent coordination and communication
    - Social learning and norm formation
    - Cooperation mechanisms
    - Resource sharing and trading
    - Conflict resolution
    """
    
    def __init__(self, max_agents: int = 100):
        """
        Initialize the multi-agent society.
        
        Args:
            max_agents: Maximum number of agents
        """
        self.max_agents = max_agents
        self.agents: Dict[str, Any] = {}  # Will be populated with PersistentAgent instances
        self.social_learning = SocialLearningSystem()
        
        # Society state
        self.society_stats = {
            "total_agents": 0,
            "active_agents": 0,
            "cooperation_rate": 0.0,
            "conflict_rate": 0.0,
            "resource_sharing_rate": 0.0
        }
    
    def add_agent(self, agent: Any) -> bool:
        """
        Add an agent to the society.
        
        Args:
            agent: Agent instance to add
            
        Returns:
            True if agent was added successfully
        """
        if len(self.agents) >= self.max_agents:
            logger.warning("Maximum agents reached, cannot add new agent")
            return False
        
        agent_id = getattr(agent, 'agent_id', f"agent_{len(self.agents)}")
        self.agents[agent_id] = agent
        self.society_stats["total_agents"] += 1
        self.society_stats["active_agents"] += 1
        
        logger.info(f"Added agent {agent_id} to society")
        return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the society.
        
        Args:
            agent_id: ID of agent to remove
            
        Returns:
            True if agent was removed successfully
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.society_stats["active_agents"] -= 1
            logger.info(f"Removed agent {agent_id} from society")
            return True
        return False
    
    def coordinate_agents(self, coordination_task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Coordinate agents for a specific task.
        
        Args:
            coordination_task: Task to coordinate
            context: Context for coordination
            
        Returns:
            Coordination results
        """
        if context is None:
            context = {}
        
        logger.info(f"Coordinating agents for task: {coordination_task}")
        
        # Get available agents
        available_agents = list(self.agents.keys())
        
        # Simple coordination strategy
        if coordination_task == "resource_sharing":
            return self._coordinate_resource_sharing(available_agents, context)
        elif coordination_task == "conflict_resolution":
            return self._coordinate_conflict_resolution(available_agents, context)
        elif coordination_task == "collective_decision":
            return self._coordinate_collective_decision(available_agents, context)
        else:
            return {"error": f"Unknown coordination task: {coordination_task}"}
    
    def _coordinate_resource_sharing(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate resource sharing among agents."""
        # Simple resource sharing coordination
        sharing_events = []
        
        for agent_id in agents:
            # Check if agent has resources to share
            agent = self.agents.get(agent_id)
            if agent and hasattr(agent, 'get_resources'):
                resources = agent.get_resources()
                if resources:
                    # Find agents in need
                    for other_agent_id in agents:
                        if other_agent_id != agent_id:
                            other_agent = self.agents.get(other_agent_id)
                            if other_agent and hasattr(other_agent, 'needs_resources'):
                                if other_agent.needs_resources():
                                    # Record sharing event
                                    sharing_events.append({
                                        "from": agent_id,
                                        "to": other_agent_id,
                                        "resources": resources,
                                        "timestamp": time.time()
                                    })
        
        return {
            "coordination_type": "resource_sharing",
            "sharing_events": sharing_events,
            "success": len(sharing_events) > 0
        }
    
    def _coordinate_conflict_resolution(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate conflict resolution among agents."""
        # Simple conflict resolution using majority rule
        conflicts = context.get("conflicts", [])
        resolutions = []
        
        for conflict in conflicts:
            # Use social learning system for majority decision
            decision = self.social_learning.get_majority_decision(
                conflict.get("initiator", ""),
                conflict.get("options", [])
            )
            
            if decision:
                resolutions.append({
                    "conflict_id": conflict.get("id", "unknown"),
                    "resolution": decision,
                    "method": "majority_rule",
                    "timestamp": time.time()
                })
        
        return {
            "coordination_type": "conflict_resolution",
            "resolutions": resolutions,
            "success": len(resolutions) > 0
        }
    
    def _coordinate_collective_decision(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate collective decision making."""
        decision_options = context.get("options", [])
        if not decision_options:
            return {"error": "No decision options provided"}
        
        # Collect votes from agents
        votes = {}
        for agent_id in agents:
            agent = self.agents.get(agent_id)
            if agent and hasattr(agent, 'vote'):
                vote = agent.vote(decision_options, context)
                if vote:
                    votes[agent_id] = vote
        
        # Determine majority decision
        vote_counts = defaultdict(int)
        for vote in votes.values():
            vote_counts[vote] += 1
        
        if vote_counts:
            majority_decision = max(vote_counts.items(), key=lambda x: x[1])[0]
            return {
                "coordination_type": "collective_decision",
                "decision": majority_decision,
                "votes": votes,
                "vote_counts": dict(vote_counts),
                "success": True
            }
        else:
            return {
                "coordination_type": "collective_decision",
                "error": "No votes collected",
                "success": False
            }
    
    def update_society(self):
        """Update the society state and statistics."""
        # Update social learning
        self.social_learning.update_cooperation_rate()
        
        # Update society statistics
        learning_stats = self.social_learning.get_social_learning_stats()
        self.society_stats["cooperation_rate"] = learning_stats["learning_stats"]["cooperation_rate"]
        
        # Calculate other metrics
        self.society_stats["active_agents"] = len(self.agents)
        
        logger.debug(f"Society updated - {self.society_stats['active_agents']} active agents")
    
    def get_society_stats(self) -> Dict[str, Any]:
        """Get society statistics."""
        self.update_society()
        
        return {
            "society_stats": self.society_stats.copy(),
            "social_learning": self.social_learning.get_social_learning_stats(),
            "agent_profiles": {
                agent_id: self.social_learning.get_agent_social_profile(agent_id)
                for agent_id in self.agents.keys()
            }
        }
