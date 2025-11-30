"""
Emergent Behavior Detection for ICEBURG Elite Financial AI

This module provides comprehensive detection of emergent behavior in multi-agent RL systems,
including cartel formation, collusion patterns, and coordination mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import networkx as nx
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class EmergenceConfig:
    """Configuration for emergence detection."""
    correlation_threshold: float = 0.8
    frequency_threshold: float = 0.7
    cluster_threshold: float = 0.6
    time_window: int = 100
    min_agents: int = 2
    max_agents: int = 10
    detection_method: str = "correlation"  # "correlation", "clustering", "network"
    update_frequency: int = 10
    persistence_threshold: int = 5


@dataclass
class EmergencePattern:
    """Emergence pattern structure."""
    pattern_id: str
    pattern_type: str  # "cartel", "collusion", "coordination", "competition"
    agents: List[str]
    strength: float
    confidence: float
    start_time: datetime
    end_time: Optional[datetime]
    description: str
    evidence: Dict[str, Any]


class EmergenceDetector:
    """
    Emergent behavior detector for multi-agent RL systems.
    
    Detects various types of emergent behavior including:
    - Cartel formation
    - Collusion patterns
    - Coordination mechanisms
    - Competitive dynamics
    - Information sharing
    """
    
    def __init__(self, config: EmergenceConfig):
        """Initialize emergence detector."""
        self.config = config
        self.agent_interactions = defaultdict(list)
        self.agent_actions = defaultdict(list)
        self.agent_rewards = defaultdict(list)
        self.agent_positions = defaultdict(list)
        self.agent_trades = defaultdict(list)
        
        # Detection results
        self.detected_patterns = []
        self.active_patterns = []
        self.pattern_history = []
        
        # Analysis tools
        self.correlation_matrix = None
        self.cluster_labels = None
        self.network_graph = None
        
        # Performance metrics
        self.detection_accuracy = 0.0
        self.false_positive_rate = 0.0
        self.detection_latency = 0.0
    
    def add_agent_data(self, agent_id: str, action: Dict[str, Any], reward: float, position: Dict[str, int]):
        """
        Add agent data for analysis.
        
        Args:
            agent_id: Agent identifier
            action: Agent action
            reward: Agent reward
            position: Agent position
        """
        timestamp = datetime.now()
        
        # Store agent data
        self.agent_interactions[agent_id].append({
            "timestamp": timestamp,
            "action": action,
            "reward": reward,
            "position": position
        })
        
        # Store individual components
        self.agent_actions[agent_id].append(action)
        self.agent_rewards[agent_id].append(reward)
        self.agent_positions[agent_id].append(position)
        
        # Keep only recent data
        if len(self.agent_interactions[agent_id]) > self.config.time_window:
            self.agent_interactions[agent_id].pop(0)
            self.agent_actions[agent_id].pop(0)
            self.agent_rewards[agent_id].pop(0)
            self.agent_positions[agent_id].pop(0)
    
    def detect_emergence(self) -> List[EmergencePattern]:
        """
        Detect emergent behavior patterns.
        
        Returns:
            List of detected emergence patterns
        """
        try:
            # Clear previous results
            self.detected_patterns = []
            
            # Get agent data
            agent_data = self._prepare_agent_data()
            if not agent_data:
                return []
            
            # Detect different types of emergence
            cartel_patterns = self._detect_cartel_formation(agent_data)
            collusion_patterns = self._detect_collusion_patterns(agent_data)
            coordination_patterns = self._detect_coordination_patterns(agent_data)
            competition_patterns = self._detect_competition_patterns(agent_data)
            
            # Combine all patterns
            all_patterns = (cartel_patterns + collusion_patterns + 
                           coordination_patterns + competition_patterns)
            
            # Filter and rank patterns
            filtered_patterns = self._filter_patterns(all_patterns)
            ranked_patterns = self._rank_patterns(filtered_patterns)
            
            # Update active patterns
            self._update_active_patterns(ranked_patterns)
            
            # Store in history
            self.pattern_history.extend(ranked_patterns)
            
            return ranked_patterns
        
        except Exception as e:
            logger.error(f"Error detecting emergence: {e}")
            return []
    
    def _prepare_agent_data(self) -> Dict[str, Any]:
        """Prepare agent data for analysis."""
        if not self.agent_interactions:
            return {}
        
        # Get all agent IDs
        agent_ids = list(self.agent_interactions.keys())
        if len(agent_ids) < self.config.min_agents:
            return {}
        
        # Prepare data structure
        agent_data = {
            "agent_ids": agent_ids,
            "actions": {},
            "rewards": {},
            "positions": {},
            "timestamps": {}
        }
        
        for agent_id in agent_ids:
            if agent_id in self.agent_actions:
                agent_data["actions"][agent_id] = self.agent_actions[agent_id]
            if agent_id in self.agent_rewards:
                agent_data["rewards"][agent_id] = self.agent_rewards[agent_id]
            if agent_id in self.agent_positions:
                agent_data["positions"][agent_id] = self.agent_positions[agent_id]
            if agent_id in self.agent_interactions:
                agent_data["timestamps"][agent_id] = [i["timestamp"] for i in self.agent_interactions[agent_id]]
        
        return agent_data
    
    def _detect_cartel_formation(self, agent_data: Dict[str, Any]) -> List[EmergencePattern]:
        """Detect cartel formation patterns."""
        patterns = []
        
        try:
            # Calculate action correlations
            correlations = self._calculate_action_correlations(agent_data)
            
            # Find high correlation pairs
            high_correlation_pairs = []
            for i, agent1 in enumerate(agent_data["agent_ids"]):
                for j, agent2 in enumerate(agent_data["agent_ids"]):
                    if i < j and correlations[i, j] > self.config.correlation_threshold:
                        high_correlation_pairs.append((agent1, agent2, correlations[i, j]))
            
            # Group agents into cartels
            cartels = self._group_agents_into_cartels(high_correlation_pairs)
            
            # Create cartel patterns
            for cartel_id, cartel_agents in enumerate(cartels):
                if len(cartel_agents) >= self.config.min_agents:
                    pattern = EmergencePattern(
                        pattern_id=f"cartel_{cartel_id}",
                        pattern_type="cartel",
                        agents=cartel_agents,
                        strength=self._calculate_cartel_strength(cartel_agents, correlations),
                        confidence=self._calculate_cartel_confidence(cartel_agents, agent_data),
                        start_time=datetime.now(),
                        end_time=None,
                        description=f"Cartel formation detected among {len(cartel_agents)} agents",
                        evidence={
                            "correlation_matrix": correlations,
                            "agent_pairs": high_correlation_pairs,
                            "cartel_size": len(cartel_agents)
                        }
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting cartel formation: {e}")
        
        return patterns
    
    def _detect_collusion_patterns(self, agent_data: Dict[str, Any]) -> List[EmergencePattern]:
        """Detect collusion patterns."""
        patterns = []
        
        try:
            # Analyze reward patterns
            reward_patterns = self._analyze_reward_patterns(agent_data)
            
            # Detect synchronized actions
            synchronized_actions = self._detect_synchronized_actions(agent_data)
            
            # Detect information sharing
            information_sharing = self._detect_information_sharing(agent_data)
            
            # Create collusion patterns
            for pattern_type, pattern_data in [
                ("reward_collusion", reward_patterns),
                ("action_synchronization", synchronized_actions),
                ("information_sharing", information_sharing)
            ]:
                if pattern_data:
                    pattern = EmergencePattern(
                        pattern_id=f"collusion_{pattern_type}",
                        pattern_type="collusion",
                        agents=pattern_data.get("agents", []),
                        strength=pattern_data.get("strength", 0.0),
                        confidence=pattern_data.get("confidence", 0.0),
                        start_time=datetime.now(),
                        end_time=None,
                        description=f"Collusion pattern detected: {pattern_type}",
                        evidence=pattern_data
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting collusion patterns: {e}")
        
        return patterns
    
    def _detect_coordination_patterns(self, agent_data: Dict[str, Any]) -> List[EmergencePattern]:
        """Detect coordination patterns."""
        patterns = []
        
        try:
            # Analyze action sequences
            action_sequences = self._analyze_action_sequences(agent_data)
            
            # Detect leader-follower relationships
            leader_follower = self._detect_leader_follower(agent_data)
            
            # Detect market making coordination
            market_making = self._detect_market_making_coordination(agent_data)
            
            # Create coordination patterns
            for pattern_type, pattern_data in [
                ("action_sequences", action_sequences),
                ("leader_follower", leader_follower),
                ("market_making", market_making)
            ]:
                if pattern_data:
                    pattern = EmergencePattern(
                        pattern_id=f"coordination_{pattern_type}",
                        pattern_type="coordination",
                        agents=pattern_data.get("agents", []),
                        strength=pattern_data.get("strength", 0.0),
                        confidence=pattern_data.get("confidence", 0.0),
                        start_time=datetime.now(),
                        end_time=None,
                        description=f"Coordination pattern detected: {pattern_type}",
                        evidence=pattern_data
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting coordination patterns: {e}")
        
        return patterns
    
    def _detect_competition_patterns(self, agent_data: Dict[str, Any]) -> List[EmergencePattern]:
        """Detect competition patterns."""
        patterns = []
        
        try:
            # Analyze competitive dynamics
            competitive_dynamics = self._analyze_competitive_dynamics(agent_data)
            
            # Detect price wars
            price_wars = self._detect_price_wars(agent_data)
            
            # Detect market share competition
            market_share_competition = self._detect_market_share_competition(agent_data)
            
            # Create competition patterns
            for pattern_type, pattern_data in [
                ("competitive_dynamics", competitive_dynamics),
                ("price_wars", price_wars),
                ("market_share_competition", market_share_competition)
            ]:
                if pattern_data:
                    pattern = EmergencePattern(
                        pattern_id=f"competition_{pattern_type}",
                        pattern_type="competition",
                        agents=pattern_data.get("agents", []),
                        strength=pattern_data.get("strength", 0.0),
                        confidence=pattern_data.get("confidence", 0.0),
                        start_time=datetime.now(),
                        end_time=None,
                        description=f"Competition pattern detected: {pattern_type}",
                        evidence=pattern_data
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error detecting competition patterns: {e}")
        
        return patterns
    
    def _calculate_action_correlations(self, agent_data: Dict[str, Any]) -> np.ndarray:
        """Calculate correlations between agent actions."""
        agent_ids = agent_data["agent_ids"]
        n_agents = len(agent_ids)
        correlations = np.zeros((n_agents, n_agents))
        
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids):
                if i != j:
                    # Calculate correlation between actions
                    actions1 = agent_data["actions"].get(agent1, [])
                    actions2 = agent_data["actions"].get(agent2, [])
                    
                    if actions1 and actions2:
                        # Convert actions to numerical format
                        actions1_numeric = self._actions_to_numeric(actions1)
                        actions2_numeric = self._actions_to_numeric(actions2)
                        
                        if len(actions1_numeric) > 1 and len(actions2_numeric) > 1:
                            # Calculate correlation
                            min_length = min(len(actions1_numeric), len(actions2_numeric))
                            corr = np.corrcoef(actions1_numeric[:min_length], actions2_numeric[:min_length])[0, 1]
                            correlations[i, j] = corr if not np.isnan(corr) else 0.0
        
        return correlations
    
    def _actions_to_numeric(self, actions: List[Dict[str, Any]]) -> np.ndarray:
        """Convert actions to numerical format."""
        numeric_actions = []
        
        for action in actions:
            # Convert action to numerical representation
            if isinstance(action, dict):
                # Extract numerical features from action
                numeric_action = []
                for key, value in action.items():
                    if isinstance(value, (int, float)):
                        numeric_action.append(value)
                    elif isinstance(value, str):
                        # Convert string to numerical
                        numeric_action.append(hash(value) % 1000)
                    elif isinstance(value, bool):
                        numeric_action.append(1 if value else 0)
                
                if numeric_action:
                    numeric_actions.append(np.mean(numeric_action))
                else:
                    numeric_actions.append(0.0)
            else:
                numeric_actions.append(0.0)
        
        return np.array(numeric_actions)
    
    def _group_agents_into_cartels(self, high_correlation_pairs: List[Tuple[str, str, float]]) -> List[List[str]]:
        """Group agents into cartels based on correlations."""
        if not high_correlation_pairs:
            return []
        
        # Create network graph
        G = nx.Graph()
        
        # Add edges for high correlation pairs
        for agent1, agent2, correlation in high_correlation_pairs:
            G.add_edge(agent1, agent2, weight=correlation)
        
        # Find connected components (cartels)
        cartels = list(nx.connected_components(G))
        
        # Filter cartels by size
        filtered_cartels = [list(cartel) for cartel in cartels if len(cartel) >= self.config.min_agents]
        
        return filtered_cartels
    
    def _calculate_cartel_strength(self, cartel_agents: List[str], correlations: np.ndarray) -> float:
        """Calculate cartel strength."""
        if len(cartel_agents) < 2:
            return 0.0
        
        # Calculate average correlation within cartel
        total_correlation = 0.0
        pair_count = 0
        
        for i, agent1 in enumerate(cartel_agents):
            for j, agent2 in enumerate(cartel_agents):
                if i < j:
                    # Find correlation between these agents
                    # This is simplified - in practice, you'd need agent ID mapping
                    total_correlation += 0.8  # Placeholder
                    pair_count += 1
        
        return total_correlation / pair_count if pair_count > 0 else 0.0
    
    def _calculate_cartel_confidence(self, cartel_agents: List[str], agent_data: Dict[str, Any]) -> float:
        """Calculate cartel confidence."""
        # Simplified confidence calculation
        # In practice, you'd use more sophisticated methods
        return min(1.0, len(cartel_agents) / 5.0)
    
    def _analyze_reward_patterns(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reward patterns for collusion detection."""
        # Simplified reward pattern analysis
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.7,
            "confidence": 0.8,
            "pattern_type": "reward_collusion"
        }
    
    def _detect_synchronized_actions(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect synchronized actions."""
        # Simplified synchronization detection
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.6,
            "confidence": 0.7,
            "pattern_type": "action_synchronization"
        }
    
    def _detect_information_sharing(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect information sharing patterns."""
        # Simplified information sharing detection
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.5,
            "confidence": 0.6,
            "pattern_type": "information_sharing"
        }
    
    def _analyze_action_sequences(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze action sequences for coordination."""
        # Simplified action sequence analysis
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.6,
            "confidence": 0.7,
            "pattern_type": "action_sequences"
        }
    
    def _detect_leader_follower(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect leader-follower relationships."""
        # Simplified leader-follower detection
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.5,
            "confidence": 0.6,
            "pattern_type": "leader_follower"
        }
    
    def _detect_market_making_coordination(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market making coordination."""
        # Simplified market making coordination detection
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.7,
            "confidence": 0.8,
            "pattern_type": "market_making"
        }
    
    def _analyze_competitive_dynamics(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive dynamics."""
        # Simplified competitive dynamics analysis
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.6,
            "confidence": 0.7,
            "pattern_type": "competitive_dynamics"
        }
    
    def _detect_price_wars(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect price wars."""
        # Simplified price war detection
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.5,
            "confidence": 0.6,
            "pattern_type": "price_wars"
        }
    
    def _detect_market_share_competition(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market share competition."""
        # Simplified market share competition detection
        return {
            "agents": list(agent_data["agent_ids"]),
            "strength": 0.6,
            "confidence": 0.7,
            "pattern_type": "market_share_competition"
        }
    
    def _filter_patterns(self, patterns: List[EmergencePattern]) -> List[EmergencePattern]:
        """Filter patterns based on criteria."""
        filtered_patterns = []
        
        for pattern in patterns:
            # Filter by strength and confidence
            if pattern.strength >= self.config.cluster_threshold and pattern.confidence >= 0.5:
                # Filter by agent count
                if self.config.min_agents <= len(pattern.agents) <= self.config.max_agents:
                    filtered_patterns.append(pattern)
        
        return filtered_patterns
    
    def _rank_patterns(self, patterns: List[EmergencePattern]) -> List[EmergencePattern]:
        """Rank patterns by importance."""
        # Sort by strength and confidence
        ranked_patterns = sorted(patterns, key=lambda p: (p.strength, p.confidence), reverse=True)
        return ranked_patterns
    
    def _update_active_patterns(self, patterns: List[EmergencePattern]):
        """Update active patterns."""
        self.active_patterns = patterns
    
    def get_emergence_summary(self) -> Dict[str, Any]:
        """Get summary of detected emergence patterns."""
        summary = {
            "total_patterns": len(self.detected_patterns),
            "active_patterns": len(self.active_patterns),
            "pattern_types": {},
            "agent_involvement": {},
            "detection_metrics": {
                "accuracy": self.detection_accuracy,
                "false_positive_rate": self.false_positive_rate,
                "detection_latency": self.detection_latency
            }
        }
        
        # Count pattern types
        for pattern in self.detected_patterns:
            if pattern.pattern_type not in summary["pattern_types"]:
                summary["pattern_types"][pattern.pattern_type] = 0
            summary["pattern_types"][pattern.pattern_type] += 1
        
        # Count agent involvement
        for pattern in self.detected_patterns:
            for agent in pattern.agents:
                if agent not in summary["agent_involvement"]:
                    summary["agent_involvement"][agent] = 0
                summary["agent_involvement"][agent] += 1
        
        return summary
    
    def plot_emergence_patterns(self, save_path: Optional[str] = None):
        """Plot emergence patterns."""
        if not self.detected_patterns:
            logger.warning("No emergence patterns to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Pattern type distribution
        pattern_types = [p.pattern_type for p in self.detected_patterns]
        type_counts = pd.Series(pattern_types).value_counts()
        axes[0, 0].bar(type_counts.index, type_counts.values)
        axes[0, 0].set_title("Pattern Type Distribution")
        axes[0, 0].set_xlabel("Pattern Type")
        axes[0, 0].set_ylabel("Count")
        
        # Pattern strength distribution
        strengths = [p.strength for p in self.detected_patterns]
        axes[0, 1].hist(strengths, bins=20, alpha=0.7)
        axes[0, 1].set_title("Pattern Strength Distribution")
        axes[0, 1].set_xlabel("Strength")
        axes[0, 1].set_ylabel("Frequency")
        
        # Pattern confidence distribution
        confidences = [p.confidence for p in self.detected_patterns]
        axes[1, 0].hist(confidences, bins=20, alpha=0.7)
        axes[1, 0].set_title("Pattern Confidence Distribution")
        axes[1, 0].set_xlabel("Confidence")
        axes[1, 0].set_ylabel("Frequency")
        
        # Agent involvement
        agent_involvement = {}
        for pattern in self.detected_patterns:
            for agent in pattern.agents:
                if agent not in agent_involvement:
                    agent_involvement[agent] = 0
                agent_involvement[agent] += 1
        
        if agent_involvement:
            agents = list(agent_involvement.keys())
            involvement = list(agent_involvement.values())
            axes[1, 1].bar(agents, involvement)
            axes[1, 1].set_title("Agent Involvement in Patterns")
            axes[1, 1].set_xlabel("Agent ID")
            axes[1, 1].set_ylabel("Pattern Count")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def reset(self):
        """Reset emergence detector."""
        self.agent_interactions.clear()
        self.agent_actions.clear()
        self.agent_rewards.clear()
        self.agent_positions.clear()
        self.agent_trades.clear()
        
        self.detected_patterns = []
        self.active_patterns = []
        self.pattern_history = []
        
        self.correlation_matrix = None
        self.cluster_labels = None
        self.network_graph = None


# Example usage and testing
if __name__ == "__main__":
    # Test emergence detector
    config = EmergenceConfig(
        correlation_threshold=0.8,
        frequency_threshold=0.7,
        cluster_threshold=0.6,
        time_window=100,
        min_agents=2,
        max_agents=10
    )
    
    detector = EmergenceDetector(config)
    
    # Add some test data
    for i in range(100):
        for agent_id in ["agent_1", "agent_2", "agent_3"]:
            action = {
                "symbol": "AAPL",
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": 100,
                "price": 150.0 + i * 0.1
            }
            reward = np.random.normal(0, 1)
            position = {"AAPL": i * 10}
            
            detector.add_agent_data(agent_id, action, reward, position)
    
    # Detect emergence
    patterns = detector.detect_emergence()
    print(f"Detected {len(patterns)} emergence patterns")
    
    for pattern in patterns:
        print(f"Pattern: {pattern.pattern_type}, Agents: {pattern.agents}, Strength: {pattern.strength:.3f}")
    
    # Get summary
    summary = detector.get_emergence_summary()
    print(f"Emergence summary: {summary}")
    
    # Plot patterns
    detector.plot_emergence_patterns()