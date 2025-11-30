"""
Analysis Tools for Elite Financial AI

This module provides analysis tools for RL agents and behavior.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class AgentAnalyzer:
    """
    Analyzer for individual agents.
    
    This class provides comprehensive analysis capabilities for individual
    reinforcement learning agents, including performance metrics, behavior
    analysis, and learning progress tracking.
    """
    
    def __init__(self):
        """Initialize agent analyzer."""
        self.analysis_data = {}
    
    def analyze_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual agent performance."""
        try:
            # Extract agent performance metrics
            rewards = agent_data.get('rewards', [])
            actions = agent_data.get('actions', [])
            states = agent_data.get('states', [])
            
            # Calculate performance metrics
            if rewards:
                avg_reward = np.mean(rewards)
                total_reward = np.sum(rewards)
                reward_std = np.std(rewards)
                max_reward = np.max(rewards)
                min_reward = np.min(rewards)
                
                # Calculate reward trend
                if len(rewards) > 10:
                    recent_rewards = rewards[-10:]
                    early_rewards = rewards[:10]
                    improvement = np.mean(recent_rewards) - np.mean(early_rewards)
                else:
                    improvement = 0.0
            else:
                avg_reward = total_reward = reward_std = max_reward = min_reward = improvement = 0.0
            
            # Analyze action distribution
            if actions:
                action_counts = {}
                for action in actions:
                    action_str = str(action)
                    action_counts[action_str] = action_counts.get(action_str, 0) + 1
                most_common_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "unknown"
            else:
                action_counts = {}
                most_common_action = "unknown"
            
            # Calculate exploration metrics
            unique_actions = len(set(str(a) for a in actions)) if actions else 0
            total_actions = len(actions)
            exploration_rate = unique_actions / total_actions if total_actions > 0 else 0.0
            
            return {
                "performance": {
                    "avg_reward": float(avg_reward),
                    "total_reward": float(total_reward),
                    "reward_std": float(reward_std),
                    "max_reward": float(max_reward),
                    "min_reward": float(min_reward),
                    "improvement": float(improvement)
                },
                "behavior": {
                    "total_actions": total_actions,
                    "unique_actions": unique_actions,
                    "exploration_rate": float(exploration_rate),
                    "most_common_action": most_common_action,
                    "action_distribution": action_counts
                },
                "episodes": len(rewards),
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "analysis": "failed"}


class BehaviorAnalyzer:
    """
    Analyzer for agent behavior patterns.
    
    This class provides detailed analysis of agent behavior patterns,
    including action analysis, state transitions, and behavioral
    consistency metrics.
    """
    
    def __init__(self):
        """Initialize behavior analyzer."""
        self.behavior_data = {}
    
    def analyze_behavior(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent behavior patterns."""
        try:
            # Extract behavior data
            actions = behavior_data.get('actions', [])
            states = behavior_data.get('states', [])
            rewards = behavior_data.get('rewards', [])
            
            # Analyze action patterns
            if actions:
                # Calculate action frequency
                action_freq = {}
                for action in actions:
                    action_str = str(action)
                    action_freq[action_str] = action_freq.get(action_str, 0) + 1
                
                # Calculate action entropy (diversity)
                total_actions = len(actions)
                entropy = 0.0
                for count in action_freq.values():
                    p = count / total_actions
                    if p > 0:
                        entropy -= p * np.log2(p)
                
                # Find dominant actions
                sorted_actions = sorted(action_freq.items(), key=lambda x: x[1], reverse=True)
                dominant_actions = sorted_actions[:3]  # Top 3 actions
            else:
                action_freq = {}
                entropy = 0.0
                dominant_actions = []
            
            # Analyze state patterns
            if states:
                # Calculate state diversity
                unique_states = len(set(str(s) for s in states))
                state_diversity = unique_states / len(states) if states else 0.0
                
                # Calculate state transitions
                transitions = {}
                for i in range(len(states) - 1):
                    current_state = str(states[i])
                    next_state = str(states[i + 1])
                    transition = f"{current_state} -> {next_state}"
                    transitions[transition] = transitions.get(transition, 0) + 1
            else:
                state_diversity = 0.0
                transitions = {}
            
            # Analyze reward patterns
            if rewards:
                # Calculate reward consistency
                reward_std = np.std(rewards)
                reward_mean = np.mean(rewards)
                consistency = 1.0 / (1.0 + reward_std) if reward_std > 0 else 1.0
                
                # Calculate reward trend
                if len(rewards) > 5:
                    recent_rewards = rewards[-5:]
                    early_rewards = rewards[:5]
                    trend = np.mean(recent_rewards) - np.mean(early_rewards)
                else:
                    trend = 0.0
            else:
                consistency = 0.0
                trend = 0.0
            
            return {
                "action_analysis": {
                    "total_actions": len(actions),
                    "unique_actions": len(action_freq),
                    "action_entropy": float(entropy),
                    "dominant_actions": dominant_actions,
                    "action_frequency": action_freq
                },
                "state_analysis": {
                    "total_states": len(states),
                    "state_diversity": float(state_diversity),
                    "unique_transitions": len(transitions),
                    "top_transitions": sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]
                },
                "reward_analysis": {
                    "total_rewards": len(rewards),
                    "reward_consistency": float(consistency),
                    "reward_trend": float(trend),
                    "avg_reward": float(np.mean(rewards)) if rewards else 0.0
                },
                "behavior_summary": {
                    "exploration_level": float(entropy),
                    "consistency_level": float(consistency),
                    "learning_progress": float(trend),
                    "overall_activity": len(actions) + len(states)
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "behavior_analysis": "failed"}
