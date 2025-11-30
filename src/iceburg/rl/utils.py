"""
Utilities for Elite Financial AI

This module provides utility functions for RL agents and performance metrics.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch


class RLUtils:
    """
    Utility functions for RL agents.
    
    This class provides utility functions for analyzing and processing
    reinforcement learning agent data, including metrics calculation,
    data preprocessing, and performance analysis.
    """
    
    def __init__(self):
        """Initialize RL utilities."""
        self.utils_data = {}
    
    def calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate RL metrics."""
        try:
            # Extract data
            rewards = data.get('rewards', [])
            actions = data.get('actions', [])
            states = data.get('states', [])
            episodes = data.get('episodes', [])
            
            metrics = {}
            
            # Reward metrics
            if rewards:
                metrics['reward'] = {
                    'mean': float(np.mean(rewards)),
                    'std': float(np.std(rewards)),
                    'min': float(np.min(rewards)),
                    'max': float(np.max(rewards)),
                    'total': float(np.sum(rewards)),
                    'count': len(rewards)
                }
                
                # Calculate reward trends
                if len(rewards) > 10:
                    recent_rewards = rewards[-10:]
                    early_rewards = rewards[:10]
                    metrics['reward']['trend'] = float(np.mean(recent_rewards) - np.mean(early_rewards))
                    metrics['reward']['improvement'] = float(np.mean(recent_rewards) / np.mean(early_rewards)) if np.mean(early_rewards) != 0 else 1.0
                else:
                    metrics['reward']['trend'] = 0.0
                    metrics['reward']['improvement'] = 1.0
            else:
                metrics['reward'] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'total': 0.0, 'count': 0, 'trend': 0.0, 'improvement': 1.0}
            
            # Action metrics
            if actions:
                unique_actions = len(set(str(a) for a in actions))
                metrics['action'] = {
                    'total': len(actions),
                    'unique': unique_actions,
                    'diversity': unique_actions / len(actions) if actions else 0.0
                }
                
                # Action frequency
                action_counts = {}
                for action in actions:
                    action_str = str(action)
                    action_counts[action_str] = action_counts.get(action_str, 0) + 1
                metrics['action']['frequency'] = action_counts
            else:
                metrics['action'] = {'total': 0, 'unique': 0, 'diversity': 0.0, 'frequency': {}}
            
            # State metrics
            if states:
                unique_states = len(set(str(s) for s in states))
                metrics['state'] = {
                    'total': len(states),
                    'unique': unique_states,
                    'diversity': unique_states / len(states) if states else 0.0
                }
            else:
                metrics['state'] = {'total': 0, 'unique': 0, 'diversity': 0.0}
            
            # Episode metrics
            if episodes:
                episode_rewards = [ep.get('total_reward', 0) for ep in episodes if isinstance(ep, dict)]
                episode_lengths = [ep.get('length', 0) for ep in episodes if isinstance(ep, dict)]
                
                metrics['episode'] = {
                    'count': len(episodes),
                    'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                    'avg_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                    'total_steps': sum(episode_lengths) if episode_lengths else 0
                }
            else:
                metrics['episode'] = {'count': 0, 'avg_reward': 0.0, 'avg_length': 0.0, 'total_steps': 0}
            
            # Overall performance score
            reward_score = metrics['reward']['mean'] if metrics['reward']['mean'] > 0 else 0.0
            diversity_score = (metrics['action']['diversity'] + metrics['state']['diversity']) / 2
            episode_score = metrics['episode']['avg_reward'] if metrics['episode']['avg_reward'] > 0 else 0.0
            
            metrics['overall'] = {
                'performance_score': float(reward_score),
                'diversity_score': float(diversity_score),
                'episode_score': float(episode_score),
                'combined_score': float((reward_score + diversity_score + episode_score) / 3)
            }
            
            return metrics
        except Exception as e:
            return {"error": str(e), "metrics": "calculation_failed"}


class PerformanceMetrics:
    """
    Performance metrics for RL agents.
    
    This class provides comprehensive performance analysis for reinforcement
    learning agents, including reward analysis, learning progress tracking,
    and overall performance scoring.
    """
    
    def __init__(self):
        """Initialize performance metrics."""
        self.metrics_data = {}
    
    def calculate_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            # Extract performance data
            rewards = performance_data.get('rewards', [])
            episodes = performance_data.get('episodes', [])
            training_time = performance_data.get('training_time', 0.0)
            convergence_data = performance_data.get('convergence', {})
            
            performance = {}
            
            # Reward-based performance
            if rewards:
                # Basic reward statistics
                performance['rewards'] = {
                    'mean': float(np.mean(rewards)),
                    'median': float(np.median(rewards)),
                    'std': float(np.std(rewards)),
                    'min': float(np.min(rewards)),
                    'max': float(np.max(rewards)),
                    'q25': float(np.percentile(rewards, 25)),
                    'q75': float(np.percentile(rewards, 75))
                }
                
                # Reward stability
                if len(rewards) > 1:
                    reward_changes = np.diff(rewards)
                    performance['rewards']['stability'] = float(1.0 / (1.0 + np.std(reward_changes)))
                    performance['rewards']['volatility'] = float(np.std(reward_changes))
                else:
                    performance['rewards']['stability'] = 1.0
                    performance['rewards']['volatility'] = 0.0
                
                # Learning progress
                if len(rewards) > 10:
                    window_size = min(10, len(rewards) // 4)
                    early_rewards = rewards[:window_size]
                    late_rewards = rewards[-window_size:]
                    performance['learning'] = {
                        'early_avg': float(np.mean(early_rewards)),
                        'late_avg': float(np.mean(late_rewards)),
                        'improvement': float(np.mean(late_rewards) - np.mean(early_rewards)),
                        'improvement_ratio': float(np.mean(late_rewards) / np.mean(early_rewards)) if np.mean(early_rewards) != 0 else 1.0
                    }
                else:
                    performance['learning'] = {
                        'early_avg': float(np.mean(rewards)),
                        'late_avg': float(np.mean(rewards)),
                        'improvement': 0.0,
                        'improvement_ratio': 1.0
                    }
            else:
                performance['rewards'] = {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0, 'stability': 0.0, 'volatility': 0.0}
                performance['learning'] = {'early_avg': 0.0, 'late_avg': 0.0, 'improvement': 0.0, 'improvement_ratio': 1.0}
            
            # Episode-based performance
            if episodes:
                episode_rewards = [ep.get('total_reward', 0) for ep in episodes if isinstance(ep, dict)]
                episode_lengths = [ep.get('length', 0) for ep in episodes if isinstance(ep, dict)]
                
                performance['episodes'] = {
                    'count': len(episodes),
                    'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                    'avg_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                    'total_steps': sum(episode_lengths) if episode_lengths else 0,
                    'success_rate': len([r for r in episode_rewards if r > 0]) / len(episode_rewards) if episode_rewards else 0.0
                }
            else:
                performance['episodes'] = {'count': 0, 'avg_reward': 0.0, 'avg_length': 0.0, 'total_steps': 0, 'success_rate': 0.0}
            
            # Training efficiency
            if training_time > 0:
                total_reward = sum(rewards) if rewards else 0.0
                performance['efficiency'] = {
                    'rewards_per_second': float(total_reward / training_time),
                    'episodes_per_second': float(len(episodes) / training_time) if episodes else 0.0,
                    'training_time': float(training_time)
                }
            else:
                performance['efficiency'] = {'rewards_per_second': 0.0, 'episodes_per_second': 0.0, 'training_time': 0.0}
            
            # Convergence analysis
            if convergence_data:
                performance['convergence'] = {
                    'converged': convergence_data.get('converged', False),
                    'convergence_episode': convergence_data.get('episode', 0),
                    'convergence_threshold': convergence_data.get('threshold', 0.0),
                    'final_performance': convergence_data.get('final_performance', 0.0)
                }
            else:
                performance['convergence'] = {'converged': False, 'convergence_episode': 0, 'convergence_threshold': 0.0, 'final_performance': 0.0}
            
            # Overall performance score
            reward_score = performance['rewards']['mean'] if performance['rewards']['mean'] > 0 else 0.0
            stability_score = performance['rewards']['stability']
            learning_score = performance['learning']['improvement_ratio']
            efficiency_score = performance['efficiency']['rewards_per_second'] / 100.0  # Normalize
            
            performance['overall'] = {
                'score': float((reward_score + stability_score + learning_score + efficiency_score) / 4),
                'grade': self._calculate_grade((reward_score + stability_score + learning_score + efficiency_score) / 4)
            }
            
            return performance
        except Exception as e:
            return {"error": str(e), "performance": "calculation_failed"}
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate performance grade based on score."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C"
        else:
            return "D"
