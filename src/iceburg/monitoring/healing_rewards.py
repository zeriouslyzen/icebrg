"""
Reward Functions for Self-Healing DRL Agent
Reward function: Fix success rate, recovery time, system stability, resource efficiency
"""

import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HealingRewards:
    """Reward functions for self-healing actions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize reward functions.
        
        Args:
            config: Configuration for reward calculation
        """
        self.config = config or {}
        
        # Reward weights
        self.weights = {
            "fix_success": self.config.get("fix_success_weight", 0.4),
            "recovery_time": self.config.get("recovery_time_weight", 0.3),
            "stability": self.config.get("stability_weight", 0.2),
            "resource_efficiency": self.config.get("resource_efficiency_weight", 0.1)
        }
        
        # Normalization factors
        self.max_recovery_time = self.config.get("max_recovery_time", 60.0)  # seconds
        self.max_resource_usage = self.config.get("max_resource_usage", 100.0)  # percentage
    
    def calculate_reward(
        self,
        action: str,
        action_result: Dict[str, Any],
        system_metrics_before: Dict[str, Any],
        system_metrics_after: Dict[str, Any],
        recovery_time: float
    ) -> float:
        """
        Calculate reward for a healing action.
        
        Args:
            action: Healing action taken
            action_result: Result of the action (success, error, etc.)
            system_metrics_before: System metrics before action
            system_metrics_after: System metrics after action
            recovery_time: Time taken to recover (seconds)
            
        Returns:
            Reward value (higher is better)
        """
        # 1. Fix success rate (0.0 to 1.0)
        fix_success = self._calculate_fix_success(action_result)
        
        # 2. Recovery time (0.0 to 1.0, faster is better)
        recovery_time_reward = self._calculate_recovery_time_reward(recovery_time)
        
        # 3. System stability (0.0 to 1.0)
        stability = self._calculate_stability(system_metrics_before, system_metrics_after)
        
        # 4. Resource efficiency (0.0 to 1.0)
        resource_efficiency = self._calculate_resource_efficiency(
            system_metrics_before, system_metrics_after
        )
        
        # Weighted sum
        reward = (
            self.weights["fix_success"] * fix_success +
            self.weights["recovery_time"] * recovery_time_reward +
            self.weights["stability"] * stability +
            self.weights["resource_efficiency"] * resource_efficiency
        )
        
        logger.debug(
            f"Reward for action {action}: "
            f"success={fix_success:.2f}, recovery={recovery_time_reward:.2f}, "
            f"stability={stability:.2f}, efficiency={resource_efficiency:.2f}, "
            f"total={reward:.2f}"
        )
        
        return reward
    
    def _calculate_fix_success(self, action_result: Dict[str, Any]) -> float:
        """Calculate fix success rate (0.0 to 1.0)."""
        if action_result.get("success", False):
            return 1.0
        elif action_result.get("partial_success", False):
            return 0.5
        else:
            return 0.0
    
    def _calculate_recovery_time_reward(self, recovery_time: float) -> float:
        """Calculate recovery time reward (0.0 to 1.0, faster is better)."""
        if recovery_time <= 0:
            return 1.0
        
        # Normalize recovery time (0 to max_recovery_time -> 1.0 to 0.0)
        normalized_time = min(recovery_time / self.max_recovery_time, 1.0)
        return 1.0 - normalized_time
    
    def _calculate_stability(
        self,
        metrics_before: Dict[str, Any],
        metrics_after: Dict[str, Any]
    ) -> float:
        """Calculate system stability (0.0 to 1.0)."""
        # Stability is measured by how much metrics improved
        improvements = []
        
        # Check latency improvement
        latency_before = metrics_before.get("latency_p95", 0.0)
        latency_after = metrics_after.get("latency_p95", 0.0)
        if latency_before > 0:
            latency_improvement = (latency_before - latency_after) / latency_before
            improvements.append(max(0.0, min(1.0, latency_improvement)))
        
        # Check memory improvement
        memory_before = metrics_before.get("memory_usage", 0.0)
        memory_after = metrics_after.get("memory_usage", 0.0)
        if memory_before > 0:
            memory_improvement = (memory_before - memory_after) / memory_before
            improvements.append(max(0.0, min(1.0, memory_improvement)))
        
        # Check error rate improvement
        error_before = metrics_before.get("error_rate", 0.0)
        error_after = metrics_after.get("error_rate", 0.0)
        if error_before > 0:
            error_improvement = (error_before - error_after) / error_before
            improvements.append(max(0.0, min(1.0, error_improvement)))
        
        # Average improvements
        if improvements:
            return sum(improvements) / len(improvements)
        else:
            return 0.5  # Neutral stability
    
    def _calculate_resource_efficiency(
        self,
        metrics_before: Dict[str, Any],
        metrics_after: Dict[str, Any]
    ) -> float:
        """Calculate resource efficiency (0.0 to 1.0)."""
        # Resource efficiency is measured by resource usage reduction
        # Lower resource usage is better
        
        # Check CPU usage
        cpu_before = metrics_before.get("cpu_usage", 0.0)
        cpu_after = metrics_after.get("cpu_usage", 0.0)
        cpu_efficiency = 1.0 - (cpu_after / self.max_resource_usage) if cpu_after > 0 else 0.5
        
        # Check memory usage
        memory_before = metrics_before.get("memory_usage", 0.0)
        memory_after = metrics_after.get("memory_usage", 0.0)
        memory_efficiency = 1.0 - (memory_after / self.max_resource_usage) if memory_after > 0 else 0.5
        
        # Average efficiency
        return (cpu_efficiency + memory_efficiency) / 2.0
    
    def get_reward_breakdown(
        self,
        action: str,
        action_result: Dict[str, Any],
        system_metrics_before: Dict[str, Any],
        system_metrics_after: Dict[str, Any],
        recovery_time: float
    ) -> Dict[str, Any]:
        """
        Get detailed reward breakdown.
        
        Args:
            action: Healing action taken
            action_result: Result of the action
            system_metrics_before: System metrics before action
            system_metrics_after: System metrics after action
            recovery_time: Time taken to recover
            
        Returns:
            Detailed reward breakdown
        """
        fix_success = self._calculate_fix_success(action_result)
        recovery_time_reward = self._calculate_recovery_time_reward(recovery_time)
        stability = self._calculate_stability(system_metrics_before, system_metrics_after)
        resource_efficiency = self._calculate_resource_efficiency(
            system_metrics_before, system_metrics_after
        )
        
        total_reward = self.calculate_reward(
            action, action_result, system_metrics_before,
            system_metrics_after, recovery_time
        )
        
        return {
            "total_reward": total_reward,
            "components": {
                "fix_success": {
                    "value": fix_success,
                    "weight": self.weights["fix_success"],
                    "contribution": self.weights["fix_success"] * fix_success
                },
                "recovery_time": {
                    "value": recovery_time_reward,
                    "weight": self.weights["recovery_time"],
                    "contribution": self.weights["recovery_time"] * recovery_time_reward
                },
                "stability": {
                    "value": stability,
                    "weight": self.weights["stability"],
                    "contribution": self.weights["stability"] * stability
                },
                "resource_efficiency": {
                    "value": resource_efficiency,
                    "weight": self.weights["resource_efficiency"],
                    "contribution": self.weights["resource_efficiency"] * resource_efficiency
                }
            }
        }

