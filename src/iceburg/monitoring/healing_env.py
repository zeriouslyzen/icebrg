"""
RL Environment for Self-Healing
State space: System metrics, bottleneck type, severity, LLM analysis results
Action space: Available healing actions (cache_clear, gc_collect, optimize_network, etc.)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from gymnasium import Env, spaces
from .bottleneck_detector import BottleneckType, Severity

logger = logging.getLogger(__name__)


class HealingEnv(Env):
    """RL environment for self-healing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize healing environment.
        
        Args:
            config: Configuration for environment
        """
        super().__init__()
        
        self.config = config or {}
        
        # Action space: Available healing actions
        self.action_space = spaces.Discrete(len(self._get_available_actions()))
        
        # State space: System metrics + bottleneck info + LLM analysis
        # State vector: [latency, memory, cpu, cache, network, disk, connections, error_rate,
        #                bottleneck_type_onehot(6), severity_onehot(4), llm_confidence]
        state_dim = 8 + 6 + 4 + 1  # 19 dimensions
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,  # Max values for metrics
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Current state
        self.current_state = None
        self.current_alert = None
        self.current_metrics = None
        self.current_llm_analysis = None
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.max_episode_steps = self.config.get("max_episode_steps", 10)
        
        # Reward calculator
        from .healing_rewards import HealingRewards
        self.reward_calculator = HealingRewards(config)
    
    def _get_available_actions(self) -> List[str]:
        """Get list of available healing actions."""
        return [
            "increase_memory",
            "optimize_memory",
            "restart_services",
            "warm_cache",
            "increase_cache_size",
            "optimize_cache_policy",
            "increase_bandwidth",
            "optimize_network",
            "load_balance",
            "cleanup_disk",
            "increase_storage",
            "optimize_io"
        ]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment.
        
        Args:
            seed: Random seed
            options: Reset options (alert, metrics, llm_analysis)
            
        Returns:
            Initial observation and info
        """
        super().reset(seed=seed)
        
        # Get initial state from options
        self.current_alert = options.get("alert") if options else None
        self.current_metrics = options.get("metrics") if options else {}
        self.current_llm_analysis = options.get("llm_analysis") if options else None
        
        # Build initial state vector
        self.current_state = self._build_state_vector(
            self.current_metrics,
            self.current_alert,
            self.current_llm_analysis
        )
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        info = {
            "alert": self.current_alert,
            "metrics": self.current_metrics,
            "llm_analysis": self.current_llm_analysis
        }
        
        return self.current_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment.
        
        Args:
            action: Action index
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Get action name
        available_actions = self._get_available_actions()
        if action >= len(available_actions):
            action = 0  # Default to first action
        
        action_name = available_actions[action]
        
        # Store metrics before action
        metrics_before = self.current_metrics.copy() if self.current_metrics else {}
        
        # Execute action (simulated - in real implementation, would call AutoHealer)
        action_result = self._execute_action(action_name)
        
        # Update metrics after action (simulated - in real implementation, would measure)
        metrics_after = self._simulate_metrics_after_action(
            metrics_before, action_name, action_result
        )
        self.current_metrics = metrics_after
        
        # Calculate reward
        recovery_time = action_result.get("recovery_time", 1.0)
        reward = self.reward_calculator.calculate_reward(
            action_name,
            action_result,
            metrics_before,
            metrics_after,
            recovery_time
        )
        
        # Update episode tracking
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Build new state
        self.current_state = self._build_state_vector(
            metrics_after,
            self.current_alert,
            self.current_llm_analysis
        )
        
        # Check termination conditions
        terminated = action_result.get("success", False)  # Terminated if action succeeded
        truncated = self.episode_steps >= self.max_episode_steps  # Truncated if max steps reached
        
        info = {
            "action": action_name,
            "action_result": action_result,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "reward": reward,
            "episode_reward": self.episode_reward,
            "episode_steps": self.episode_steps
        }
        
        return self.current_state, reward, terminated, truncated, info
    
    def _build_state_vector(
        self,
        metrics: Dict[str, Any],
        alert: Optional[Any],
        llm_analysis: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Build state vector from metrics, alert, and LLM analysis."""
        state = []
        
        # System metrics (8 dimensions)
        state.append(metrics.get("latency_p95", 0.0))
        state.append(metrics.get("memory_usage", 0.0))
        state.append(metrics.get("cpu_usage", 0.0))
        state.append(metrics.get("cache_hit_rate", 0.0))
        state.append(metrics.get("network_throughput", 0.0))
        state.append(metrics.get("disk_usage", 0.0))
        state.append(float(metrics.get("active_connections", 0)))
        state.append(metrics.get("error_rate", 0.0))
        
        # Bottleneck type one-hot encoding (6 dimensions)
        bottleneck_type_onehot = [0.0] * 6
        if alert:
            bottleneck_type = alert.bottleneck_type
            type_index = list(BottleneckType).index(bottleneck_type)
            if 0 <= type_index < 6:
                bottleneck_type_onehot[type_index] = 1.0
        state.extend(bottleneck_type_onehot)
        
        # Severity one-hot encoding (4 dimensions)
        severity_onehot = [0.0] * 4
        if alert:
            severity = alert.severity
            severity_index = list(Severity).index(severity)
            if 0 <= severity_index < 4:
                severity_onehot[severity_index] = 1.0
        state.extend(severity_onehot)
        
        # LLM analysis confidence (1 dimension)
        llm_confidence = 0.0
        if llm_analysis:
            llm_confidence = llm_analysis.get("confidence", 0.0)
        state.append(llm_confidence)
        
        return np.array(state, dtype=np.float32)
    
    def _execute_action(self, action_name: str) -> Dict[str, Any]:
        """Execute healing action (simulated)."""
        # In real implementation, would call AutoHealer.apply_healing()
        # For now, simulate action execution
        
        # Simulate success based on action type
        success_probability = {
            "increase_memory": 0.9,
            "optimize_memory": 0.8,
            "restart_services": 0.7,
            "warm_cache": 0.6,
            "increase_cache_size": 0.5,
            "optimize_cache_policy": 0.7,
            "increase_bandwidth": 0.4,
            "optimize_network": 0.8,
            "load_balance": 0.6,
            "cleanup_disk": 0.9,
            "increase_storage": 0.3,
            "optimize_io": 0.7
        }
        
        import random
        success = random.random() < success_probability.get(action_name, 0.5)
        
        return {
            "success": success,
            "partial_success": not success and random.random() < 0.3,
            "recovery_time": random.uniform(0.5, 5.0) if success else random.uniform(5.0, 10.0),
            "action": action_name
        }
    
    def _simulate_metrics_after_action(
        self,
        metrics_before: Dict[str, Any],
        action_name: str,
        action_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate metrics after action (simulated)."""
        metrics_after = metrics_before.copy()
        
        if action_result.get("success", False):
            # Simulate improvement
            if "memory" in action_name:
                metrics_after["memory_usage"] = max(0.0, metrics_after.get("memory_usage", 0.0) - 10.0)
            if "cache" in action_name:
                metrics_after["cache_hit_rate"] = min(100.0, metrics_after.get("cache_hit_rate", 0.0) + 5.0)
            if "network" in action_name or "bandwidth" in action_name:
                metrics_after["network_throughput"] = metrics_after.get("network_throughput", 0.0) + 1.0
            if "latency" in action_name or "optimize" in action_name:
                metrics_after["latency_p95"] = max(0.0, metrics_after.get("latency_p95", 0.0) - 50.0)
        
        return metrics_after

