"""
DRL Agent for Self-Healing
Adapts existing RL framework (PPO/SAC) for self-healing rewards
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
from .healing_env import HealingEnv
from .healing_rewards import HealingRewards

logger = logging.getLogger(__name__)


class DRLHealingAgent:
    """DRL agent for self-healing."""
    
    def __init__(self, config: Dict[str, Any] = None, cfg=None):
        """
        Initialize DRL healing agent.
        
        Args:
            config: Configuration for DRL agent
            cfg: ICEBURG configuration
        """
        self.config = config or {}
        self.cfg = cfg
        
        # RL configuration
        self.algorithm = self.config.get("algorithm", "PPO")
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.batch_size = self.config.get("batch_size", 256)
        self.total_timesteps = self.config.get("total_timesteps", 100000)
        
        # Environment
        self.env = None
        self.model = None
        
        # Training state
        self.training_history = []
        self.is_trained = False
        
        # Load existing RL framework
        self._load_rl_framework()
    
    def _load_rl_framework(self):
        """Load existing RL framework (PPO/SAC)."""
        try:
            from ..rl.optimized_training import OptimizedRLTrainer, RLOptimizationConfig
            from ..rl.config import RLConfig
            
            # Create RL config
            rl_config = RLConfig(
                algorithm=self.algorithm,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size
            )
            
            # Create optimization config
            opt_config = RLOptimizationConfig(
                use_vectorized_envs=True,
                n_envs=4,
                use_experience_replay=True,
                replay_buffer_size=10000,
                use_prioritized_replay=True,
                use_curriculum_learning=True,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )
            
            # Create trainer
            self.trainer = OptimizedRLTrainer(opt_config)
            logger.info(f"Loaded RL framework: {self.algorithm}")
        except Exception as e:
            logger.warning(f"Failed to load RL framework: {e}")
            self.trainer = None
    
    def create_environment(
        self,
        alert: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        llm_analysis: Optional[Dict[str, Any]] = None
    ):
        """
        Create healing environment.
        
        Args:
            alert: Bottleneck alert
            metrics: System metrics
            llm_analysis: LLM analysis results
        """
        self.env = HealingEnv(config=self.config)
        
        # Reset environment with initial state
        options = {
            "alert": alert,
            "metrics": metrics or {},
            "llm_analysis": llm_analysis
        }
        self.env.reset(options=options)
        
        logger.info("Created healing environment")
    
    def select_action(
        self,
        state: Optional[np.ndarray] = None,
        llm_analysis: Optional[Dict[str, Any]] = None,
        deterministic: bool = False
    ) -> int:
        """
        Select healing action based on state and LLM analysis.
        
        Args:
            state: Current state vector
            llm_analysis: LLM analysis results
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action index
        """
        if self.model is None:
            # Fallback to random action if model not trained
            logger.warning("Model not trained, using random action")
            return self.env.action_space.sample()
        
        # Use model to predict action
        try:
            if state is None:
                state = self.env.current_state
            
            # Predict action
            action, _ = self.model.predict(state, deterministic=deterministic)
            return int(action)
        except Exception as e:
            logger.error(f"Failed to select action: {e}")
            return self.env.action_space.sample()
    
    def train(
        self,
        n_episodes: int = 100,
        total_timesteps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train DRL agent on historical fix outcomes.
        
        Args:
            n_episodes: Number of training episodes
            total_timesteps: Total training timesteps
            
        Returns:
            Training results
        """
        if self.trainer is None:
            logger.error("RL framework not available, cannot train")
            return {"success": False, "error": "RL framework not available"}
        
        if self.env is None:
            logger.error("Environment not created, cannot train")
            return {"success": False, "error": "Environment not created"}
        
        # Create vectorized environment
        def env_factory():
            return HealingEnv(config=self.config)
        
        self.trainer.create_vectorized_environment(env_factory, n_envs=4)
        
        # Train model
        total_timesteps = total_timesteps or self.total_timesteps
        
        if self.algorithm == "PPO":
            results = self.trainer.train_ppo_agent(total_timesteps=total_timesteps)
        elif self.algorithm == "SAC":
            results = self.trainer.train_sac_agent(total_timesteps=total_timesteps)
        else:
            logger.error(f"Unknown algorithm: {self.algorithm}")
            return {"success": False, "error": f"Unknown algorithm: {self.algorithm}"}
        
        # Store model
        self.model = results.get("model")
        self.is_trained = True
        
        # Evaluate agent
        eval_results = self.trainer.evaluate_agent(n_episodes=50)
        
        # Store training history
        self.training_history.append({
            "algorithm": self.algorithm,
            "total_timesteps": total_timesteps,
            "training_time": results.get("training_time", 0.0),
            "eval_results": eval_results
        })
        
        logger.info(f"DRL agent trained: {eval_results.get('mean_reward', 0.0):.2f}")
        
        return {
            "success": True,
            "training_time": results.get("training_time", 0.0),
            "eval_results": eval_results,
            "training_history": self.training_history
        }
    
    def learn_from_historical_fixes(
        self,
        historical_fixes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Learn from historical fix outcomes stored in UnifiedMemory.
        
        Args:
            historical_fixes: List of historical fix outcomes
            
        Returns:
            Learning results
        """
        if not historical_fixes:
            logger.warning("No historical fixes provided")
            return {"success": False, "error": "No historical fixes provided"}
        
        # Convert historical fixes to training episodes
        episodes = []
        for fix in historical_fixes:
            # Extract state, action, reward from historical fix
            state = fix.get("state")
            action = fix.get("action")
            reward = fix.get("reward", 0.0)
            next_state = fix.get("next_state")
            done = fix.get("done", False)
            
            if state and action is not None:
                episodes.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                })
        
        if not episodes:
            logger.warning("No valid episodes extracted from historical fixes")
            return {"success": False, "error": "No valid episodes extracted"}
        
        # Train on historical episodes
        # In real implementation, would use experience replay
        logger.info(f"Learning from {len(episodes)} historical fixes")
        
        # For now, just log that we would train on historical data
        # In full implementation, would integrate with experience replay buffer
        return {
            "success": True,
            "episodes": len(episodes),
            "message": "Historical fixes loaded for training"
        }
    
    def get_action_recommendations(
        self,
        alert: Any,
        metrics: Dict[str, Any],
        llm_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get action recommendations for a bottleneck alert.
        
        Args:
            alert: Bottleneck alert
            metrics: System metrics
            llm_analysis: LLM analysis results
            
        Returns:
            List of action recommendations with expected rewards
        """
        if self.env is None:
            self.create_environment(alert=alert, metrics=metrics, llm_analysis=llm_analysis)
        
        # Get all available actions
        available_actions = self.env._get_available_actions()
        
        # Evaluate each action
        recommendations = []
        for i, action_name in enumerate(available_actions):
            # Simulate action
            action_result = self.env._execute_action(action_name)
            metrics_after = self.env._simulate_metrics_after_action(
                metrics, action_name, action_result
            )
            
            # Calculate expected reward
            recovery_time = action_result.get("recovery_time", 1.0)
            expected_reward = self.env.reward_calculator.calculate_reward(
                action_name,
                action_result,
                metrics,
                metrics_after,
                recovery_time
            )
            
            recommendations.append({
                "action": action_name,
                "action_index": i,
                "expected_reward": expected_reward,
                "success_probability": action_result.get("success", False),
                "recovery_time": recovery_time
            })
        
        # Sort by expected reward (descending)
        recommendations.sort(key=lambda x: x["expected_reward"], reverse=True)
        
        return recommendations
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status."""
        return {
            "is_trained": self.is_trained,
            "algorithm": self.algorithm,
            "training_history": self.training_history,
            "model_available": self.model is not None
        }

