"""
Training Loop for DRL Healing Agent
Training loop for DRL healing agent with historical fix outcomes
"""

import logging
from typing import Dict, Any, Optional, List
from .drl_healing_agent import DRLHealingAgent
from .healing_env import HealingEnv

logger = logging.getLogger(__name__)


class HealingTraining:
    """Training loop for DRL healing agent."""
    
    def __init__(self, config: Dict[str, Any] = None, cfg=None):
        """
        Initialize training loop.
        
        Args:
            config: Configuration for training
            cfg: ICEBURG configuration
        """
        self.config = config or {}
        self.cfg = cfg
        
        # Training configuration
        self.n_episodes = self.config.get("n_episodes", 100)
        self.total_timesteps = self.config.get("total_timesteps", 100000)
        self.eval_freq = self.config.get("eval_freq", 10000)
        self.save_freq = self.config.get("save_freq", 50000)
        
        # Create DRL agent
        self.agent = DRLHealingAgent(config=self.config, cfg=cfg)
        
        # Training state
        self.training_history = []
        self.best_reward = -float('inf')
        self.best_model = None
    
    def train_on_historical_fixes(
        self,
        historical_fixes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Train agent on historical fix outcomes.
        
        Args:
            historical_fixes: List of historical fix outcomes
            
        Returns:
            Training results
        """
        logger.info(f"Training on {len(historical_fixes)} historical fixes")
        
        # Learn from historical fixes
        learn_results = self.agent.learn_from_historical_fixes(historical_fixes)
        
        if not learn_results.get("success", False):
            logger.warning("Failed to learn from historical fixes")
            return learn_results
        
        # Train agent
        train_results = self.agent.train(
            n_episodes=self.n_episodes,
            total_timesteps=self.total_timesteps
        )
        
        if not train_results.get("success", False):
            logger.error("Failed to train agent")
            return train_results
        
        # Store training history
        self.training_history.append({
            "historical_fixes": len(historical_fixes),
            "training_results": train_results
        })
        
        # Update best model
        eval_results = train_results.get("eval_results", {})
        mean_reward = eval_results.get("mean_reward", -float('inf'))
        
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.best_model = self.agent.model
            logger.info(f"New best model: {mean_reward:.2f}")
        
        return {
            "success": True,
            "training_results": train_results,
            "best_reward": self.best_reward,
            "training_history": self.training_history
        }
    
    def train_on_simulated_episodes(
        self,
        n_episodes: int = 100
    ) -> Dict[str, Any]:
        """
        Train agent on simulated episodes.
        
        Args:
            n_episodes: Number of training episodes
            
        Returns:
            Training results
        """
        logger.info(f"Training on {n_episodes} simulated episodes")
        
        # Train agent
        train_results = self.agent.train(
            n_episodes=n_episodes,
            total_timesteps=self.total_timesteps
        )
        
        if not train_results.get("success", False):
            logger.error("Failed to train agent")
            return train_results
        
        # Store training history
        self.training_history.append({
            "n_episodes": n_episodes,
            "training_results": train_results
        })
        
        # Update best model
        eval_results = train_results.get("eval_results", {})
        mean_reward = eval_results.get("mean_reward", -float('inf'))
        
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.best_model = self.agent.model
            logger.info(f"New best model: {mean_reward:.2f}")
        
        return {
            "success": True,
            "training_results": train_results,
            "best_reward": self.best_reward,
            "training_history": self.training_history
        }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status."""
        return {
            "agent_status": self.agent.get_training_status(),
            "best_reward": self.best_reward,
            "training_history": self.training_history,
            "best_model_available": self.best_model is not None
        }

