"""
PPO Trading Agent for ICEBURG Elite Financial AI

This module provides Proximal Policy Optimization (PPO) trading agents
for multi-agent RL in financial markets.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
import gymnasium as gym
from gymnasium import spaces

from .base_agent import BaseAgent, AgentConfig, Action, State

logger = logging.getLogger(__name__)


@dataclass
class PPOTraderConfig(AgentConfig):
    """Configuration for PPO trading agent."""
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False
    normalize_advantage: bool = True
    target_kl: Optional[float] = None
    tensorboard_log: Optional[str] = None
    policy_kwargs: Optional[Dict[str, Any]] = None


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for trading data.
    
    Extracts relevant features from market data and agent state
    for PPO training.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        """Initialize features extractor."""
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimension
        input_dim = observation_space.shape[0]
        
        # Define network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        return self.network(observations)


class PPOTrader(BaseAgent):
    """
    PPO trading agent for financial markets.
    
    Uses Proximal Policy Optimization for learning trading strategies
    in multi-agent environments.
    """
    
    def __init__(self, config: PPOTraderConfig):
        """Initialize PPO trader."""
        super().__init__(config)
        self.agent_type = "ppo"
        
        # PPO configuration
        self.n_steps = config.n_steps
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_range = config.clip_range
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = config.max_grad_norm
        self.learning_rate = config.learning_rate
        self.use_sde = config.use_sde
        self.sde_sample_freq = config.sde_sample_freq
        self.use_sde_at_warmup = config.use_sde_at_warmup
        self.normalize_advantage = config.normalize_advantage
        self.target_kl = config.target_kl
        self.tensorboard_log = config.tensorboard_log
        self.policy_kwargs = config.policy_kwargs or {}
        
        # Initialize PPO model
        self.model = None
        self.env = None
        self.training_step = 0
        
        # Experience buffer
        self.experience_buffer = []
        self.current_episode = []
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
    
    def _initialize_agent(self):
        """Initialize PPO-specific components."""
        # Set up policy kwargs
        if "features_extractor_class" not in self.policy_kwargs:
            self.policy_kwargs["features_extractor_class"] = TradingFeaturesExtractor
            self.policy_kwargs["features_extractor_kwargs"] = {"features_dim": 128}
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            learning_rate=self.learning_rate,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            use_sde_at_warmup=self.use_sde_at_warmup,
            normalize_advantage=self.normalize_advantage,
            target_kl=self.target_kl,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs=self.policy_kwargs,
            verbose=1,
            device=self.device
        )
    
    def set_environment(self, env):
        """Set training environment."""
        self.env = env
        if self.model is None:
            self._initialize_agent()
    
    def act(self, state: State) -> Action:
        """
        Choose action using PPO policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Action to take
        """
        if self.model is None:
            # Fallback to random action if model not initialized
            return self._random_action(state)
        
        try:
            # Convert state to observation
            observation = self._state_to_observation(state)
            
            # Get action from PPO model
            action, _ = self.model.predict(observation, deterministic=False)
            
            # Convert action to trading action
            trading_action = self._action_to_trading_action(action, state)
            
            return trading_action
        
        except Exception as e:
            logger.error(f"Error in PPO act: {e}")
            return self._random_action(state)
    
    def _state_to_observation(self, state: State) -> np.ndarray:
        """Convert state to observation array."""
        observation = []
        
        # Add market data
        for symbol, data in state.market_data.items():
            observation.extend([
                data["price"],
                data["volume"],
                data["bid"],
                data["ask"],
                data["spread"],
                data["trades"]
            ])
        
        # Add agent data
        observation.extend([
            state.agent_data.get("capital", self.capital),
            state.agent_data.get("positions", {}).get("AAPL", 0),
            state.agent_data.get("pnl", self.pnl),
            state.agent_data.get("total_volume", self.total_volume),
            state.agent_data.get("trade_count", self.trade_count)
        ])
        
        return np.array(observation, dtype=np.float32)
    
    def _action_to_trading_action(self, action: np.ndarray, state: State) -> Action:
        """Convert PPO action to trading action."""
        # PPO action is [symbol_idx, side, quantity, price]
        symbol_idx = int(action[0])
        side = "buy" if action[1] > 0.5 else "sell"
        quantity = int(action[2] * 100)  # Scale to reasonable quantity
        price = float(action[3] * 1000)  # Scale to reasonable price
        
        # Get symbol
        symbols = list(state.market_data.keys())
        if symbol_idx >= len(symbols):
            symbol_idx = 0
        symbol = symbols[symbol_idx]
        
        # Ensure reasonable values
        quantity = max(1, min(quantity, 1000))
        price = max(0.01, price)
        
        return Action(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )
    
    def _random_action(self, state: State) -> Action:
        """Generate random action as fallback."""
        symbols = list(state.market_data.keys())
        if not symbols:
            return None
        
        symbol = np.random.choice(symbols)
        side = np.random.choice(["buy", "sell"])
        quantity = np.random.randint(1, 100)
        price = state.market_data[symbol]["price"] * (1 + np.random.uniform(-0.05, 0.05))
        
        return Action(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """
        Learn from experience using PPO.
        
        Args:
            experience: Experience data
        """
        if self.model is None:
            return
        
        try:
            # Add experience to buffer
            self.experience_buffer.append(experience)
            
            # Train if buffer is full
            if len(self.experience_buffer) >= self.n_steps:
                self._train_ppo()
                self.experience_buffer = []
        
        except Exception as e:
            logger.error(f"Error in PPO learn: {e}")
    
    def _train_ppo(self):
        """Train PPO model on experience buffer."""
        if self.model is None or len(self.experience_buffer) < self.batch_size:
            return
        
        try:
            # Convert experiences to training data
            observations = []
            actions = []
            rewards = []
            dones = []
            values = []
            log_probs = []
            
            for exp in self.experience_buffer:
                observations.append(self._state_to_observation(exp["state"]))
                actions.append(self._trading_action_to_action(exp["action"]))
                rewards.append(exp["reward"])
                dones.append(exp["done"])
                values.append(0.0)  # Placeholder
                log_probs.append(0.0)  # Placeholder
            
            # Convert to tensors
            observations = torch.FloatTensor(observations)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.BoolTensor(dones)
            
            # Calculate advantages
            advantages = self._calculate_advantages(rewards, dones)
            
            # Normalize advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Train model
            self.model.learn(
                total_timesteps=len(self.experience_buffer),
                reset_num_timesteps=False
            )
            
            # Update training step
            self.training_step += 1
            
            # Track performance
            self.episode_rewards.append(np.sum(rewards))
            self.episode_lengths.append(len(self.experience_buffer))
            
        except Exception as e:
            logger.error(f"Error training PPO: {e}")
    
    def _trading_action_to_action(self, trading_action: Action) -> np.ndarray:
        """Convert trading action to PPO action format."""
        # Convert trading action to [symbol_idx, side, quantity, price]
        symbol_idx = 0  # Simplified - would need symbol mapping
        side = 1.0 if trading_action.side == "buy" else 0.0
        quantity = trading_action.quantity / 100.0  # Normalize
        price = trading_action.price / 1000.0  # Normalize
        
        return np.array([symbol_idx, side, quantity, price], dtype=np.float32)
    
    def _calculate_advantages(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using GAE."""
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0
            advantage = rewards[t] + self.gamma * advantage
            advantages[t] = advantage
        
        return advantages
    
    def train(self, env, total_timesteps: int = 100000):
        """
        Train PPO agent.
        
        Args:
            env: Training environment
            total_timesteps: Total training timesteps
        """
        if self.model is None:
            self.set_environment(env)
        
        try:
            # Train PPO model
            self.model.learn(total_timesteps=total_timesteps)
            
            logger.info(f"PPO training completed for {total_timesteps} timesteps")
        
        except Exception as e:
            logger.error(f"Error in PPO training: {e}")
    
    def save_model(self, path: str):
        """Save PPO model."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"PPO model saved to {path}")
    
    def load_model(self, path: str):
        """Load PPO model."""
        if self.model is not None:
            self.model = PPO.load(path)
            logger.info(f"PPO model loaded from {path}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        return {
            "training_step": self.training_step,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
            "buffer_size": len(self.experience_buffer)
        }
    
    def reset(self):
        """Reset agent to initial state."""
        super().reset()
        self.experience_buffer = []
        self.current_episode = []
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []


class PPOTraderWithCustomPolicy(PPOTrader):
    """
    PPO trader with custom policy architecture.
    
    Extends base PPO trader with custom policy modifications
    for financial trading applications.
    """
    
    def __init__(self, config: PPOTraderConfig):
        """Initialize PPO trader with custom policy."""
        super().__init__(config)
        
        # Custom policy modifications
        self.policy_kwargs.update({
            "net_arch": [{"pi": [256, 128], "vf": [256, 128]}],
            "activation_fn": nn.ReLU,
            "ortho_init": True,
            "log_std_init": -0.5
        })
    
    def _initialize_agent(self):
        """Initialize with custom policy."""
        # Set up custom policy kwargs
        if "features_extractor_class" not in self.policy_kwargs:
            self.policy_kwargs["features_extractor_class"] = TradingFeaturesExtractor
            self.policy_kwargs["features_extractor_kwargs"] = {"features_dim": 128}
        
        # Initialize PPO model with custom policy
        self.model = PPO(
            "MlpPolicy",
            self.env,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            learning_rate=self.learning_rate,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            use_sde_at_warmup=self.use_sde_at_warmup,
            normalize_advantage=self.normalize_advantage,
            target_kl=self.target_kl,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs=self.policy_kwargs,
            verbose=1,
            device=self.device
        )


# Example usage and testing
if __name__ == "__main__":
    # Test PPO trader
    config = PPOTraderConfig(
        agent_id="ppo_trader",
        agent_type="ppo",
        capital=100000.0,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4
    )
    
    trader = PPOTrader(config)
    
    # Test state and action
    state = State(
        market_data={"AAPL": {"price": 150.0, "volume": 1000, "bid": 149.9, "ask": 150.1, "spread": 0.2, "trades": 10}},
        agent_data={"capital": 100000.0, "positions": {}, "pnl": 0.0, "total_volume": 0.0, "trade_count": 0}
    )
    
    # Test action (will be random since model not trained)
    action = trader.act(state)
    print(f"PPO trader action: {action}")
    
    # Test experience learning
    experience = {
        "state": state,
        "action": action,
        "reward": 1.0,
        "next_state": state,
        "done": False
    }
    
    trader.learn(experience)
    print(f"PPO trader learning info: {trader.get_training_info()}")
    
    # Test custom policy trader
    custom_config = PPOTraderConfig(
        agent_id="custom_ppo_trader",
        agent_type="ppo",
        capital=100000.0
    )
    
    custom_trader = PPOTraderWithCustomPolicy(custom_config)
    custom_action = custom_trader.act(state)
    print(f"Custom PPO trader action: {custom_action}")
