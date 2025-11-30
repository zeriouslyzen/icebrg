"""
SAC Trading Agent for ICEBURG Elite Financial AI

This module provides Soft Actor-Critic (SAC) trading agents
for multi-agent RL in financial markets.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
import gymnasium as gym
from gymnasium import spaces

from .base_agent import BaseAgent, AgentConfig, Action, State

logger = logging.getLogger(__name__)


@dataclass
class SACTraderConfig(AgentConfig):
    """Configuration for SAC trading agent."""
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 64
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    target_update_interval: int = 1
    target_entropy: str = "auto"
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False
    learning_rate: float = 3e-4
    policy_kwargs: Optional[Dict[str, Any]] = None


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for trading data.
    
    Extracts relevant features from market data and agent state
    for SAC training.
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


class SACTrader(BaseAgent):
    """
    SAC trading agent for financial markets.
    
    Uses Soft Actor-Critic for learning trading strategies
    in multi-agent environments with continuous action spaces.
    """
    
    def __init__(self, config: SACTraderConfig):
        """Initialize SAC trader."""
        super().__init__(config)
        self.agent_type = "sac"
        
        # SAC configuration
        self.buffer_size = config.buffer_size
        self.learning_starts = config.learning_starts
        self.batch_size = config.batch_size
        self.tau = config.tau
        self.gamma = config.gamma
        self.train_freq = config.train_freq
        self.gradient_steps = config.gradient_steps
        self.ent_coef = config.ent_coef
        self.target_update_interval = config.target_update_interval
        self.target_entropy = config.target_entropy
        self.use_sde = config.use_sde
        self.sde_sample_freq = config.sde_sample_freq
        self.use_sde_at_warmup = config.use_sde_at_warmup
        self.learning_rate = config.learning_rate
        self.policy_kwargs = config.policy_kwargs or {}
        
        # Initialize SAC model
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
        self.entropy_losses = []
        self.value_losses = []
        self.policy_losses = []
    
    def _initialize_agent(self):
        """Initialize SAC-specific components."""
        # Set up policy kwargs
        if "features_extractor_class" not in self.policy_kwargs:
            self.policy_kwargs["features_extractor_class"] = TradingFeaturesExtractor
            self.policy_kwargs["features_extractor_kwargs"] = {"features_dim": 128}
        
        # Initialize SAC model
        self.model = SAC(
            "MlpPolicy",
            self.env,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            ent_coef=self.ent_coef,
            target_update_interval=self.target_update_interval,
            target_entropy=self.target_entropy,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            use_sde_at_warmup=self.use_sde_at_warmup,
            learning_rate=self.learning_rate,
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
        Choose action using SAC policy.
        
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
            
            # Get action from SAC model
            action, _ = self.model.predict(observation, deterministic=False)
            
            # Convert action to trading action
            trading_action = self._action_to_trading_action(action, state)
            
            return trading_action
        
        except Exception as e:
            logger.error(f"Error in SAC act: {e}")
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
        """Convert SAC action to trading action."""
        # SAC action is [symbol_idx, side, quantity, price]
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
        Learn from experience using SAC.
        
        Args:
            experience: Experience data
        """
        if self.model is None:
            return
        
        try:
            # Add experience to buffer
            self.experience_buffer.append(experience)
            
            # Train if buffer is full enough
            if len(self.experience_buffer) >= self.learning_starts:
                self._train_sac()
        
        except Exception as e:
            logger.error(f"Error in SAC learn: {e}")
    
    def _train_sac(self):
        """Train SAC model on experience buffer."""
        if self.model is None or len(self.experience_buffer) < self.batch_size:
            return
        
        try:
            # Convert experiences to training data
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            
            for exp in self.experience_buffer:
                observations.append(self._state_to_observation(exp["state"]))
                actions.append(self._trading_action_to_action(exp["action"]))
                rewards.append(exp["reward"])
                next_observations.append(self._state_to_observation(exp["next_state"]))
                dones.append(exp["done"])
            
            # Convert to tensors
            observations = torch.FloatTensor(observations)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_observations = torch.FloatTensor(next_observations)
            dones = torch.BoolTensor(dones)
            
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
            logger.error(f"Error training SAC: {e}")
    
    def _trading_action_to_action(self, trading_action: Action) -> np.ndarray:
        """Convert trading action to SAC action format."""
        # Convert trading action to [symbol_idx, side, quantity, price]
        symbol_idx = 0  # Simplified - would need symbol mapping
        side = 1.0 if trading_action.side == "buy" else 0.0
        quantity = trading_action.quantity / 100.0  # Normalize
        price = trading_action.price / 1000.0  # Normalize
        
        return np.array([symbol_idx, side, quantity, price], dtype=np.float32)
    
    def train(self, env, total_timesteps: int = 100000):
        """
        Train SAC agent.
        
        Args:
            env: Training environment
            total_timesteps: Total training timesteps
        """
        if self.model is None:
            self.set_environment(env)
        
        try:
            # Train SAC model
            self.model.learn(total_timesteps=total_timesteps)
            
            logger.info(f"SAC training completed for {total_timesteps} timesteps")
        
        except Exception as e:
            logger.error(f"Error in SAC training: {e}")
    
    def save_model(self, path: str):
        """Save SAC model."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"SAC model saved to {path}")
    
    def load_model(self, path: str):
        """Load SAC model."""
        if self.model is not None:
            self.model = SAC.load(path)
            logger.info(f"SAC model loaded from {path}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get training information."""
        return {
            "training_step": self.training_step,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
            "entropy_losses": self.entropy_losses,
            "value_losses": self.value_losses,
            "policy_losses": self.policy_losses,
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
        self.entropy_losses = []
        self.value_losses = []
        self.policy_losses = []


class SACTraderWithCustomPolicy(SACTrader):
    """
    SAC trader with custom policy architecture.
    
    Extends base SAC trader with custom policy modifications
    for financial trading applications.
    """
    
    def __init__(self, config: SACTraderConfig):
        """Initialize SAC trader with custom policy."""
        super().__init__(config)
        
        # Custom policy modifications
        self.policy_kwargs.update({
            "net_arch": [{"pi": [256, 128], "qf": [256, 128]}],
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
        
        # Initialize SAC model with custom policy
        self.model = SAC(
            "MlpPolicy",
            self.env,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            ent_coef=self.ent_coef,
            target_update_interval=self.target_update_interval,
            target_entropy=self.target_entropy,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            use_sde_at_warmup=self.use_sde_at_warmup,
            learning_rate=self.learning_rate,
            policy_kwargs=self.policy_kwargs,
            verbose=1,
            device=self.device
        )


class SACTraderWithExploration(SACTrader):
    """
    SAC trader with enhanced exploration.
    
    Extends base SAC trader with additional exploration strategies
    for financial trading applications.
    """
    
    def __init__(self, config: SACTraderConfig):
        """Initialize SAC trader with exploration."""
        super().__init__(config)
        
        # Exploration parameters
        self.exploration_noise = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.curiosity_bonus = 0.1
        
        # Curiosity-driven exploration
        self.curiosity_model = None
        self.curiosity_buffer = []
    
    def act(self, state: State) -> Action:
        """
        Choose action with enhanced exploration.
        
        Args:
            state: Current state observation
            
        Returns:
            Action to take
        """
        if self.model is None:
            return self._random_action(state)
        
        try:
            # Convert state to observation
            observation = self._state_to_observation(state)
            
            # Get action from SAC model
            action, _ = self.model.predict(observation, deterministic=False)
            
            # Add exploration noise
            if self.exploration_noise > self.min_exploration:
                noise = np.random.normal(0, self.exploration_noise, action.shape)
                action = action + noise
                action = np.clip(action, -1, 1)  # Clip to valid range
            
            # Convert action to trading action
            trading_action = self._action_to_trading_action(action, state)
            
            # Add curiosity bonus
            if self.curiosity_model is not None:
                curiosity_reward = self._calculate_curiosity_reward(state, trading_action)
                trading_action.curiosity_bonus = curiosity_reward
            
            return trading_action
        
        except Exception as e:
            logger.error(f"Error in SAC act with exploration: {e}")
            return self._random_action(state)
    
    def _calculate_curiosity_reward(self, state: State, action: Action) -> float:
        """Calculate curiosity reward for exploration."""
        # Simplified curiosity calculation
        # In practice, you'd use a curiosity model
        return np.random.uniform(0, self.curiosity_bonus)
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """
        Learn from experience with curiosity.
        
        Args:
            experience: Experience data
        """
        if self.model is None:
            return
        
        try:
            # Add experience to buffer
            self.experience_buffer.append(experience)
            
            # Add to curiosity buffer
            if hasattr(experience["action"], "curiosity_bonus"):
                self.curiosity_buffer.append({
                    "state": experience["state"],
                    "action": experience["action"],
                    "curiosity_bonus": experience["action"].curiosity_bonus
                })
            
            # Train if buffer is full enough
            if len(self.experience_buffer) >= self.learning_starts:
                self._train_sac()
                
                # Decay exploration noise
                self.exploration_noise = max(
                    self.min_exploration,
                    self.exploration_noise * self.exploration_decay
                )
        
        except Exception as e:
            logger.error(f"Error in SAC learn with exploration: {e}")
    
    def get_exploration_info(self) -> Dict[str, Any]:
        """Get exploration information."""
        return {
            "exploration_noise": self.exploration_noise,
            "curiosity_buffer_size": len(self.curiosity_buffer),
            "exploration_decay": self.exploration_decay,
            "min_exploration": self.min_exploration
        }


# Example usage and testing
if __name__ == "__main__":
    # Test SAC trader
    config = SACTraderConfig(
        agent_id="sac_trader",
        agent_type="sac",
        capital=100000.0,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        learning_rate=3e-4
    )
    
    trader = SACTrader(config)
    
    # Test state and action
    state = State(
        market_data={"AAPL": {"price": 150.0, "volume": 1000, "bid": 149.9, "ask": 150.1, "spread": 0.2, "trades": 10}},
        agent_data={"capital": 100000.0, "positions": {}, "pnl": 0.0, "total_volume": 0.0, "trade_count": 0}
    )
    
    # Test action (will be random since model not trained)
    action = trader.act(state)
    print(f"SAC trader action: {action}")
    
    # Test experience learning
    experience = {
        "state": state,
        "action": action,
        "reward": 1.0,
        "next_state": state,
        "done": False
    }
    
    trader.learn(experience)
    print(f"SAC trader learning info: {trader.get_training_info()}")
    
    # Test custom policy trader
    custom_config = SACTraderConfig(
        agent_id="custom_sac_trader",
        agent_type="sac",
        capital=100000.0
    )
    
    custom_trader = SACTraderWithCustomPolicy(custom_config)
    custom_action = custom_trader.act(state)
    print(f"Custom SAC trader action: {custom_action}")
    
    # Test exploration trader
    exploration_config = SACTraderConfig(
        agent_id="exploration_sac_trader",
        agent_type="sac",
        capital=100000.0
    )
    
    exploration_trader = SACTraderWithExploration(exploration_config)
    exploration_action = exploration_trader.act(state)
    print(f"Exploration SAC trader action: {exploration_action}")
    print(f"Exploration info: {exploration_trader.get_exploration_info()}")
