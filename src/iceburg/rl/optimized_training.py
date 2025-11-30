"""
Optimized Reinforcement Learning Training for ICEBURG Elite Financial AI

This module provides optimized RL training capabilities with vectorized environments,
experience replay optimization, and distributed training for financial applications.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import threading
from collections import deque
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

logger = logging.getLogger(__name__)


@dataclass
class RLOptimizationConfig:
    """Configuration for RL optimization."""
    use_vectorized_envs: bool = True
    n_envs: int = 8
    use_experience_replay: bool = True
    replay_buffer_size: int = 100000
    use_prioritized_replay: bool = True
    use_curriculum_learning: bool = True
    use_distributed_training: bool = False
    n_workers: int = 4
    use_gpu: bool = True
    batch_size: int = 256
    learning_rate: float = 3e-4
    use_hyperparameter_optimization: bool = True


class OptimizedExperienceReplay:
    """
    Optimized experience replay buffer with prioritized sampling.
    
    Provides efficient experience storage and sampling for RL training.
    """
    
    def __init__(self, capacity: int, use_prioritized: bool = True):
        """
        Initialize experience replay buffer.
        
        Args:
            capacity: Buffer capacity
            use_prioritized: Whether to use prioritized sampling
        """
        self.capacity = capacity
        self.use_prioritized = use_prioritized
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        self.alpha = 0.6  # Prioritization exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
        self.lock = threading.Lock()
    
    def add(self, experience: Dict[str, Any], priority: float = None):
        """
        Add experience to buffer.
        
        Args:
            experience: Experience dictionary
            priority: Experience priority
        """
        with self.lock:
            if priority is None:
                priority = self.max_priority
            
            self.buffer.append(experience)
            self.priorities.append(priority)
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Experiences, indices, and importance weights
        """
        with self.lock:
            if not self.use_prioritized:
                # Uniform sampling
                indices = random.sample(range(len(self.buffer)), batch_size)
                experiences = [self.buffer[i] for i in indices]
                weights = np.ones(batch_size)
                return experiences, indices, weights
            
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            experiences = [self.buffer[i] for i in indices]
            
            # Calculate importance weights
            weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()
            
            # Update beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Experience indices
            priorities: New priorities
        """
        with self.lock:
            for idx, priority in zip(indices, priorities):
                if idx < len(self.priorities):
                    self.priorities[idx] = priority
                    self.max_priority = max(self.max_priority, priority)
    
    def size(self) -> int:
        """Get buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
            self.priorities.clear()


class VectorizedEnvironment:
    """
    Vectorized environment for parallel RL training.
    
    Provides efficient parallel environment execution for faster training.
    """
    
    def __init__(self, env_factory, n_envs: int = 8, use_subprocess: bool = True):
        """
        Initialize vectorized environment.
        
        Args:
            env_factory: Environment factory function
            n_envs: Number of environments
            use_subprocess: Whether to use subprocess environments
        """
        self.env_factory = env_factory
        self.n_envs = n_envs
        self.use_subprocess = use_subprocess
        self.vec_env = None
        self._create_vectorized_env()
    
    def _create_vectorized_env(self):
        """Create vectorized environment."""
        if self.use_subprocess:
            # Use subprocess environments for true parallelism
            self.vec_env = SubprocVecEnv([
                lambda: self.env_factory() for _ in range(self.n_envs)
            ])
        else:
            # Use dummy vectorized environment
            self.vec_env = DummyVecEnv([
                lambda: self.env_factory() for _ in range(self.n_envs)
            ])
    
    def reset(self):
        """Reset all environments."""
        return self.vec_env.reset()
    
    def step(self, actions):
        """Step all environments."""
        return self.vec_env.step(actions)
    
    def close(self):
        """Close all environments."""
        self.vec_env.close()
    
    def get_attr(self, attr_name: str):
        """Get attribute from environments."""
        return self.vec_env.get_attr(attr_name)
    
    def set_attr(self, attr_name: str, value):
        """Set attribute in environments."""
        self.vec_env.set_attr(attr_name, value)


class CurriculumLearning:
    """
    Curriculum learning for RL training.
    
    Gradually increases task difficulty during training for better convergence.
    """
    
    def __init__(self, initial_difficulty: float = 0.1, max_difficulty: float = 1.0):
        """
        Initialize curriculum learning.
        
        Args:
            initial_difficulty: Initial difficulty level
            max_difficulty: Maximum difficulty level
        """
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = initial_difficulty
        self.difficulty_increment = 0.01
        self.performance_threshold = 0.8
        self.performance_history = deque(maxlen=100)
    
    def update_difficulty(self, performance: float):
        """
        Update difficulty based on performance.
        
        Args:
            performance: Current performance metric
        """
        self.performance_history.append(performance)
        
        # Increase difficulty if performance is good
        if len(self.performance_history) >= 10:
            avg_performance = np.mean(list(self.performance_history)[-10:])
            if avg_performance > self.performance_threshold:
                self.current_difficulty = min(
                    self.max_difficulty,
                    self.current_difficulty + self.difficulty_increment
                )
                logger.info(f"Increased difficulty to {self.current_difficulty:.3f}")
    
    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty
    
    def reset(self):
        """Reset curriculum learning."""
        self.current_difficulty = self.initial_difficulty
        self.performance_history.clear()


class OptimizedRLTrainer:
    """
    Optimized RL trainer with advanced training techniques.
    
    Provides optimized training with vectorized environments, experience replay,
    and curriculum learning for financial applications.
    """
    
    def __init__(self, config: RLOptimizationConfig):
        """
        Initialize optimized RL trainer.
        
        Args:
            config: RL optimization configuration
        """
        self.config = config
        self.experience_replay = OptimizedExperienceReplay(
            config.replay_buffer_size,
            config.use_prioritized_replay
        )
        self.curriculum_learning = CurriculumLearning()
        self.vectorized_env = None
        self.model = None
        self.training_metrics = {}
    
    def create_vectorized_environment(self, env_factory, n_envs: int = None):
        """
        Create vectorized environment.
        
        Args:
            env_factory: Environment factory function
            n_envs: Number of environments
        """
        if n_envs is None:
            n_envs = self.config.n_envs
        
        self.vectorized_env = VectorizedEnvironment(
            env_factory,
            n_envs,
            use_subprocess=True
        )
        
        logger.info(f"Created vectorized environment with {n_envs} environments")
    
    def train_ppo_agent(self, total_timesteps: int, learning_rate: float = None) -> Dict[str, Any]:
        """
        Train PPO agent with optimizations.
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vectorized_env.vec_env,
            learning_rate=learning_rate,
            batch_size=self.config.batch_size,
            verbose=1
        )
        
        # Train model
        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps)
        training_time = time.time() - start_time
        
        # Collect training metrics
        results = {
            "training_time": training_time,
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "model": self.model
        }
        
        logger.info(f"PPO training completed in {training_time:.2f}s")
        
        return results
    
    def train_sac_agent(self, total_timesteps: int, learning_rate: float = None) -> Dict[str, Any]:
        """
        Train SAC agent with optimizations.
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        # Create SAC model
        self.model = SAC(
            "MlpPolicy",
            self.vectorized_env.vec_env,
            learning_rate=learning_rate,
            batch_size=self.config.batch_size,
            verbose=1
        )
        
        # Train model
        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps)
        training_time = time.time() - start_time
        
        # Collect training metrics
        results = {
            "training_time": training_time,
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "model": self.model
        }
        
        logger.info(f"SAC training completed in {training_time:.2f}s")
        
        return results
    
    def evaluate_agent(self, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate trained agent.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        # Evaluate agent
        start_time = time.time()
        obs = self.vectorized_env.reset()
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.vectorized_env.step(action)
                
                episode_reward += reward.sum()
                episode_length += 1
                
                if done.any():
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        evaluation_time = time.time() - start_time
        
        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "evaluation_time": evaluation_time,
            "n_episodes": n_episodes
        }
        
        logger.info(f"Agent evaluation completed: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        
        return results
    
    def hyperparameter_optimization(self, param_space: Dict[str, List], 
                                   n_trials: int = 20) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.
        
        Args:
            param_space: Parameter space for optimization
            n_trials: Number of optimization trials
            
        Returns:
            Optimization results
        """
        if not self.config.use_hyperparameter_optimization:
            logger.warning("Hyperparameter optimization disabled")
            return {}
        
        best_params = None
        best_performance = -float('inf')
        optimization_results = []
        
        for trial in range(n_trials):
            # Sample parameters
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = random.choice(param_values)
            
            # Train model with sampled parameters
            try:
                if params.get('algorithm') == 'PPO':
                    results = self.train_ppo_agent(
                        total_timesteps=10000,
                        learning_rate=params.get('learning_rate', 3e-4)
                    )
                else:
                    results = self.train_sac_agent(
                        total_timesteps=10000,
                        learning_rate=params.get('learning_rate', 3e-4)
                    )
                
                # Evaluate performance
                eval_results = self.evaluate_agent(n_episodes=50)
                performance = eval_results['mean_reward']
                
                optimization_results.append({
                    'params': params,
                    'performance': performance
                })
                
                # Update best parameters
                if performance > best_performance:
                    best_performance = performance
                    best_params = params
                
                logger.info(f"Trial {trial+1}/{n_trials}: Performance = {performance:.2f}")
                
            except Exception as e:
                logger.warning(f"Trial {trial+1} failed: {e}")
                continue
        
        results = {
            "best_params": best_params,
            "best_performance": best_performance,
            "optimization_results": optimization_results,
            "n_trials": n_trials
        }
        
        logger.info(f"Hyperparameter optimization completed. Best performance: {best_performance:.2f}")
        
        return results


class RLPerformanceMonitor:
    """
    Performance monitoring for RL training.
    
    Tracks training metrics, convergence, and performance indicators.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.training_history = []
        self.convergence_metrics = []
    
    def start_training(self, algorithm: str, total_timesteps: int):
        """
        Start monitoring training session.
        
        Args:
            algorithm: RL algorithm name
            total_timesteps: Total training timesteps
        """
        self.metrics = {
            "algorithm": algorithm,
            "total_timesteps": total_timesteps,
            "start_time": time.time(),
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": []
        }
    
    def record_episode(self, reward: float, length: int):
        """
        Record episode metrics.
        
        Args:
            reward: Episode reward
            length: Episode length
        """
        self.metrics["episode_rewards"].append(reward)
        self.metrics["episode_lengths"].append(length)
    
    def record_loss(self, loss: float):
        """
        Record training loss.
        
        Args:
            loss: Training loss
        """
        self.metrics["losses"].append(loss)
    
    def end_training(self) -> Dict[str, Any]:
        """
        End training session and return metrics.
        
        Returns:
            Training metrics
        """
        self.metrics["end_time"] = time.time()
        self.metrics["training_time"] = self.metrics["end_time"] - self.metrics["start_time"]
        
        # Calculate summary statistics
        if self.metrics["episode_rewards"]:
            self.metrics["mean_reward"] = np.mean(self.metrics["episode_rewards"])
            self.metrics["std_reward"] = np.std(self.metrics["episode_rewards"])
            self.metrics["max_reward"] = np.max(self.metrics["episode_rewards"])
            self.metrics["min_reward"] = np.min(self.metrics["episode_rewards"])
        
        if self.metrics["episode_lengths"]:
            self.metrics["mean_length"] = np.mean(self.metrics["episode_lengths"])
            self.metrics["std_length"] = np.std(self.metrics["episode_lengths"])
        
        if self.metrics["losses"]:
            self.metrics["mean_loss"] = np.mean(self.metrics["losses"])
            self.metrics["std_loss"] = np.std(self.metrics["losses"])
        
        # Store in training history
        self.training_history.append(self.metrics.copy())
        
        return self.metrics
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """
        Analyze training convergence.
        
        Returns:
            Convergence analysis
        """
        if not self.metrics["episode_rewards"]:
            return {}
        
        rewards = np.array(self.metrics["episode_rewards"])
        
        # Calculate convergence metrics
        window_size = min(100, len(rewards) // 4)
        if window_size > 0:
            recent_rewards = rewards[-window_size:]
            early_rewards = rewards[:window_size]
            
            convergence_ratio = np.mean(recent_rewards) / np.mean(early_rewards)
            stability = np.std(recent_rewards) / np.mean(recent_rewards)
            
            return {
                "convergence_ratio": convergence_ratio,
                "stability": stability,
                "improvement": np.mean(recent_rewards) - np.mean(early_rewards),
                "window_size": window_size
            }
        
        return {}


# Example usage and testing
if __name__ == "__main__":
    # Test optimized RL training
    config = RLOptimizationConfig(
        use_vectorized_envs=True,
        n_envs=4,
        use_experience_replay=True,
        replay_buffer_size=10000,
        use_prioritized_replay=True,
        use_curriculum_learning=True,
        use_distributed_training=False,
        n_workers=2,
        use_gpu=False,
        batch_size=128,
        learning_rate=3e-4,
        use_hyperparameter_optimization=True
    )
    
    # Create trainer
    trainer = OptimizedRLTrainer(config)
    
    # Create vectorized environment
    def create_env():
        return gym.make("CartPole-v1")
    
    trainer.create_vectorized_environment(create_env, n_envs=4)
    
    # Test training
    results = trainer.train_ppo_agent(total_timesteps=10000)
    print(f"Training results: {results}")
    
    # Test evaluation
    eval_results = trainer.evaluate_agent(n_episodes=50)
    print(f"Evaluation results: {eval_results}")
    
    # Test hyperparameter optimization
    param_space = {
        "algorithm": ["PPO", "SAC"],
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "batch_size": [64, 128, 256]
    }
    
    opt_results = trainer.hyperparameter_optimization(param_space, n_trials=5)
    print(f"Optimization results: {opt_results}")
    
    # Test performance monitoring
    monitor = RLPerformanceMonitor()
    monitor.start_training("PPO", 10000)
    
    # Simulate training
    for episode in range(100):
        reward = np.random.randn() * 10 + 100
        length = np.random.randint(50, 200)
        monitor.record_episode(reward, length)
        monitor.record_loss(np.random.randn() * 0.1 + 0.5)
    
    training_metrics = monitor.end_training()
    print(f"Training metrics: {training_metrics}")
    
    convergence_analysis = monitor.get_convergence_analysis()
    print(f"Convergence analysis: {convergence_analysis}")
    
    # Test experience replay
    replay = OptimizedExperienceReplay(capacity=1000, use_prioritized=True)
    
    # Add experiences
    for i in range(100):
        experience = {
            "state": np.random.randn(4),
            "action": np.random.randint(0, 2),
            "reward": np.random.randn(),
            "next_state": np.random.randn(4),
            "done": np.random.choice([True, False])
        }
        replay.add(experience, priority=np.random.rand())
    
    # Sample experiences
    experiences, indices, weights = replay.sample(batch_size=32)
    print(f"Sampled {len(experiences)} experiences with weights {weights.shape}")
    
    # Test curriculum learning
    curriculum = CurriculumLearning()
    
    for i in range(100):
        performance = np.random.rand()
        curriculum.update_difficulty(performance)
        if i % 20 == 0:
            print(f"Episode {i}: Difficulty = {curriculum.get_difficulty():.3f}")
    
    print("All tests completed successfully!")
