"""
Reinforcement Learning Configuration for ICEBURG Elite Financial AI

This module provides configuration settings for RL training operations,
including hyperparameters, optimization settings, and performance tuning.
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for reinforcement learning operations."""
    
    # Algorithm settings
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    
    # Environment settings
    n_envs: int = 8
    use_vectorized_envs: bool = True
    use_subprocess_envs: bool = True
    max_episode_steps: int = 1000
    
    # Training settings
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    save_freq: int = 50000
    log_interval: int = 1000
    
    # Optimization settings
    use_experience_replay: bool = True
    replay_buffer_size: int = 100000
    use_prioritized_replay: bool = True
    use_curriculum_learning: bool = True
    use_hyperparameter_optimization: bool = False
    
    # Performance settings
    use_gpu: bool = True
    use_distributed_training: bool = False
    n_workers: int = 4
    use_mixed_precision: bool = False
    
    # Memory settings
    max_memory_usage: float = 8.0  # GB
    memory_cleanup_threshold: float = 0.8
    
    # Logging settings
    log_level: str = "INFO"
    log_training_metrics: bool = True
    log_evaluation_metrics: bool = True
    log_performance_metrics: bool = True


class RLHyperparameterOptimizer:
    """
    Hyperparameter optimization for RL algorithms.
    
    Provides automated hyperparameter tuning for RL training.
    """
    
    def __init__(self, config: RLConfig):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            config: RL configuration
        """
        self.config = config
        self.optimization_history = []
        self.best_params = None
        self.best_performance = -float('inf')
    
    def get_parameter_space(self, algorithm: str) -> Dict[str, List]:
        """
        Get parameter space for specified algorithm.
        
        Args:
            algorithm: RL algorithm name
            
        Returns:
            Parameter space
        """
        if algorithm == "PPO":
            return {
                "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                "batch_size": [64, 128, 256, 512],
                "n_epochs": [3, 5, 10, 20],
                "clip_range": [0.1, 0.2, 0.3, 0.4],
                "ent_coef": [0.0, 0.01, 0.1, 0.2],
                "vf_coef": [0.25, 0.5, 1.0, 2.0]
            }
        elif algorithm == "SAC":
            return {
                "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                "batch_size": [64, 128, 256, 512],
                "tau": [0.005, 0.01, 0.02, 0.05],
                "gamma": [0.9, 0.95, 0.99, 0.995],
                "ent_coef": [0.0, 0.01, 0.1, 0.2]
            }
        elif algorithm == "A2C":
            return {
                "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                "batch_size": [64, 128, 256, 512],
                "n_steps": [5, 10, 20, 50],
                "gamma": [0.9, 0.95, 0.99, 0.995],
                "ent_coef": [0.0, 0.01, 0.1, 0.2]
            }
        else:
            logger.warning(f"Unknown algorithm: {algorithm}")
            return {}
    
    def optimize_hyperparameters(self, algorithm: str, env_factory, 
                                n_trials: int = 20, total_timesteps: int = 50000) -> Dict[str, Any]:
        """
        Optimize hyperparameters for specified algorithm.
        
        Args:
            algorithm: RL algorithm name
            env_factory: Environment factory function
            n_trials: Number of optimization trials
            total_timesteps: Training timesteps per trial
            
        Returns:
            Optimization results
        """
        if not self.config.use_hyperparameter_optimization:
            logger.warning("Hyperparameter optimization disabled")
            return {}
        
        param_space = self.get_parameter_space(algorithm)
        if not param_space:
            logger.error(f"No parameter space available for {algorithm}")
            return {}
        
        import random
        
        for trial in range(n_trials):
            # Sample parameters
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = random.choice(param_values)
            
            # Train model with sampled parameters
            try:
                performance = self._train_and_evaluate(algorithm, env_factory, params, total_timesteps)
                
                # Record trial
                trial_result = {
                    "trial": trial + 1,
                    "params": params,
                    "performance": performance
                }
                self.optimization_history.append(trial_result)
                
                # Update best parameters
                if performance > self.best_performance:
                    self.best_performance = performance
                    self.best_params = params
                
                logger.info(f"Trial {trial+1}/{n_trials}: Performance = {performance:.2f}")
                
            except Exception as e:
                logger.warning(f"Trial {trial+1} failed: {e}")
                continue
        
        results = {
            "best_params": self.best_params,
            "best_performance": self.best_performance,
            "optimization_history": self.optimization_history,
            "n_trials": n_trials
        }
        
        logger.info(f"Hyperparameter optimization completed. Best performance: {self.best_performance:.2f}")
        
        return results
    
    def _train_and_evaluate(self, algorithm: str, env_factory, params: Dict[str, Any], 
                           total_timesteps: int) -> float:
        """
        Train and evaluate model with given parameters.
        
        Args:
            algorithm: RL algorithm name
            env_factory: Environment factory function
            params: Hyperparameters
            total_timesteps: Training timesteps
            
        Returns:
            Performance metric
        """
        # This is a simplified implementation
        # In practice, use proper RL training and evaluation
        
        import gymnasium as gym
        from stable_baselines3 import PPO, SAC, A2C
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Create environment
        env = DummyVecEnv([env_factory])
        
        # Create model
        if algorithm == "PPO":
            model = PPO("MlpPolicy", env, **params, verbose=0)
        elif algorithm == "SAC":
            model = SAC("MlpPolicy", env, **params, verbose=0)
        elif algorithm == "A2C":
            model = A2C("MlpPolicy", env, **params, verbose=0)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train model
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate model
        obs = env.reset()
        episode_rewards = []
        
        for _ in range(100):  # 100 evaluation episodes
            episode_reward = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward.sum()
                
                if done.any():
                    break
            
            episode_rewards.append(episode_reward)
        
        # Calculate performance metric
        performance = np.mean(episode_rewards)
        
        env.close()
        
        return performance


class RLPerformanceConfig:
    """
    Configuration for RL performance optimization.
    
    Provides settings for optimizing RL training performance.
    """
    
    def __init__(self, config: RLConfig):
        """
        Initialize performance configuration.
        
        Args:
            config: RL configuration
        """
        self.config = config
        self.performance_levels = {
            "basic": {
                "use_vectorized_envs": False,
                "n_envs": 1,
                "use_experience_replay": False,
                "use_curriculum_learning": False,
                "use_distributed_training": False
            },
            "standard": {
                "use_vectorized_envs": True,
                "n_envs": 4,
                "use_experience_replay": True,
                "use_curriculum_learning": False,
                "use_distributed_training": False
            },
            "aggressive": {
                "use_vectorized_envs": True,
                "n_envs": 8,
                "use_experience_replay": True,
                "use_curriculum_learning": True,
                "use_distributed_training": True
            }
        }
    
    def get_performance_config(self, level: str = "standard") -> Dict[str, Any]:
        """
        Get performance configuration for specified level.
        
        Args:
            level: Performance level
            
        Returns:
            Performance configuration
        """
        if level not in self.performance_levels:
            logger.warning(f"Unknown performance level: {level}, using standard")
            level = "standard"
        
        return self.performance_levels[level]
    
    def get_optimal_batch_size(self, n_envs: int, memory_available: float) -> int:
        """
        Calculate optimal batch size based on environment count and memory.
        
        Args:
            n_envs: Number of environments
            memory_available: Available memory in GB
            
        Returns:
            Optimal batch size
        """
        # Simple heuristic for batch size calculation
        base_batch_size = 256
        env_factor = min(n_envs / 4, 2.0)
        memory_factor = min(memory_available / 4, 2.0)
        
        optimal_batch_size = int(base_batch_size * env_factor * memory_factor)
        
        # Clamp to reasonable range
        return max(32, min(optimal_batch_size, 1024))
    
    def get_optimal_learning_rate(self, algorithm: str, total_timesteps: int) -> float:
        """
        Calculate optimal learning rate based on algorithm and training length.
        
        Args:
            algorithm: RL algorithm name
            total_timesteps: Total training timesteps
            
        Returns:
            Optimal learning rate
        """
        # Base learning rates for different algorithms
        base_rates = {
            "PPO": 3e-4,
            "SAC": 3e-4,
            "A2C": 3e-4,
            "TD3": 3e-4
        }
        
        base_rate = base_rates.get(algorithm, 3e-4)
        
        # Adjust based on training length
        if total_timesteps > 1000000:
            # Longer training, use smaller learning rate
            return base_rate * 0.5
        elif total_timesteps < 100000:
            # Shorter training, use larger learning rate
            return base_rate * 2.0
        else:
            return base_rate


class RLMemoryManager:
    """
    Memory management for RL training.
    
    Monitors memory usage and performs cleanup during training.
    """
    
    def __init__(self, config: RLConfig):
        """
        Initialize memory manager.
        
        Args:
            config: RL configuration
        """
        self.config = config
        self.memory_usage = 0.0
        self.training_objects = {}
    
    def check_memory_usage(self) -> bool:
        """
        Check if memory usage is within limits.
        
        Returns:
            True if memory usage is acceptable
        """
        import psutil
        
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        memory_usage_gb = memory_info.used / (1024**3)
        
        self.memory_usage = memory_usage_gb
        
        # Check if memory usage exceeds threshold
        if memory_usage_gb > self.config.max_memory_usage:
            logger.warning(f"Memory usage {memory_usage_gb:.2f}GB exceeds limit {self.config.max_memory_usage}GB")
            return False
        
        return True
    
    def cleanup_memory(self):
        """Clean up memory by clearing unused objects."""
        # Clear training objects if memory usage is high
        if self.memory_usage > self.config.max_memory_usage * self.config.memory_cleanup_threshold:
            self.training_objects.clear()
            logger.info("Cleared training objects due to high memory usage")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get current memory status.
        
        Returns:
            Memory status information
        """
        import psutil
        
        memory_info = psutil.virtual_memory()
        
        return {
            "total_memory_gb": memory_info.total / (1024**3),
            "available_memory_gb": memory_info.available / (1024**3),
            "used_memory_gb": memory_info.used / (1024**3),
            "memory_percentage": memory_info.percent,
            "training_objects": len(self.training_objects)
        }


def load_rl_config(config_path: str = None) -> RLConfig:
    """
    Load RL configuration from file or environment.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        RL configuration
    """
    config = RLConfig()
    
    # Load from environment variables
    config.algorithm = os.getenv("RL_ALGORITHM", config.algorithm)
    config.learning_rate = float(os.getenv("RL_LEARNING_RATE", str(config.learning_rate)))
    config.batch_size = int(os.getenv("RL_BATCH_SIZE", str(config.batch_size)))
    config.n_envs = int(os.getenv("RL_N_ENVS", str(config.n_envs)))
    config.total_timesteps = int(os.getenv("RL_TOTAL_TIMESTEPS", str(config.total_timesteps)))
    config.use_vectorized_envs = os.getenv("RL_USE_VECTORIZED", "true").lower() == "true"
    config.use_experience_replay = os.getenv("RL_USE_EXPERIENCE_REPLAY", "true").lower() == "true"
    config.use_curriculum_learning = os.getenv("RL_USE_CURRICULUM", "true").lower() == "true"
    config.use_hyperparameter_optimization = os.getenv("RL_USE_HYPEROPT", "false").lower() == "true"
    
    # Load from configuration file if provided
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Update configuration with file values
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Loaded RL configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load RL configuration from {config_path}: {e}")
    
    return config


def save_rl_config(config: RLConfig, config_path: str):
    """
    Save RL configuration to file.
    
    Args:
        config: RL configuration
        config_path: Path to save configuration
    """
    try:
        import yaml
        
        # Convert dataclass to dictionary
        config_dict = {
            "algorithm": config.algorithm,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "clip_range": config.clip_range,
            "ent_coef": config.ent_coef,
            "vf_coef": config.vf_coef,
            "n_envs": config.n_envs,
            "use_vectorized_envs": config.use_vectorized_envs,
            "use_subprocess_envs": config.use_subprocess_envs,
            "max_episode_steps": config.max_episode_steps,
            "total_timesteps": config.total_timesteps,
            "eval_freq": config.eval_freq,
            "save_freq": config.save_freq,
            "log_interval": config.log_interval,
            "use_experience_replay": config.use_experience_replay,
            "replay_buffer_size": config.replay_buffer_size,
            "use_prioritized_replay": config.use_prioritized_replay,
            "use_curriculum_learning": config.use_curriculum_learning,
            "use_hyperparameter_optimization": config.use_hyperparameter_optimization,
            "use_gpu": config.use_gpu,
            "use_distributed_training": config.use_distributed_training,
            "n_workers": config.n_workers,
            "use_mixed_precision": config.use_mixed_precision,
            "max_memory_usage": config.max_memory_usage,
            "memory_cleanup_threshold": config.memory_cleanup_threshold,
            "log_level": config.log_level,
            "log_training_metrics": config.log_training_metrics,
            "log_evaluation_metrics": config.log_evaluation_metrics,
            "log_performance_metrics": config.log_performance_metrics
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Saved RL configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save RL configuration to {config_path}: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test RL configuration
    config = load_rl_config()
    print(f"RL configuration: {config}")
    
    # Test hyperparameter optimizer
    optimizer = RLHyperparameterOptimizer(config)
    param_space = optimizer.get_parameter_space("PPO")
    print(f"PPO parameter space: {param_space}")
    
    # Test performance configuration
    perf_config = RLPerformanceConfig(config)
    opt_config = perf_config.get_performance_config("aggressive")
    print(f"Performance configuration: {opt_config}")
    
    # Test memory manager
    memory_manager = RLMemoryManager(config)
    memory_status = memory_manager.get_memory_status()
    print(f"Memory status: {memory_status}")
    
    # Test optimal parameters
    optimal_batch_size = perf_config.get_optimal_batch_size(n_envs=8, memory_available=8.0)
    optimal_lr = perf_config.get_optimal_learning_rate("PPO", 1000000)
    print(f"Optimal batch size: {optimal_batch_size}")
    print(f"Optimal learning rate: {optimal_lr}")
    
    # Save configuration
    save_rl_config(config, "rl_config.yaml")
    print("RL configuration saved to rl_config.yaml")