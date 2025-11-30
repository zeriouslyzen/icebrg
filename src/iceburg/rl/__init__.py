"""
ICEBURG Multi-Agent Reinforcement Learning Module

This module provides multi-agent reinforcement learning capabilities for elite financial AI,
including trading environments, agent implementations, and emergent behavior detection.

Key Components:
- environments: Trading and financial simulation environments
- agents: Multi-agent RL implementations (PPO, SAC, etc.)
- rewards: Reward function definitions
- emergence_detector: Emergent behavior and cartel formation detection
- analysis: Agent behavior analysis and visualization
- config: RL system configuration
- utils: RL utilities and helpers
"""

from .config import RLConfig
from .environments import TradingEnvironment, OrderBook, MarketSimulator
from .agents import BaseAgent, PPOTrader, SACTrader
from .rewards import RewardFunction, SharpeReward, RiskAdjustedReward
from .emergence_detector import EmergenceDetector
from .analysis import AgentAnalyzer, BehaviorAnalyzer
from .visualization import RLVisualizer, EmergenceVisualizer
from .utils import RLUtils, PerformanceMetrics

__all__ = [
    "RLConfig",
    "TradingEnvironment",
    "OrderBook", 
    "MarketSimulator",
    "BaseAgent",
    "PPOTrader",
    "SACTrader", 
    "RewardFunction",
    "SharpeReward",
    "RiskAdjustedReward",
    "EmergenceDetector",
    "AgentAnalyzer",
    "BehaviorAnalyzer",
    "RLVisualizer",
    "EmergenceVisualizer",
    "RLUtils",
    "PerformanceMetrics"
]

__version__ = "1.0.0"
__author__ = "ICEBURG Protocol"
__description__ = "Multi-agent RL module for elite financial AI"
