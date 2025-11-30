"""
Trading Agents for ICEBURG Elite Financial AI

This module provides trading agents for multi-agent RL,
including base agents, PPO traders, SAC traders, and specialized agents.
"""

from .base_agent import BaseAgent, AgentConfig, Action, State, RandomAgent, MomentumAgent, MeanReversionAgent
from .ppo_trader import PPOTrader, PPOTraderConfig, PPOTraderWithCustomPolicy
from .sac_trader import SACTrader, SACTraderConfig, SACTraderWithCustomPolicy, SACTraderWithExploration

__all__ = [
    "BaseAgent",
    "AgentConfig", 
    "Action",
    "State",
    "RandomAgent",
    "MomentumAgent",
    "MeanReversionAgent",
    "PPOTrader",
    "PPOTraderConfig",
    "PPOTraderWithCustomPolicy",
    "SACTrader",
    "SACTraderConfig",
    "SACTraderWithCustomPolicy",
    "SACTraderWithExploration"
]

__version__ = "1.0.0"
__author__ = "ICEBURG Protocol"
__description__ = "Trading agents for elite financial AI"
