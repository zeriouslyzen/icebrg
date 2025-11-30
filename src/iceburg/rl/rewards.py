"""
Reward Functions for Elite Financial AI

This module provides reward function implementations for RL agents.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for reward functions."""
    
    @abstractmethod
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                        next_state: Dict[str, Any]) -> float:
        """Calculate reward for given state, action, and next state."""
        pass


class SharpeReward(RewardFunction):
    """Sharpe ratio-based reward function."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                        next_state: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio reward."""
        try:
            # Extract portfolio values
            current_value = state.get('portfolio_value', 100000)
            next_value = next_state.get('portfolio_value', current_value)
            
            # Calculate return
            if current_value > 0:
                return_rate = (next_value - current_value) / current_value
            else:
                return_rate = 0.0
            
            # Calculate volatility (simplified)
            returns_history = state.get('returns_history', [])
            if len(returns_history) > 1:
                volatility = np.std(returns_history)
            else:
                volatility = 0.1  # Default volatility
            
            # Calculate Sharpe ratio
            if volatility > 0:
                sharpe_ratio = (return_rate - self.risk_free_rate) / volatility
            else:
                sharpe_ratio = 0.0
            
            # Scale and clip the reward
            reward = np.clip(sharpe_ratio * 100, -10, 10)
            return float(reward)
        except Exception:
            return 0.0


class RiskAdjustedReward(RewardFunction):
    """Risk-adjusted reward function."""
    
    def __init__(self, risk_penalty: float = 0.1):
        self.risk_penalty = risk_penalty
    
    def calculate_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                        next_state: Dict[str, Any]) -> float:
        """Calculate risk-adjusted reward."""
        try:
            # Extract portfolio values
            current_value = state.get('portfolio_value', 100000)
            next_value = next_state.get('portfolio_value', current_value)
            
            # Calculate return
            if current_value > 0:
                return_rate = (next_value - current_value) / current_value
            else:
                return_rate = 0.0
            
            # Calculate risk metrics
            returns_history = state.get('returns_history', [])
            if len(returns_history) > 1:
                volatility = np.std(returns_history)
                max_drawdown = np.min(np.cumprod(1 + np.array(returns_history)) / 
                                    np.maximum.accumulate(np.cumprod(1 + np.array(returns_history))) - 1)
            else:
                volatility = 0.1
                max_drawdown = 0.0
            
            # Calculate risk-adjusted reward
            base_reward = return_rate * 100  # Scale return
            
            # Apply risk penalty
            risk_penalty = self.risk_penalty * (volatility * 100 + abs(max_drawdown) * 100)
            
            # Calculate final reward
            reward = base_reward - risk_penalty
            
            # Clip the reward
            reward = np.clip(reward, -10, 10)
            return float(reward)
        except Exception:
            return 0.0
