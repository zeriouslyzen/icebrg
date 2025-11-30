"""
Base Agent for ICEBURG Elite Financial AI

This module provides the base agent class and common functionality
for all trading agents in the multi-agent RL system.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from datetime import datetime
import random

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for trading agents."""
    agent_id: str
    agent_type: str
    capital: float = 100000.0
    max_position: float = 10000.0
    risk_tolerance: float = 0.1
    latency: float = 0.001
    strategy: str = "momentum"
    learning_rate: float = 3e-4
    exploration_rate: float = 0.1
    memory_size: int = 10000
    batch_size: int = 64
    device: str = "cpu"


@dataclass
class Action:
    """Action structure for trading agents."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    order_type: str = "limit"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class State:
    """State structure for trading agents."""
    market_data: Dict[str, Any]
    agent_data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseAgent(ABC):
    """
    Base class for all trading agents.
    
    Provides common functionality and interface for all agent types.
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize base agent."""
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.capital = config.capital
        self.max_position = config.max_position
        self.risk_tolerance = config.risk_tolerance
        self.latency = config.latency
        self.strategy = config.strategy
        
        # Agent state
        self.positions = {}
        self.orders = []
        self.trades = []
        self.pnl = 0.0
        self.total_volume = 0
        self.trade_count = 0
        
        # Learning parameters
        self.learning_rate = config.learning_rate
        self.exploration_rate = config.exploration_rate
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.device = config.device
        
        # Memory
        self.memory = []
        self.experience_buffer = []
        
        # Performance metrics
        self.performance_metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_trade_size": 0.0,
            "total_trades": 0,
            "total_volume": 0.0
        }
        
        # Initialize agent
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize agent-specific components."""
        pass
    
    @abstractmethod
    def act(self, state: State) -> Action:
        """
        Choose action based on current state.
        
        Args:
            state: Current state observation
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def learn(self, experience: Dict[str, Any]) -> None:
        """
        Learn from experience.
        
        Args:
            experience: Experience data
        """
        pass
    
    def get_action(self, state: State) -> Action:
        """
        Get action with exploration.
        
        Args:
            state: Current state
            
        Returns:
            Action to take
        """
        # Add exploration noise
        if random.random() < self.exploration_rate:
            return self._explore_action(state)
        else:
            return self.act(state)
    
    def _explore_action(self, state: State) -> Action:
        """Generate exploratory action."""
        # Random action for exploration
        symbol = random.choice(list(state.market_data.keys()))
        side = random.choice(["buy", "sell"])
        quantity = random.randint(1, 100)
        price = state.market_data[symbol]["price"] * (1 + random.uniform(-0.01, 0.01))
        
        return Action(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )
    
    def update_position(self, symbol: str, quantity: int, price: float):
        """Update agent position."""
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        self.positions[symbol] += quantity
        self.total_volume += abs(quantity) * price
        self.trade_count += 1
        
        # Update P&L
        if quantity > 0:  # Buy
            self.pnl -= quantity * price
        else:  # Sell
            self.pnl += abs(quantity) * price
    
    def get_position(self, symbol: str) -> int:
        """Get current position for symbol."""
        return self.positions.get(symbol, 0)
    
    def get_total_position_value(self) -> float:
        """Get total value of all positions."""
        total_value = 0.0
        for symbol, position in self.positions.items():
            # This would need current market prices
            # For now, return position count
            total_value += abs(position)
        return total_value
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics for the agent."""
        if not self.trades:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_trade_size": 0.0
            }
        
        # Calculate returns
        returns = []
        for trade in self.trades:
            if trade["side"] == "sell":
                returns.append(trade["quantity"] * trade["price"])
            else:
                returns.append(-trade["quantity"] * trade["price"])
        
        if returns:
            total_return = sum(returns)
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe ratio
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # Max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            # Win rate
            winning_trades = sum(1 for r in returns if r > 0)
            win_rate = winning_trades / len(returns) if returns else 0
            
            # Average trade size
            avg_trade_size = np.mean([abs(trade["quantity"]) for trade in self.trades])
            
            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "avg_trade_size": avg_trade_size
            }
        else:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_trade_size": 0.0
            }
    
    def add_experience(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        """Add experience to memory."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "timestamp": datetime.now()
        }
        
        self.memory.append(experience)
        
        # Keep memory size limited
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def get_memory_sample(self, batch_size: int = None) -> List[Dict[str, Any]]:
        """Get random sample from memory."""
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return self.memory
        
        return random.sample(self.memory, batch_size)
    
    def update_performance_metrics(self):
        """Update performance metrics."""
        self.performance_metrics = self.calculate_risk_metrics()
        self.performance_metrics.update({
            "total_trades": self.trade_count,
            "total_volume": self.total_volume
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        self.update_performance_metrics()
        return self.performance_metrics
    
    def reset(self):
        """Reset agent to initial state."""
        self.positions = {}
        self.orders = []
        self.trades = []
        self.pnl = 0.0
        self.total_volume = 0
        self.trade_count = 0
        self.memory = []
        self.experience_buffer = []
        
        # Reset performance metrics
        self.performance_metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_trade_size": 0.0,
            "total_trades": 0,
            "total_volume": 0.0
        }
    
    def save_state(self, path: str):
        """Save agent state to file."""
        state = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capital": self.capital,
            "positions": self.positions,
            "pnl": self.pnl,
            "performance_metrics": self.performance_metrics,
            "config": self.config
        }
        
        torch.save(state, path)
        logger.info(f"Agent state saved to {path}")
    
    def load_state(self, path: str):
        """Load agent state from file."""
        state = torch.load(path)
        
        self.agent_id = state["agent_id"]
        self.agent_type = state["agent_type"]
        self.capital = state["capital"]
        self.positions = state["positions"]
        self.pnl = state["pnl"]
        self.performance_metrics = state["performance_metrics"]
        
        logger.info(f"Agent state loaded from {path}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capital": self.capital,
            "positions": self.positions,
            "pnl": self.pnl,
            "performance_metrics": self.performance_metrics,
            "memory_size": len(self.memory),
            "total_trades": self.trade_count,
            "total_volume": self.total_volume
        }


class RandomAgent(BaseAgent):
    """
    Random trading agent for baseline comparison.
    
    Takes random actions to establish baseline performance.
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize random agent."""
        super().__init__(config)
        self.agent_type = "random"
    
    def act(self, state: State) -> Action:
        """Take random action."""
        # Get available symbols
        symbols = list(state.market_data.keys())
        if not symbols:
            return None
        
        # Random symbol
        symbol = random.choice(symbols)
        
        # Random side
        side = random.choice(["buy", "sell"])
        
        # Random quantity
        quantity = random.randint(1, 100)
        
        # Random price around current price
        current_price = state.market_data[symbol]["price"]
        price = current_price * (1 + random.uniform(-0.05, 0.05))
        
        return Action(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """Random agent doesn't learn."""
        pass


class MomentumAgent(BaseAgent):
    """
    Momentum trading agent.
    
    Buys when prices are rising and sells when prices are falling.
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize momentum agent."""
        super().__init__(config)
        self.agent_type = "momentum"
        self.lookback_period = 20
        self.momentum_threshold = 0.02  # 2% momentum threshold
    
    def act(self, state: State) -> Action:
        """Take momentum-based action."""
        # Get available symbols
        symbols = list(state.market_data.keys())
        if not symbols:
            return None
        
        # Choose symbol with highest momentum
        best_symbol = None
        best_momentum = 0
        
        for symbol in symbols:
            momentum = self._calculate_momentum(symbol, state)
            if abs(momentum) > abs(best_momentum):
                best_momentum = momentum
                best_symbol = symbol
        
        if best_symbol is None:
            return None
        
        # Determine action based on momentum
        if best_momentum > self.momentum_threshold:
            # Strong upward momentum - buy
            side = "buy"
            quantity = min(100, int(self.capital * 0.1 / state.market_data[best_symbol]["price"]))
        elif best_momentum < -self.momentum_threshold:
            # Strong downward momentum - sell
            side = "sell"
            quantity = min(100, abs(self.get_position(best_symbol)))
        else:
            # No clear momentum - hold
            return None
        
        # Set price
        current_price = state.market_data[best_symbol]["price"]
        if side == "buy":
            price = current_price * 1.001  # Slightly above market
        else:
            price = current_price * 0.999  # Slightly below market
        
        return Action(
            symbol=best_symbol,
            side=side,
            quantity=quantity,
            price=price
        )
    
    def _calculate_momentum(self, symbol: str, state: State) -> float:
        """Calculate momentum for a symbol."""
        # This is a simplified momentum calculation
        # In practice, you'd use historical price data
        current_price = state.market_data[symbol]["price"]
        
        # Simulate momentum calculation
        # In reality, you'd use historical prices
        momentum = random.uniform(-0.05, 0.05)
        
        return momentum
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """Momentum agent doesn't learn."""
        pass


class MeanReversionAgent(BaseAgent):
    """
    Mean reversion trading agent.
    
    Buys when prices are below mean and sells when prices are above mean.
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize mean reversion agent."""
        super().__init__(config)
        self.agent_type = "mean_reversion"
        self.lookback_period = 50
        self.reversion_threshold = 0.02  # 2% deviation threshold
    
    def act(self, state: State) -> Action:
        """Take mean reversion action."""
        # Get available symbols
        symbols = list(state.market_data.keys())
        if not symbols:
            return None
        
        # Choose symbol with highest deviation from mean
        best_symbol = None
        best_deviation = 0
        
        for symbol in symbols:
            deviation = self._calculate_deviation(symbol, state)
            if abs(deviation) > abs(best_deviation):
                best_deviation = deviation
                best_symbol = symbol
        
        if best_symbol is None:
            return None
        
        # Determine action based on deviation
        if best_deviation > self.reversion_threshold:
            # Price above mean - sell
            side = "sell"
            quantity = min(100, abs(self.get_position(best_symbol)))
        elif best_deviation < -self.reversion_threshold:
            # Price below mean - buy
            side = "buy"
            quantity = min(100, int(self.capital * 0.1 / state.market_data[best_symbol]["price"]))
        else:
            # Price near mean - hold
            return None
        
        # Set price
        current_price = state.market_data[best_symbol]["price"]
        if side == "buy":
            price = current_price * 1.001  # Slightly above market
        else:
            price = current_price * 0.999  # Slightly below market
        
        return Action(
            symbol=best_symbol,
            side=side,
            quantity=quantity,
            price=price
        )
    
    def _calculate_deviation(self, symbol: str, state: State) -> float:
        """Calculate deviation from mean for a symbol."""
        # This is a simplified deviation calculation
        # In practice, you'd use historical price data
        current_price = state.market_data[symbol]["price"]
        
        # Simulate deviation calculation
        # In reality, you'd use historical prices
        deviation = random.uniform(-0.05, 0.05)
        
        return deviation
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """Mean reversion agent doesn't learn."""
        pass


# Example usage and testing
if __name__ == "__main__":
    # Test base agent
    config = AgentConfig(
        agent_id="test_agent",
        agent_type="base",
        capital=100000.0,
        risk_tolerance=0.1
    )
    
    agent = BaseAgent(config)
    
    # Test state and action
    state = State(
        market_data={"AAPL": {"price": 150.0, "volume": 1000}},
        agent_data={"capital": 100000.0, "positions": {}}
    )
    
    action = agent.get_action(state)
    print(f"Agent action: {action}")
    
    # Test performance metrics
    metrics = agent.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Test random agent
    random_config = AgentConfig(
        agent_id="random_agent",
        agent_type="random",
        capital=100000.0
    )
    
    random_agent = RandomAgent(random_config)
    random_action = random_agent.act(state)
    print(f"Random agent action: {random_action}")
    
    # Test momentum agent
    momentum_config = AgentConfig(
        agent_id="momentum_agent",
        agent_type="momentum",
        capital=100000.0
    )
    
    momentum_agent = MomentumAgent(momentum_config)
    momentum_action = momentum_agent.act(state)
    print(f"Momentum agent action: {momentum_action}")
    
    # Test mean reversion agent
    mean_reversion_config = AgentConfig(
        agent_id="mean_reversion_agent",
        agent_type="mean_reversion",
        capital=100000.0
    )
    
    mean_reversion_agent = MeanReversionAgent(mean_reversion_config)
    mean_reversion_action = mean_reversion_agent.act(state)
    print(f"Mean reversion agent action: {mean_reversion_action}")
