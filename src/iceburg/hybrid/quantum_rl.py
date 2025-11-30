"""
Quantum-RL Integration for ICEBURG Elite Financial AI

This module provides integration between quantum computing and reinforcement learning,
including quantum oracles, hybrid policies, and quantum-enhanced RL agents.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import pennylane as qml
from scipy.optimize import minimize

from ..quantum.circuits import VQC, QuantumCircuit
from ..quantum.sampling import QuantumSampler, MonteCarloAccelerator
from ..quantum.qgan import QuantumGAN
from ..rl.agents import BaseAgent, Action, State
from ..rl.agents.ppo_trader import PPOTrader
from ..rl.agents.sac_trader import SACTrader

logger = logging.getLogger(__name__)


@dataclass
class QuantumRLConfig:
    """Configuration for quantum-RL integration."""
    n_qubits: int = 8
    n_layers: int = 3
    quantum_device: str = "default.qubit"
    shots: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    quantum_advantage_threshold: float = 0.1
    hybrid_weight: float = 0.5
    oracle_update_frequency: int = 10


class QuantumOracle:
    """
    Quantum oracle for RL agents.
    
    Provides quantum-enhanced decision making capabilities
    for RL agents in financial markets.
    """
    
    def __init__(self, config: QuantumRLConfig):
        """Initialize quantum oracle."""
        self.config = config
        self.device = qml.device(config.quantum_device, wires=config.n_qubits, shots=config.shots)
        self.quantum_circuit = None
        self.quantum_sampler = QuantumSampler(config)
        self.monte_carlo_accelerator = MonteCarloAccelerator(config)
        
        # Oracle state
        self.oracle_state = None
        self.oracle_history = []
        self.performance_metrics = {}
        
        # Initialize quantum circuit
        self._initialize_quantum_circuit()
    
    def _initialize_quantum_circuit(self):
        """Initialize quantum circuit for oracle."""
        @qml.qnode(device=self.device, interface="torch")
        def quantum_oracle_circuit(inputs, weights):
            # Encode inputs
            for i, val in enumerate(inputs):
                if i < self.config.n_qubits:
                    qml.RY(val, wires=i)
            
            # Variational layers
            for layer in range(self.config.n_layers):
                # Single-qubit rotations
                for qubit in range(self.config.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                
                # Entangling layer
                for qubit in range(self.config.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.config.n_qubits)]
        
        self.quantum_circuit = quantum_oracle_circuit
    
    def query_oracle(self, state: State, action: Action) -> Dict[str, Any]:
        """
        Query quantum oracle for decision support.
        
        Args:
            state: Current state
            action: Proposed action
            
        Returns:
            Oracle response
        """
        try:
            # Convert state to quantum input
            quantum_input = self._state_to_quantum_input(state)
            
            # Generate random weights (in practice, these would be learned)
            weights = torch.randn(self.config.n_layers, self.config.n_qubits, 3)
            
            # Query quantum circuit
            quantum_output = self.quantum_circuit(quantum_input, weights)
            
            # Process quantum output
            oracle_response = self._process_quantum_output(quantum_output, action)
            
            # Update oracle state
            self._update_oracle_state(oracle_response)
            
            return oracle_response
        
        except Exception as e:
            logger.error(f"Error querying quantum oracle: {e}")
            return self._default_oracle_response()
    
    def _state_to_quantum_input(self, state: State) -> torch.Tensor:
        """Convert state to quantum input format."""
        # Extract numerical features from state
        features = []
        
        # Market data features
        for symbol, data in state.market_data.items():
            features.extend([
                data.get("price", 0.0),
                data.get("volume", 0.0),
                data.get("bid", 0.0),
                data.get("ask", 0.0),
                data.get("spread", 0.0),
                data.get("trades", 0.0)
            ])
        
        # Agent data features
        features.extend([
            state.agent_data.get("capital", 0.0),
            state.agent_data.get("pnl", 0.0),
            state.agent_data.get("total_volume", 0.0),
            state.agent_data.get("trade_count", 0.0)
        ])
        
        # Normalize features
        features = np.array(features)
        if len(features) > 0:
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # Pad or truncate to match qubit count
        if len(features) < self.config.n_qubits:
            features = np.pad(features, (0, self.config.n_qubits - len(features)))
        else:
            features = features[:self.config.n_qubits]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _process_quantum_output(self, quantum_output: torch.Tensor, action: Action) -> Dict[str, Any]:
        """Process quantum circuit output."""
        # Convert quantum output to oracle response
        oracle_response = {
            "quantum_advantage": float(torch.mean(quantum_output)),
            "action_confidence": float(torch.std(quantum_output)),
            "quantum_features": quantum_output.tolist(),
            "recommendation": self._generate_recommendation(quantum_output, action),
            "risk_assessment": self._assess_risk(quantum_output),
            "market_insight": self._generate_market_insight(quantum_output)
        }
        
        return oracle_response
    
    def _generate_recommendation(self, quantum_output: torch.Tensor, action: Action) -> str:
        """Generate action recommendation based on quantum output."""
        mean_output = torch.mean(quantum_output)
        
        if mean_output > 0.5:
            return "strong_buy"
        elif mean_output > 0.0:
            return "buy"
        elif mean_output > -0.5:
            return "hold"
        else:
            return "sell"
    
    def _assess_risk(self, quantum_output: torch.Tensor) -> Dict[str, float]:
        """Assess risk based on quantum output."""
        risk_metrics = {
            "volatility": float(torch.std(quantum_output)),
            "uncertainty": float(torch.var(quantum_output)),
            "confidence": float(1.0 - torch.std(quantum_output)),
            "quantum_advantage": float(torch.mean(quantum_output))
        }
        
        return risk_metrics
    
    def _generate_market_insight(self, quantum_output: torch.Tensor) -> str:
        """Generate market insight based on quantum output."""
        mean_output = torch.mean(quantum_output)
        std_output = torch.std(quantum_output)
        
        if std_output > 0.5:
            return "high_volatility_market"
        elif mean_output > 0.3:
            return "bullish_market"
        elif mean_output < -0.3:
            return "bearish_market"
        else:
            return "neutral_market"
    
    def _update_oracle_state(self, oracle_response: Dict[str, Any]):
        """Update oracle state with new response."""
        self.oracle_state = oracle_response
        self.oracle_history.append({
            "timestamp": datetime.now(),
            "response": oracle_response
        })
        
        # Keep only recent history
        if len(self.oracle_history) > 1000:
            self.oracle_history.pop(0)
    
    def _default_oracle_response(self) -> Dict[str, Any]:
        """Default oracle response when quantum circuit fails."""
        return {
            "quantum_advantage": 0.0,
            "action_confidence": 0.5,
            "quantum_features": [0.0] * self.config.n_qubits,
            "recommendation": "hold",
            "risk_assessment": {
                "volatility": 0.5,
                "uncertainty": 0.5,
                "confidence": 0.5,
                "quantum_advantage": 0.0
            },
            "market_insight": "neutral_market"
        }
    
    def get_oracle_performance(self) -> Dict[str, Any]:
        """Get oracle performance metrics."""
        if not self.oracle_history:
            return {"performance": "no_data"}
        
        # Calculate performance metrics
        advantages = [r["response"]["quantum_advantage"] for r in self.oracle_history]
        confidences = [r["response"]["action_confidence"] for r in self.oracle_history]
        
        performance = {
            "avg_quantum_advantage": np.mean(advantages),
            "avg_confidence": np.mean(confidences),
            "total_queries": len(self.oracle_history),
            "quantum_advantage_std": np.std(advantages),
            "confidence_std": np.std(confidences)
        }
        
        return performance


class HybridPolicy:
    """
    Hybrid quantum-classical policy for RL agents.
    
    Combines classical RL policies with quantum oracles
    for enhanced decision making in financial markets.
    """
    
    def __init__(self, config: QuantumRLConfig):
        """Initialize hybrid policy."""
        self.config = config
        self.quantum_oracle = QuantumOracle(config)
        self.classical_policy = None
        self.hybrid_weight = config.hybrid_weight
        
        # Policy state
        self.policy_history = []
        self.performance_metrics = {}
    
    def set_classical_policy(self, classical_policy):
        """Set classical policy component."""
        self.classical_policy = classical_policy
    
    def get_action(self, state: State) -> Action:
        """
        Get action from hybrid policy.
        
        Args:
            state: Current state
            
        Returns:
            Action from hybrid policy
        """
        try:
            # Get classical action
            classical_action = self._get_classical_action(state)
            
            # Get quantum oracle response
            oracle_response = self.quantum_oracle.query_oracle(state, classical_action)
            
            # Combine classical and quantum decisions
            hybrid_action = self._combine_decisions(classical_action, oracle_response)
            
            # Update policy history
            self._update_policy_history(state, classical_action, oracle_response, hybrid_action)
            
            return hybrid_action
        
        except Exception as e:
            logger.error(f"Error in hybrid policy: {e}")
            return self._fallback_action(state)
    
    def _get_classical_action(self, state: State) -> Action:
        """Get action from classical policy."""
        if self.classical_policy is None:
            return self._random_action(state)
        
        try:
            return self.classical_policy.act(state)
        except Exception as e:
            logger.error(f"Error in classical policy: {e}")
            return self._random_action(state)
    
    def _combine_decisions(self, classical_action: Action, oracle_response: Dict[str, Any]) -> Action:
        """Combine classical and quantum decisions."""
        # Extract quantum recommendations
        quantum_recommendation = oracle_response.get("recommendation", "hold")
        quantum_confidence = oracle_response.get("action_confidence", 0.5)
        quantum_advantage = oracle_response.get("quantum_advantage", 0.0)
        
        # Determine if quantum advantage is significant
        if abs(quantum_advantage) > self.config.quantum_advantage_threshold:
            # Use quantum guidance
            hybrid_action = self._apply_quantum_guidance(classical_action, oracle_response)
        else:
            # Use classical action with quantum confidence adjustment
            hybrid_action = self._adjust_classical_action(classical_action, oracle_response)
        
        return hybrid_action
    
    def _apply_quantum_guidance(self, classical_action: Action, oracle_response: Dict[str, Any]) -> Action:
        """Apply quantum guidance to action."""
        quantum_recommendation = oracle_response.get("recommendation", "hold")
        quantum_confidence = oracle_response.get("action_confidence", 0.5)
        
        # Modify action based on quantum recommendation
        if quantum_recommendation == "strong_buy" and classical_action.side == "sell":
            classical_action.side = "buy"
            classical_action.quantity = int(classical_action.quantity * 1.5)
        elif quantum_recommendation == "strong_sell" and classical_action.side == "buy":
            classical_action.side = "sell"
            classical_action.quantity = int(classical_action.quantity * 1.5)
        elif quantum_recommendation == "hold":
            classical_action.quantity = int(classical_action.quantity * 0.5)
        
        # Adjust price based on quantum confidence
        if quantum_confidence > 0.8:
            classical_action.price *= 1.001  # Slightly more aggressive
        elif quantum_confidence < 0.3:
            classical_action.price *= 0.999  # More conservative
        
        return classical_action
    
    def _adjust_classical_action(self, classical_action: Action, oracle_response: Dict[str, Any]) -> Action:
        """Adjust classical action based on quantum confidence."""
        quantum_confidence = oracle_response.get("action_confidence", 0.5)
        
        # Adjust quantity based on quantum confidence
        if quantum_confidence > 0.7:
            classical_action.quantity = int(classical_action.quantity * 1.2)
        elif quantum_confidence < 0.3:
            classical_action.quantity = int(classical_action.quantity * 0.8)
        
        return classical_action
    
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
    
    def _fallback_action(self, state: State) -> Action:
        """Fallback action when hybrid policy fails."""
        return self._random_action(state)
    
    def _update_policy_history(self, state: State, classical_action: Action, 
                              oracle_response: Dict[str, Any], hybrid_action: Action):
        """Update policy history."""
        self.policy_history.append({
            "timestamp": datetime.now(),
            "state": state,
            "classical_action": classical_action,
            "oracle_response": oracle_response,
            "hybrid_action": hybrid_action
        })
        
        # Keep only recent history
        if len(self.policy_history) > 1000:
            self.policy_history.pop(0)
    
    def get_policy_performance(self) -> Dict[str, Any]:
        """Get hybrid policy performance metrics."""
        if not self.policy_history:
            return {"performance": "no_data"}
        
        # Calculate performance metrics
        oracle_advantages = [h["oracle_response"]["quantum_advantage"] for h in self.policy_history]
        oracle_confidences = [h["oracle_response"]["action_confidence"] for h in self.policy_history]
        
        performance = {
            "avg_quantum_advantage": np.mean(oracle_advantages),
            "avg_quantum_confidence": np.mean(oracle_confidences),
            "total_decisions": len(self.policy_history),
            "quantum_advantage_std": np.std(oracle_advantages),
            "quantum_confidence_std": np.std(oracle_confidences)
        }
        
        return performance


class QuantumRLIntegration:
    """
    Main integration class for quantum-RL systems.
    
    Orchestrates the integration between quantum computing
    and reinforcement learning for financial applications.
    """
    
    def __init__(self, config: QuantumRLConfig):
        """Initialize quantum-RL integration."""
        self.config = config
        self.quantum_oracle = QuantumOracle(config)
        self.hybrid_policy = HybridPolicy(config)
        self.quantum_sampler = QuantumSampler(config)
        self.monte_carlo_accelerator = MonteCarloAccelerator(config)
        
        # Integration state
        self.integration_active = False
        self.performance_metrics = {}
        self.integration_history = []
    
    def integrate_with_agent(self, agent: BaseAgent) -> BaseAgent:
        """
        Integrate quantum capabilities with RL agent.
        
        Args:
            agent: RL agent to enhance
            
        Returns:
            Enhanced agent with quantum capabilities
        """
        try:
            # Set hybrid policy for agent
            self.hybrid_policy.set_classical_policy(agent)
            
            # Create enhanced agent
            enhanced_agent = QuantumEnhancedAgent(agent, self.hybrid_policy, self.config)
            
            # Activate integration
            self.integration_active = True
            
            return enhanced_agent
        
        except Exception as e:
            logger.error(f"Error integrating with agent: {e}")
            return agent
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            "integration_active": self.integration_active,
            "quantum_oracle_performance": self.quantum_oracle.get_oracle_performance(),
            "hybrid_policy_performance": self.hybrid_policy.get_policy_performance(),
            "total_integrations": len(self.integration_history)
        }
    
    def reset_integration(self):
        """Reset integration state."""
        self.integration_active = False
        self.performance_metrics = {}
        self.integration_history = []
        
        # Reset components
        self.quantum_oracle = QuantumOracle(self.config)
        self.hybrid_policy = HybridPolicy(self.config)


class QuantumEnhancedAgent(BaseAgent):
    """
    Quantum-enhanced RL agent.
    
    Wraps a classical RL agent with quantum capabilities
    for enhanced decision making.
    """
    
    def __init__(self, classical_agent: BaseAgent, hybrid_policy: HybridPolicy, config: QuantumRLConfig):
        """Initialize quantum-enhanced agent."""
        super().__init__(classical_agent.config)
        
        self.classical_agent = classical_agent
        self.hybrid_policy = hybrid_policy
        self.config = config
        
        # Enhanced capabilities
        self.quantum_advantage = 0.0
        self.quantum_confidence = 0.0
        self.quantum_features = []
        
        # Performance tracking
        self.quantum_performance = {}
        self.hybrid_performance = {}
    
    def act(self, state: State) -> Action:
        """Get action using hybrid quantum-classical policy."""
        try:
            # Use hybrid policy
            action = self.hybrid_policy.get_action(state)
            
            # Update quantum metrics
            self._update_quantum_metrics(action)
            
            return action
        
        except Exception as e:
            logger.error(f"Error in quantum-enhanced agent act: {e}")
            return self.classical_agent.act(state)
    
    def learn(self, experience: Dict[str, Any]) -> None:
        """Learn from experience."""
        try:
            # Learn in classical agent
            self.classical_agent.learn(experience)
            
            # Update hybrid policy
            self.hybrid_policy._update_policy_history(
                experience["state"],
                experience["action"],
                {"quantum_advantage": 0.0, "action_confidence": 0.5},
                experience["action"]
            )
        
        except Exception as e:
            logger.error(f"Error in quantum-enhanced agent learn: {e}")
    
    def _update_quantum_metrics(self, action: Action):
        """Update quantum performance metrics."""
        # This would be implemented based on specific quantum metrics
        pass
    
    def get_quantum_performance(self) -> Dict[str, Any]:
        """Get quantum performance metrics."""
        return {
            "quantum_advantage": self.quantum_advantage,
            "quantum_confidence": self.quantum_confidence,
            "quantum_features": self.quantum_features,
            "hybrid_policy_performance": self.hybrid_policy.get_policy_performance()
        }


# Example usage and testing
if __name__ == "__main__":
    # Test quantum-RL integration
    config = QuantumRLConfig(
        n_qubits=4,
        n_layers=2,
        quantum_device="default.qubit",
        shots=1000,
        learning_rate=0.001
    )
    
    # Test quantum oracle
    oracle = QuantumOracle(config)
    
    # Test state
    state = State(
        market_data={"AAPL": {"price": 150.0, "volume": 1000, "bid": 149.9, "ask": 150.1, "spread": 0.2, "trades": 10}},
        agent_data={"capital": 100000.0, "positions": {}, "pnl": 0.0, "total_volume": 0.0, "trade_count": 0}
    )
    
    action = Action("AAPL", "buy", 100, 150.0)
    
    # Query oracle
    oracle_response = oracle.query_oracle(state, action)
    print(f"Oracle response: {oracle_response}")
    
    # Test hybrid policy
    hybrid_policy = HybridPolicy(config)
    
    # Get hybrid action
    hybrid_action = hybrid_policy.get_action(state)
    print(f"Hybrid action: {hybrid_action}")
    
    # Test integration
    integration = QuantumRLIntegration(config)
    
    # Create mock agent
    from ..rl.agents.base_agent import RandomAgent
    mock_agent = RandomAgent(config)
    
    # Integrate with agent
    enhanced_agent = integration.integrate_with_agent(mock_agent)
    print(f"Enhanced agent: {enhanced_agent}")
    
    # Test enhanced agent
    enhanced_action = enhanced_agent.act(state)
    print(f"Enhanced agent action: {enhanced_action}")
    
    # Get integration status
    status = integration.get_integration_status()
    print(f"Integration status: {status}")
