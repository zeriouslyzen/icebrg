"""
Hybrid Quantum-Classical Integration for ICEBURG Elite Financial AI

This module provides integration between quantum computing and reinforcement learning
for elite financial AI applications, including quantum oracles, hybrid policies,
and quantum-enhanced RL agents.
"""

from .quantum_rl import QuantumRLIntegration, QuantumOracle, HybridPolicy
from .quantum_policy import QuantumPolicyNetwork, QuantumValueNetwork
from .orchestrator import HybridOrchestrator, QuantumRLOrchestrator

__all__ = [
    "QuantumRLIntegration",
    "QuantumOracle", 
    "HybridPolicy",
    "QuantumPolicyNetwork",
    "QuantumValueNetwork",
    "HybridOrchestrator",
    "QuantumRLOrchestrator"
]

__version__ = "1.0.0"
__author__ = "ICEBURG Protocol"
__description__ = "Hybrid quantum-classical integration for elite financial AI"
