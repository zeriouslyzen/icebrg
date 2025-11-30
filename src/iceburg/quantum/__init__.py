"""
ICEBURG Quantum Computing Module

This module provides quantum computing capabilities for elite financial AI,
including Variational Quantum Circuits (VQCs), Quantum GANs (QGANs),
quantum sampling methods, and portfolio optimization algorithms.

Key Components:
- circuits: VQC definitions and quantum circuit utilities
- kernels: Quantum kernel methods for machine learning
- qgan: Quantum GAN implementation for financial data generation
- sampling: Quantum sampling methods for market scenarios
- portfolio_opt: QAOA portfolio optimization
- hybrid_training: Quantum-classical hybrid training
- config: Quantum system configuration
- utils: Quantum computing utilities
"""

from .config import QuantumConfig
from .circuits import VQC, QuantumCircuit
from .kernels import QuantumKernel, QuantumSVM
from .qgan import QuantumGAN
from .sampling import QuantumSampler, MonteCarloAccelerator
from .portfolio_opt import QAOAOptimizer, PortfolioOptimizer
from .hybrid_training import HybridTrainer
from .utils import QuantumUtils

__all__ = [
    "QuantumConfig",
    "VQC",
    "QuantumCircuit", 
    "QuantumKernel",
    "QuantumSVM",
    "QuantumGAN",
    "QuantumSampler",
    "MonteCarloAccelerator",
    "QAOAOptimizer",
    "PortfolioOptimizer",
    "HybridTrainer",
    "QuantumUtils"
]

__version__ = "1.0.0"
__author__ = "ICEBURG Protocol"
__description__ = "Quantum computing module for elite financial AI"
