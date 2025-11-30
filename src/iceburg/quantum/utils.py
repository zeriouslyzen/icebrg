"""
Quantum Computing Utilities for ICEBURG Elite Financial AI

This module provides utility functions and helper classes for quantum computing
operations in financial applications.
"""

import numpy as np
import pennylane as qml
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class QuantumUtils:
    """
    Utility class for quantum computing operations.
    
    Provides helper functions for quantum circuit manipulation,
    state preparation, and measurement operations.
    """
    
    def __init__(self, n_qubits: int = 4):
        """
        Initialize quantum utilities.
        
        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)
    
    def prepare_bell_state(self, qubit1: int, qubit2: int) -> qml.QNode:
        """
        Prepare Bell state between two qubits.
        
        Args:
            qubit1: First qubit index
            qubit2: Second qubit index
            
        Returns:
            Bell state circuit
        """
        @qml.qnode(device=self.device)
        def bell_circuit():
            qml.Hadamard(wires=qubit1)
            qml.CNOT(wires=[qubit1, qubit2])
            return qml.state()
        
        return bell_circuit
    
    def prepare_ghz_state(self, n_qubits: int = None) -> qml.QNode:
        """
        Prepare GHZ (Greenberger-Horne-Zeilinger) state.
        
        Args:
            n_qubits: Number of qubits (defaults to self.n_qubits)
            
        Returns:
            GHZ state circuit
        """
        if n_qubits is None:
            n_qubits = self.n_qubits
        
        @qml.qnode(device=self.device)
        def ghz_circuit():
            qml.Hadamard(wires=0)
            for i in range(1, n_qubits):
                qml.CNOT(wires=[0, i])
            return qml.state()
        
        return ghz_circuit
    
    def measure_expectation_values(self, observables: List[str]) -> qml.QNode:
        """
        Measure expectation values of observables.
        
        Args:
            observables: List of observable names
            
        Returns:
            Measurement circuit
        """
        @qml.qnode(device=self.device)
        def expectation_circuit():
            # Prepare state
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Measure observables
            measurements = []
            for i, obs in enumerate(observables):
                if i < self.n_qubits:
                    if obs == "X":
                        measurements.append(qml.expval(qml.PauliX(i)))
                    elif obs == "Y":
                        measurements.append(qml.expval(qml.PauliY(i)))
                    elif obs == "Z":
                        measurements.append(qml.expval(qml.PauliZ(i)))
                    else:
                        measurements.append(qml.expval(qml.PauliZ(i)))
            
            return measurements
        
        return expectation_circuit
    
    def create_parameterized_circuit(self, n_layers: int = 2) -> qml.QNode:
        """
        Create parameterized quantum circuit.
        
        Args:
            n_layers: Number of layers
            
        Returns:
            Parameterized circuit
        """
        @qml.qnode(device=self.device)
        def parameterized_circuit(params):
            # Initial state preparation
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                # Single-qubit rotations
                for qubit in range(self.n_qubits):
                    qml.RX(params[layer, qubit, 0], wires=qubit)
                    qml.RY(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)
                
                # Entangling layer
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return parameterized_circuit
    
    def calculate_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate fidelity between two quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Fidelity value
        """
        # Normalize states
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)
        
        # Calculate fidelity
        fidelity = np.abs(np.dot(state1.conj(), state2))**2
        
        return fidelity
    
    def calculate_entanglement_entropy(self, state: np.ndarray, partition: int) -> float:
        """
        Calculate entanglement entropy of a quantum state.
        
        Args:
            state: Quantum state vector
            partition: Partition point for entropy calculation
            
        Returns:
            Entanglement entropy
        """
        # Reshape state for bipartition
        dim1 = 2**partition
        dim2 = 2**(self.n_qubits - partition)
        
        # Reshape state matrix
        state_matrix = state.reshape(dim1, dim2)
        
        # Calculate reduced density matrix
        rho = np.dot(state_matrix, state_matrix.conj().T)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical errors
        
        # Calculate von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return entropy


def create_quantum_circuit(n_qubits: int, circuit_type: str = "variational") -> qml.QNode:
    """
    Create quantum circuit of specified type.
    
    Args:
        n_qubits: Number of qubits
        circuit_type: Type of circuit
        
    Returns:
        Quantum circuit
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    if circuit_type == "variational":
        @qml.qnode(dev)
        def variational_circuit(params):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            for layer in range(len(params)):
                for qubit in range(n_qubits):
                    qml.RX(params[layer, qubit, 0], wires=qubit)
                    qml.RY(params[layer, qubit, 1], wires=qubit)
                    qml.RZ(params[layer, qubit, 2], wires=qubit)
                
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return variational_circuit
    
    elif circuit_type == "encoding":
        @qml.qnode(dev)
        def encoding_circuit(data):
            for i, val in enumerate(data):
                if i < n_qubits:
                    qml.RY(val, wires=i)
            return qml.state()
        
        return encoding_circuit
    
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")


def optimize_quantum_circuit(circuit: qml.QNode, cost_function, initial_params: np.ndarray, 
                           optimizer: str = "adam", max_iterations: int = 100) -> Dict[str, Any]:
    """
    Optimize quantum circuit parameters.
    
    Args:
        circuit: Quantum circuit to optimize
        cost_function: Cost function to minimize
        initial_params: Initial parameters
        optimizer: Optimizer type
        max_iterations: Maximum iterations
        
    Returns:
        Optimization results
    """
    # Simple optimization (in practice, use proper optimizer)
    best_params = initial_params.copy()
    best_cost = float('inf')
    costs = []
    
    for iteration in range(max_iterations):
        # Random parameter update
        params = best_params + 0.1 * np.random.randn(*best_params.shape)
        
        # Calculate cost
        cost = cost_function(params)
        costs.append(cost)
        
        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
        
        if iteration % 20 == 0:
            logger.info(f"Iteration {iteration}, Cost: {cost:.4f}")
    
    return {
        "best_params": best_params,
        "best_cost": best_cost,
        "costs": costs,
        "iterations": max_iterations
    }


def measure_quantum_state(state: np.ndarray, observable: str = "Z") -> float:
    """
    Measure quantum state with specified observable.
    
    Args:
        state: Quantum state vector
        observable: Observable to measure
        
    Returns:
        Expectation value
    """
    # Normalize state
    state = state / np.linalg.norm(state)
    
    if observable == "X":
        # Pauli-X measurement
        return np.real(np.dot(state.conj(), np.array([1, 0, 0, 1]) * state))
    elif observable == "Y":
        # Pauli-Y measurement
        return np.real(np.dot(state.conj(), np.array([0, -1j, 1j, 0]) * state))
    elif observable == "Z":
        # Pauli-Z measurement
        return np.real(np.dot(state.conj(), np.array([1, 0, 0, -1]) * state))
    else:
        raise ValueError(f"Unknown observable: {observable}")


# Example usage and testing
if __name__ == "__main__":
    # Test quantum utilities
    utils = QuantumUtils(n_qubits=4)
    
    # Test Bell state preparation
    bell_circuit = utils.prepare_bell_state(0, 1)
    bell_state = bell_circuit()
    print(f"Bell state: {bell_state}")
    
    # Test GHZ state preparation
    ghz_circuit = utils.prepare_ghz_state()
    ghz_state = ghz_circuit()
    print(f"GHZ state: {ghz_state}")
    
    # Test parameterized circuit
    param_circuit = utils.create_parameterized_circuit(n_layers=2)
    params = np.random.randn(2, 4, 3)
    result = param_circuit(params)
    print(f"Parameterized circuit result: {result}")
    
    # Test fidelity calculation
    state1 = np.array([1, 0, 0, 0])
    state2 = np.array([0.707, 0, 0, 0.707])
    fidelity = utils.calculate_fidelity(state1, state2)
    print(f"Fidelity: {fidelity:.4f}")
    
    # Test entanglement entropy
    state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    entropy = utils.calculate_entanglement_entropy(state, 1)
    print(f"Entanglement entropy: {entropy:.4f}")
    
    # Test circuit creation
    circuit = create_quantum_circuit(n_qubits=2, circuit_type="variational")
    params = np.random.randn(1, 2, 3)
    result = circuit(params)
    print(f"Circuit result: {result}")
    
    # Test optimization
    def cost_function(params):
        return np.sum(params**2)
    
    initial_params = np.random.randn(2, 2, 3)
    result = optimize_quantum_circuit(circuit, cost_function, initial_params, max_iterations=50)
    print(f"Optimization result: {result['best_cost']:.4f}")
    
    # Test state measurement
    state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    expectation = measure_quantum_state(state, "Z")
    print(f"Z expectation value: {expectation:.4f}")
