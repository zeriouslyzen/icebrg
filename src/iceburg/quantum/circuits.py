"""
Quantum Circuits for ICEBURG Elite Financial AI

This module provides Variational Quantum Circuits (VQCs) and quantum circuit utilities
for financial applications, including quantum state preparation and measurement.
"""

import numpy as np
import pennylane as qml
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class QuantumCircuitConfig:
    """Configuration for quantum circuits."""
    n_qubits: int = 8
    n_layers: int = 3
    n_wires: Optional[int] = None
    device: str = "default.qubit"
    shots: int = 1000
    interface: str = "torch"


class VQC(nn.Module):
    """
    Variational Quantum Circuit for financial applications.
    
    Implements a parameterized quantum circuit that can be optimized
    for specific financial tasks like classification, regression, or sampling.
    """
    
    def __init__(
        self, 
        n_qubits: int = 8, 
        n_layers: int = 3, 
        n_inputs: int = 4,
        n_outputs: int = 1,
        device: str = "default.qubit",
        interface: str = "torch"
    ):
        """
        Initialize VQC.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of variational layers
            n_inputs: Number of input features
            n_outputs: Number of output values
            device: Quantum device
            interface: Classical interface (torch, tf, numpy)
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.device = device
        self.interface = interface
        
        # Initialize quantum device
        self.qdevice = qml.device(device, wires=n_qubits, shots=1000)
        
        # Create quantum circuit
        self.qnode = qml.QNode(self._quantum_circuit, self.qdevice, interface=interface)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize variational parameters."""
        # Input encoding parameters
        self.input_weights = nn.Parameter(torch.randn(self.n_inputs, self.n_qubits))
        
        # Variational layer parameters
        self.variational_weights = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 3)  # 3 parameters per qubit (RX, RY, RZ)
        )
        
        # Entangling layer parameters
        self.entangling_weights = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits - 1)  # CNOT gates between adjacent qubits
        )
        
        # Output measurement parameters
        self.output_weights = nn.Parameter(torch.randn(self.n_outputs, self.n_qubits))
    
    def _quantum_circuit(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Define the quantum circuit.
        
        Args:
            inputs: Input data tensor
            
        Returns:
            Output measurements
        """
        # Input encoding with proper dimension handling
        for i in range(self.n_qubits):
            if i < len(inputs) and i < self.n_inputs:
                # Encode input data with proper bounds checking
                qml.RY(inputs[i] * self.input_weights[i, i], wires=i)
            else:
                # Initialize with random state
                qml.RY(np.pi / 4, wires=i)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.RX(self.variational_weights[layer, qubit, 0], wires=qubit)
                qml.RY(self.variational_weights[layer, qubit, 1], wires=qubit)
                qml.RZ(self.variational_weights[layer, qubit, 2], wires=qubit)
            
            # Entangling layer
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
                qml.RY(self.entangling_weights[layer, qubit], wires=qubit + 1)
        
        # Output measurements
        measurements = []
        for i in range(self.n_outputs):
            if i < self.n_qubits:
                measurements.append(qml.expval(qml.PauliZ(i)))
            else:
                measurements.append(qml.expval(qml.PauliZ(0)))  # Default measurement
        
        return measurements
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.
        
        Args:
            inputs: Input data tensor
            
        Returns:
            Output predictions
        """
        # Ensure inputs are the right size
        if inputs.shape[-1] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {inputs.shape[-1]}")
        
        # Process through quantum circuit
        outputs = self.qnode(inputs)
        
        # Convert to tensor
        if isinstance(outputs, (list, tuple)):
            outputs = torch.stack(outputs)
        else:
            outputs = torch.tensor(outputs)
        
        # Apply output weights
        outputs = outputs * self.output_weights[:len(outputs)]
        
        return outputs
    
    def get_parameters(self) -> List[torch.Tensor]:
        """Get all trainable parameters."""
        return list(self.parameters())
    
    def set_parameters(self, params: List[torch.Tensor]):
        """Set parameters from list."""
        for param, new_param in zip(self.parameters(), params):
            param.data = new_param.data.clone()


class QuantumCircuit:
    """
    General quantum circuit class for financial applications.
    
    Provides utilities for creating, manipulating, and executing quantum circuits.
    """
    
    def __init__(self, config: QuantumCircuitConfig):
        """Initialize quantum circuit with configuration."""
        self.config = config
        self.device = qml.device(
            config.device, 
            wires=config.n_qubits, 
            shots=config.shots
        )
        self.circuit = None
        self.parameters = {}
    
    def create_circuit(self, circuit_type: str = "variational") -> qml.QNode:
        """
        Create a quantum circuit of specified type.
        
        Args:
            circuit_type: Type of circuit to create
            
        Returns:
            Quantum circuit QNode
        """
        if circuit_type == "variational":
            return self._create_variational_circuit()
        elif circuit_type == "encoding":
            return self._create_encoding_circuit()
        elif circuit_type == "measurement":
            return self._create_measurement_circuit()
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    def _create_variational_circuit(self) -> qml.QNode:
        """Create variational quantum circuit."""
        @qml.qnode(device=self.device, interface=self.config.interface)
        def variational_circuit(inputs, weights):
            # Input encoding
            for i in range(self.config.n_qubits):
                if i < len(inputs):
                    qml.RY(inputs[i], wires=i)
            
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
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.config.n_qubits)]
        
        return variational_circuit
    
    def _create_encoding_circuit(self) -> qml.QNode:
        """Create data encoding circuit."""
        @qml.qnode(device=self.device, interface=self.config.interface)
        def encoding_circuit(inputs):
            # Encode input data into quantum state
            for i, input_val in enumerate(inputs):
                if i < self.config.n_qubits:
                    qml.RY(input_val, wires=i)
            
            # Return state vector
            return qml.state()
        
        return encoding_circuit
    
    def _create_measurement_circuit(self) -> qml.QNode:
        """Create measurement circuit."""
        @qml.qnode(device=self.device, interface=self.config.interface)
        def measurement_circuit(inputs):
            # Prepare state
            for i, input_val in enumerate(inputs):
                if i < self.config.n_qubits:
                    qml.RY(input_val, wires=i)
            
            # Measure in different bases
            measurements = []
            for i in range(self.config.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
                measurements.append(qml.expval(qml.PauliX(i)))
                measurements.append(qml.expval(qml.PauliY(i)))
            
            return measurements
        
        return measurement_circuit
    
    def execute_circuit(self, inputs: np.ndarray, parameters: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute quantum circuit with given inputs and parameters.
        
        Args:
            inputs: Input data
            parameters: Circuit parameters
            
        Returns:
            Circuit outputs
        """
        if self.circuit is None:
            self.circuit = self.create_circuit()
        
        if parameters is not None:
            return self.circuit(inputs, parameters)
        else:
            return self.circuit(inputs)
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the quantum circuit."""
        return {
            "n_qubits": self.config.n_qubits,
            "n_layers": self.config.n_layers,
            "device": self.config.device,
            "shots": self.config.shots,
            "interface": self.config.interface,
            "parameters": self.parameters
        }


class QuantumStatePreparation:
    """Quantum state preparation utilities for financial data."""
    
    def __init__(self, n_qubits: int = 8):
        """Initialize state preparation."""
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)
    
    def prepare_financial_state(self, data: np.ndarray) -> qml.QNode:
        """
        Prepare quantum state from financial data.
        
        Args:
            data: Financial data array
            
        Returns:
            Quantum circuit for state preparation
        """
        @qml.qnode(device=self.device)
        def financial_state_circuit():
            # Normalize data to [0, 2π]
            normalized_data = 2 * np.pi * (data - np.min(data)) / (np.max(data) - np.min(data))
            
            # Encode data into quantum state
            for i, val in enumerate(normalized_data):
                if i < self.n_qubits:
                    qml.RY(val, wires=i)
            
            return qml.state()
        
        return financial_state_circuit
    
    def prepare_superposition_state(self, n_states: int = 4) -> qml.QNode:
        """
        Prepare superposition state for multiple scenarios.
        
        Args:
            n_states: Number of superposition states
            
        Returns:
            Quantum circuit for superposition preparation
        """
        @qml.qnode(device=self.device)
        def superposition_circuit():
            # Create equal superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Apply controlled rotations for different states
            for state in range(n_states):
                if state < self.n_qubits:
                    qml.CRY(np.pi / (state + 1), wires=[0, state])
            
            return qml.state()
        
        return superposition_circuit
    
    def prepare_entangled_state(self, pairs: List[Tuple[int, int]]) -> qml.QNode:
        """
        Prepare entangled state for correlated financial variables.
        
        Args:
            pairs: List of qubit pairs to entangle
            
        Returns:
            Quantum circuit for entanglement preparation
        """
        @qml.qnode(device=self.device)
        def entangled_circuit():
            # Initialize with Hadamard gates
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Create entanglement between specified pairs
            for qubit1, qubit2 in pairs:
                if qubit1 < self.n_qubits and qubit2 < self.n_qubits:
                    qml.CNOT(wires=[qubit1, qubit2])
            
            return qml.state()
        
        return entangled_circuit


def simple_vqc(features: torch.Tensor, n_qubits: int = 2, n_layers: int = 1) -> torch.Tensor:
    """
    Simple Variational Quantum Circuit with proper dimension handling.
    
    Args:
        features: Input features tensor
        n_qubits: Number of qubits
        n_layers: Number of layers
        
    Returns:
        Quantum circuit output
    """
    # Ensure features are the right size
    if len(features) > n_qubits:
        features = features[:n_qubits]  # Truncate if too many
    elif len(features) < n_qubits:
        # Pad with zeros if too few
        padding = torch.zeros(n_qubits - len(features))
        features = torch.cat([features, padding])
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def circuit(weights):
        # Angle embedding with proper dimensions
        qml.AngleEmbedding(features, wires=range(n_qubits))
        
        # Variational layers
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            
            # Entangling layer
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Initialize weights
    weights = torch.randn(n_layers, n_qubits, 3, requires_grad=True)
    
    return circuit(weights)


class QuantumMeasurement:
    """Quantum measurement utilities for financial applications."""
    
    def __init__(self, n_qubits: int = 8):
        """Initialize measurement utilities."""
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)
    
    def measure_expectation_values(self, observables: List[str]) -> qml.QNode:
        """
        Measure expectation values of specified observables.
        
        Args:
            observables: List of observable names (PauliX, PauliY, PauliZ)
            
        Returns:
            Quantum circuit for expectation value measurement
        """
        @qml.qnode(device=self.device)
        def expectation_circuit(inputs):
            # Prepare state
            for i, input_val in enumerate(inputs):
                if i < self.n_qubits:
                    qml.RY(input_val, wires=i)
            
            # Measure expectation values
            measurements = []
            for i, obs in enumerate(observables):
                if i < self.n_qubits:
                    if obs == "PauliX":
                        measurements.append(qml.expval(qml.PauliX(i)))
                    elif obs == "PauliY":
                        measurements.append(qml.expval(qml.PauliY(i)))
                    elif obs == "PauliZ":
                        measurements.append(qml.expval(qml.PauliZ(i)))
                    else:
                        measurements.append(qml.expval(qml.PauliZ(i)))  # Default
            
            return measurements
        
        return expectation_circuit
    
    def measure_probability_distribution(self) -> qml.QNode:
        """
        Measure probability distribution over computational basis states.
        
        Returns:
            Quantum circuit for probability measurement
        """
        @qml.qnode(device=self.device)
        def probability_circuit(inputs):
            # Prepare state
            for i, input_val in enumerate(inputs):
                if i < self.n_qubits:
                    qml.RY(input_val, wires=i)
            
            # Return probabilities
            return qml.probs(wires=range(self.n_qubits))
        
        return probability_circuit
    
    def measure_correlation(self, qubit_pairs: List[Tuple[int, int]]) -> qml.QNode:
        """
        Measure correlation between qubit pairs.
        
        Args:
            qubit_pairs: List of qubit pairs to measure correlation
            
        Returns:
            Quantum circuit for correlation measurement
        """
        @qml.qnode(device=self.device)
        def correlation_circuit(inputs):
            # Prepare state
            for i, input_val in enumerate(inputs):
                if i < self.n_qubits:
                    qml.RY(input_val, wires=i)
            
            # Measure correlations
            correlations = []
            for qubit1, qubit2 in qubit_pairs:
                if qubit1 < self.n_qubits and qubit2 < self.n_qubits:
                    correlations.append(qml.expval(qml.PauliZ(qubit1) @ qml.PauliZ(qubit2)))
            
            return correlations
        
        return correlation_circuit


# Example usage and testing
if __name__ == "__main__":
    # Test VQC
    vqc = VQC(n_qubits=4, n_layers=2, n_inputs=4, n_outputs=1)
    
    # Test input
    test_input = torch.randn(4)
    output = vqc(test_input)
    print(f"VQC output: {output}")
    
    # Test quantum circuit
    config = QuantumCircuitConfig(n_qubits=4, n_layers=2)
    circuit = QuantumCircuit(config)
    qnode = circuit.create_circuit("variational")
    
    # Test execution
    test_inputs = np.random.randn(4)
    test_weights = np.random.randn(2, 4, 3)
    result = qnode(test_inputs, test_weights)
    print(f"Circuit output: {result}")
    
    # Test state preparation
    state_prep = QuantumStatePreparation(n_qubits=4)
    financial_data = np.random.randn(4)
    state_circuit = state_prep.prepare_financial_state(financial_data)
    state = state_circuit()
    print(f"Financial state: {state}")
    
    # Test measurement
    measurement = QuantumMeasurement(n_qubits=4)
    exp_circuit = measurement.measure_expectation_values(["PauliZ", "PauliX", "PauliY", "PauliZ"])
    exp_values = exp_circuit(financial_data)
    print(f"Expectation values: {exp_values}")
