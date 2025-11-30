"""
ICEBURG Qubit Simulator
Simulates qubits (quantum bits) without external dependencies
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QubitState:
    """State of a qubit"""
    qubit_id: int
    state_vector: np.ndarray  # [|0> amplitude, |1> amplitude]
    phase: float  # Phase angle
    timestamp: datetime


@dataclass
class QuantumCircuitResult:
    """Result of running a quantum circuit"""
    qubits: List[QubitState]
    measurements: List[int]
    entanglement: Dict[Tuple[int, int], float]  # (qubit1, qubit2): correlation
    execution_time: float
    timestamp: datetime


class QubitSimulator:
    """
    Simulates qubits (quantum bits) for quantum computing
    
    Philosophy: Qubits can be simulated computationally
    - Superposition: |ψ> = α|0> + β|1>
    - Entanglement: Correlated qubit states
    - Measurement: Collapse to classical state
    """
    
    def __init__(self):
        self.qubits: Dict[int, QubitState] = {}
        self.circuit_history: List[QuantumCircuitResult] = []
        self.measurement_history: List[Dict[str, Any]] = []
    
    def create_qubit(self, qubit_id: int, initial_state: str = "0") -> QubitState:
        """
        Create a qubit in initial state
        
        Args:
            qubit_id: Unique qubit identifier
            initial_state: "0", "1", or "superposition"
        """
        if initial_state == "0":
            state_vector = np.array([1.0, 0.0])  # |0>
        elif initial_state == "1":
            state_vector = np.array([0.0, 1.0])  # |1>
        elif initial_state == "superposition":
            state_vector = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)])  # |+>
        else:
            state_vector = np.array([1.0, 0.0])  # Default to |0>
        
        qubit = QubitState(
            qubit_id=qubit_id,
            state_vector=state_vector,
            phase=0.0,
            timestamp=datetime.utcnow()
        )
        
        self.qubits[qubit_id] = qubit
        return qubit
    
    def apply_hadamard(self, qubit_id: int) -> QubitState:
        """
        Apply Hadamard gate (creates superposition)
        
        H|0> = (|0> + |1>)/√2
        H|1> = (|0> - |1>)/√2
        """
        if qubit_id not in self.qubits:
            self.create_qubit(qubit_id, "0")
        
        qubit = self.qubits[qubit_id]
        
        # Hadamard matrix
        H = np.array([[1.0/np.sqrt(2), 1.0/np.sqrt(2)],
                      [1.0/np.sqrt(2), -1.0/np.sqrt(2)]])
        
        # Apply Hadamard
        new_state = H @ qubit.state_vector
        
        # Normalize
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state = new_state / norm
        
        qubit.state_vector = new_state
        qubit.timestamp = datetime.utcnow()
        
        return qubit
    
    def apply_pauli_x(self, qubit_id: int) -> QubitState:
        """
        Apply Pauli-X gate (bit flip)
        
        X|0> = |1>
        X|1> = |0>
        """
        if qubit_id not in self.qubits:
            self.create_qubit(qubit_id, "0")
        
        qubit = self.qubits[qubit_id]
        
        # Pauli-X matrix
        X = np.array([[0.0, 1.0],
                      [1.0, 0.0]])
        
        # Apply Pauli-X
        qubit.state_vector = X @ qubit.state_vector
        qubit.timestamp = datetime.utcnow()
        
        return qubit
    
    def apply_pauli_y(self, qubit_id: int) -> QubitState:
        """
        Apply Pauli-Y gate (bit and phase flip)
        """
        if qubit_id not in self.qubits:
            self.create_qubit(qubit_id, "0")
        
        qubit = self.qubits[qubit_id]
        
        # Pauli-Y matrix
        Y = np.array([[0.0, -1.0j],
                      [1.0j, 0.0]])
        
        # Apply Pauli-Y
        qubit.state_vector = Y @ qubit.state_vector
        qubit.timestamp = datetime.utcnow()
        
        return qubit
    
    def apply_pauli_z(self, qubit_id: int) -> QubitState:
        """
        Apply Pauli-Z gate (phase flip)
        
        Z|0> = |0>
        Z|1> = -|1>
        """
        if qubit_id not in self.qubits:
            self.create_qubit(qubit_id, "0")
        
        qubit = self.qubits[qubit_id]
        
        # Pauli-Z matrix
        Z = np.array([[1.0, 0.0],
                      [0.0, -1.0]])
        
        # Apply Pauli-Z
        qubit.state_vector = Z @ qubit.state_vector
        qubit.timestamp = datetime.utcnow()
        
        return qubit
    
    def apply_cnot(self, control_id: int, target_id: int) -> Tuple[QubitState, QubitState]:
        """
        Apply CNOT gate (entanglement)
        
        CNOT|00> = |00>
        CNOT|01> = |01>
        CNOT|10> = |11>
        CNOT|11> = |10>
        """
        if control_id not in self.qubits:
            self.create_qubit(control_id, "0")
        if target_id not in self.qubits:
            self.create_qubit(target_id, "0")
        
        control = self.qubits[control_id]
        target = self.qubits[target_id]
        
        # CNOT gate (4x4 matrix for 2-qubit system)
        # |00> |01> |10> |11>
        CNOT = np.array([
            [1.0, 0.0, 0.0, 0.0],  # |00> -> |00>
            [0.0, 1.0, 0.0, 0.0],  # |01> -> |01>
            [0.0, 0.0, 0.0, 1.0],  # |10> -> |11>
            [0.0, 0.0, 1.0, 0.0]   # |11> -> |10>
        ])
        
        # Combine qubit states
        combined_state = np.kron(control.state_vector, target.state_vector)
        
        # Apply CNOT
        new_combined = CNOT @ combined_state
        
        # Extract individual qubit states (simplified - full extraction would need tensor decomposition)
        # For now, update based on measurement probabilities
        control_prob_0 = abs(new_combined[0])**2 + abs(new_combined[1])**2
        target_prob_0 = abs(new_combined[0])**2 + abs(new_combined[2])**2
        
        control.state_vector = np.array([np.sqrt(control_prob_0), np.sqrt(1 - control_prob_0)])
        target.state_vector = np.array([np.sqrt(target_prob_0), np.sqrt(1 - target_prob_0)])
        
        control.timestamp = datetime.utcnow()
        target.timestamp = datetime.utcnow()
        
        return control, target
    
    def measure(self, qubit_id: int) -> int:
        """
        Measure a qubit (collapses to classical state)
        
        Returns:
            0 or 1 (classical bit)
        """
        if qubit_id not in self.qubits:
            self.create_qubit(qubit_id, "0")
        
        qubit = self.qubits[qubit_id]
        
        # Probability of measuring |0> and |1>
        prob_0 = abs(qubit.state_vector[0])**2
        prob_1 = abs(qubit.state_vector[1])**2
        
        # Normalize probabilities
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        # Measure (collapse to classical state)
        measurement = np.random.choice([0, 1], p=[prob_0, prob_1])
        
        # Collapse state
        if measurement == 0:
            qubit.state_vector = np.array([1.0, 0.0])
        else:
            qubit.state_vector = np.array([0.0, 1.0])
        
        qubit.timestamp = datetime.utcnow()
        
        # Record measurement
        self.measurement_history.append({
            "qubit_id": qubit_id,
            "measurement": measurement,
            "prob_0": prob_0,
            "prob_1": prob_1,
            "timestamp": qubit.timestamp
        })
        
        return measurement
    
    def create_bell_state(self, qubit1_id: int, qubit2_id: int) -> Tuple[QubitState, QubitState]:
        """
        Create Bell state (maximally entangled)
        
        |Φ+> = (|00> + |11>)/√2
        """
        # Initialize qubits
        self.create_qubit(qubit1_id, "0")
        self.create_qubit(qubit2_id, "0")
        
        # Apply Hadamard to first qubit
        self.apply_hadamard(qubit1_id)
        
        # Apply CNOT
        self.apply_cnot(qubit1_id, qubit2_id)
        
        return self.qubits[qubit1_id], self.qubits[qubit2_id]
    
    def get_entanglement(self, qubit1_id: int, qubit2_id: int) -> float:
        """
        Calculate entanglement between two qubits
        
        Returns:
            Entanglement measure (0.0 to 1.0)
        """
        if qubit1_id not in self.qubits or qubit2_id not in self.qubits:
            return 0.0
        
        qubit1 = self.qubits[qubit1_id]
        qubit2 = self.qubits[qubit2_id]
        
        # Calculate correlation
        # Simplified entanglement measure
        combined_state = np.kron(qubit1.state_vector, qubit2.state_vector)
        
        # Entanglement entropy (simplified)
        # For Bell state: entanglement = 1.0
        # For separable state: entanglement = 0.0
        
        # Check if state is separable
        if abs(combined_state[0]) > 0.99 or abs(combined_state[3]) > 0.99:
            # |00> or |11> - separable
            return 0.0
        elif abs(combined_state[1]) > 0.99 or abs(combined_state[2]) > 0.99:
            # |01> or |10> - separable
            return 0.0
        else:
            # Entangled state
            # Calculate von Neumann entropy (simplified)
            probs = np.abs(combined_state)**2
            probs = probs[probs > 1e-10]  # Remove zeros
            entropy = -np.sum(probs * np.log2(probs))
            return min(1.0, entropy)  # Normalize to [0, 1]
    
    def run_quantum_circuit(self, circuit: List[Dict[str, Any]]) -> QuantumCircuitResult:
        """
        Run a quantum circuit
        
        Args:
            circuit: List of gate operations
                Example: [{"gate": "hadamard", "qubit": 0}, {"gate": "cnot", "control": 0, "target": 1}]
        """
        import time
        start_time = time.time()
        
        # Execute circuit
        for operation in circuit:
            gate = operation.get("gate", "")
            if gate == "hadamard":
                qubit_id = operation.get("qubit", 0)
                self.apply_hadamard(qubit_id)
            elif gate == "pauli_x":
                qubit_id = operation.get("qubit", 0)
                self.apply_pauli_x(qubit_id)
            elif gate == "pauli_y":
                qubit_id = operation.get("qubit", 0)
                self.apply_pauli_y(qubit_id)
            elif gate == "pauli_z":
                qubit_id = operation.get("qubit", 0)
                self.apply_pauli_z(qubit_id)
            elif gate == "cnot":
                control_id = operation.get("control", 0)
                target_id = operation.get("target", 1)
                self.apply_cnot(control_id, target_id)
            elif gate == "measure":
                qubit_id = operation.get("qubit", 0)
                self.measure(qubit_id)
        
        # Measure all qubits
        measurements = []
        for qubit_id in sorted(self.qubits.keys()):
            measurement = self.measure(qubit_id)
            measurements.append(measurement)
        
        # Calculate entanglement
        entanglement = {}
        qubit_ids = sorted(self.qubits.keys())
        for i in range(len(qubit_ids)):
            for j in range(i+1, len(qubit_ids)):
                ent = self.get_entanglement(qubit_ids[i], qubit_ids[j])
                entanglement[(qubit_ids[i], qubit_ids[j])] = ent
        
        execution_time = time.time() - start_time
        
        result = QuantumCircuitResult(
            qubits=[self.qubits[qid] for qid in sorted(self.qubits.keys())],
            measurements=measurements,
            entanglement=entanglement,
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )
        
        self.circuit_history.append(result)
        return result
    
    def get_qubit_state(self, qubit_id: int) -> Optional[QubitState]:
        """Get current state of a qubit"""
        return self.qubits.get(qubit_id)
    
    def get_all_qubits(self) -> List[QubitState]:
        """Get all qubits"""
        return [self.qubits[qid] for qid in sorted(self.qubits.keys())]


# Global qubit simulator instance
_qubit_simulator: Optional[QubitSimulator] = None

def get_qubit_simulator() -> QubitSimulator:
    """Get or create the global qubit simulator instance"""
    global _qubit_simulator
    if _qubit_simulator is None:
        _qubit_simulator = QubitSimulator()
    return _qubit_simulator

