"""
Quantum Configuration for ICEBURG Elite Financial AI

This module provides configuration settings for quantum computing operations,
including device selection, optimization parameters, and performance tuning.
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumConfig:
    """Configuration for quantum computing operations."""
    
    # Device configuration
    device: str = "default.qubit"
    use_gpu: bool = True
    gpu_device: str = "lightning.gpu"
    fallback_device: str = "default.qubit"
    
    # Performance settings
    shots: int = 1000
    max_shots: int = 10000
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Optimization settings
    use_caching: bool = True
    cache_size: int = 1000
    circuit_compilation: bool = True
    depth_reduction: bool = True
    
    # Memory settings
    max_memory_usage: float = 8.0  # GB
    memory_cleanup_threshold: float = 0.8
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    error_threshold: float = 0.1
    
    # Logging
    log_level: str = "INFO"
    log_quantum_operations: bool = True
    log_performance_metrics: bool = True


class QuantumDeviceManager:
    """
    Manages quantum device selection and configuration.
    
    Automatically selects the best available quantum device based on
    system capabilities and configuration.
    """
    
    def __init__(self, config: QuantumConfig):
        """
        Initialize quantum device manager.
        
        Args:
            config: Quantum configuration
        """
        self.config = config
        self.available_devices = []
        self.best_device = None
        self._detect_available_devices()
    
    def _detect_available_devices(self):
        """Detect available quantum devices."""
        # List of devices to try in order of preference
        device_candidates = [
            "lightning.gpu",
            "qiskit.aer",
            "default.qubit",
            "default.qubit.legacy"
        ]
        
        for device_name in device_candidates:
            try:
                # Test if device is available
                if self._test_device(device_name):
                    self.available_devices.append(device_name)
                    logger.info(f"Available device: {device_name}")
            except Exception as e:
                logger.debug(f"Device {device_name} not available: {e}")
        
        # Select best device
        self.best_device = self._select_best_device()
        logger.info(f"Selected best device: {self.best_device}")
    
    def _test_device(self, device_name: str) -> bool:
        """Test if a quantum device is available."""
        try:
            import pennylane as qml
            
            # Create test device
            if "gpu" in device_name.lower():
                if not self.config.use_gpu:
                    return False
                device = qml.device(device_name, wires=2)
            else:
                device = qml.device(device_name, wires=2)
            
            # Test basic operation
            @qml.qnode(device)
            def test_circuit():
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))
            
            # Execute test
            result = test_circuit()
            
            return True
            
        except Exception as e:
            logger.debug(f"Device {device_name} test failed: {e}")
            return False
    
    def _select_best_device(self) -> str:
        """Select the best available quantum device."""
        if not self.available_devices:
            logger.warning("No quantum devices available, using default")
            return "default.qubit"
        
        # Prefer GPU devices if available and enabled
        if self.config.use_gpu:
            gpu_devices = [d for d in self.available_devices if "gpu" in d.lower()]
            if gpu_devices:
                return gpu_devices[0]
        
        # Return first available device
        return self.available_devices[0]
    
    def get_device(self, n_qubits: int, shots: int = None) -> str:
        """
        Get quantum device for specified parameters.
        
        Args:
            n_qubits: Number of qubits
            shots: Number of shots
            
        Returns:
            Device name
        """
        if shots is None:
            shots = self.config.shots
        
        # Check if device supports required parameters
        try:
            import pennylane as qml
            device = qml.device(self.best_device, wires=n_qubits, shots=shots)
            return self.best_device
        except Exception:
            # Fallback to default device
            logger.warning(f"Best device {self.best_device} failed, using fallback")
            return self.config.fallback_device


class QuantumPerformanceConfig:
    """
    Configuration for quantum performance optimization.
    
    Provides settings for optimizing quantum circuit performance.
    """
    
    def __init__(self, config: QuantumConfig):
        """
        Initialize performance configuration.
        
        Args:
            config: Quantum configuration
        """
        self.config = config
        self.optimization_levels = {
            "basic": {
                "use_caching": True,
                "cache_size": 100,
                "circuit_compilation": False,
                "depth_reduction": False
            },
            "standard": {
                "use_caching": True,
                "cache_size": 1000,
                "circuit_compilation": True,
                "depth_reduction": False
            },
            "aggressive": {
                "use_caching": True,
                "cache_size": 10000,
                "circuit_compilation": True,
                "depth_reduction": True
            }
        }
    
    def get_optimization_config(self, level: str = "standard") -> Dict[str, Any]:
        """
        Get optimization configuration for specified level.
        
        Args:
            level: Optimization level
            
        Returns:
            Optimization configuration
        """
        if level not in self.optimization_levels:
            logger.warning(f"Unknown optimization level: {level}, using standard")
            level = "standard"
        
        return self.optimization_levels[level]
    
    def get_optimal_shots(self, circuit_depth: int, accuracy_required: float) -> int:
        """
        Calculate optimal number of shots based on circuit depth and accuracy.
        
        Args:
            circuit_depth: Circuit depth
            accuracy_required: Required accuracy
            
        Returns:
            Optimal number of shots
        """
        # Simple heuristic for shot calculation
        base_shots = 1000
        depth_factor = min(circuit_depth / 10, 2.0)
        accuracy_factor = 1.0 / accuracy_required
        
        optimal_shots = int(base_shots * depth_factor * accuracy_factor)
        
        # Clamp to reasonable range
        return max(100, min(optimal_shots, self.config.max_shots))
    
    def get_optimal_workers(self, n_circuits: int) -> int:
        """
        Calculate optimal number of workers for parallel execution.
        
        Args:
            n_circuits: Number of circuits to execute
            
        Returns:
            Optimal number of workers
        """
        if not self.config.parallel_execution:
            return 1
        
        # Simple heuristic for worker calculation
        optimal_workers = min(n_circuits, self.config.max_workers)
        
        return max(1, optimal_workers)


class QuantumMemoryManager:
    """
    Manages quantum circuit memory usage and cleanup.
    
    Monitors memory usage and performs cleanup when necessary.
    """
    
    def __init__(self, config: QuantumConfig):
        """
        Initialize memory manager.
        
        Args:
            config: Quantum configuration
        """
        self.config = config
        self.memory_usage = 0.0
        self.circuit_cache = {}
    
    def check_memory_usage(self) -> bool:
        """
        Check if memory usage is within limits.
        
        Returns:
            True if memory usage is acceptable
        """
        import psutil
        
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        memory_usage_gb = memory_info.used / (1024**3)
        
        self.memory_usage = memory_usage_gb
        
        # Check if memory usage exceeds threshold
        if memory_usage_gb > self.config.max_memory_usage:
            logger.warning(f"Memory usage {memory_usage_gb:.2f}GB exceeds limit {self.config.max_memory_usage}GB")
            return False
        
        return True
    
    def cleanup_memory(self):
        """Clean up memory by clearing caches and unused objects."""
        # Clear circuit cache if memory usage is high
        if self.memory_usage > self.config.max_memory_usage * self.config.memory_cleanup_threshold:
            self.circuit_cache.clear()
            logger.info("Cleared circuit cache due to high memory usage")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get current memory status.
        
        Returns:
            Memory status information
        """
        import psutil
        
        memory_info = psutil.virtual_memory()
        
        return {
            "total_memory_gb": memory_info.total / (1024**3),
            "available_memory_gb": memory_info.available / (1024**3),
            "used_memory_gb": memory_info.used / (1024**3),
            "memory_percentage": memory_info.percent,
            "circuit_cache_size": len(self.circuit_cache)
        }


def load_quantum_config(config_path: str = None) -> QuantumConfig:
    """
    Load quantum configuration from file or environment.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Quantum configuration
    """
    config = QuantumConfig()
    
    # Load from environment variables
    config.device = os.getenv("QUANTUM_DEVICE", config.device)
    config.use_gpu = os.getenv("QUANTUM_USE_GPU", "true").lower() == "true"
    config.shots = int(os.getenv("QUANTUM_SHOTS", str(config.shots)))
    config.parallel_execution = os.getenv("QUANTUM_PARALLEL", "true").lower() == "true"
    config.use_caching = os.getenv("QUANTUM_CACHING", "true").lower() == "true"
    config.cache_size = int(os.getenv("QUANTUM_CACHE_SIZE", str(config.cache_size)))
    
    # Load from configuration file if provided
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Update configuration with file values
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_path}: {e}")
    
    return config


def save_quantum_config(config: QuantumConfig, config_path: str):
    """
    Save quantum configuration to file.
    
    Args:
        config: Quantum configuration
        config_path: Path to save configuration
    """
    try:
        import yaml
        
        # Convert dataclass to dictionary
        config_dict = {
            "device": config.device,
            "use_gpu": config.use_gpu,
            "gpu_device": config.gpu_device,
            "fallback_device": config.fallback_device,
            "shots": config.shots,
            "max_shots": config.max_shots,
            "parallel_execution": config.parallel_execution,
            "max_workers": config.max_workers,
            "use_caching": config.use_caching,
            "cache_size": config.cache_size,
            "circuit_compilation": config.circuit_compilation,
            "depth_reduction": config.depth_reduction,
            "max_memory_usage": config.max_memory_usage,
            "memory_cleanup_threshold": config.memory_cleanup_threshold,
            "max_retries": config.max_retries,
            "retry_delay": config.retry_delay,
            "error_threshold": config.error_threshold,
            "log_level": config.log_level,
            "log_quantum_operations": config.log_quantum_operations,
            "log_performance_metrics": config.log_performance_metrics
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test quantum configuration
    config = load_quantum_config()
    print(f"Quantum configuration: {config}")
    
    # Test device manager
    device_manager = QuantumDeviceManager(config)
    print(f"Available devices: {device_manager.available_devices}")
    print(f"Best device: {device_manager.best_device}")
    
    # Test performance configuration
    perf_config = QuantumPerformanceConfig(config)
    opt_config = perf_config.get_optimization_config("aggressive")
    print(f"Optimization configuration: {opt_config}")
    
    # Test memory manager
    memory_manager = QuantumMemoryManager(config)
    memory_status = memory_manager.get_memory_status()
    print(f"Memory status: {memory_status}")
    
    # Test optimal parameters
    optimal_shots = perf_config.get_optimal_shots(circuit_depth=10, accuracy_required=0.95)
    optimal_workers = perf_config.get_optimal_workers(n_circuits=8)
    print(f"Optimal shots: {optimal_shots}")
    print(f"Optimal workers: {optimal_workers}")
    
    # Save configuration
    save_quantum_config(config, "quantum_config.yaml")
    print("Configuration saved to quantum_config.yaml")