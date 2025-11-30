"""
ICEBURG Stochastic Simulator
Enhanced randomness modeling for biological systems using Monte Carlo and stochastic processes
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StochasticSimulationResult:
    """Result of stochastic simulation"""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    percentiles: Dict[str, float]
    n_samples: int
    distribution_type: str


class StochasticSimulator:
    """
    Enhanced randomness modeling for biological systems
    
    Philosophy: Biological systems are stochastic - we need to model randomness properly
    """
    
    def __init__(self):
        self.simulation_counter = 0
    
    def monte_carlo_simulation(self, simulation_func, parameters: Dict[str, Any], 
                            n_samples: int = 1000, 
                            random_seed: Optional[int] = None) -> StochasticSimulationResult:
        """
        Run Monte Carlo simulation
        
        Args:
            simulation_func: Function that takes parameters and returns a value
            parameters: Base parameters for simulation
            n_samples: Number of Monte Carlo samples
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        results = []
        
        for i in range(n_samples):
            # Add random variation to parameters
            varied_params = self._add_random_variation(parameters)
            
            # Run simulation
            result = simulation_func(varied_params)
            
            # Extract numeric value if result is dict
            if isinstance(result, dict):
                # Get first numeric value
                value = next((v for v in result.values() if isinstance(v, (int, float))), 0.0)
            elif isinstance(result, (int, float)):
                value = result
            else:
                value = 0.0
            
            results.append(value)
        
        # Calculate statistics
        results_array = np.array(results)
        mean = np.mean(results_array)
        std = np.std(results_array)
        
        # Confidence interval (95%)
        confidence_interval = (
            np.percentile(results_array, 2.5),
            np.percentile(results_array, 97.5)
        )
        
        # Percentiles
        percentiles = {
            "5th": np.percentile(results_array, 5),
            "25th": np.percentile(results_array, 25),
            "50th": np.percentile(results_array, 50),
            "75th": np.percentile(results_array, 75),
            "95th": np.percentile(results_array, 95)
        }
        
        return StochasticSimulationResult(
            mean=mean,
            std=std,
            confidence_interval=confidence_interval,
            percentiles=percentiles,
            n_samples=n_samples,
            distribution_type="monte_carlo"
        )
    
    def _add_random_variation(self, parameters: Dict[str, Any], 
                            variation_level: float = 0.1) -> Dict[str, Any]:
        """Add random variation to parameters"""
        varied = {}
        
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                # Add Gaussian noise
                noise = np.random.normal(0, abs(value) * variation_level)
                varied[key] = value + noise
            elif isinstance(value, list):
                # Add noise to each element
                varied[key] = [v + np.random.normal(0, abs(v) * variation_level) 
                              if isinstance(v, (int, float)) else v 
                              for v in value]
            else:
                varied[key] = value
        
        return varied
    
    def stochastic_process_simulation(self, process_type: str, 
                                    parameters: Dict[str, Any],
                                    n_steps: int = 1000,
                                    dt: float = 0.01) -> np.ndarray:
        """
        Simulate stochastic processes (random walks, Ornstein-Uhlenbeck, etc.)
        
        Args:
            process_type: Type of stochastic process ("random_walk", "ornstein_uhlenbeck", "geometric_brownian")
            parameters: Process parameters
            n_steps: Number of time steps
            dt: Time step size
        """
        if process_type == "random_walk":
            return self._random_walk(parameters, n_steps, dt)
        elif process_type == "ornstein_uhlenbeck":
            return self._ornstein_uhlenbeck(parameters, n_steps, dt)
        elif process_type == "geometric_brownian":
            return self._geometric_brownian(parameters, n_steps, dt)
        else:
            raise ValueError(f"Unknown process type: {process_type}")
    
    def _random_walk(self, parameters: Dict[str, Any], n_steps: int, dt: float) -> np.ndarray:
        """Random walk process"""
        initial_value = parameters.get("initial_value", 0.0)
        drift = parameters.get("drift", 0.0)
        volatility = parameters.get("volatility", 1.0)
        
        # Generate random increments
        increments = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_steps)
        
        # Cumulative sum
        process = np.cumsum(increments) + initial_value
        
        return process
    
    def _ornstein_uhlenbeck(self, parameters: Dict[str, Any], n_steps: int, dt: float) -> np.ndarray:
        """Ornstein-Uhlenbeck process (mean-reverting)"""
        initial_value = parameters.get("initial_value", 0.0)
        mean = parameters.get("mean", 0.0)
        theta = parameters.get("theta", 1.0)  # Mean reversion speed
        sigma = parameters.get("sigma", 1.0)  # Volatility
        
        process = np.zeros(n_steps)
        process[0] = initial_value
        
        for i in range(1, n_steps):
            # Mean-reverting drift
            drift = theta * (mean - process[i-1]) * dt
            # Random noise
            noise = sigma * np.sqrt(dt) * np.random.normal(0, 1)
            process[i] = process[i-1] + drift + noise
        
        return process
    
    def _geometric_brownian(self, parameters: Dict[str, Any], n_steps: int, dt: float) -> np.ndarray:
        """Geometric Brownian motion"""
        initial_value = parameters.get("initial_value", 1.0)
        mu = parameters.get("mu", 0.0)  # Drift
        sigma = parameters.get("sigma", 1.0)  # Volatility
        
        process = np.zeros(n_steps)
        process[0] = initial_value
        
        for i in range(1, n_steps):
            # Geometric Brownian motion
            process[i] = process[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + 
                                              sigma * np.sqrt(dt) * np.random.normal(0, 1))
        
        return process
    
    def biological_stochastic_simulation(self, biological_process: str,
                                        parameters: Dict[str, Any],
                                        n_samples: int = 1000) -> StochasticSimulationResult:
        """
        Simulate biological stochastic processes
        
        Args:
            biological_process: Type of biological process ("gene_expression", "protein_folding", "ion_channel", etc.)
            parameters: Process parameters
            n_samples: Number of samples
        """
        if biological_process == "gene_expression":
            return self._simulate_gene_expression(parameters, n_samples)
        elif biological_process == "protein_folding":
            return self._simulate_protein_folding(parameters, n_samples)
        elif biological_process == "ion_channel":
            return self._simulate_ion_channel(parameters, n_samples)
        else:
            raise ValueError(f"Unknown biological process: {biological_process}")
    
    def _simulate_gene_expression(self, parameters: Dict[str, Any], n_samples: int) -> StochasticSimulationResult:
        """Simulate gene expression (Poisson-like process)"""
        mean_expression = parameters.get("mean_expression", 100.0)
        noise_level = parameters.get("noise_level", 0.2)
        
        # Gene expression follows approximately Poisson distribution
        results = np.random.poisson(mean_expression * (1 + np.random.normal(0, noise_level, n_samples)))
        
        mean = np.mean(results)
        std = np.std(results)
        confidence_interval = (np.percentile(results, 2.5), np.percentile(results, 97.5))
        
        return StochasticSimulationResult(
            mean=mean,
            std=std,
            confidence_interval=confidence_interval,
            percentiles={
                "5th": np.percentile(results, 5),
                "25th": np.percentile(results, 25),
                "50th": np.percentile(results, 50),
                "75th": np.percentile(results, 75),
                "95th": np.percentile(results, 95)
            },
            n_samples=n_samples,
            distribution_type="gene_expression"
        )
    
    def _simulate_protein_folding(self, parameters: Dict[str, Any], n_samples: int) -> StochasticSimulationResult:
        """Simulate protein folding (stochastic process)"""
        base_efficiency = parameters.get("base_efficiency", 0.9)
        temperature = parameters.get("temperature", 300.0)
        noise_level = parameters.get("noise_level", 0.1)
        
        # Protein folding efficiency with temperature-dependent noise
        thermal_noise = np.random.normal(0, noise_level * (temperature / 300.0), n_samples)
        results = base_efficiency * (1 + thermal_noise)
        results = np.clip(results, 0.0, 1.0)  # Clamp to [0, 1]
        
        mean = np.mean(results)
        std = np.std(results)
        confidence_interval = (np.percentile(results, 2.5), np.percentile(results, 97.5))
        
        return StochasticSimulationResult(
            mean=mean,
            std=std,
            confidence_interval=confidence_interval,
            percentiles={
                "5th": np.percentile(results, 5),
                "25th": np.percentile(results, 25),
                "50th": np.percentile(results, 50),
                "75th": np.percentile(results, 75),
                "95th": np.percentile(results, 95)
            },
            n_samples=n_samples,
            distribution_type="protein_folding"
        )
    
    def _simulate_ion_channel(self, parameters: Dict[str, Any], n_samples: int) -> StochasticSimulationResult:
        """Simulate ion channel gating (stochastic opening/closing)"""
        open_probability = parameters.get("open_probability", 0.5)
        noise_level = parameters.get("noise_level", 0.1)
        
        # Ion channel gating follows binomial distribution
        results = np.random.binomial(1, open_probability * (1 + np.random.normal(0, noise_level, n_samples)), n_samples)
        results = np.clip(results, 0.0, 1.0)
        
        mean = np.mean(results)
        std = np.std(results)
        confidence_interval = (np.percentile(results, 2.5), np.percentile(results, 97.5))
        
        return StochasticSimulationResult(
            mean=mean,
            std=std,
            confidence_interval=confidence_interval,
            percentiles={
                "5th": np.percentile(results, 5),
                "25th": np.percentile(results, 25),
                "50th": np.percentile(results, 50),
                "75th": np.percentile(results, 75),
                "95th": np.percentile(results, 95)
            },
            n_samples=n_samples,
            distribution_type="ion_channel"
        )


# Global simulator instance
_simulator: Optional[StochasticSimulator] = None

def get_stochastic_simulator() -> StochasticSimulator:
    """Get or create the global stochastic simulator instance"""
    global _simulator
    if _simulator is None:
        _simulator = StochasticSimulator()
    return _simulator

