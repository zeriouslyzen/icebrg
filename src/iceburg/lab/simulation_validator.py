"""
ICEBURG Simulation Validator
Validates biological simulations against experimental data and theoretical predictions
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json


@dataclass
class ValidationResult:
    """Result of validating a simulation against experimental data"""
    simulation_type: str
    validation_metric: str  # "accuracy", "correlation", "rmse", etc.
    experimental_value: float
    simulated_value: float
    difference: float
    relative_error: float
    confidence_level: float
    validation_status: str  # "pass", "fail", "partial"
    timestamp: datetime


class SimulationValidator:
    """
    Validates biological simulations against experimental data
    
    Philosophy: Simulations must match reality to be useful
    """
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.experimental_data: Dict[str, Dict[str, float]] = {}
        self._load_experimental_data()
    
    def _load_experimental_data(self):
        """Load experimental data for validation"""
        # Quantum coherence in photosynthesis (Engel et al. 2007, Nature)
        self.experimental_data["quantum_coherence_photosynthesis"] = {
            "coherence_time_ps": 9.0,  # Experimental value
            "energy_transfer_efficiency": 0.95,  # Experimental value
            "temperature_K": 77.0,  # Experimental conditions
            "source": "Engel et al. 2007, Nature"
        }
        
        # Bioelectric signaling (Hodgkin-Huxley model)
        self.experimental_data["bioelectric_signaling"] = {
            "membrane_potential_mV": -70.0,  # Resting potential
            "action_potential_amplitude_mV": 100.0,  # Typical amplitude
            "signal_frequency_Hz": 0.01,  # Typical frequency
            "source": "Hodgkin-Huxley model"
        }
        
        # Quantum entanglement in biological systems
        self.experimental_data["quantum_entanglement_biological"] = {
            "entanglement_probability": 0.8,  # Estimated from literature
            "correlation_function": 0.05,  # Estimated from literature
            "source": "Literature estimates"
        }
        
        # Pancreatic bioelectric-brain synchronization
        self.experimental_data["pancreatic_bioelectric"] = {
            "synchronization_index": 0.6,  # Estimated from literature
            "correlation_coefficient": 0.7,  # Estimated from literature
            "source": "Literature estimates"
        }
    
    def validate_simulation(self, simulation_type: str, 
                          simulation_metrics: Dict[str, float]) -> ValidationResult:
        """Validate simulation results against experimental data"""
        
        if simulation_type not in self.experimental_data:
            return ValidationResult(
                simulation_type=simulation_type,
                validation_metric="unknown",
                experimental_value=0.0,
                simulated_value=0.0,
                difference=0.0,
                relative_error=0.0,
                confidence_level=0.0,
                validation_status="fail",
                timestamp=datetime.utcnow()
            )
        
        experimental = self.experimental_data[simulation_type]
        
        # Find matching metrics
        validation_metrics = []
        for key, exp_value in experimental.items():
            if key == "source" or key == "temperature_K":
                continue
            
            # Find corresponding simulated value
            sim_value = simulation_metrics.get(key, None)
            if sim_value is None:
                continue
            
            # Calculate validation metrics
            difference = abs(sim_value - exp_value)
            relative_error = difference / abs(exp_value) if exp_value != 0 else float('inf')
            
            # Determine validation status
            if relative_error < 0.1:  # Within 10%
                status = "pass"
            elif relative_error < 0.3:  # Within 30%
                status = "partial"
            else:
                status = "fail"
            
            validation_metrics.append({
                "metric": key,
                "experimental": exp_value,
                "simulated": sim_value,
                "difference": difference,
                "relative_error": relative_error,
                "status": status
            })
        
        # Overall validation
        if not validation_metrics:
            return ValidationResult(
                simulation_type=simulation_type,
                validation_metric="no_match",
                experimental_value=0.0,
                simulated_value=0.0,
                difference=0.0,
                relative_error=0.0,
                confidence_level=0.0,
                validation_status="fail",
                timestamp=datetime.utcnow()
            )
        
        # Calculate overall metrics
        avg_relative_error = np.mean([m["relative_error"] for m in validation_metrics])
        pass_count = sum(1 for m in validation_metrics if m["status"] == "pass")
        total_count = len(validation_metrics)
        
        # Overall status
        if pass_count == total_count:
            overall_status = "pass"
        elif pass_count > total_count / 2:
            overall_status = "partial"
        else:
            overall_status = "fail"
        
        # Confidence level (inverse of relative error)
        confidence_level = max(0.0, min(1.0, 1.0 - avg_relative_error))
        
        # Use first metric for result
        first_metric = validation_metrics[0]
        
        result = ValidationResult(
            simulation_type=simulation_type,
            validation_metric=first_metric["metric"],
            experimental_value=first_metric["experimental"],
            simulated_value=first_metric["simulated"],
            difference=first_metric["difference"],
            relative_error=first_metric["relative_error"],
            confidence_level=confidence_level,
            validation_status=overall_status,
            timestamp=datetime.utcnow()
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_multiple_simulations(self, simulation_results: List[Tuple[str, Dict[str, float]]]) -> List[ValidationResult]:
        """Validate multiple simulations"""
        results = []
        for sim_type, sim_metrics in simulation_results:
            result = self.validate_simulation(sim_type, sim_metrics)
            results.append(result)
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.validation_results:
            return {"status": "no_validations", "count": 0}
        
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.validation_status == "pass")
        partial = sum(1 for r in self.validation_results if r.validation_status == "partial")
        failed = sum(1 for r in self.validation_results if r.validation_status == "fail")
        
        avg_confidence = np.mean([r.confidence_level for r in self.validation_results])
        avg_relative_error = np.mean([r.relative_error for r in self.validation_results])
        
        return {
            "status": "validated",
            "total_validations": total,
            "passed": passed,
            "partial": partial,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "average_confidence": avg_confidence,
            "average_relative_error": avg_relative_error
        }
    
    def add_experimental_data(self, simulation_type: str, data: Dict[str, float]):
        """Add experimental data for validation"""
        self.experimental_data[simulation_type] = data
    
    def get_experimental_data(self, simulation_type: str) -> Optional[Dict[str, float]]:
        """Get experimental data for a simulation type"""
        return self.experimental_data.get(simulation_type)


# Global validator instance
_validator: Optional[SimulationValidator] = None

def get_validator() -> SimulationValidator:
    """Get or create the global validator instance"""
    global _validator
    if _validator is None:
        _validator = SimulationValidator()
    return _validator

