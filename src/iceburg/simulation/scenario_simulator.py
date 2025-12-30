"""
Scenario Simulator - Phase 4
Monte Carlo simulation engine for V2 prediction system

Runs massive parallel simulations to generate confidence intervals,
counterfactual timelines, and adversarial scenarios.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
import logging

from ..prediction.event_prediction_engine import EventPrediction

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of scenarios"""
    BASELINE = "baseline"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    BLACK_SWAN = "black_swan"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class SimulationParameters:
    """Parameters for simulation run"""
    n_simulations: int = 10000
    time_horizon_days: int = 365
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    parallel_workers: int = 4
    include_extreme_scenarios: bool = True


@dataclass
class ScenarioOutcome:
    """Single scenario outcome"""
    scenario_id: str
    scenario_type: ScenarioType
    outcome_description: str
    probability: float
    expected_value: float
    timeline: Dict[str, Any]
    key_assumptions: List[str] = field(default_factory=list)


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation"""
    simulation_id: str
    total_runs: int
    scenarios: List[ScenarioOutcome]
    mean_outcome: float
    median_outcome: float
    std_dev: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    percentiles: Dict[int, float]
    extreme_outcomes: Dict[str, ScenarioOutcome]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ScenarioSimulator:
    """
    Massive-scale Monte Carlo simulation engine.
    
    Capabilities:
    - 1M+ simulation runs
    - Parallel execution
    - Counterfactual generation
    - Confidence interval estimation
    - Extreme scenario modeling
    """
    
    def __init__(self):
        self.simulation_history: List[SimulationResults] = []
        logger.info("Scenario Simulator initialized")
    
    def run_monte_carlo(
        self,
        prediction: EventPrediction,
        params: Optional[SimulationParameters] = None
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation for event prediction.
        
        Args:
            prediction: Event prediction to simulate
            params: Simulation parameters
            
        Returns:
            Simulation results with confidence intervals
        """
        if params is None:
            params = SimulationParameters()
        
        logger.info(f"Running {params.n_simulations:,} Monte Carlo simulations...")
        
        # Set random seed for reproducibility
        if params.random_seed:
            np.random.seed(params.random_seed)
        
        # Run simulations in parallel
        outcomes = []
        batch_size = params.n_simulations // params.parallel_workers
        
        with ProcessPoolExecutor(max_workers=params.parallel_workers) as executor:
            futures = []
            for worker_id in range(params.parallel_workers):
                future = executor.submit(
                    self._simulate_batch,
                    prediction,
                    batch_size,
                    worker_id
                )
                futures.append(future)
            
            for future in as_completed(futures):
                batch_outcomes = future.result()
                outcomes.extend(batch_outcomes)
        
        # Calculate statistics
        outcome_values = [o.expected_value for o in outcomes]
        
        mean_outcome = np.mean(outcome_values)
        median_outcome = np.median(outcome_values)
        std_dev = np.std(outcome_values)
        
        # Confidence intervals
        ci_level = params.confidence_level
        lower_percentile = (1 - ci_level) / 2 * 100
        upper_percentile = (1 + ci_level) / 2 * 100
        
        confidence_intervals = {
            f"{int(ci_level * 100)}%": (
                np.percentile(outcome_values, lower_percentile),
                np.percentile(outcome_values, upper_percentile)
            )
        }
        
        # Percentiles
        percentiles = {
            p: np.percentile(outcome_values, p)
            for p in [5, 10, 25, 50, 75, 90, 95]
        }
        
        # Extreme outcomes
        extreme_outcomes = {
            "best_case": max(outcomes, key=lambda x: x.expected_value),
            "worst_case": min(outcomes, key=lambda x: x.expected_value),
            "most_likely": min(outcomes, key=lambda x: abs(x.expected_value - median_outcome))
        }
        
        results = SimulationResults(
            simulation_id=f"sim_{datetime.utcnow().timestamp()}",
            total_runs=len(outcomes),
            scenarios=outcomes[:100],  # Sample for storage
            mean_outcome=mean_outcome,
            median_outcome=median_outcome,
            std_dev=std_dev,
            confidence_intervals=confidence_intervals,
            percentiles=percentiles,
            extreme_outcomes=extreme_outcomes
        )
        
        self.simulation_history.append(results)
        logger.info(f"Simulation complete: mean={mean_outcome:.3f}, std={std_dev:.3f}")
        
        return results
    
    def generate_counterfactuals(
        self,
        base_prediction: EventPrediction,
        intervention: str,
        n_scenarios: int = 1000
    ) -> List[ScenarioOutcome]:
        """
        Generate counterfactual scenarios (what-if analysis).
        
        Example: "What if X intervention had happened instead?"
        
        Args:
            base_prediction: Original prediction
            intervention: Description of intervention/change
            n_scenarios: Number of counterfactual scenarios
            
        Returns:
            List of counterfactual outcomes
        """
        logger.info(f"Generating {n_scenarios} counterfactual scenarios...")
        
        counterfactuals = []
        
        # Base probability
        base_prob = base_prediction.probability
        
        for i in range(n_scenarios):
            # Model intervention effect (simplified)
            intervention_effect = np.random.normal(0.2, 0.1)  # Intervention changes probability
            new_prob = np.clip(base_prob + intervention_effect, 0.0, 1.0)
            
            # Calculate outcome value
            outcome_value = self._calculate_outcome_value(new_prob, base_prediction)
            
            scenario = ScenarioOutcome(
                scenario_id=f"counterfactual_{i}",
                scenario_type=ScenarioType.COUNTERFACTUAL,
                outcome_description=f"If {intervention}, probability becomes {new_prob:.2%}",
                probability=new_prob,
                expected_value=outcome_value,
                timeline={},
                key_assumptions=[intervention]
            )
            counterfactuals.append(scenario)
        
        logger.info(f"Generated {len(counterfactuals)} counterfactual scenarios")
        return counterfactuals
    
    def generate_adversarial_scenarios(
        self,
        base_prediction: EventPrediction,
        adversary_capability: float = 0.8
    ) -> List[ScenarioOutcome]:
        """
        Generate adversarial scenarios (worst-case with capable opponent).
        
        Args:
            base_prediction: Original prediction
            adversary_capability: Capability of adversary (0-1)
            
        Returns:
            Adversarial scenarios
        """
        scenarios = []
        
        # Pessimistic scenario - adversary acts optimally
        pessimistic_prob = base_prediction.probability * (1 + adversary_capability * 0.5)
        pessimistic_prob = min(pessimistic_prob, 0.99)
        
        scenarios.append(ScenarioOutcome(
            scenario_id="adversarial_pessimistic",
            scenario_type=ScenarioType.PESSIMISTIC,
            outcome_description="Adversary acts optimally against our interests",
            probability=pessimistic_prob,
            expected_value=self._calculate_outcome_value(pessimistic_prob, base_prediction) * -1.5,
            timeline={},
            key_assumptions=[f"Adversary capability: {adversary_capability:.0%}"]
        ))
        
        # Black swan adversarial
        black_swan_prob = min(pessimistic_prob * 1.3, 0.99)
        scenarios.append(ScenarioOutcome(
            scenario_id="adversarial_black_swan",
            scenario_type=ScenarioType.BLACK_SWAN,
            outcome_description="Adversary uses unknown capability (black swan)",
            probability=black_swan_prob,
            expected_value=self._calculate_outcome_value(black_swan_prob, base_prediction) * -2.0,
            timeline={},
            key_assumptions=["Unknown adversary capability deployed"]
        ))
        
        return scenarios
    
    def _simulate_batch(
        self,
        prediction: EventPrediction,
        batch_size: int,
        worker_id: int
    ) -> List[ScenarioOutcome]:
        """Simulate batch of scenarios (runs in parallel process)."""
        outcomes = []
        
        for i in range(batch_size):
            # Sample from probability distribution
            prob_sample = np.random.beta(
                prediction.probability * 10,
                (1 - prediction.probability) * 10
            )
            
            # Calculate outcome value
            outcome_value = self._calculate_outcome_value(prob_sample, prediction)
            
            # Determine scenario type
            if prob_sample > 0.9:
                scenario_type = ScenarioType.OPTIMISTIC
            elif prob_sample < 0.1:
                scenario_type = ScenarioType.PESSIMISTIC
            else:
                scenario_type = ScenarioType.BASELINE
            
            outcome = ScenarioOutcome(
                scenario_id=f"sim_{worker_id}_{i}",
                scenario_type=scenario_type,
                outcome_description=f"Simulated outcome with p={prob_sample:.3f}",
                probability=prob_sample,
                expected_value=outcome_value,
                timeline={}
            )
            outcomes.append(outcome)
        
        return outcomes
    
    def _calculate_outcome_value(
        self,
        probability: float,
        prediction: EventPrediction
    ) -> float:
        """Calculate outcome value from probability."""
        # Simple value function: impact * probability
        base_impact = prediction.expected_impact
        value = base_impact * probability
        
        # Add noise
        noise = np.random.normal(0, 0.1)
        return value + noise


# Global simulator instance
_simulator: Optional[ScenarioSimulator] = None


def get_scenario_simulator() -> ScenarioSimulator:
    """Get or create global scenario simulator."""
    global _simulator
    if _simulator is None:
        _simulator = ScenarioSimulator()
    return _simulator
