"""
V2 Simulation & OpSec API Controller
Exposes simulation and security capabilities
"""

from fastapi import APIRouter, HTTPException, Query, Header
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..simulation.scenario_simulator import (
    get_scenario_simulator,
    SimulationParameters,
    ScenarioType
)
from ..opsec.prediction_opsec import (
    get_prediction_opsec,
    SecurityLevel
)
from ..prediction.event_prediction_engine import get_event_prediction_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/simulation", tags=["v2-simulation"])
opsec_router = APIRouter(prefix="/api/v2/opsec", tags=["v2-opsec"])


# ============================================
# SIMULATION ENDPOINTS
# ============================================

@router.post("/monte-carlo")
async def run_monte_carlo(request_data: Dict[str, Any]):
    """
    Run Monte Carlo simulation.
    
    Request body:
        {
            "prediction_id": "geo_123",
            "n_simulations": 100000,
            "confidence_level": 0.95,
            "parallel_workers": 4
        }
        
    Returns:
        Simulation results with confidence intervals
    """
    try:
        simulator = get_scenario_simulator()
        engine = get_event_prediction_engine()
        
        prediction_id = request_data.get("prediction_id")
        if not prediction_id:
            raise ValueError("prediction_id required")
        
        # Find prediction
        prediction = next(
            (p for p in engine.predictions if p.prediction_id == prediction_id),
            None
        )
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Build parameters
        params = SimulationParameters(
            n_simulations=request_data.get("n_simulations", 10000),
            confidence_level=request_data.get("confidence_level", 0.95),
            parallel_workers=request_data.get("parallel_workers", 4),
            random_seed=request_data.get("random_seed")
        )
        
        # Run simulation
        results = simulator.run_monte_carlo(prediction, params)
        
        return {
            "simulation_id": results.simulation_id,
            "total_runs": results.total_runs,
            "mean_outcome": results.mean_outcome,
            "median_outcome": results.median_outcome,
            "std_dev": results.std_dev,
            "confidence_intervals": {
                k: {"lower": v[0], "upper": v[1]}
                for k, v in results.confidence_intervals.items()
            },
            "percentiles": results.percentiles,
            "extreme_outcomes": {
                k: {
                    "description": v.outcome_description,
                    "probability": v.probability,
                    "expected_value": v.expected_value
                }
                for k, v in results.extreme_outcomes.items()
            }
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Monte Carlo error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counterfactual")
async def generate_counterfactual(request_data: Dict[str, Any]):
    """
    Generate counterfactual scenarios.
    
    Request body:
        {
            "prediction_id": "geo_123",
            "intervention": "Diplomatic intervention by UN",
            "n_scenarios": 1000
        }
    """
    try:
        simulator = get_scenario_simulator()
        engine = get_event_prediction_engine()
        
        prediction_id = request_data["prediction_id"]
        intervention = request_data["intervention"]
        n_scenarios = request_data.get("n_scenarios", 1000)
        
        # Find prediction
        prediction = next(
            (p for p in engine.predictions if p.prediction_id == prediction_id),
            None
        )
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Generate counterfactuals
        scenarios = simulator.generate_counterfactuals(
            prediction,
            intervention,
            n_scenarios
        )
        
        # Calculate statistics
        probs = [s.probability for s in scenarios]
        
        return {
            "intervention": intervention,
            "scenario_count": len(scenarios),
            "mean_probability": sum(probs) / len(probs) if probs else 0,
            "probability_range": {
                "min": min(probs) if probs else 0,
                "max": max(probs) if probs else 0
            },
            "sample_scenarios": [
                {
                    "description": s.outcome_description,
                    "probability": s.probability,
                    "expected_value": s.expected_value
                }
                for s in scenarios[:10]
            ]
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        logger.error(f"Counterfactual generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adversarial-scenarios")
async def generate_adversarial_scenarios(request_data: Dict[str, Any]):
    """
    Generate adversarial scenarios (worst-case with capable opponent).
    """
    try:
        simulator = get_scenario_simulator()
        engine = get_event_prediction_engine()
        
        prediction_id = request_data["prediction_id"]
        adversary_capability = request_data.get("adversary_capability", 0.8)
        
        prediction = next(
            (p for p in engine.predictions if p.prediction_id == prediction_id),
            None
        )
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        scenarios = simulator.generate_adversarial_scenarios(
            prediction,
            adversary_capability
        )
        
        return {
            "adversary_capability": adversary_capability,
            "scenarios": [
                {
                    "type": s.scenario_type.value,
                    "description": s.outcome_description,
                    "probability": s.probability,
                    "expected_value": s.expected_value,
                    "assumptions": s.key_assumptions
                }
                for s in scenarios
            ]
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        logger.error(f"Adversarial scenario error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# OPSEC ENDPOINTS
# ============================================

@opsec_router.post("/encrypt-prediction")
async def encrypt_prediction(
    request_data: Dict[str, Any],
    x_api_key: Optional[str] = Header(None)
):
    """
    Encrypt sensitive prediction.
    
    Request body:
        {
            "prediction_data": "Classified prediction content",
            "security_level": "secret",
            "authorized_entities": ["entity1", "entity2"]
        }
    """
    try:
        opsec = get_prediction_opsec()
        
        prediction_data = request_data["prediction_data"]
        security_level_str = request_data.get("security_level", "confidential")
        security_level = SecurityLevel(security_level_str)
        authorized_entities = request_data.get("authorized_entities", [])
        
        encrypted = opsec.encrypt_sensitive_prediction(
            prediction_data,
            security_level,
            authorized_entities
        )
        
        return {
            "prediction_id": encrypted.prediction_id,
            "encryption_method": encrypted.encryption_method,
            "security_level": encrypted.security_level.value,
            "access_control": encrypted.access_control,
            "timestamp": encrypted.timestamp.isoformat()
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@opsec_router.post("/zero-knowledge-proof")
async def generate_zkp(request_data: Dict[str, Any]):
    """
    Generate zero-knowledge proof of prediction.
    
    Allows proving prediction was made without revealing content.
    """
    try:
        opsec = get_prediction_opsec()
        
        prediction_data = request_data["prediction_data"]
        secret_value = request_data.get("secret_value")
        
        zkp = opsec.generate_zero_knowledge_proof(prediction_data, secret_value)
        
        return {
            "proof_id": zkp.proof_id,
            "commitment": zkp.commitment,
            "challenge": zkp.challenge,
            "response": zkp.response,
            "timestamp": zkp.timestamp.isoformat()
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        logger.error(f"ZKP generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@opsec_router.post("/verify-proof")
async def verify_zkp(request_data: Dict[str, Any]):
    """Verify zero-knowledge proof."""
    try:
        opsec = get_prediction_opsec()
        
        proof_id = request_data["proof_id"]
        claimed_prediction = request_data["claimed_prediction"]
        secret_value = request_data.get("secret_value")
        
        valid = opsec.verify_zero_knowledge_proof(
            proof_id,
            claimed_prediction,
            secret_value
        )
        
        return {
            "proof_id": proof_id,
            "valid":  valid
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        logger.error(f"ZKP verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@opsec_router.post("/detect-surveillance")
async def detect_surveillance(request_data: Dict[str, Any]):
    """
    Detect potential counter-surveillance.
    
    Analyzes access patterns for anomalies.
    """
    try:
        opsec = get_prediction_opsec()
        
        request_metadata = request_data.get("metadata", {})
        
        detection = opsec.detect_counter_surveillance(request_metadata)
        
        return detection
    
    except Exception as e:
        logger.error(f"Surveillance detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
