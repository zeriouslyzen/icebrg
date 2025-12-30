"""
V2 Prediction API Controller
Exposes event prediction capabilities
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..prediction.event_prediction_engine import (
    get_event_prediction_engine,
    EventCategory,
    EventPrediction,
    BlackSwanAlert
)
from ..intelligence.multi_source_aggregator import get_intelligence_aggregator
from ..config import IceburgConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/prediction", tags=["v2-prediction"])


@router.post("/event")
async def predict_event(request_data: Dict[str, Any]):
    """
    Predict specific event.
    
    Request body:
        {
            "category": "geopolitical",  // or "economic", "technological", etc.
            "scenario": "US-China trade conflict escalation",
            "context": {...}  // optional context
        }
        
    Returns:
        Event prediction with probability, timeframe, and impact
    """
    try:
        engine = get_event_prediction_engine()
        
        category_str = request_data.get("category", "geopolitical")
        scenario = request_data["scenario"]
        context = request_data.get("context", {})
        
        # Get relevant intelligence
        aggregator = get_intelligence_aggregator()
        correlations = aggregator.get_correlations(min_confidence=0.5, limit=10)
        intelligence = correlations[0] if correlations else None
        
        # Predict based on category
        if category_str == "geopolitical":
            prediction = engine.predict_geopolitical_event(scenario, intelligence)
        elif category_str == "economic":
            indicators = context.get("indicators", {})
            market = context.get("market", "global")
            prediction = engine.predict_economic_regime_change(market, indicators)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported category: {category_str}")
        
        return {
            "prediction_id": prediction.prediction_id,
            "event_category": prediction.event_category.value,
            "event_description": prediction.event_description,
            "probability": prediction.probability,
            "timeframe": prediction.timeframe,
            "expected_impact": prediction.expected_impact,
            "confidence": prediction.confidence,
            "key_indicators": prediction.key_indicators,
            "cascade_effects": prediction.cascade_effects,
            "historical_precedents": prediction.historical_precedents,
            "timestamp": prediction.timestamp.isoformat()
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Event prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenarios/{event_id}")
async def get_event_scenarios(event_id: str):
    """
    Get scenario simulations for predicted event.
    
    Returns:
        Multiple scenario outcomes (best/worst/likely case)
    """
    try:
        engine = get_event_prediction_engine()
        
        # Find prediction
        prediction = next(
            (p for p in engine.predictions if p.prediction_id == event_id),
            None
        )
        
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        # Generate scenarios (simplified - would use simulation engine)
        scenarios = [
            {
                "scenario_type": "best_case",
                "probability": 0.2,
                "outcome": "Conflict avoided, diplomatic resolution",
                "impact_reduction": 0.7
            },
            {
                "scenario_type": "most_likely",
                "probability": 0.6,
                "outcome": prediction.event_description,
                "impact": prediction.expected_impact
            },
            {
                "scenario_type": "worst_case",
                "probability": 0.2,
                "outcome": "Escalation beyond initial prediction",
                "impact_multiplier": 1.5
            }
        ]
        
        return {
            "event_id": event_id,
            "base_prediction": {
                "description": prediction.event_description,
                "probability": prediction.probability,
                "impact": prediction.expected_impact
            },
            "scenarios": scenarios
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scenario generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adversarial")
async def model_adversary(request_data: Dict[str, Any]):
    """
    Model adversarial behavior.
    
    Request body:
        {
            "actor": "China",
            "context": {"situation": "Taiwan tensions"}
        }
        
    Returns:
        Predicted strategies and counter-moves
    """
    try:
        engine = get_event_prediction_engine()
        
        actor = request_data["actor"]
        context = request_data.get("context", {})
        
        analysis = engine.adversarial_modeling(actor, context)
        
        return analysis
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Adversarial modeling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blackswan/radar")
async def get_black_swan_radar(
    domain: str = Query("finance", description="Domain to monitor"),
    observation_days: int = Query(90, ge=7, le=365)
):
    """
    Get black swan early warning radar.
    
    Returns:
        Potential black swan alerts
    """
    try:
        engine = get_event_prediction_engine()
        
        alerts = engine.detect_black_swan(domain, observation_days)
        
        alerts_data = [
            {
                "alert_id": a.alert_id,
                "signal_strength": a.signal_strength,
                "emergence_score": a.emergence_score,
                "description": a.description,
                "early_warning_indicators": a.early_warning_indicators,
                "probability_range": {
                    "min": a.probability_range[0],
                    "max": a.probability_range[1]
                },
                "timestamp": a.timestamp.isoformat()
            }
            for a in alerts
        ]
        
        return {
            "domain": domain,
            "observation_period_days": observation_days,
            "alerts": alerts_data,
            "alert_count": len(alerts_data)
        }
    
    except Exception as e:
        logger.error(f"Black swan radar error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_predictions(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_probability: float = Query(0.3, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500)
):
    """
    Get all event predictions.
    
    Returns:
        List of predicted events
    """
    try:
        engine = get_event_prediction_engine()
        
        category_filter = None
        if category:
            try:
                category_filter = EventCategory(category.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        predictions = engine.get_predictions(
            category=category_filter,
            min_probability=min_probability,
            limit=limit
        )
        
        predictions_data = [
            {
                "prediction_id": p.prediction_id,
                "category": p.event_category.value,
                "description": p.event_description,
                "probability": p.probability,
                "timeframe": p.timeframe,
                "impact": p.expected_impact,
                "confidence": p.confidence,
                "timestamp": p.timestamp.isoformat()
            }
            for p in predictions
        ]
        
        return {
            "predictions": predictions_data,
            "count": len(predictions_data)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get predictions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
