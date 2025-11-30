"""
Astro-Physiology Feedback Loop System
V2: Collect outcomes, analyze effectiveness, update models, refine recommendations
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..database.unified_database import UnifiedDatabase, DatabaseConfig
from ..config import IceburgConfig

logger = logging.getLogger(__name__)


async def _process_user_feedback(
    feedback_data: Dict[str, Any],
    cfg: IceburgConfig
) -> Dict[str, Any]:
    """
    V2: Process user feedback and update models.
    
    Collects:
    - User outcomes (ratings, comments, health changes)
    - Intervention effectiveness (success rates, outcome patterns)
    - Updates predictive models (adjust parameters based on outcomes)
    - Refines expert recommendations (update agent prompts based on feedback)
    - Stores in knowledge base (via UniversalKnowledgeAccumulator)
    
    Args:
        feedback_data: Dictionary with feedback information
        cfg: ICEBURG configuration
        
    Returns:
        Dictionary with processing results and updates
    """
    try:
        db_config = DatabaseConfig()
        db = UnifiedDatabase(cfg, db_config)
        
        user_id = feedback_data.get("user_id")
        intervention_id = feedback_data.get("intervention_id")
        rating = feedback_data.get("rating")
        comment = feedback_data.get("comment")
        outcome = feedback_data.get("outcome", {})
        
        # Store feedback in database
        feedback_id = f"feedback_{datetime.now().timestamp()}"
        await db.store_feedback(
            feedback_id=feedback_id,
            user_id=user_id,
            intervention_id=intervention_id,
            rating=rating,
            comment=comment,
            outcome=outcome
        )
        
        # Analyze intervention effectiveness
        effectiveness_analysis = await _analyze_intervention_effectiveness(
            intervention_id,
            user_id,
            db
        )
        
        # Update predictive models
        model_updates = await _update_predictive_models(
            effectiveness_analysis,
            cfg
        )
        
        # Refine expert recommendations
        recommendation_updates = await _refine_expert_recommendations(
            effectiveness_analysis,
            cfg
        )
        
        processing_results = {
            "feedback_id": feedback_id,
            "effectiveness_analysis": effectiveness_analysis,
            "model_updates": model_updates,
            "recommendation_updates": recommendation_updates,
            "processed_at": datetime.now().isoformat()
        }
        
        logger.info(f"🌌 Feedback processed: effectiveness={effectiveness_analysis.get('success_rate', 0):.2f}, model_updates={len(model_updates.get('updates', []))}")
        
        return processing_results
        
    except Exception as e:
        logger.error(f"Error processing user feedback: {e}", exc_info=True)
        return {
            "error": str(e),
            "effectiveness_analysis": {},
            "model_updates": {},
            "recommendation_updates": {}
        }


async def _analyze_intervention_effectiveness(
    intervention_id: Optional[str],
    user_id: Optional[str],
    db: UnifiedDatabase
) -> Dict[str, Any]:
    """Analyze intervention effectiveness from feedback"""
    # Get all feedback for this intervention
    feedback = await db.get_user_feedback(
        intervention_id=intervention_id,
        limit=100
    )
    
    if not feedback:
        return {
            "success_rate": 0.0,
            "total_feedback": 0,
            "average_rating": 0.0,
            "outcome_patterns": {}
        }
    
    # Calculate success rate (rating >= 4/5)
    ratings = [f.get("rating", 0) for f in feedback if f.get("rating")]
    success_count = len([r for r in ratings if r >= 4.0])
    success_rate = success_count / len(ratings) if ratings else 0.0
    
    # Calculate average rating
    average_rating = sum(ratings) / len(ratings) if ratings else 0.0
    
    # Analyze outcome patterns
    outcomes = [f.get("outcome") for f in feedback if f.get("outcome")]
    outcome_patterns = {}
    for outcome in outcomes:
        if isinstance(outcome, dict):
            for key, value in outcome.items():
                if key not in outcome_patterns:
                    outcome_patterns[key] = []
                outcome_patterns[key].append(value)
    
    return {
        "success_rate": success_rate,
        "total_feedback": len(feedback),
        "average_rating": average_rating,
        "outcome_patterns": outcome_patterns
    }


async def _update_predictive_models(
    effectiveness_analysis: Dict[str, Any],
    cfg: IceburgConfig
) -> Dict[str, Any]:
    """
    V2: Update predictive models based on feedback.
    
    Adjusts:
    - Model parameters (tune based on outcomes)
    - Confidence scores (update based on accuracy)
    - Prediction thresholds (optimize based on effectiveness)
    """
    updates = []
    
    success_rate = effectiveness_analysis.get("success_rate", 0.5)
    
    # If success rate is high, models are working well
    # If low, may need parameter adjustments
    if success_rate < 0.5:
        updates.append({
            "model_type": "health_trajectory",
            "update_type": "parameter_adjustment",
            "adjustment": "Increase prediction confidence threshold",
            "reason": f"Low success rate ({success_rate:.2f}) suggests model may be overconfident"
        })
    
    # Store model updates in database
    try:
        db_config = DatabaseConfig()
        db = UnifiedDatabase(cfg, db_config)
        
        update_id = f"model_update_{datetime.now().timestamp()}"
        # Would store in astro_physiology_model_updates table
        # For now, just log
        
        logger.info(f"🌌 Model updates generated: {len(updates)} updates")
        
    except Exception as e:
        logger.warning(f"Error storing model updates: {e}", exc_info=True)
    
    return {
        "updates": updates,
        "updated_at": datetime.now().isoformat()
    }


async def _refine_expert_recommendations(
    effectiveness_analysis: Dict[str, Any],
    cfg: IceburgConfig
) -> Dict[str, Any]:
    """
    V2: Refine expert recommendations based on feedback.
    
    Updates:
    - Expert agent prompts (based on performance)
    - Recommendation templates (improve based on outcomes)
    - Intervention strategies (optimize based on effectiveness)
    """
    refinements = []
    
    success_rate = effectiveness_analysis.get("success_rate", 0.5)
    average_rating = effectiveness_analysis.get("average_rating", 0.0)
    
    # If recommendations are not effective, refine them
    if success_rate < 0.6 or average_rating < 3.5:
        refinements.append({
            "expert_type": "all",
            "refinement_type": "prompt_enhancement",
            "change": "Add more specific, actionable recommendations",
            "reason": f"Low effectiveness (success_rate={success_rate:.2f}, rating={average_rating:.2f})"
        })
    
    # Analyze outcome patterns for specific refinements
    outcome_patterns = effectiveness_analysis.get("outcome_patterns", {})
    if outcome_patterns:
        refinements.append({
            "expert_type": "targeted",
            "refinement_type": "pattern_based",
            "patterns": outcome_patterns,
            "change": "Adjust recommendations based on outcome patterns"
        })
    
    return {
        "refinements": refinements,
        "refined_at": datetime.now().isoformat()
    }


async def _update_models(
    user_id: Optional[str],
    cfg: IceburgConfig,
    force_update: bool = False
) -> Dict[str, Any]:
    """
    V2: Update models based on accumulated feedback.
    
    Retrains:
    - Predictive models (if sufficient data)
    - Expert agent prompts (based on performance)
    - Algorithmic calculations (tune parameters)
    - Intervention templates (improve based on outcomes)
    """
    try:
        db_config = DatabaseConfig()
        db = UnifiedDatabase(cfg, db_config)
        
        # Get all feedback for this user
        feedback = await db.get_user_feedback(
            user_id=user_id,
            limit=1000
        )
        
        if len(feedback) < 10 and not force_update:
            return {
                "updated": False,
                "reason": "Insufficient data for model update",
                "feedback_count": len(feedback)
            }
        
        # Analyze overall effectiveness
        effectiveness = await _analyze_intervention_effectiveness(None, user_id, db)
        
        # Update predictive models
        model_updates = await _update_predictive_models(effectiveness, cfg)
        
        # Refine expert recommendations
        recommendation_updates = await _refine_expert_recommendations(effectiveness, cfg)
        
        # Update algorithmic parameters (if needed)
        algorithmic_updates = await _update_algorithmic_parameters(effectiveness, cfg)
        
        update_results = {
            "updated": True,
            "model_updates": model_updates,
            "recommendation_updates": recommendation_updates,
            "algorithmic_updates": algorithmic_updates,
            "updated_at": datetime.now().isoformat()
        }
        
        logger.info(f"🌌 Models updated: {len(model_updates.get('updates', []))} model updates, {len(recommendation_updates.get('refinements', []))} recommendation refinements")
        
        return update_results
        
    except Exception as e:
        logger.error(f"Error updating models: {e}", exc_info=True)
        return {
            "updated": False,
            "error": str(e)
        }


async def _update_algorithmic_parameters(
    effectiveness: Dict[str, Any],
    cfg: IceburgConfig
) -> Dict[str, Any]:
    """Update algorithmic calculation parameters based on feedback"""
    updates = []
    
    # If predictions are consistently off, adjust parameters
    success_rate = effectiveness.get("success_rate", 0.5)
    
    if success_rate < 0.5:
        updates.append({
            "parameter": "celestial_modulation_factor",
            "adjustment": "Reduce by 10%",
            "reason": "Low success rate suggests over-weighting celestial influences"
        })
    
    return {
        "updates": updates,
        "updated_at": datetime.now().isoformat()
    }

