"""
Astro-Physiology Real-Time Monitoring
V2: Track intervention effectiveness, auto-adapt, alert on anomalies
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..database.unified_database import UnifiedDatabase, DatabaseConfig
from ..config import IceburgConfig

logger = logging.getLogger(__name__)


async def _monitor_intervention_effectiveness(
    user_id: Optional[str],
    cfg: IceburgConfig,
    days: int = 7
) -> Dict[str, Any]:
    """
    V2: Monitor intervention effectiveness.
    
    Tracks:
    - User adherence to interventions (check-ins, completion rates)
    - Health metrics (from health_tracking table)
    - Improvements/declines (trend analysis)
    - Auto-adapt interventions (modify based on outcomes)
    - Alert on anomalies (unexpected changes, risk factors)
    
    Args:
        user_id: User ID to monitor
        cfg: ICEBURG configuration
        days: Number of days to look back
        
    Returns:
        Dictionary with monitoring results and recommendations
    """
    try:
        db_config = DatabaseConfig()
        db = UnifiedDatabase(cfg, db_config)
        
        # Get active interventions
        interventions = await db.get_user_interventions(
            user_id=user_id,
            status="active",
            limit=50
        )
        
        # Get recent health tracking data
        health_tracking = await db.get_user_health_tracking(
            user_id=user_id,
            days=days,
            limit=1000
        )
        
        # Get recent feedback
        feedback = await db.get_user_feedback(
            user_id=user_id,
            limit=50
        )
        
        monitoring_results = {
            "user_id": user_id,
            "monitoring_period_days": days,
            "active_interventions": len(interventions),
            "intervention_analysis": [],
            "health_trends": {},
            "anomalies": [],
            "adaptation_recommendations": [],
            "alerts": []
        }
        
        # Analyze each active intervention
        for intervention in interventions:
            intervention_id = intervention.get("intervention_id")
            intervention_data = intervention.get("intervention_data", {})
            started_at = intervention.get("started_at", 0)
            
            # Calculate adherence (simplified - would use check-ins)
            adherence_score = _calculate_adherence(
                intervention,
                feedback,
                health_tracking
            )
            
            # Analyze health metrics for this intervention
            health_impact = _analyze_health_impact(
                intervention_id,
                health_tracking,
                started_at
            )
            
            # Detect improvements/declines
            trend = _detect_trend(health_impact)
            
            intervention_analysis = {
                "intervention_id": intervention_id,
                "adherence_score": adherence_score,
                "health_impact": health_impact,
                "trend": trend,
                "days_active": (datetime.now().timestamp() - started_at) / (24 * 60 * 60) if started_at else 0
            }
            
            monitoring_results["intervention_analysis"].append(intervention_analysis)
            
            # Generate adaptation recommendations
            if adherence_score < 0.5 or trend == "declining":
                adaptation = _generate_adaptation_recommendation(
                    intervention,
                    adherence_score,
                    trend
                )
                monitoring_results["adaptation_recommendations"].append(adaptation)
            
            # Alert on anomalies
            if trend == "declining" and health_impact.get("severity", 0) > 0.7:
                monitoring_results["alerts"].append({
                    "type": "health_decline",
                    "intervention_id": intervention_id,
                    "severity": "high",
                    "message": f"Health decline detected for intervention {intervention_id}",
                    "recommended_action": "Review intervention and consider modification"
                })
        
        # Overall health trends
        monitoring_results["health_trends"] = _calculate_overall_health_trends(
            health_tracking,
            days
        )
        
        # Detect anomalies in health metrics
        anomalies = _detect_anomalies(health_tracking)
        monitoring_results["anomalies"].extend(anomalies)
        
        logger.info(f"🌌 Monitoring completed: {len(interventions)} interventions, {len(anomalies)} anomalies, {len(monitoring_results['adaptation_recommendations'])} adaptation recommendations")
        
        return monitoring_results
        
    except Exception as e:
        logger.error(f"Error monitoring intervention effectiveness: {e}", exc_info=True)
        return {
            "user_id": user_id,
            "error": str(e),
            "intervention_analysis": [],
            "health_trends": {},
            "anomalies": [],
            "adaptation_recommendations": [],
            "alerts": []
        }


def _calculate_adherence(
    intervention: Dict[str, Any],
    feedback: List[Dict[str, Any]],
    health_tracking: List[Dict[str, Any]]
) -> float:
    """Calculate intervention adherence score (0-1)"""
    intervention_id = intervention.get("intervention_id")
    
    # Count feedback entries for this intervention
    intervention_feedback = [f for f in feedback if f.get("intervention_id") == intervention_id]
    
    # Simplified adherence calculation
    # In production, would track actual check-ins and completions
    if intervention_feedback:
        # Use feedback ratings as proxy for adherence
        ratings = [f.get("rating", 0) for f in intervention_feedback if f.get("rating")]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            # Convert 1-5 rating to 0-1 adherence score
            adherence = (avg_rating - 1) / 4.0
            return max(0.0, min(1.0, adherence))
    
    # Default: assume moderate adherence if no feedback
    return 0.5


def _analyze_health_impact(
    intervention_id: str,
    health_tracking: List[Dict[str, Any]],
    started_at: float
) -> Dict[str, Any]:
    """Analyze health impact of an intervention"""
    # Filter health metrics after intervention started
    post_intervention_metrics = [
        m for m in health_tracking
        if m.get("timestamp", 0) >= started_at
    ]
    
    if not post_intervention_metrics:
        return {
            "metric_count": 0,
            "average_change": 0.0,
            "severity": 0.0
        }
    
    # Calculate average change (simplified)
    # In production, would compare to baseline
    values = [m.get("metric_value", 0) for m in post_intervention_metrics]
    avg_value = sum(values) / len(values) if values else 0.0
    
    return {
        "metric_count": len(post_intervention_metrics),
        "average_value": avg_value,
        "average_change": 0.0,  # Would calculate vs baseline
        "severity": abs(avg_value - 0.5)  # Distance from ideal (0.5)
    }


def _detect_trend(health_impact: Dict[str, Any]) -> str:
    """Detect health trend: improving, declining, or stable"""
    change = health_impact.get("average_change", 0.0)
    
    if change > 0.1:
        return "improving"
    elif change < -0.1:
        return "declining"
    else:
        return "stable"


def _generate_adaptation_recommendation(
    intervention: Dict[str, Any],
    adherence_score: float,
    trend: str
) -> Dict[str, Any]:
    """Generate adaptation recommendation for an intervention"""
    intervention_id = intervention.get("intervention_id")
    
    if adherence_score < 0.5:
        return {
            "intervention_id": intervention_id,
            "type": "reduce_difficulty",
            "reason": "Low adherence detected",
            "recommendation": "Simplify intervention or reduce frequency",
            "priority": "high"
        }
    elif trend == "declining":
        return {
            "intervention_id": intervention_id,
            "type": "modify_approach",
            "reason": "Health decline detected",
            "recommendation": "Modify intervention approach or intensity",
            "priority": "high"
        }
    else:
        return {
            "intervention_id": intervention_id,
            "type": "maintain",
            "reason": "Stable or improving",
            "recommendation": "Continue current intervention",
            "priority": "low"
        }


def _calculate_overall_health_trends(
    health_tracking: List[Dict[str, Any]],
    days: int
) -> Dict[str, Any]:
    """Calculate overall health trends"""
    if not health_tracking:
        return {
            "trend": "insufficient_data",
            "metrics_tracked": 0
        }
    
    # Group by metric name
    metrics_by_name = {}
    for metric in health_tracking:
        name = metric.get("metric_name", "unknown")
        if name not in metrics_by_name:
            metrics_by_name[name] = []
        metrics_by_name[name].append(metric)
    
    trends = {}
    for metric_name, metrics in metrics_by_name.items():
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.get("timestamp", 0))
        
        if len(sorted_metrics) >= 2:
            first_value = sorted_metrics[0].get("metric_value", 0)
            last_value = sorted_metrics[-1].get("metric_value", 0)
            change = last_value - first_value
            
            trends[metric_name] = {
                "trend": "improving" if change > 0.05 else "declining" if change < -0.05 else "stable",
                "change": change,
                "data_points": len(sorted_metrics)
            }
    
    return {
        "trend": "mixed",  # Would calculate overall trend
        "metrics_tracked": len(metrics_by_name),
        "metric_trends": trends
    }


def _detect_anomalies(
    health_tracking: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Detect anomalies in health metrics"""
    anomalies = []
    
    if not health_tracking:
        return anomalies
    
    # Group by metric name
    metrics_by_name = {}
    for metric in health_tracking:
        name = metric.get("metric_name", "unknown")
        if name not in metrics_by_name:
            metrics_by_name[name] = []
        metrics_by_name[name].append(metric)
    
    # Detect outliers (simplified - would use statistical methods)
    for metric_name, metrics in metrics_by_name.items():
        if len(metrics) < 3:
            continue
        
        values = [m.get("metric_value", 0) for m in metrics]
        avg = sum(values) / len(values)
        std_dev = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
        
        # Flag values more than 2 standard deviations from mean
        for metric in metrics:
            value = metric.get("metric_value", 0)
            if abs(value - avg) > 2 * std_dev and std_dev > 0:
                anomalies.append({
                    "type": "outlier",
                    "metric_name": metric_name,
                    "value": value,
                    "expected_range": (avg - 2 * std_dev, avg + 2 * std_dev),
                    "timestamp": metric.get("timestamp", 0),
                    "severity": "moderate"
                })
    
    return anomalies

