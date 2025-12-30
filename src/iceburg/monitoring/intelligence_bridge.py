"""
Intelligence Bridge Controller
Connects V2 Intelligence System to V10 Finance Dashboard

This bridges the gap between event intelligence and tradeable alpha signals.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..intelligence.multi_source_aggregator import (
    get_intelligence_aggregator,
    SignalPriority
)
from ..intelligence.quant_signal_processor import get_quant_processor
from ..prediction.event_prediction_engine import get_event_prediction_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/finance", tags=["finance-intelligence"])


# Tradeable symbols (expand as needed)
TRADEABLE_SYMBOLS = {
    # Crypto
    'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'MATIC', 'AVAX', 'DOT',
    # Major stocks
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
    # Other
    'SPY', 'QQQ', 'DIA'
}


def has_financial_impact(intelligence) -> bool:
    """Check if intelligence has financial trading implications."""
    # Check if any entities are tradeable symbols
    for entity in intelligence.entities:
        if entity.upper() in TRADEABLE_SYMBOLS:
            return True
    
    # Check narrative for financial keywords
    financial_keywords = [
        'earnings', 'revenue', 'profit', 'stock', 'price', 'market',
        'trading', 'buy', 'sell', 'investment', 'valuation', 'ipo',
        'bankruptcy', 'merger', 'acquisition', 'fed', 'interest rate',
        'inflation', 'recession', 'gdp', 'unemployment'
    ]
    
    narrative_lower = intelligence.narrative.lower()
    return any(keyword in narrative_lower for keyword in financial_keywords)


@router.get("/intelligence-signals")
async def get_finance_intelligence(
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get intelligence signals relevant to trading.
    
    This endpoint is called by V10 Finance Dashboard to get V2 intelligence
    that has tradeable implications.
    
    Returns:
        {
            "intelligence_signals": [...],  # Raw V2 signals with financial relevance
            "alpha_signals": [...],          # Converted to tradeable alpha
            "event_predictions": [...],      # Relevant event predictions
            "summary": {...}
        }
    """
    try:
        aggregator = get_intelligence_aggregator()
        processor = get_quant_processor()
        prediction_engine = get_event_prediction_engine()
        
        # Get intelligence signals
        priority_filter = SignalPriority(priority) if priority else None
        all_signals = aggregator.get_intelligence_feed(
            priority=priority_filter,
            limit=limit * 2  # Get more to filter
        )
        
        # Filter for financial relevance
        financial_signals = []
        for signal in all_signals:
            # Check if any entities are tradeable
            if any(e.upper() in TRADEABLE_SYMBOLS for e in signal.entities):
                financial_signals.append({
                    "signal_id": signal.signal_id,
                    "source_type": signal.source_type.value,
                    "content": signal.content,
                    "priority": signal.priority.value,
                    "confidence": signal.confidence,
                    "entities": signal.entities,
                    "timestamp": signal.timestamp.isoformat()
                })
        
        financial_signals = financial_signals[:limit]
        
        # Get correlated intelligence
        correlations = aggregator.get_correlations(min_confidence=0.5, limit=10)
        
        # Convert to alpha signals
        alpha_signals = []
        for intelligence in correlations:
            if has_financial_impact(intelligence):
                try:
                    alphas = processor.intelligence_to_alpha(intelligence)
                    for alpha in alphas:
                        alpha_signals.append({
                            "signal_id": alpha.signal_id,
                            "symbol": alpha.symbol,
                            "direction": "LONG" if alpha.direction == 1 else "SHORT" if alpha.direction == -1 else "NEUTRAL",
                            "strength": alpha.strength,
                            "confidence": alpha.confidence,
                            "expected_return": alpha.expected_return,
                            "sharpe_estimate": alpha.sharpe_estimate,
                            "position_size": alpha.position_size,
                            "stop_loss": alpha.stop_loss,
                            "take_profit": alpha.take_profit,
                            "time_horizon": alpha.time_horizon,
                            "intelligence_source": intelligence.narrative[:100],
                            "timestamp": alpha.timestamp.isoformat()
                        })
                except Exception as e:
                    logger.warning(f"Failed to convert intelligence to alpha: {e}")
                    continue
        
        # Get relevant event predictions (financial impact)
        event_predictions = []
        for pred in prediction_engine.predictions:
            if pred.event_category.value in ['economic', 'corporate'] or has_financial_impact(pred):
                event_predictions.append({
                    "prediction_id": pred.prediction_id,
                    "category": pred.event_category.value,
                    "description": pred.event_description,
                    "probability": pred.probability,
                    "timeframe": pred.timeframe,
                    "expected_impact": pred.expected_impact
                })
        
        return {
            "intelligence_signals": financial_signals,
            "alpha_signals": alpha_signals,
            "event_predictions": event_predictions[:10],
            "summary": {
                "total_signals": len(financial_signals),
                "total_alpha_signals": len(alpha_signals),
                "total_predictions": len(event_predictions),
                "avg_confidence": sum(s["confidence"] for s in financial_signals) / len(financial_signals) if financial_signals else 0,
                "high_priority_count": len([s for s in financial_signals if s["priority"] in ["critical", "high"]]),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"Finance intelligence error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-alpha-signal")
async def execute_alpha_signal(request_data: Dict[str, Any]):
    """
    Execute alpha signal via RL agent (or manual confirmation).
    
    Request body:
        {
            "signal_id": "alpha_123",
            "symbol": "BTC",
            "direction": "LONG",
            "position_size": 0.05,
            "execution_mode": "auto" | "paper" | "manual"
        }
    """
    try:
        signal_id = request_data["signal_id"]
        symbol = request_data["symbol"]
        direction = request_data["direction"]
        position_size = request_data.get("position_size", 0.05)
        execution_mode = request_data.get("execution_mode", "paper")
        
        # For now, return execution plan
        # TODO: Integrate with actual RL agent and trading execution
        
        return {
            "status": "queued",
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "position_size": position_size,
            "execution_mode": execution_mode,
            "message": f"Signal queued for {execution_mode} execution",
            "estimated_execution_time": "~30 seconds",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except Exception as e:
        logger.error(f"Execute alpha signal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intelligence-summary")
async def get_intelligence_summary():
    """
    Get quick summary of V2 intelligence for V10 dashboard widget.
    
    Returns high-level metrics without full signal details.
    """
    try:
        aggregator = get_intelligence_aggregator()
        processor = get_quant_processor()
        
        # Get recent signals
        recent_signals = aggregator.get_intelligence_feed(limit=50)
        
        # Count by priority
        priority_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for signal in recent_signals:
            priority_counts[signal.priority.value] = priority_counts.get(signal.priority.value, 0) + 1
        
        # Get active alpha signals
        alpha_signals = processor.alpha_signals[-20:]  # Last 20
        
        tradeable_count = len([
            s for s in recent_signals 
            if any(e.upper() in TRADEABLE_SYMBOLS for e in s.entities)
        ])
        
        return {
            "total_signals": len(recent_signals),
            "tradeable_signals": tradeable_count,
            "active_alpha_signals": len(alpha_signals),
            "priority_breakdown": priority_counts,
            "avg_confidence": sum(s.confidence for s in recent_signals) / len(recent_signals) if recent_signals else 0,
            "last_update": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Intelligence summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
