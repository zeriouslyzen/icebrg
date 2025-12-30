"""
V2 Intelligence API Controller
Exposes intelligence aggregation capabilities via REST API
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..intelligence.multi_source_aggregator import (
    get_intelligence_aggregator,
    IntelligenceSource,
    SignalPriority,
    IntelligenceSignal,
    CorrelatedIntelligence
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/intelligence", tags=["v2-intelligence"])


@router.get("/feed")
async def get_intelligence_feed(
    priority: Optional[str] = Query(None, description="Filter by priority: critical/high/medium/low"),
    limit: int = Query(100, ge=1, le=1000, description="Max signals to return")
):
    """
    Get real-time intelligence feed.
    
    Returns:
        {
            "timestamp": "2025-12-29T15:30:00Z",
            "signals": [
                {
                    "signal_id": "sig_123",
                    "source_type": "osint",
                    "content": "...",
                    "priority": "high",
                    "confidence": 0.85,
                    "entities": ["AAPL", "Tim Cook"],
                    ...
                },
                ...
            ],
            "count": 42
        }
    """
    try:
        aggregator = get_intelligence_aggregator()
        
        # Parse priority filter
        priority_filter = None
        if priority:
            try:
                priority_filter = SignalPriority(priority.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        # Get signals
        signals = aggregator.get_intelligence_feed(priority=priority_filter, limit=limit)
        
        # Convert to dict format
        signals_data = [
            {
                "signal_id": s.signal_id,
                "source_type": s.source_type.value,
                "source_name": s.source_name,
                "content": s.content,
                "timestamp": s.timestamp.isoformat(),
                "priority": s.priority.value,
                "confidence": s.confidence,
                "tags": s.tags,
                "entities": s.entities,
                "metadata": s.metadata
            }
            for s in signals
        ]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "signals": signals_data,
            "count": len(signals_data)
        }
    
    except Exception as e:
        logger.error(f"Intelligence feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_type}")
async def get_intelligence_by_source(
    source_type: str,
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Get intelligence filtered by source type.
    
    Args:
        source_type: Source type (osint, corpint, darkint, netint, geoint, psyint)
        limit: Max signals to return
        
    Returns:
        Intelligence signals from specific source
    """
    try:
        # Validate source type
        try:
            source_enum = IntelligenceSource(source_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source type: {source_type}")
        
        aggregator = get_intelligence_aggregator()
        all_signals = aggregator.get_intelligence_feed(limit=limit * 2)
        
        # Filter by source type
        filtered = [s for s in all_signals if s.source_type == source_enum][:limit]
        
        signals_data = [
            {
                "signal_id": s.signal_id,
                "source_name": s.source_name,
                "content": s.content,
                "timestamp": s.timestamp.isoformat(),
                "priority": s.priority.value,
                "confidence": s.confidence,
                "entities": s.entities
            }
            for s in filtered
        ]
        
        return {
            "source_type": source_type,
            "signals": signals_data,
            "count": len(signals_data)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Source filter error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlations")
async def get_correlated_intelligence(
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500)
):
    """
    Get correlated intelligence narratives.
    
    Returns:
        {
            "timestamp": "2025-12-29T15:30:00Z",
            "correlations": [
                {
                    "correlation_id": "corr_123",
                    "narrative": "...",
                    "confidence": 0.85,
                    "predicted_event": "...",
                    "signal_count": 5,
                    "entities": ["AAPL", "Tim Cook"],
                    ...
                },
                ...
            ]
        }
    """
    try:
        aggregator = get_intelligence_aggregator()
        correlations = aggregator.get_correlations(min_confidence=min_confidence, limit=limit)
        
        correlations_data = [
            {
                "correlation_id": c.correlation_id,
                "narrative": c.narrative,
                "confidence": c.confidence,
                "predicted_event": c.predicted_event,
                "timeframe": c.timeframe,
                "impact_score": c.impact_score,
                "signal_count": len(c.signals),
                "entities": list(set(e for s in c.signals for e in s.entities)),
                "timestamp": c.timestamp.isoformat()
            }
            for c in correlations
        ]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "correlations": correlations_data,
            "count": len(correlations_data)
        }
    
    except Exception as e:
        logger.error(f"Correlations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def manual_signal_ingest(signal_data: Dict[str, Any]):
    """
    Manually ingest an intelligence signal (admin only).
    
    Request body:
        {
            "source_type": "osint",
            "source_name": "Manual Entry",
            "content": "...",
            "entities": ["AAPL"],
            "tags": ["tech", "earnings"],
            "priority": "high",
            "confidence": 0.8
        }
    """
    try:
        aggregator = get_intelligence_aggregator()
        
        # Parse and validate
        source_type = IntelligenceSource(signal_data.get("source_type", "osint"))
        priority = SignalPriority(signal_data.get("priority", "medium"))
        
        signal = IntelligenceSignal(
            signal_id=f"manual_{datetime.utcnow().timestamp()}",
            source_type=source_type,
            source_name=signal_data.get("source_name", "Manual"),
            content=signal_data["content"],
            timestamp=datetime.utcnow(),
            priority=priority,
            confidence=signal_data.get("confidence", 0.5),
            tags=signal_data.get("tags", []),
            entities=signal_data.get("entities", []),
            metadata=signal_data.get(" metadata", {})
        )
        
        aggregator.ingest_signal(signal)
        
        return {
            "status": "ingested",
            "signal_id": signal.signal_id,
            "timestamp": signal.timestamp.isoformat()
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value: {e}")
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_intelligence_system_status():
    """
    Get V2 intelligence system status.
    
    Returns:
        {
            "status": "operational",
            "signal_count": 1234,
            "correlation_count": 56,
            "active_sources": ["osint", "corpint"],
            "version": "2.0.0"
        }
    """
    try:
        aggregator = get_intelligence_aggregator()
        
        # Count signals by source
        source_counts = {}
        for signal in aggregator.signals:
            source_type = signal.source_type.value
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        return {
            "status": "operational",
            "signal_count": len(aggregator.signals),
            "correlation_count": len(aggregator.correlations),
            "active_sources": list(source_counts.keys()),
            "source_breakdown": source_counts,
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
