from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from datetime import datetime
import json
import os

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Mock metrics store - in production this would come from a database or monitoring service
SYSTEM_METRICS = {
    "avg_latency": 142,
    "total_requests": 1247,
    "accuracy": 98.5,
    "total_tokens": 523400,
    "ttft": 85,
    "tpot": 12,
    "tps": 83,
    "rps": 2.4,
    "input_tokens": 312500,
    "output_tokens": 210900,
    "total_cost": 0.00,
    "cost_per_query": 0.0000,
    "avg_confidence": 92,
    "hallucination_rate": 2.1,
    "symbolic_rate": 65,
    "memory_gb": 4.2,
    "memory_percent": 52,
    "gpu_percent": 35
}

@router.get("/metrics")
async def get_metrics():
    """Get system metrics for dashboard"""
    # In a real app, collect these from telemetry
    return SYSTEM_METRICS

@router.get("/settings")
async def get_settings():
    """Get metrics settings"""
    # Load from a config file or return defaults
    return {
        "defaultModel": "qwen2.5:7b",
        "synthesistModel": "qwen2.5:7b",
        "vjepaModel": "vjepa2-vitl",
        "formalReasoning": True,
        "vjepaEnabled": True,
        "smartRouting": True,
        "hybridMode": False,
        "ollamaUrl": "http://localhost:11434",
        "timeout": 300
    }

@router.post("/settings")
async def save_settings(settings: Dict[str, Any]):
    """Save admin settings"""
    # In a real app, write to config.yaml or DB
    # For now, we just acknowledge
    return {"status": "success", "settings": settings}

@router.get("/alerts")
async def get_alerts():
    """Get active system alerts"""
    return {
        "alerts": [],
        "count": 0
    }

@router.get("/hallucination-stats")
async def get_hallucination_stats():
    """Get hallucination statistics"""
    return {
        "stats": {
            "rate": 2.1,
            "detected": 14,
            "corrected": 12
        }
    }

@router.get("/sessions")
async def get_sessions(limit: int = 50, search: str = ""):
    """Get recent sessions"""
    # Return mock sessions for now to populate the UI
    mock_sessions = [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "query_type": "Hierarchical",
            "success": True,
            "query": "Analysis of complex system architecture",
            "response": "System decomposed into 4 subsystems...",
            "latency_ms": 145,
            "tokens_used": 450
        }
    ]
    return {
        "sessions": mock_sessions,
        "count": len(mock_sessions)
    }

@router.get("/data/files")
async def list_files(path: str = ""):
    """List files for Data Explorer (V5)"""
    return {
        "files": [],
        "path": path
    }
