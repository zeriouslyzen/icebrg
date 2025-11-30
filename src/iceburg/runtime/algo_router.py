"""
ICEBURG Runtime Algorithm Router
Routes algorithm execution based on query requirements and system capabilities
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..config import IceburgConfig


def extract_claims_simple(text: str, max_claims: int = 10) -> List[Dict[str, Any]]:
    """Simple claim extraction from text"""
    claims = []

    try:
        # Basic sentence splitting and claim identification
        sentences = text.split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum sentence length
                claims.append({
                    "text": sentence,
                    "confidence": 0.7,
                    "type": "statement"
                })
                
                # Limit claims if max_claims specified
                if len(claims) >= max_claims:
                    break

    except Exception:
        claims = []

    return claims


def route_algorithm_execution(query: str, algorithm_type: str, cfg: IceburgConfig) -> Dict[str, Any]:
    """Route algorithm execution based on query requirements"""

    routing_decision = {
        "algorithm_type": algorithm_type,
        "execution_path": "standard",
        "optimization_level": "balanced",
        "fallback_options": ["simple", "basic"]
    }

    # Route based on query complexity
    if len(query) > 1000:  # Long query
        routing_decision["execution_path"] = "optimized"
        routing_decision["optimization_level"] = "high"
    elif len(query) < 100:  # Short query
        routing_decision["execution_path"] = "fast"
        routing_decision["optimization_level"] = "low"

    return routing_decision


def optimize_algorithm_parameters(algorithm_config: Dict[str, Any], hardware_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize algorithm parameters based on hardware profile"""

    optimized_config = algorithm_config.copy()

    # Memory optimization
    available_memory = hardware_profile.get("memory_available_gb", 8)
    if available_memory < 8:
        optimized_config["batch_size"] = min(optimized_config.get("batch_size", 1), 1)
        optimized_config["context_window"] = min(optimized_config.get("context_window", 4096), 2048)
    elif available_memory > 16:
        optimized_config["batch_size"] = max(optimized_config.get("batch_size", 1), 4)
        optimized_config["context_window"] = max(optimized_config.get("context_window", 4096), 8192)

    return optimized_config
