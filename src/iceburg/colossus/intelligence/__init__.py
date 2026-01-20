"""
COLOSSUS Intelligence Layer

AI-powered entity extraction, resolution, and analysis.
Optimized for M4 Mac with local Ollama inference.
"""

from .extraction import EntityExtractor, ExtractedEntity
from .resolution import EntityResolver, ResolvedEntity
from .risk import RiskScorer, RiskAssessment

__all__ = [
    "EntityExtractor",
    "ExtractedEntity",
    "EntityResolver",
    "ResolvedEntity",
    "RiskScorer",
    "RiskAssessment",
]
