"""
ICEBURG Research Module
General-purpose device generation and research methodology
"""

from .methodology_analyzer import MethodologyAnalyzer
from .insight_generator import InsightGenerator
from .historical_analyzer import HistoricalAnalyzer
from .pattern_extractor import PatternExtractor
from .self_study import SelfStudy
from .validation_engine import ValidationEngine

__all__ = [
    "MethodologyAnalyzer",
    "InsightGenerator",
    "HistoricalAnalyzer",
    "PatternExtractor",
    "SelfStudy",
    "ValidationEngine",
]

