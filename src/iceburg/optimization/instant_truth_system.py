"""
ICEBURG Instant Truth System - Experimental Optimization
========================================================

This module implements instant truth recognition and smart routing
to make ICEBURG faster and more direct while preserving breakthrough capabilities.

SAFETY FEATURES:
- Can be disabled with ICEBURG_INSTANT_TRUTH=false
- Fallback to original system if errors occur
- Easy rollback by removing this module
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TruthType(Enum):
    """Types of instant truths"""
    KNOWN_PATTERN = "known_pattern"
    VERIFIED_INSIGHT = "verified_insight"
    SUPPRESSION_DETECTED = "suppression_detected"
    BREAKTHROUGH_DISCOVERY = "breakthrough_discovery"

@dataclass
class InstantTruth:
    """Instant truth record"""
    query_pattern: str
    truth_type: TruthType
    direct_insight: str
    confidence: float
    evidence: List[str]
    timestamp: float
    verification_count: int = 0

class TruthCache:
    """Cache of instant truths for rapid access"""
    
    def __init__(self, cache_file: Path = Path("data/instant_truth_cache.json")):
        self.cache_file = cache_file
        self.truths: Dict[str, InstantTruth] = {}
        self.pattern_matcher = PatternMatcher()
        self.load_cache()
    
    def load_cache(self):
        """Load cached truths from file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for key, truth_data in data.items():
                        self.truths[key] = InstantTruth(**truth_data)
                logger.info(f"Loaded {len(self.truths)} instant truths from cache")
        except Exception as e:
            logger.warning(f"Failed to load truth cache: {e}")
    
    def save_cache(self):
        """Save truths to cache file"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for k, truth in self.truths.items():
                truth_dict = truth.__dict__.copy()
                truth_dict['truth_type'] = truth_dict['truth_type'].value  # Convert enum to string
                data[k] = truth_dict
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save truth cache: {e}")
    
    def add_truth(self, query: str, truth: InstantTruth):
        """Add a new instant truth"""
        self.truths[query] = truth
        self.save_cache()
    
    def get_instant_truth(self, query: str) -> Optional[InstantTruth]:
        """Get instant truth if available"""
        # Direct match
        if query in self.truths:
            return self.truths[query]
        
        # Pattern match
        for pattern, truth in self.truths.items():
            if self.pattern_matcher.matches(query, pattern):
                return truth
        
        return None

class PatternMatcher:
    """Smart pattern matching for queries using semantic similarity"""
    
    def __init__(self):
        # Import the new semantic pattern matcher
        try:
            from .semantic_pattern_matcher import advanced_pattern_matcher
            self.semantic_matcher = advanced_pattern_matcher
            self.use_semantic = True
            logger.info("Using advanced semantic pattern matching")
        except ImportError:
            self.semantic_matcher = None
            self.use_semantic = False
            logger.warning("Semantic pattern matcher not available, using fallback")
        
        # Fallback patterns (kept for compatibility)
        self.medical_patterns = [
            "medical research", "healthcare funding", "disease research",
            "clinical trials", "medical breakthrough", "treatment development"
        ]
        self.cancer_patterns = [
            "cancer research", "oncology", "tumor", "carcinoma", "malignancy",
            "chemotherapy", "radiation therapy", "cancer treatment", "cancer cure"
        ]
        self.suppression_patterns = [
            "suppressed research", "research suppression", "institutional corruption",
            "funding bias", "career destruction"
        ]
        self.breakthrough_patterns = [
            "breakthrough", "discovery", "innovation", "revolutionary",
            "unprecedented", "groundbreaking"
        ]
    
    def matches(self, query: str, pattern: str) -> bool:
        """Check if query matches pattern using semantic similarity"""
        
        # Use semantic matching if available
        if self.use_semantic and self.semantic_matcher:
            try:
                return self.semantic_matcher.matches(query, pattern)
            except Exception as e:
                logger.error(f"Semantic matching error: {e}, falling back to keyword matching")
        
        # Fallback to exact match only (more conservative)
        query_lower = query.lower()
        pattern_lower = pattern.lower()
        
        # Only exact matches for fallback
        return query_lower == pattern_lower

class SmartRouter:
    """Smart routing system for query processing"""
    
    def __init__(self):
        self.truth_cache = TruthCache()
        self.complexity_analyzer = ComplexityAnalyzer()
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route query to appropriate processing mode"""
        
        # Check for instant truth
        instant_truth = self.truth_cache.get_instant_truth(query)
        if instant_truth and instant_truth.confidence > 0.8:
            return {
                "mode": "instant_truth",
                "truth": instant_truth,
                "processing_time": "< 1 second",
                "reason": "Known truth with high confidence"
            }
        
        # Analyze complexity
        complexity = self.complexity_analyzer.analyze(query)
        
        if complexity["level"] == "simple":
            return {
                "mode": "fast_path",
                "complexity": complexity,
                "processing_time": "2-5 seconds",
                "reason": "Simple query, fast processing"
            }
        elif complexity["level"] == "known_pattern":
            return {
                "mode": "pattern_recognition",
                "complexity": complexity,
                "processing_time": "5-15 seconds",
                "reason": "Known pattern, optimized processing"
            }
        else:
            return {
                "mode": "full_analysis",
                "complexity": complexity,
                "processing_time": "2-5 minutes",
                "reason": "Complex query, full analysis required"
            }

class ComplexityAnalyzer:
    """Analyze query complexity for smart routing"""
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity"""
        query_lower = query.lower()
        
        # Simple queries (factual, direct)
        simple_indicators = ["what is", "define", "explain", "how does", "when did"]
        if any(indicator in query_lower for indicator in simple_indicators):
            return {"level": "simple", "indicators": simple_indicators}
        
        # Known patterns (cancer, suppression, etc.)
        known_patterns = ["cancer", "suppression", "breakthrough", "research"]
        if any(pattern in query_lower for pattern in known_patterns):
            return {"level": "known_pattern", "patterns": known_patterns}
        
        # Complex queries (analysis, synthesis, discovery)
        complex_indicators = ["analyze", "synthesize", "discover", "investigate", "find patterns"]
        if any(indicator in query_lower for indicator in complex_indicators):
            return {"level": "complex", "indicators": complex_indicators}
        
        return {"level": "medium", "indicators": []}

class DirectInsightGenerator:
    """Generate direct insights without verbose explanations"""
    
    def __init__(self):
        self.truth_cache = TruthCache()
    
    def generate_direct_insight(self, query: str, analysis_result: Dict[str, Any]) -> str:
        """Generate direct insight without process explanation"""
        
        # Check for instant truth first
        instant_truth = self.truth_cache.get_instant_truth(query)
        if instant_truth:
            return f"**TRUTH:** {instant_truth.direct_insight}"
        
        # Extract key insights from analysis
        insights = self._extract_key_insights(analysis_result)
        
        # Generate direct response
        if insights:
            return f"**INSIGHT:** {insights[0]}\n\n**EVIDENCE:** {', '.join(insights[1:3])}"
        else:
            return "**ANALYSIS:** Complex query requires full processing"
    
    def _extract_key_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Extract key insights from analysis result"""
        insights = []
        
        # Extract from different layers
        if "surveyor" in analysis_result:
            insights.append("Consensus research completed")
        
        if "synthesist" in analysis_result:
            insights.append("Cross-domain synthesis achieved")
        
        if "suppression_detected" in analysis_result:
            insights.append("Suppression patterns identified")
        
        return insights

class InstantTruthSystem:
    """Main instant truth system"""
    
    def __init__(self):
        self.enabled = self._is_enabled()
        self.truth_cache = TruthCache()
        self.smart_router = SmartRouter()
        self.direct_insight = DirectInsightGenerator()
        
        # Initialize suppression detector
        try:
            from ..truth.suppression_detector import SuppressionDetector
            self.suppression_detector = SuppressionDetector()
        except ImportError:
            self.suppression_detector = None
            logger.warning("Suppression detector not available")
        
        # Initialize with known truths
        self._initialize_known_truths()
    
    def _is_enabled(self) -> bool:
        """Check if instant truth system is enabled"""
        import os
        return os.getenv("ICEBURG_INSTANT_TRUTH", "true").lower() == "true"
    
    def _initialize_known_truths(self):
        """Initialize with known breakthrough truths"""
        
        # Medical research funding patterns (evidence-based)
        research_funding_truth = InstantTruth(
            query_pattern="medical research funding",
            truth_type=TruthType.KNOWN_PATTERN,
            direct_insight="Medical research funding is influenced by multiple factors including disease prevalence, treatment costs, market potential, and public health priorities. Funding distribution varies significantly across different disease areas.",
            confidence=0.85,
            evidence=[
                "NIH funding statistics by disease area",
                "Pharmaceutical industry investment patterns",
                "Academic research funding distribution studies"
            ],
            timestamp=time.time(),
            verification_count=1
        )
        
        self.truth_cache.add_truth("medical research funding", research_funding_truth)
        
        # Research suppression pattern
        suppression_truth = InstantTruth(
            query_pattern="research suppression",
            truth_type=TruthType.KNOWN_PATTERN,
            direct_insight="Systematic research suppression occurs across academic, financial, regulatory, and social systems to protect vested interests and maintain power structures.",
            confidence=0.90,
            evidence=[
                "Cross-domain suppression patterns",
                "Institutional corruption evidence",
                "Career destruction cases"
            ],
            timestamp=time.time(),
            verification_count=1
        )
        
        self.truth_cache.add_truth("research suppression", suppression_truth)
    
    async def process_query(self, query: str, documents: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Process query with instant truth system"""
        
        if not self.enabled:
            return {"mode": "disabled", "reason": "Instant truth system disabled"}
        
        try:
            # Route query
            routing = self.smart_router.route_query(query)
            
            if routing["mode"] == "instant_truth":
                return {
                    "mode": "instant_truth",
                    "result": routing["truth"].direct_insight,
                    "processing_time": "< 1 second",
                    "confidence": routing["truth"].confidence,
                    "evidence": routing["truth"].evidence
                }
            
            # Check for suppression if documents provided
            if documents and self.suppression_detector:
                suppression_result = self.suppression_detector.detect_suppression(documents)
                if suppression_result.get("suppression_detected"):
                    return {
                        "mode": "suppression_detected",
                        "result": "Suppression patterns detected",
                        "suppression_score": suppression_result.get("overall_suppression_score", 0.0),
                        "details": suppression_result,
                        "processing_time": "5-10 seconds"
                    }
            
            return routing
            
        except Exception as e:
            logger.error(f"Instant truth system error: {e}")
            return {"mode": "fallback", "reason": f"Error: {e}"}
    
    def get_instant_truth(self, query: str) -> Optional[InstantTruth]:
        """Get instant truth for query"""
        return self.truth_cache.get_instant_truth(query)
    
    def add_truth(self, query: str, truth: InstantTruth):
        """Add new instant truth"""
        self.truth_cache.add_truth(query, truth)
    
    def disable(self):
        """Disable instant truth system"""
        self.enabled = False
        logger.info("Instant truth system disabled")
    
    def enable(self):
        """Enable instant truth system"""
        self.enabled = True
        logger.info("Instant truth system enabled")

# Global instance
instant_truth_system = InstantTruthSystem()
