"""Temporal Emergence Detector - Detects temporal patterns and evolution in emergence."""

from __future__ import annotations

import json
import time
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalEmergencePattern:
    """Represents a detected temporal emergence pattern."""
    pattern_type: str
    confidence: float
    temporal_signature: str
    evolution_stage: str
    time_scale: str
    causality_indicators: List[str]
    metadata: Dict[str, Any]


class TemporalEmergenceDetector:
    """Detects temporal patterns and evolution in emergence phenomena."""
    
    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg
        self.temporal_keywords = {
            "evolution", "development", "progression", "emergence", "arise",
            "evolve", "develop", "progress", "emerge", "manifest",
            "temporal", "time", "sequence", "chronological", "historical",
            "phase", "stage", "period", "epoch", "era", "cycle"
        }
        self.causality_terms = {
            "cause", "effect", "result", "consequence", "outcome",
            "leads_to", "results_in", "produces", "generates", "creates",
            "influences", "affects", "determines", "shapes", "drives"
        }
        self.evolution_stages = {
            "initial": ["beginning", "start", "origin", "genesis", "birth"],
            "development": ["growth", "expansion", "building", "construction", "formation"],
            "maturation": ["mature", "complete", "full", "developed", "established"],
            "transformation": ["change", "shift", "transition", "metamorphosis", "evolution"],
            "emergence": ["emerge", "arise", "manifest", "appear", "surface"]
        }
        self.time_scales = {
            "instantaneous": ["immediate", "instant", "sudden", "abrupt", "rapid"],
            "short_term": ["minutes", "hours", "days", "weeks", "quick"],
            "medium_term": ["months", "years", "decades", "gradual", "progressive"],
            "long_term": ["centuries", "millennia", "evolutionary", "historical", "deep_time"]
        }
    
    def detect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect temporal emergence patterns in the given data."""
        try:
            # Extract text content for analysis
            text_content = self._extract_text_content(data)
            if not text_content:
                return {"temporal_emergence_detected": False, "confidence": 0.0, "reason": "no_text_content"}
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(text_content)
            causality_score = self._analyze_causality_patterns(text_content)
            evolution_stage = self._identify_evolution_stage(text_content)
            time_scale = self._identify_time_scale(text_content)
            
            # Calculate overall temporal emergence score
            emergence_score = self._calculate_temporal_emergence_score(
                temporal_patterns, causality_score, evolution_stage, time_scale
            )
            
            if emergence_score > 0.5:  # Threshold for temporal emergence
                temporal_signature = self._generate_temporal_signature(
                    text_content, temporal_patterns, evolution_stage, time_scale
                )
                causality_indicators = self._identify_causality_indicators(text_content)
                
                pattern = TemporalEmergencePattern(
                    pattern_type="temporal_emergence",
                    confidence=emergence_score,
                    temporal_signature=temporal_signature,
                    evolution_stage=evolution_stage,
                    time_scale=time_scale,
                    causality_indicators=causality_indicators,
                    metadata={
                        "temporal_patterns": temporal_patterns,
                        "causality_score": causality_score,
                        "timestamp": time.time(),
                        "text_hash": hashlib.md5(text_content.encode()).hexdigest()[:8]
                    }
                )
                
                return {
                    "temporal_emergence_detected": True,
                    "confidence": emergence_score,
                    "pattern": pattern,
                    "temporal_signature": temporal_signature,
                    "evolution_stage": evolution_stage,
                    "time_scale": time_scale,
                    "causality_indicators": causality_indicators,
                    "metadata": pattern.metadata
                }
            
            return {
                "temporal_emergence_detected": False,
                "confidence": emergence_score,
                "reason": f"below_threshold_{emergence_score:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Temporal emergence detection error: {e}")
            return {"temporal_emergence_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _extract_text_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from various data formats."""
        if isinstance(data, str):
            return data
        
        # Try common text fields
        text_fields = ["content", "text", "output", "result", "response", "analysis"]
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                return data[field]
        
        # Try nested structures
        if "agent_output" in data:
            return self._extract_text_content(data["agent_output"])
        
        if "reasoning_chain" in data:
            chain = data["reasoning_chain"]
            if isinstance(chain, dict):
                texts = []
                for key, value in chain.items():
                    if isinstance(value, str):
                        texts.append(value)
                    elif isinstance(value, dict) and "output" in value:
                        texts.append(value["output"])
                return " ".join(texts)
        
        # Fallback: convert to string
        return str(data)
    
    def _analyze_temporal_patterns(self, text: str) -> List[str]:
        """Analyze text for temporal-related patterns."""
        text_lower = text.lower()
        patterns = []
        
        # Check for temporal keywords
        temporal_matches = [kw for kw in self.temporal_keywords if kw in text_lower]
        if temporal_matches:
            patterns.append(f"temporal_terminology:{len(temporal_matches)}")
        
        # Check for sequence indicators
        sequence_indicators = ["first", "then", "next", "finally", "subsequently", "afterwards"]
        sequence_matches = [si for si in sequence_indicators if si in text_lower]
        if sequence_matches:
            patterns.append(f"sequence_indicators:{len(sequence_matches)}")
        
        # Check for process descriptions
        process_terms = ["process", "mechanism", "pathway", "trajectory", "dynamics"]
        process_matches = [pt for pt in process_terms if pt in text_lower]
        if process_matches:
            patterns.append(f"process_descriptions:{len(process_matches)}")
        
        # Check for historical context
        if any(term in text_lower for term in ["history", "historical", "past", "tradition", "legacy"]):
            patterns.append("historical_context")
        
        # Check for future predictions
        if any(term in text_lower for term in ["future", "prediction", "forecast", "projection", "will"]):
            patterns.append("future_predictions")
        
        return patterns
    
    def _analyze_causality_patterns(self, text: str) -> float:
        """Analyze causality patterns in the text."""
        text_lower = text.lower()
        
        # Count causality terms
        causality_count = sum(1 for term in self.causality_terms if term in text_lower)
        
        # Look for causal structures
        causal_structures = [
            "because", "due to", "as a result", "consequently", "therefore",
            "thus", "hence", "so", "leads to", "results in", "causes"
        ]
        
        structure_count = sum(1 for structure in causal_structures if structure in text_lower)
        
        # Calculate causality score
        causality_score = min(1.0, (causality_count * 0.1) + (structure_count * 0.15))
        return causality_score
    
    def _identify_evolution_stage(self, text: str) -> str:
        """Identify the evolution stage described in the text."""
        text_lower = text.lower()
        
        stage_scores = {}
        for stage, keywords in self.evolution_stages.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            stage_scores[stage] = score
        
        # Return the stage with highest score, or "unknown" if no clear match
        if stage_scores:
            best_stage = max(stage_scores, key=stage_scores.get)
            if stage_scores[best_stage] > 0:
                return best_stage
        
        return "unknown"
    
    def _identify_time_scale(self, text: str) -> str:
        """Identify the time scale described in the text."""
        text_lower = text.lower()
        
        scale_scores = {}
        for scale, keywords in self.time_scales.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scale_scores[scale] = score
        
        # Return the scale with highest score, or "unknown" if no clear match
        if scale_scores:
            best_scale = max(scale_scores, key=scale_scores.get)
            if scale_scores[best_scale] > 0:
                return best_scale
        
        return "unknown"
    
    def _calculate_temporal_emergence_score(
        self, 
        temporal_patterns: List[str], 
        causality_score: float, 
        evolution_stage: str, 
        time_scale: str
    ) -> float:
        """Calculate overall temporal emergence score."""
        base_score = 0.0
        
        # Pattern-based scoring
        for pattern in temporal_patterns:
            if "temporal_terminology" in pattern:
                base_score += 0.3
            elif "sequence_indicators" in pattern:
                base_score += 0.2
            elif "process_descriptions" in pattern:
                base_score += 0.25
            elif "historical_context" in pattern:
                base_score += 0.15
            elif "future_predictions" in pattern:
                base_score += 0.2
        
        # Add causality score
        total_score = base_score + (causality_score * 0.3)
        
        # Bonus for clear evolution stage and time scale
        if evolution_stage != "unknown":
            total_score += 0.1
        if time_scale != "unknown":
            total_score += 0.1
        
        return min(1.0, total_score)
    
    def _generate_temporal_signature(
        self, 
        text: str, 
        temporal_patterns: List[str], 
        evolution_stage: str, 
        time_scale: str
    ) -> str:
        """Generate a temporal signature for the emergence pattern."""
        signature_parts = []
        
        # Add pattern indicators
        for pattern in temporal_patterns:
            if "temporal_terminology" in pattern:
                signature_parts.append("T")
            elif "sequence_indicators" in pattern:
                signature_parts.append("S")
            elif "process_descriptions" in pattern:
                signature_parts.append("P")
            elif "historical_context" in pattern:
                signature_parts.append("H")
            elif "future_predictions" in pattern:
                signature_parts.append("F")
        
        # Add evolution stage indicator
        stage_map = {"initial": "I", "development": "D", "maturation": "M", 
                    "transformation": "X", "emergence": "E", "unknown": "U"}
        signature_parts.append(stage_map.get(evolution_stage, "U"))
        
        # Add time scale indicator
        scale_map = {"instantaneous": "I", "short_term": "S", "medium_term": "M", 
                    "long_term": "L", "unknown": "U"}
        signature_parts.append(scale_map.get(time_scale, "U"))
        
        return "".join(signature_parts) + "_" + hashlib.md5(text.encode()).hexdigest()[:6]
    
    def _identify_causality_indicators(self, text: str) -> List[str]:
        """Identify causality indicators in the text."""
        text_lower = text.lower()
        indicators = []
        
        # Look for explicit causal relationships
        causal_patterns = [
            "because", "due to", "as a result", "consequently", "therefore",
            "thus", "hence", "so", "leads to", "results in", "causes",
            "influences", "affects", "determines", "shapes", "drives"
        ]
        
        for pattern in causal_patterns:
            if pattern in text_lower:
                indicators.append(f"causal:{pattern}")
        
        # Look for temporal causality
        if any(term in text_lower for term in ["after", "before", "during", "while", "when"]):
            indicators.append("temporal_causality")
        
        # Look for conditional relationships
        if any(term in text_lower for term in ["if", "unless", "provided", "assuming", "given"]):
            indicators.append("conditional_causality")
        
        return indicators
