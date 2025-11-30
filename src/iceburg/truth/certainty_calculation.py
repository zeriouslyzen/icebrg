"""
Certainty Calculation System
Calculates certainty levels with multi-source verification
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ..agents.scrutineer import run as scrutineer_run
from ..agents.oracle import run as oracle_run
from ..config import IceburgConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Represents a source of evidence"""
    source_id: str
    source_type: str  # "research", "expert", "data", "pattern", "matrix"
    content: str
    reliability: float
    timestamp: str


class CertaintyCalculation:
    """
    Calculates certainty levels for truth claims.
    
    Performs multi-source verification, pattern confirmation, and matrix validation.
    """
    
    def __init__(self, cfg: Optional[IceburgConfig] = None):
        """
        Initialize certainty calculation system.
        
        Args:
            cfg: ICEBURG config (loads if None)
        """
        self.cfg = cfg or load_config()
        logger.info("Certainty Calculation System initialized")
    
    def calculate_certainty(self, evidence: Dict[str, Any]) -> float:
        """
        Calculate certainty level from evidence.
        
        Args:
            evidence: Dictionary with evidence data
            
        Returns:
            Certainty score (0.0 to 1.0)
        """
        certainty = 0.0
        
        # Evidence strength
        evidence_strength = evidence.get("strength", 0.5)
        certainty += evidence_strength * 0.3
        
        # Source reliability
        source_reliability = evidence.get("source_reliability", 0.5)
        certainty += source_reliability * 0.2
        
        # Multi-source verification
        multi_source = evidence.get("multi_source", False)
        if multi_source:
            certainty += 0.2
        
        # Pattern confirmation
        pattern_confirmation = evidence.get("pattern_confirmation", 0.5)
        certainty += pattern_confirmation * 0.2
        
        # Matrix validation
        matrix_validation = evidence.get("matrix_validation", 0.5)
        certainty += matrix_validation * 0.1
        
        return min(certainty, 1.0)
    
    def multi_source_verification(self, sources: List[Source]) -> float:
        """
        Perform multi-source verification.
        
        Args:
            sources: List of sources to verify
            
        Returns:
            Verification score (0.0 to 1.0)
        """
        if not sources:
            return 0.0
        
        if len(sources) == 1:
            return sources[0].reliability
        
        # Calculate average reliability
        avg_reliability = sum(s.reliability for s in sources) / len(sources)
        
        # Bonus for multiple sources
        multi_source_bonus = min(len(sources) / 5.0, 0.3)
        
        # Check for source agreement
        source_types = [s.source_type for s in sources]
        type_diversity = len(set(source_types)) / len(source_types) if source_types else 0.0
        
        verification_score = avg_reliability + multi_source_bonus + (type_diversity * 0.2)
        
        return min(verification_score, 1.0)
    
    def pattern_confirmation(self, patterns: List[Dict[str, Any]]) -> float:
        """
        Confirm patterns.
        
        Args:
            patterns: List of patterns to confirm
            
        Returns:
            Confirmation score (0.0 to 1.0)
        """
        if not patterns:
            return 0.0
        
        # Calculate average pattern confidence
        avg_confidence = sum(p.get("confidence", 0.5) for p in patterns) / len(patterns)
        
        # Bonus for multiple patterns
        multi_pattern_bonus = min(len(patterns) / 5.0, 0.2)
        
        # Check for pattern consistency
        pattern_types = [p.get("pattern_type", "unknown") for p in patterns]
        type_consistency = len(set(pattern_types)) / len(pattern_types) if pattern_types else 0.0
        
        confirmation_score = avg_confidence + multi_pattern_bonus + (type_consistency * 0.1)
        
        return min(confirmation_score, 1.0)
    
    def matrix_validation(self, matrix: Dict[str, Any]) -> float:
        """
        Validate matrix structure.
        
        Args:
            matrix: Matrix dictionary to validate
            
        Returns:
            Validation score (0.0 to 1.0)
        """
        validation_score = 0.0
        
        # Matrix completeness
        nodes = matrix.get("nodes", [])
        edges = matrix.get("edges", [])
        
        if nodes and edges:
            completeness = min(len(edges) / (len(nodes) * (len(nodes) - 1) / 2), 1.0) if len(nodes) > 1 else 0.0
            validation_score += completeness * 0.5
        
        # Matrix confidence
        confidence = matrix.get("confidence", 0.5)
        validation_score += confidence * 0.3
        
        # Pattern presence
        patterns = matrix.get("patterns", [])
        if patterns:
            validation_score += 0.2
        
        return min(validation_score, 1.0)
    
    def calculate_with_scrutineer(self, query: str, evidence: str) -> float:
        """
        Calculate certainty using Scrutineer agent.
        
        Args:
            query: User query
            evidence: Evidence string
            
        Returns:
            Certainty score (0.0 to 1.0)
        """
        try:
            # Use Scrutineer to grade evidence
            scrutineer_result = scrutineer_run(
                cfg=self.cfg,
                vs=None,  # VectorStore not needed for simple grading
                query=query,
                evidence=evidence,
                verbose=False
            )
            
            # Parse evidence level from result
            # Scrutineer returns evidence levels: A, B, C, D
            evidence_levels = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3}
            
            for level, score in evidence_levels.items():
                if level in scrutineer_result.upper():
                    return score
            
            return 0.5  # Default
        except Exception as e:
            logger.warning(f"Error using Scrutineer for certainty calculation: {e}")
            return 0.5
    
    def calculate_with_oracle(self, query: str, evidence: str) -> float:
        """
        Calculate certainty using Oracle agent.
        
        Args:
            query: User query
            evidence: Evidence string
            
        Returns:
            Certainty score (0.0 to 1.0)
        """
        try:
            # Use Oracle to generate principles
            oracle_result = oracle_run(
                cfg=self.cfg,
                vs=None,  # VectorStore not needed for simple principle generation
                query=query,
                evidence=evidence,
                verbose=False
            )
            
            # Parse principle confidence from result
            # Oracle returns principles with confidence indicators
            if "high confidence" in oracle_result.lower() or "strong evidence" in oracle_result.lower():
                return 0.8
            elif "moderate confidence" in oracle_result.lower() or "some evidence" in oracle_result.lower():
                return 0.6
            elif "low confidence" in oracle_result.lower() or "weak evidence" in oracle_result.lower():
                return 0.4
            else:
                return 0.5  # Default
        except Exception as e:
            logger.warning(f"Error using Oracle for certainty calculation: {e}")
            return 0.5

