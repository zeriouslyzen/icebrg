"""
Pattern Correlation Engine
Correlates patterns across matrices and moves from truth-seeking to truth-knowing
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ..awareness.matrix_detection import Pattern, Matrix
from ..awareness.matrix_reasoning import MatrixReasoning
from ..emergence_detector import EmergenceDetector

logger = logging.getLogger(__name__)


@dataclass
class TruthKnowingResult:
    """Result of truth knowing analysis"""
    query: str
    truth_certainty: float
    evidence: List[str]
    patterns_correlated: List[str]
    matrices_used: List[str]
    conclusion: str
    confidence: float
    metadata: Dict[str, Any]


class PatternCorrelationEngine:
    """
    Correlates patterns across matrices.
    
    Moves from truth-seeking to truth-knowing through pattern correlation,
    calculates certainty levels, and provides truth knowing (not just seeking).
    """
    
    def __init__(self):
        """Initialize pattern correlation engine."""
        self.matrix_reasoning = MatrixReasoning()
        self.emergence_detector = EmergenceDetector()
        logger.info("Pattern Correlation Engine initialized")
    
    def correlate_patterns(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """
        Correlate patterns across matrices.
        
        Args:
            patterns: List of patterns to correlate
            
        Returns:
            Dictionary with correlation results
        """
        correlation_result = {
            "patterns": [p.pattern_id for p in patterns],
            "correlations": [],
            "common_elements": [],
            "cross_pattern_insights": []
        }
        
        # Find common elements
        all_elements = {}
        for pattern in patterns:
            for element in pattern.evidence:
                if element not in all_elements:
                    all_elements[element] = []
                all_elements[element].append(pattern.pattern_id)
        
        common_elements = {element: patterns for element, patterns in all_elements.items() if len(patterns) > 1}
        if common_elements:
            correlation_result["correlations"].append(f"Found {len(common_elements)} common elements across patterns")
            correlation_result["common_elements"].extend(list(common_elements.keys())[:5])
        
        # Find pattern similarities
        pattern_similarities = []
        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i+1:]:
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                if similarity > 0.5:
                    pattern_similarities.append({
                        "pattern1": pattern1.pattern_id,
                        "pattern2": pattern2.pattern_id,
                        "similarity": similarity
                    })
        
        if pattern_similarities:
            correlation_result["correlations"].append(f"Found {len(pattern_similarities)} pattern similarities")
            correlation_result["cross_pattern_insights"].extend(pattern_similarities[:3])
        
        logger.info(f"Correlated {len(patterns)} patterns: {len(common_elements)} common elements, "
                   f"{len(pattern_similarities)} similarities")
        return correlation_result
    
    def calculate_truth_certainty(self, evidence: Dict[str, Any]) -> float:
        """
        Calculate truth certainty from evidence.
        
        Args:
            evidence: Dictionary with evidence data
            
        Returns:
            Truth certainty score (0.0 to 1.0)
        """
        certainty = 0.0
        
        # Evidence strength
        evidence_strength = evidence.get("strength", 0.5)
        certainty += evidence_strength * 0.4
        
        # Pattern correlation
        pattern_correlation = evidence.get("pattern_correlation", 0.5)
        certainty += pattern_correlation * 0.3
        
        # Multi-source verification
        multi_source = evidence.get("multi_source", False)
        if multi_source:
            certainty += 0.2
        
        # Matrix validation
        matrix_validation = evidence.get("matrix_validation", 0.5)
        certainty += matrix_validation * 0.1
        
        return min(certainty, 1.0)
    
    def move_to_truth_knowing(self, query: str) -> Dict[str, Any]:
        """
        Move from truth-seeking to truth-knowing.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with truth knowing results
        """
        result = {
            "query": query,
            "truth_seeking": False,
            "truth_knowing": True,
            "certainty": 0.0,
            "evidence": [],
            "patterns": [],
            "matrices": [],
            "conclusion": ""
        }
        
        # Use matrix reasoning
        matrices = self.matrix_reasoning.identify_underlying_matrices(query)
        if matrices:
            result["matrices"] = [m.matrix_id for m in matrices]
            
            # Correlate across matrices
            all_patterns = []
            for matrix in matrices:
                all_patterns.extend(matrix.patterns)
            
            if all_patterns:
                correlation = self.correlate_patterns(all_patterns)
                result["patterns"] = [p.pattern_id for p in all_patterns]
                result["evidence"].extend(correlation.get("common_elements", []))
                
                # Calculate certainty
                evidence_dict = {
                    "strength": 0.7,
                    "pattern_correlation": len(correlation.get("correlations", [])) / 5.0,
                    "multi_source": len(matrices) > 1,
                    "matrix_validation": 0.8
                }
                result["certainty"] = self.calculate_truth_certainty(evidence_dict)
        
        # Generate conclusion
        if result["certainty"] > 0.7:
            result["conclusion"] = f"Truth known with {result['certainty']:.2f} certainty based on pattern correlation across {len(result['matrices'])} matrices"
        else:
            result["conclusion"] = f"Truth seeking with {result['certainty']:.2f} certainty, more evidence needed"
        
        logger.info(f"Truth knowing analysis complete: certainty={result['certainty']:.2f}, "
                   f"matrices={len(result['matrices'])}, patterns={len(result['patterns'])}")
        return result
    
    def provide_truth_knowing(self, query: str) -> TruthKnowingResult:
        """
        Provide truth knowing (not just seeking).
        
        Args:
            query: User query
            
        Returns:
            TruthKnowingResult with complete truth knowing analysis
        """
        # Move to truth knowing
        truth_result = self.move_to_truth_knowing(query)
        
        # Create truth knowing result
        result = TruthKnowingResult(
            query=query,
            truth_certainty=truth_result["certainty"],
            evidence=truth_result["evidence"],
            patterns_correlated=truth_result["patterns"],
            matrices_used=truth_result["matrices"],
            conclusion=truth_result["conclusion"],
            confidence=truth_result["certainty"],
            metadata=truth_result
        )
        
        logger.info(f"Truth knowing provided: certainty={result.truth_certainty:.2f}, "
                   f"matrices={len(result.matrices_used)}, patterns={len(result.patterns_correlated)}")
        return result
    
    def _calculate_pattern_similarity(self, pattern1: Pattern, pattern2: Pattern) -> float:
        """Calculate similarity between two patterns."""
        similarity = 0.0
        
        # Type similarity
        if pattern1.pattern_type == pattern2.pattern_type:
            similarity += 0.3
        
        # Description similarity
        desc1_lower = pattern1.description.lower()
        desc2_lower = pattern2.description.lower()
        
        words1 = set(desc1_lower.split())
        words2 = set(desc2_lower.split())
        common_words = words1.intersection(words2)
        
        if words1 or words2:
            similarity += (len(common_words) / max(len(words1), len(words2))) * 0.4
        
        # Evidence overlap
        evidence1 = set(pattern1.evidence)
        evidence2 = set(pattern2.evidence)
        common_evidence = evidence1.intersection(evidence2)
        
        if evidence1 or evidence2:
            similarity += (len(common_evidence) / max(len(evidence1), len(evidence2))) * 0.3
        
        return min(similarity, 1.0)

