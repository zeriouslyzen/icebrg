"""
Enhanced Hallucination Detection System
Multi-layer detection beyond quarantine
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Enhanced hallucination detection system with multiple detection layers.
    
    Detection Layers:
    1. Factual Consistency Check
    2. Source Verification
    3. Internal Coherence Analysis
    4. Confidence Scoring
    5. Pattern-Based Detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hallucination detector"""
        self.config = config or {}
        
        # Detection thresholds (configurable)
        self.hallucination_threshold = self.config.get("hallucination_threshold", 0.15)  # Lowered from 0.2
        self.confidence_threshold = self.config.get("confidence_threshold", 0.85)
        self.consistency_threshold = self.config.get("consistency_threshold", 0.8)
        
        # Pattern-based detection
        self.hallucination_patterns = [
            r'\b(?:definitely|absolutely|certainly|undoubtedly)\s+(?:is|are|was|were)\s+',
            r'\b(?:proven|confirmed|established)\s+(?:fact|truth|reality)',
            r'\b(?:all|every|none|never|always)\s+(?:scientists|researchers|studies)',
            r'\b(?:universally|completely|totally)\s+(?:accepted|agreed)',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.hallucination_patterns]
    
    def detect_hallucination(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        evidence_level: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Multi-layer hallucination detection.
        
        Args:
            content: Response content to check
            sources: List of sources used
            evidence_level: Evidence grade (A/B/C/S/X)
            context: Additional context for detection
            
        Returns:
            Detection result with scores and flags
        """
        result = {
            "content": content[:200],  # Truncate for logging
            "timestamp": datetime.now().isoformat(),
            "hallucination_detected": False,
            "hallucination_score": 0.0,
            "confidence": 0.0,
            "detection_layers": {},
            "flags": [],
            "recommendations": []
        }
        
        # Layer 1: Factual Consistency Check
        consistency_result = self._check_factual_consistency(content, sources)
        result["detection_layers"]["factual_consistency"] = consistency_result
        
        # Layer 2: Source Verification
        source_result = self._verify_sources(content, sources)
        result["detection_layers"]["source_verification"] = source_result
        
        # Layer 3: Internal Coherence Analysis
        coherence_result = self._analyze_coherence(content)
        result["detection_layers"]["internal_coherence"] = coherence_result
        
        # Layer 4: Confidence Scoring
        confidence_result = self._calculate_confidence(content, sources, evidence_level)
        result["detection_layers"]["confidence_scoring"] = confidence_result
        result["confidence"] = confidence_result.get("confidence_score", 0.0)
        
        # Layer 5: Pattern-Based Detection
        pattern_result = self._detect_patterns(content)
        result["detection_layers"]["pattern_detection"] = pattern_result
        
        # Calculate overall hallucination score
        scores = [
            consistency_result.get("score", 0.0),
            source_result.get("score", 0.0),
            coherence_result.get("score", 0.0),
            pattern_result.get("score", 0.0)
        ]
        
        # Weighted average (factual consistency and source verification weighted higher)
        weights = [0.3, 0.3, 0.2, 0.2]
        result["hallucination_score"] = sum(s * w for s, w in zip(scores, weights))
        
        # Determine if hallucination detected
        result["hallucination_detected"] = (
            result["hallucination_score"] > self.hallucination_threshold or
            result["confidence"] < self.confidence_threshold
        )
        
        # Generate flags and recommendations
        if result["hallucination_detected"]:
            result["flags"].append("hallucination_risk")
            if consistency_result.get("issues"):
                result["flags"].extend(consistency_result["issues"])
            if source_result.get("issues"):
                result["flags"].extend(source_result["issues"])
            
            result["recommendations"].append("Review content for factual accuracy")
            result["recommendations"].append("Verify sources and citations")
            if evidence_level in ["C", "S", "X"]:
                result["recommendations"].append(f"Evidence level {evidence_level} requires additional verification")
        
        return result
    
    def _check_factual_consistency(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Layer 1: Check factual consistency"""
        result = {
            "score": 0.0,
            "issues": [],
            "checks": []
        }
        
        # Check for unsupported claims
        unsupported_patterns = [
            r'\b(?:proven|confirmed|established)\s+(?:fact|truth)',
            r'\b(?:all|every|none)\s+(?:studies|research|scientists)',
            r'\b(?:universally|completely)\s+(?:accepted|agreed)',
        ]
        
        issues = []
        for pattern in unsupported_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                issues.append(f"Unsupported claim pattern: {pattern}")
        
        # Check for contradictory statements
        contradiction_indicators = [
            ("always", "never"),
            ("all", "none"),
            ("proven", "unproven"),
            ("confirmed", "unconfirmed"),
        ]
        
        for indicator1, indicator2 in contradiction_indicators:
            if indicator1 in content.lower() and indicator2 in content.lower():
                issues.append(f"Contradictory indicators: {indicator1} and {indicator2}")
        
        # Calculate score (more issues = higher hallucination score)
        result["score"] = min(1.0, len(issues) * 0.2)
        result["issues"] = issues
        result["checks"] = [
            "Unsupported claims check",
            "Contradiction detection",
            "Source alignment check"
        ]
        
        return result
    
    def _verify_sources(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Layer 2: Verify sources"""
        result = {
            "score": 0.0,
            "issues": [],
            "source_count": 0,
            "verified_sources": 0
        }
        
        if not sources:
            result["score"] = 0.3  # Moderate risk without sources
            result["issues"].append("No sources provided")
            return result
        
        result["source_count"] = len(sources)
        
        # Check source quality
        verified = 0
        for source in sources:
            # Check if source has URL
            if source.get("url"):
                verified += 1
            # Check if source has title
            if source.get("title"):
                verified += 1
            # Check source type
            source_type = source.get("source_type", "unknown")
            if source_type in ["mainstream", "peer-reviewed", "academic"]:
                verified += 1
        
        result["verified_sources"] = verified
        
        # Calculate score (fewer verified sources = higher hallucination score)
        if result["source_count"] == 0:
            result["score"] = 0.5
        else:
            verification_ratio = verified / (result["source_count"] * 3)  # 3 checks per source
            result["score"] = 1.0 - verification_ratio
        
        if result["score"] > 0.3:
            result["issues"].append("Insufficient source verification")
        
        return result
    
    def _analyze_coherence(
        self,
        content: str
    ) -> Dict[str, Any]:
        """Layer 3: Analyze internal coherence"""
        result = {
            "score": 0.0,
            "issues": [],
            "coherence_checks": []
        }
        
        # Check for logical inconsistencies
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Check for repeated contradictory statements
        contradictions = 0
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i+1:]:
                # Simple contradiction detection
                if any(word in sent1.lower() and antonym in sent2.lower() 
                       for word, antonym in [("yes", "no"), ("true", "false"), ("proven", "unproven")]):
                    contradictions += 1
        
        # Check for abrupt topic changes (potential hallucination)
        topic_changes = 0
        keywords_per_sentence = []
        for sentence in sentences[:10]:  # Check first 10 sentences
            words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            keywords_per_sentence.append(words)
        
        for i in range(len(keywords_per_sentence) - 1):
            overlap = len(keywords_per_sentence[i] & keywords_per_sentence[i+1])
            if overlap < 2:  # Very little overlap
                topic_changes += 1
        
        # Calculate score
        result["score"] = min(1.0, (contradictions * 0.3) + (topic_changes * 0.1))
        
        if contradictions > 0:
            result["issues"].append(f"{contradictions} logical contradictions detected")
        if topic_changes > 3:
            result["issues"].append(f"{topic_changes} abrupt topic changes detected")
        
        result["coherence_checks"] = [
            "Logical consistency",
            "Topic coherence",
            "Statement alignment"
        ]
        
        return result
    
    def _calculate_confidence(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        evidence_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """Layer 4: Calculate confidence score"""
        result = {
            "confidence_score": 0.5,
            "factors": {}
        }
        
        # Base confidence from evidence level
        evidence_confidence = {
            "A": 0.9,  # Well-established
            "B": 0.7,  # Plausible
            "C": 0.5,  # Speculative
            "S": 0.6,  # Suppressed but valid
            "X": 0.4   # Actively censored
        }
        
        if evidence_level:
            result["confidence_score"] = evidence_confidence.get(evidence_level.upper(), 0.5)
            result["factors"]["evidence_level"] = evidence_level
        
        # Adjust based on sources
        if sources:
            source_count = len(sources)
            source_boost = min(0.2, source_count * 0.05)
            result["confidence_score"] = min(1.0, result["confidence_score"] + source_boost)
            result["factors"]["source_count"] = source_count
        
        # Adjust based on content length (longer = more context)
        content_length = len(content)
        if content_length > 500:
            length_boost = min(0.1, (content_length - 500) / 5000)
            result["confidence_score"] = min(1.0, result["confidence_score"] + length_boost)
        
        result["factors"]["content_length"] = content_length
        
        return result
    
    def _detect_patterns(
        self,
        content: str
    ) -> Dict[str, Any]:
        """Layer 5: Pattern-based detection"""
        result = {
            "score": 0.0,
            "issues": [],
            "patterns_detected": []
        }
        
        # Check for hallucination patterns
        for pattern in self.compiled_patterns:
            matches = pattern.findall(content)
            if matches:
                result["patterns_detected"].extend(matches)
                result["score"] += 0.2
        
        result["score"] = min(1.0, result["score"])
        
        if result["patterns_detected"]:
            result["issues"].append(f"{len(result['patterns_detected'])} hallucination patterns detected")
        
        return result
    
    def should_quarantine(
        self,
        detection_result: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Determine if content should be quarantined.
        
        Returns:
            (should_quarantine, reason)
        """
        if detection_result["hallucination_detected"]:
            if detection_result["hallucination_score"] > 0.5:
                return True, "High hallucination score"
            elif detection_result["confidence"] < 0.5:
                return True, "Low confidence score"
            elif "hallucination_risk" in detection_result["flags"]:
                return True, "Hallucination risk detected"
        
        return False, "No quarantine needed"

