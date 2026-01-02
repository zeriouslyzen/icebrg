"""
ICEBURG Emergence Processor
===========================

Processes training data for emergence patterns and novelty detection.
Scores data by emergence probability and creates emergence-weighted curricula.

Features:
- Integration with ICEBURG's EmergenceDetector
- Novelty scoring based on embedding distances
- Emergence-weighted sampling for curriculum learning
- Pattern categorization (novel, familiar, transitional)
"""

from __future__ import annotations
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class EmergenceCategory(Enum):
    """Categories of emergence level."""
    NOVEL = "novel"                 # High emergence, new patterns (>0.8)
    TRANSITIONAL = "transitional"   # Medium emergence, evolving patterns (0.5-0.8)
    FAMILIAR = "familiar"           # Low emergence, known patterns (0.2-0.5)
    BASELINE = "baseline"           # Minimal emergence, standard patterns (<0.2)


@dataclass
class EmergenceDatapoint:
    """Training datapoint with emergence scoring."""
    id: str
    messages: List[Dict[str, str]]
    emergence_score: float
    emergence_category: EmergenceCategory
    novelty_features: List[str] = field(default_factory=list)
    curriculum_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "messages": self.messages,
            "emergence_score": self.emergence_score,
            "emergence_category": self.emergence_category.value,
            "novelty_features": self.novelty_features,
            "curriculum_weight": self.curriculum_weight,
            "metadata": self.metadata
        }


@dataclass
class EmergenceStats:
    """Statistics from emergence processing."""
    total_processed: int = 0
    novel_count: int = 0
    transitional_count: int = 0
    familiar_count: int = 0
    baseline_count: int = 0
    average_emergence: float = 0.0
    novel_features_found: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_processed": self.total_processed,
            "novel_count": self.novel_count,
            "transitional_count": self.transitional_count,
            "familiar_count": self.familiar_count,
            "baseline_count": self.baseline_count,
            "average_emergence": self.average_emergence,
            "novel_features_found": self.novel_features_found
        }


class EmergenceProcessor:
    """
    Processes training data for emergence patterns.
    
    Integrates with ICEBURG's emergence detection system to identify
    novel patterns and create emergence-aware training curricula.
    """
    
    # Thresholds for emergence categorization
    NOVEL_THRESHOLD = 0.8
    TRANSITIONAL_THRESHOLD = 0.5
    FAMILIAR_THRESHOLD = 0.2
    
    # Keywords that indicate novelty in different domains
    NOVELTY_KEYWORDS = {
        "scientific": ["breakthrough", "discovery", "novel", "unprecedented", "first"],
        "technical": ["innovative", "new approach", "alternative method", "optimization"],
        "conceptual": ["paradigm shift", "reframing", "synthesis", "connection"],
        "contradictory": ["however", "contrary to", "challenges", "disputes", "alternative"],
    }
    
    def __init__(
        self,
        novel_weight: float = 2.0,
        transitional_weight: float = 1.5,
        familiar_weight: float = 1.0,
        baseline_weight: float = 0.5,
        use_emergence_detector: bool = True
    ):
        """
        Initialize emergence processor.
        
        Args:
            novel_weight: Curriculum weight for novel patterns
            transitional_weight: Curriculum weight for transitional patterns
            familiar_weight: Curriculum weight for familiar patterns
            baseline_weight: Curriculum weight for baseline patterns
            use_emergence_detector: Whether to use ICEBURG's EmergenceDetector
        """
        self.novel_weight = novel_weight
        self.transitional_weight = transitional_weight
        self.familiar_weight = familiar_weight
        self.baseline_weight = baseline_weight
        self.use_emergence_detector = use_emergence_detector
        
        # Try to import EmergenceDetector
        self._emergence_detector = None
        if use_emergence_detector:
            try:
                from ..emergence_detector import EmergenceDetector
                self._emergence_detector = EmergenceDetector()
                logger.info("EmergenceProcessor: EmergenceDetector integration enabled")
            except ImportError:
                logger.warning("EmergenceProcessor: EmergenceDetector not available, using keyword-based scoring")
                
        # Pattern history for novelty detection
        self._pattern_history: Dict[str, int] = defaultdict(int)
        
    def process(
        self,
        data: List[Dict[str, Any]],
        agent_type: Optional[str] = None
    ) -> Tuple[List[EmergenceDatapoint], EmergenceStats]:
        """
        Process training data for emergence patterns.
        
        Args:
            data: List of training data entries
            agent_type: Optional agent type for specialized processing
            
        Returns:
            Tuple of (emergence datapoints, statistics)
        """
        stats = EmergenceStats(total_processed=len(data))
        processed_data: List[EmergenceDatapoint] = []
        emergence_scores: List[float] = []
        
        for i, entry in enumerate(data):
            # Extract messages
            messages = self._extract_messages(entry)
            if not messages:
                continue
                
            # Calculate emergence score
            emergence_score, novelty_features = self._calculate_emergence_score(
                entry, messages, agent_type
            )
            emergence_scores.append(emergence_score)
            
            # Categorize
            if emergence_score >= self.NOVEL_THRESHOLD:
                category = EmergenceCategory.NOVEL
                weight = self.novel_weight
                stats.novel_count += 1
            elif emergence_score >= self.TRANSITIONAL_THRESHOLD:
                category = EmergenceCategory.TRANSITIONAL
                weight = self.transitional_weight
                stats.transitional_count += 1
            elif emergence_score >= self.FAMILIAR_THRESHOLD:
                category = EmergenceCategory.FAMILIAR
                weight = self.familiar_weight
                stats.familiar_count += 1
            else:
                category = EmergenceCategory.BASELINE
                weight = self.baseline_weight
                stats.baseline_count += 1
                
            # Track novel features
            for feature in novelty_features:
                if feature not in stats.novel_features_found:
                    stats.novel_features_found.append(feature)
                    
            # Create datapoint
            datapoint = EmergenceDatapoint(
                id=entry.get("id", f"emergence_{i}"),
                messages=messages,
                emergence_score=emergence_score,
                emergence_category=category,
                novelty_features=novelty_features,
                curriculum_weight=weight,
                metadata=entry.get("metadata", {})
            )
            
            processed_data.append(datapoint)
            
        # Calculate average emergence
        if emergence_scores:
            stats.average_emergence = sum(emergence_scores) / len(emergence_scores)
            
        logger.info(f"EmergenceProcessor: Processed {stats.total_processed} entries")
        logger.info(f"  Novel: {stats.novel_count}, Transitional: {stats.transitional_count}")
        logger.info(f"  Familiar: {stats.familiar_count}, Baseline: {stats.baseline_count}")
        logger.info(f"  Average emergence: {stats.average_emergence:.3f}")
        
        return processed_data, stats
        
    def _extract_messages(self, entry: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract messages from entry."""
        if "messages" in entry:
            return entry["messages"]
        elif "conversation" in entry:
            return entry["conversation"]
        elif "input" in entry and "output" in entry:
            return [
                {"role": "user", "content": entry["input"]},
                {"role": "assistant", "content": entry["output"]}
            ]
        return []
        
    def _calculate_emergence_score(
        self,
        entry: Dict[str, Any],
        messages: List[Dict[str, str]],
        agent_type: Optional[str] = None
    ) -> Tuple[float, List[str]]:
        """
        Calculate emergence score for a training entry.
        
        Args:
            entry: Original entry dict
            messages: Extracted messages
            agent_type: Optional agent type
            
        Returns:
            Tuple of (emergence score, novelty features found)
        """
        novelty_features: List[str] = []
        
        # 1. Use existing emergence score if present
        if "emergence_score" in entry:
            base_score = entry["emergence_score"]
        else:
            base_score = 0.0
            
        # 2. Keyword-based novelty detection
        content = " ".join(msg.get("content", "") for msg in messages).lower()
        keyword_score, keyword_features = self._score_keywords(content)
        novelty_features.extend(keyword_features)
        
        # 3. Pattern novelty (based on history)
        pattern_score = self._score_pattern_novelty(content)
        
        # 4. Content complexity scoring
        complexity_score = self._score_complexity(messages)
        
        # 5. Agent-specific emergence scoring
        agent_score = 0.0
        if agent_type:
            agent_score = self._score_agent_emergence(content, agent_type)
            
        # 6. Use EmergenceDetector if available
        detector_score = 0.0
        if self._emergence_detector:
            try:
                detector_result = self._emergence_detector.detect(content)
                if detector_result:
                    detector_score = detector_result.get("emergence_probability", 0.0)
            except Exception as e:
                logger.debug(f"EmergenceProcessor: Detector error: {e}")
                
        # Combine scores with weights
        if base_score > 0:
            # If we have an existing score, weight it heavily
            combined_score = (
                base_score * 0.4 +
                keyword_score * 0.2 +
                pattern_score * 0.15 +
                complexity_score * 0.1 +
                agent_score * 0.1 +
                detector_score * 0.05
            )
        else:
            # Without existing score, rely more on our calculations
            combined_score = (
                keyword_score * 0.3 +
                pattern_score * 0.25 +
                complexity_score * 0.2 +
                agent_score * 0.15 +
                detector_score * 0.1
            )
            
        return min(1.0, max(0.0, combined_score)), novelty_features
        
    def _score_keywords(self, content: str) -> Tuple[float, List[str]]:
        """Score based on novelty keywords."""
        score = 0.0
        features: List[str] = []
        
        for category, keywords in self.NOVELTY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content:
                    score += 0.1
                    features.append(f"{category}:{keyword}")
                    
        return min(1.0, score), features
        
    def _score_pattern_novelty(self, content: str) -> float:
        """Score based on pattern novelty in history."""
        # Simple n-gram based novelty
        words = content.split()[:50]  # First 50 words
        if len(words) < 5:
            return 0.3
            
        # Create simple patterns
        patterns = []
        for i in range(len(words) - 2):
            pattern = " ".join(words[i:i+3])
            patterns.append(pattern)
            
        # Calculate novelty based on pattern frequency
        novel_count = 0
        for pattern in patterns:
            if self._pattern_history[pattern] == 0:
                novel_count += 1
            self._pattern_history[pattern] += 1
            
        if patterns:
            return novel_count / len(patterns)
        return 0.3
        
    def _score_complexity(self, messages: List[Dict[str, str]]) -> float:
        """Score based on content complexity."""
        score = 0.3  # Base score
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Length indicates complexity
            if len(content) > 1000:
                score += 0.2
            elif len(content) > 500:
                score += 0.1
                
            # Code indicates technical complexity
            if "```" in content:
                score += 0.15
                
            # Multiple sections/paragraphs
            if content.count("\n\n") > 3:
                score += 0.1
                
            # Lists and structure
            if content.count("- ") > 5 or content.count("1. ") > 3:
                score += 0.1
                
        return min(1.0, score)
        
    def _score_agent_emergence(self, content: str, agent_type: str) -> float:
        """Score emergence potential for specific agent types."""
        score = 0.3
        
        if agent_type == "surveyor":
            # Novel research findings
            if "new study" in content or "recent research" in content:
                score += 0.3
            if "contrary to" in content or "challenges" in content:
                score += 0.2
                
        elif agent_type == "dissident":
            # Contradictions and challenges
            if "however" in content or "but" in content:
                score += 0.2
            if "alternative" in content or "instead" in content:
                score += 0.3
                
        elif agent_type == "synthesist":
            # Cross-domain connections
            if "connects" in content or "synthesis" in content:
                score += 0.3
            if "cross-domain" in content or "interdisciplinary" in content:
                score += 0.3
                
        elif agent_type == "oracle":
            # Truth validation and principles
            if "principle" in content or "truth" in content:
                score += 0.2
            if "validated" in content or "confirmed" in content:
                score += 0.2
                
        return min(1.0, score)
        
    def create_curriculum(
        self,
        data: List[EmergenceDatapoint],
        strategy: str = "weighted"
    ) -> List[EmergenceDatapoint]:
        """
        Create a training curriculum based on emergence patterns.
        
        Args:
            data: List of emergence datapoints
            strategy: Curriculum strategy ("weighted", "novel_first", "progressive")
            
        Returns:
            Ordered list of datapoints for training
        """
        if strategy == "weighted":
            # Weighted random sampling based on curriculum weights
            return self._weighted_curriculum(data)
        elif strategy == "novel_first":
            # Novel patterns first, then transitional, then familiar
            return self._novel_first_curriculum(data)
        elif strategy == "progressive":
            # Start with familiar, gradually increase novelty
            return self._progressive_curriculum(data)
        else:
            return data
            
    def _weighted_curriculum(self, data: List[EmergenceDatapoint]) -> List[EmergenceDatapoint]:
        """Create weighted random curriculum."""
        # Expand based on weights
        expanded = []
        for dp in data:
            # Round weight to integer for repetition
            repeats = max(1, int(dp.curriculum_weight))
            expanded.extend([dp] * repeats)
            
        # Shuffle
        random.shuffle(expanded)
        
        # Remove duplicates while preserving order (keep first occurrence)
        seen_ids = set()
        result = []
        for dp in expanded:
            if dp.id not in seen_ids:
                seen_ids.add(dp.id)
                result.append(dp)
                
        return result
        
    def _novel_first_curriculum(self, data: List[EmergenceDatapoint]) -> List[EmergenceDatapoint]:
        """Create curriculum with novel patterns first."""
        # Sort by emergence score descending
        return sorted(data, key=lambda x: x.emergence_score, reverse=True)
        
    def _progressive_curriculum(self, data: List[EmergenceDatapoint]) -> List[EmergenceDatapoint]:
        """Create curriculum with progressive difficulty."""
        # Sort by emergence score ascending (familiar first)
        return sorted(data, key=lambda x: x.emergence_score)


# Convenience function
def process_for_emergence(
    data: List[Dict[str, Any]],
    agent_type: Optional[str] = None
) -> Tuple[List[EmergenceDatapoint], EmergenceStats]:
    """
    Process training data for emergence patterns.
    
    Args:
        data: List of training data entries
        agent_type: Optional agent type
        
    Returns:
        Tuple of (emergence datapoints, statistics)
    """
    processor = EmergenceProcessor()
    return processor.process(data, agent_type)

