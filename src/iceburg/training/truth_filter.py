"""
ICEBURG Truth Filter
====================

Filters training data using ICEBURG's Instant Truth System and quality metrics.
Prioritizes high-quality, verified insights for fine-tuning.

Features:
- Integration with InstantTruthSystem for pattern matching
- Quality-based filtering (min score threshold)
- Duplicate detection and removal
- Truth-seeking metric scoring
- Data validation and sanitization
"""

from __future__ import annotations
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TruthCategory(Enum):
    """Categories of truth assessment."""
    VERIFIED = "verified"           # High confidence, verified by multiple sources
    HIGH_QUALITY = "high_quality"   # Good quality, passed quality threshold
    STANDARD = "standard"           # Meets minimum requirements
    LOW_QUALITY = "low_quality"     # Below threshold, filtered out
    DUPLICATE = "duplicate"         # Duplicate of existing data


@dataclass
class FilteredDatapoint:
    """Represents a filtered training datapoint."""
    id: str
    messages: List[Dict[str, str]]
    quality_score: float
    truth_category: TruthCategory
    metadata: Dict[str, Any] = field(default_factory=dict)
    emergence_score: float = 0.0
    content_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "messages": self.messages,
            "quality_score": self.quality_score,
            "truth_category": self.truth_category.value,
            "metadata": self.metadata,
            "emergence_score": self.emergence_score,
            "content_hash": self.content_hash
        }


@dataclass
class FilterStats:
    """Statistics from filtering operation."""
    total_input: int = 0
    verified: int = 0
    high_quality: int = 0
    standard: int = 0
    low_quality_removed: int = 0
    duplicates_removed: int = 0
    final_output: int = 0
    average_quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_input": self.total_input,
            "verified": self.verified,
            "high_quality": self.high_quality,
            "standard": self.standard,
            "low_quality_removed": self.low_quality_removed,
            "duplicates_removed": self.duplicates_removed,
            "final_output": self.final_output,
            "average_quality_score": self.average_quality_score
        }


class TruthFilter:
    """
    Filters training data using truth-seeking metrics.
    
    Integrates with ICEBURG's InstantTruthSystem to identify and prioritize
    high-quality, verified training data for fine-tuning.
    """
    
    def __init__(
        self,
        min_quality_score: float = 0.7,
        verified_threshold: float = 0.9,
        high_quality_threshold: float = 0.8,
        remove_duplicates: bool = True,
        use_instant_truth: bool = True
    ):
        """
        Initialize truth filter.
        
        Args:
            min_quality_score: Minimum quality score to include (0.0-1.0)
            verified_threshold: Threshold for "verified" category
            high_quality_threshold: Threshold for "high_quality" category
            remove_duplicates: Whether to remove duplicate entries
            use_instant_truth: Whether to use InstantTruthSystem for pattern matching
        """
        self.min_quality_score = min_quality_score
        self.verified_threshold = verified_threshold
        self.high_quality_threshold = high_quality_threshold
        self.remove_duplicates = remove_duplicates
        self.use_instant_truth = use_instant_truth
        
        # Content hashes for duplicate detection
        self._seen_hashes: set = set()
        
        # Try to import InstantTruthSystem
        self._truth_system = None
        if use_instant_truth:
            try:
                from ..optimization.instant_truth_system import InstantTruthSystem, get_truth_system
                self._truth_system = get_truth_system()
                logger.info("TruthFilter: InstantTruthSystem integration enabled")
            except ImportError:
                logger.warning("TruthFilter: InstantTruthSystem not available, using fallback scoring")
                
    def filter(
        self,
        data: List[Dict[str, Any]],
        agent_type: Optional[str] = None
    ) -> Tuple[List[FilteredDatapoint], FilterStats]:
        """
        Filter training data for quality and truth.
        
        Args:
            data: List of training data entries
            agent_type: Optional agent type for agent-specific filtering
            
        Returns:
            Tuple of (filtered datapoints, filter statistics)
        """
        stats = FilterStats(total_input=len(data))
        filtered_data: List[FilteredDatapoint] = []
        
        self._seen_hashes.clear()
        quality_scores: List[float] = []
        
        for i, entry in enumerate(data):
            # Extract messages
            messages = self._extract_messages(entry)
            if not messages:
                stats.low_quality_removed += 1
                continue
                
            # Calculate content hash for duplicate detection
            content_hash = self._compute_hash(messages)
            
            # Check for duplicates
            if self.remove_duplicates and content_hash in self._seen_hashes:
                stats.duplicates_removed += 1
                continue
                
            self._seen_hashes.add(content_hash)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(entry, messages, agent_type)
            
            # Categorize based on quality
            if quality_score >= self.verified_threshold:
                category = TruthCategory.VERIFIED
                stats.verified += 1
            elif quality_score >= self.high_quality_threshold:
                category = TruthCategory.HIGH_QUALITY
                stats.high_quality += 1
            elif quality_score >= self.min_quality_score:
                category = TruthCategory.STANDARD
                stats.standard += 1
            else:
                category = TruthCategory.LOW_QUALITY
                stats.low_quality_removed += 1
                continue
                
            # Create filtered datapoint
            datapoint = FilteredDatapoint(
                id=f"data_{i}_{content_hash[:8]}",
                messages=messages,
                quality_score=quality_score,
                truth_category=category,
                metadata=entry.get("metadata", {}),
                emergence_score=entry.get("emergence_score", 0.0),
                content_hash=content_hash
            )
            
            filtered_data.append(datapoint)
            quality_scores.append(quality_score)
            
        # Calculate final stats
        stats.final_output = len(filtered_data)
        if quality_scores:
            stats.average_quality_score = sum(quality_scores) / len(quality_scores)
            
        logger.info(f"TruthFilter: {stats.final_output}/{stats.total_input} entries passed filtering")
        logger.info(f"  Verified: {stats.verified}, High Quality: {stats.high_quality}, Standard: {stats.standard}")
        logger.info(f"  Removed: {stats.low_quality_removed} low quality, {stats.duplicates_removed} duplicates")
        
        return filtered_data, stats
        
    def _extract_messages(self, entry: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract messages from entry in ChatML format."""
        # Try different formats
        if "messages" in entry:
            return entry["messages"]
        elif "conversation" in entry:
            return entry["conversation"]
        elif "input" in entry and "output" in entry:
            # Convert input/output format to messages
            return [
                {"role": "user", "content": entry["input"]},
                {"role": "assistant", "content": entry["output"]}
            ]
        elif "prompt" in entry and "response" in entry:
            return [
                {"role": "user", "content": entry["prompt"]},
                {"role": "assistant", "content": entry["response"]}
            ]
        else:
            return []
            
    def _compute_hash(self, messages: List[Dict[str, str]]) -> str:
        """Compute content hash for duplicate detection."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
        
    def _calculate_quality_score(
        self,
        entry: Dict[str, Any],
        messages: List[Dict[str, str]],
        agent_type: Optional[str] = None
    ) -> float:
        """
        Calculate quality score for a training entry.
        
        Args:
            entry: Original entry dict
            messages: Extracted messages
            agent_type: Optional agent type for specialized scoring
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        weights_sum = 0.0
        
        # 1. Existing quality score (if present)
        if "quality_score" in entry:
            score += entry["quality_score"] * 0.4
            weights_sum += 0.4
        elif "quality" in entry:
            score += entry["quality"] * 0.4
            weights_sum += 0.4
            
        # 2. Message content quality
        content_score = self._score_content_quality(messages)
        score += content_score * 0.3
        weights_sum += 0.3
        
        # 3. Conversation structure quality
        structure_score = self._score_structure_quality(messages)
        score += structure_score * 0.15
        weights_sum += 0.15
        
        # 4. Agent-specific scoring
        if agent_type:
            agent_score = self._score_agent_specific(messages, agent_type)
            score += agent_score * 0.1
            weights_sum += 0.1
            
        # 5. InstantTruthSystem pattern matching
        if self._truth_system:
            truth_score = self._score_with_truth_system(messages)
            score += truth_score * 0.05
            weights_sum += 0.05
            
        # Normalize
        if weights_sum > 0:
            score = score / weights_sum
            
        return min(1.0, max(0.0, score))
        
    def _score_content_quality(self, messages: List[Dict[str, str]]) -> float:
        """Score based on content quality indicators."""
        score = 0.5  # Base score
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Length scoring (longer responses often higher quality)
            if len(content) > 500:
                score += 0.1
            elif len(content) > 200:
                score += 0.05
            elif len(content) < 50:
                score -= 0.1
                
            # Check for code blocks (higher quality for code)
            if "```" in content:
                score += 0.1
                
            # Check for structured content (lists, headers)
            if any(marker in content for marker in ["- ", "* ", "1. ", "## ", "### "]):
                score += 0.05
                
            # Penalize very short or generic responses
            generic_phrases = ["I don't know", "I'm not sure", "I can't help"]
            if any(phrase.lower() in content.lower() for phrase in generic_phrases):
                score -= 0.2
                
        return min(1.0, max(0.0, score))
        
    def _score_structure_quality(self, messages: List[Dict[str, str]]) -> float:
        """Score based on conversation structure."""
        score = 0.5
        
        # Check for proper role alternation
        roles = [msg.get("role") for msg in messages]
        
        # Should have at least user and assistant messages
        if "user" in roles and "assistant" in roles:
            score += 0.2
            
        # Proper alternation is good
        for i in range(len(roles) - 1):
            if roles[i] != roles[i + 1]:
                score += 0.05
            else:
                score -= 0.1
                
        # System message at start is good
        if roles and roles[0] == "system":
            score += 0.1
            
        return min(1.0, max(0.0, score))
        
    def _score_agent_specific(self, messages: List[Dict[str, str]], agent_type: str) -> float:
        """Score based on agent-specific criteria."""
        score = 0.5
        
        content = " ".join(msg.get("content", "") for msg in messages).lower()
        
        if agent_type == "surveyor":
            # Surveyor should have research-like content
            research_keywords = ["research", "study", "evidence", "source", "finding"]
            score += 0.1 * sum(1 for kw in research_keywords if kw in content)
            
        elif agent_type == "dissident":
            # Dissident should challenge and question
            challenge_keywords = ["however", "but", "alternatively", "contradiction", "challenge"]
            score += 0.1 * sum(1 for kw in challenge_keywords if kw in content)
            
        elif agent_type == "synthesist":
            # Synthesist should connect and integrate
            synthesis_keywords = ["connect", "integrate", "combine", "synthesis", "cross-domain"]
            score += 0.1 * sum(1 for kw in synthesis_keywords if kw in content)
            
        elif agent_type == "oracle":
            # Oracle should validate and conclude
            oracle_keywords = ["truth", "validate", "conclude", "principle", "insight"]
            score += 0.1 * sum(1 for kw in oracle_keywords if kw in content)
            
        return min(1.0, max(0.0, score))
        
    def _score_with_truth_system(self, messages: List[Dict[str, str]]) -> float:
        """Score using InstantTruthSystem pattern matching."""
        if not self._truth_system:
            return 0.5
            
        try:
            # Extract query from messages
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                return 0.5
                
            query = user_messages[0].get("content", "")
            
            # Check if this matches a known truth pattern
            instant_truth = self._truth_system.get_instant_truth(query)
            if instant_truth:
                return 0.9  # High score for verified patterns
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"TruthFilter: Error in truth system scoring: {e}")
            return 0.5
            
    def filter_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        agent_type: Optional[str] = None
    ) -> Tuple[List[FilteredDatapoint], FilterStats]:
        """
        Filter a JSONL file of training data.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Optional path to save filtered data
            agent_type: Optional agent type for filtering
            
        Returns:
            Tuple of (filtered datapoints, statistics)
        """
        # Load data
        data = []
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        # Filter
        filtered, stats = self.filter(data, agent_type)
        
        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for dp in filtered:
                    f.write(json.dumps(dp.to_dict()) + "\n")
            logger.info(f"TruthFilter: Saved {len(filtered)} entries to {output_path}")
            
        return filtered, stats


# Convenience function
def filter_training_data(
    data: List[Dict[str, Any]],
    min_quality: float = 0.7,
    agent_type: Optional[str] = None
) -> Tuple[List[FilteredDatapoint], FilterStats]:
    """
    Filter training data for quality.
    
    Args:
        data: List of training data entries
        min_quality: Minimum quality score
        agent_type: Optional agent type
        
    Returns:
        Tuple of (filtered data, statistics)
    """
    filter = TruthFilter(min_quality_score=min_quality)
    return filter.filter(data, agent_type)

