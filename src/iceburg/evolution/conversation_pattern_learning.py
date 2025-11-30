"""
Conversation Pattern Learning
Learns conversation patterns and adapts response style
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .user_profile_builder import UserProfile
from ..memory.unified_memory import UnifiedMemory

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeModel:
    """Represents a user-specific knowledge model"""
    user_id: str
    knowledge_domains: List[str] = field(default_factory=list)
    knowledge_levels: Dict[str, float] = field(default_factory=dict)
    preferred_topics: List[str] = field(default_factory=list)
    learning_patterns: Dict[str, Any] = field(default_factory=dict)


class ConversationPatternLearning:
    """
    Learns conversation patterns from user.
    
    Adapts response style to user preferences, builds user-specific knowledge models,
    and evolves conversation structure.
    """
    
    def __init__(self, memory: Optional[UnifiedMemory] = None):
        """
        Initialize conversation pattern learning.
        
        Args:
            memory: UnifiedMemory instance (creates new if None)
        """
        self.memory = memory or UnifiedMemory()
        logger.info("Conversation Pattern Learning initialized")
    
    def learn_conversation_patterns(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Learn conversation patterns from user.
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            Dictionary of learned patterns
        """
        patterns = {
            "query_patterns": [],
            "response_patterns": [],
            "topic_patterns": [],
            "style_patterns": [],
            "frequency_patterns": {}
        }
        
        # Analyze conversations
        for conversation in conversations:
            query = conversation.get("query", "")
            response = conversation.get("response", "")
            metadata = conversation.get("metadata", {})
            
            # Learn query patterns
            query_pattern = self._analyze_query_pattern(query)
            if query_pattern:
                patterns["query_patterns"].append(query_pattern)
            
            # Learn response patterns
            response_pattern = self._analyze_response_pattern(response)
            if response_pattern:
                patterns["response_patterns"].append(response_pattern)
            
            # Learn topic patterns
            topics = metadata.get("domains", [])
            for topic in topics:
                patterns["frequency_patterns"][topic] = patterns["frequency_patterns"].get(topic, 0) + 1
        
        logger.info(f"Learned {len(patterns['query_patterns'])} query patterns, "
                   f"{len(patterns['response_patterns'])} response patterns, "
                   f"{len(patterns['frequency_patterns'])} topic patterns")
        return patterns
    
    def adapt_response_style(self, user_profile: UserProfile) -> Dict[str, Any]:
        """
        Adapt response style to user preferences.
        
        Args:
            user_profile: User profile
            
        Returns:
            Dictionary with adapted response style
        """
        style = {
            "response_style": user_profile.preferences.get("response_style", "balanced"),
            "format": user_profile.preferences.get("format", "text"),
            "detail_level": "high" if user_profile.preferences.get("response_style") == "detailed" else "medium",
            "domains": user_profile.interests[:5] if user_profile.interests else [],
            "expertise_level": self._calculate_expertise_level(user_profile)
        }
        
        logger.info(f"Adapted response style for user {user_profile.user_id}: {style['response_style']}")
        return style
    
    def build_user_knowledge_model(self, user_profile: UserProfile) -> KnowledgeModel:
        """
        Build user-specific knowledge model.
        
        Args:
            user_profile: User profile
            
        Returns:
            User-specific knowledge model
        """
        model = KnowledgeModel(
            user_id=user_profile.user_id,
            knowledge_domains=list(user_profile.domain_expertise.keys()),
            knowledge_levels=user_profile.domain_expertise.copy(),
            preferred_topics=user_profile.interests.copy(),
            learning_patterns={
                "total_conversations": user_profile.total_conversations,
                "preferences": user_profile.preferences.copy(),
                "conversation_patterns": user_profile.conversation_patterns.copy()
            }
        )
        
        logger.info(f"Built knowledge model for user {user_profile.user_id}: "
                   f"{len(model.knowledge_domains)} domains, {len(model.preferred_topics)} topics")
        return model
    
    def evolve_conversation_structure(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve conversation structure based on patterns.
        
        Args:
            patterns: Learned conversation patterns
            
        Returns:
            Dictionary with evolved conversation structure
        """
        structure = {
            "query_structure": self._evolve_query_structure(patterns.get("query_patterns", [])),
            "response_structure": self._evolve_response_structure(patterns.get("response_patterns", [])),
            "topic_prioritization": self._evolve_topic_prioritization(patterns.get("frequency_patterns", {})),
            "style_adaptation": self._evolve_style_adaptation(patterns.get("style_patterns", []))
        }
        
        logger.info("Evolved conversation structure")
        return structure
    
    def _analyze_query_pattern(self, query: str) -> Optional[Dict[str, Any]]:
        """Analyze query pattern."""
        query_lower = query.lower()
        
        # Detect question type
        question_type = "general"
        if query_lower.startswith(("what", "who", "where", "when", "why", "how")):
            question_type = query_lower.split()[0]
        
        # Detect complexity
        complexity = "simple" if len(query.split()) < 10 else "complex"
        
        return {
            "question_type": question_type,
            "complexity": complexity,
            "length": len(query.split())
        }
    
    def _analyze_response_pattern(self, response: str) -> Optional[Dict[str, Any]]:
        """Analyze response pattern."""
        response_lower = response.lower()
        
        # Detect response style
        style = "balanced"
        if "detailed" in response_lower or len(response.split()) > 200:
            style = "detailed"
        elif len(response.split()) < 50:
            style = "brief"
        
        # Detect format
        format_type = "text"
        if "```" in response:
            format_type = "code"
        elif "|" in response and "-" in response:
            format_type = "table"
        
        return {
            "style": style,
            "format": format_type,
            "length": len(response.split())
        }
    
    def _calculate_expertise_level(self, user_profile: UserProfile) -> str:
        """Calculate user expertise level."""
        if not user_profile.domain_expertise:
            return "beginner"
        
        avg_expertise = sum(user_profile.domain_expertise.values()) / len(user_profile.domain_expertise)
        
        if avg_expertise > 0.7:
            return "expert"
        elif avg_expertise > 0.4:
            return "intermediate"
        else:
            return "beginner"
    
    def _evolve_query_structure(self, query_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve query structure based on patterns."""
        if not query_patterns:
            return {"structure": "default"}
        
        # Find most common question type
        question_types = [p.get("question_type", "general") for p in query_patterns]
        most_common = max(set(question_types), key=question_types.count) if question_types else "general"
        
        return {
            "structure": "evolved",
            "preferred_question_type": most_common,
            "average_complexity": sum(p.get("length", 0) for p in query_patterns) / len(query_patterns) if query_patterns else 0
        }
    
    def _evolve_response_structure(self, response_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve response structure based on patterns."""
        if not response_patterns:
            return {"structure": "default"}
        
        # Find most common style
        styles = [p.get("style", "balanced") for p in response_patterns]
        most_common_style = max(set(styles), key=styles.count) if styles else "balanced"
        
        # Find most common format
        formats = [p.get("format", "text") for p in response_patterns]
        most_common_format = max(set(formats), key=formats.count) if formats else "text"
        
        return {
            "structure": "evolved",
            "preferred_style": most_common_style,
            "preferred_format": most_common_format,
            "average_length": sum(p.get("length", 0) for p in response_patterns) / len(response_patterns) if response_patterns else 0
        }
    
    def _evolve_topic_prioritization(self, frequency_patterns: Dict[str, int]) -> Dict[str, Any]:
        """Evolve topic prioritization based on patterns."""
        if not frequency_patterns:
            return {"prioritization": "default"}
        
        # Sort topics by frequency
        sorted_topics = sorted(frequency_patterns.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "prioritization": "evolved",
            "top_topics": [topic for topic, _ in sorted_topics[:5]],
            "topic_frequencies": frequency_patterns
        }
    
    def _evolve_style_adaptation(self, style_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve style adaptation based on patterns."""
        if not style_patterns:
            return {"adaptation": "default"}
        
        return {
            "adaptation": "evolved",
            "style_patterns": style_patterns
        }

