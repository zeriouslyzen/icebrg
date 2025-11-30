"""
Conversation Structure Builder
Builds conversation structure that accumulates knowledge
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..gnosis.conversation_insight_extractor import ConversationInsightExtractor
from ..gnosis.universal_knowledge_accumulator import Insight

logger = logging.getLogger(__name__)


@dataclass
class ConversationStructure:
    """Represents a conversation structure"""
    conversation_id: str
    query: str
    response: str
    insights: List[Insight] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    structure_type: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ConversationStructureBuilder:
    """
    Builds conversation structure that accumulates knowledge.
    
    Each conversation contributes to gnosis, automatically extracts insights,
    and proactively fills knowledge gaps.
    """
    
    def __init__(self):
        """Initialize conversation structure builder."""
        self.insight_extractor = ConversationInsightExtractor()
        logger.info("Conversation Structure Builder initialized")
    
    def build_conversation_structure(self, query: str) -> Dict[str, Any]:
        """
        Build conversation structure for query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with conversation structure
        """
        structure = {
            "query": query,
            "structure_type": "standard",
            "insights": [],
            "knowledge_gaps": [],
            "metadata": {}
        }
        
        # Identify knowledge gaps
        gaps = self.fill_knowledge_gaps(query)
        structure["knowledge_gaps"] = gaps
        
        # Determine structure type
        if len(gaps) > 0:
            structure["structure_type"] = "gap_filling"
        
        logger.info(f"Built conversation structure for query: {query[:50]}...")
        return structure
    
    def extract_insights_automatically(self, conversation: Dict[str, Any]) -> List[Insight]:
        """
        Extract insights from conversation automatically.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            List of extracted insights
        """
        insights = self.insight_extractor.extract_insights(conversation)
        
        logger.info(f"Extracted {len(insights)} insights automatically from conversation")
        return insights
    
    def fill_knowledge_gaps(self, query: str) -> List[str]:
        """
        Fill knowledge gaps in query.
        
        Args:
            query: User query
            
        Returns:
            List of identified knowledge gaps
        """
        gaps = []
        
        query_lower = query.lower()
        
        # Identify gaps based on query structure
        gap_indicators = [
            "what is", "how does", "why is", "when does", "where is",
            "unknown", "unclear", "uncertain", "mystery", "puzzle"
        ]
        
        for indicator in gap_indicators:
            if indicator in query_lower:
                gap = f"Gap identified: {indicator} in query"
                gaps.append(gap)
        
        # Identify missing context
        if len(query.split()) < 5:
            gaps.append("Gap: Query lacks sufficient context")
        
        logger.info(f"Identified {len(gaps)} knowledge gaps for query: {query[:50]}...")
        return gaps
    
    def contribute_to_gnosis(self, conversation: Dict[str, Any], accumulator) -> None:
        """
        Contribute conversation to gnosis knowledge base.
        
        Args:
            conversation: Conversation dictionary
            accumulator: UniversalKnowledgeAccumulator instance
        """
        # Extract insights
        insights = self.extract_insights_automatically(conversation)
        
        # Accumulate to gnosis
        accumulator.accumulate_to_gnosis(insights)
        
        logger.info(f"Contributed conversation to gnosis: {len(insights)} insights")

