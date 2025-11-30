"""
Gnosis Query System
Queries complete knowledge base and searches across all accumulated research
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from .universal_knowledge_accumulator import Knowledge, UniversalKnowledgeAccumulator
from ..memory.unified_memory import UnifiedMemory
from ..graph_store import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class Research:
    """Represents a research item"""
    research_id: str
    title: str
    content: str
    domains: List[str]
    sources: List[str]
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class Connection:
    """Represents a connection between knowledge items"""
    connection_id: str
    source_id: str
    target_id: str
    connection_type: str
    description: str
    confidence: float


class GnosisQuery:
    """
    Queries complete knowledge base (gnosis).
    
    Searches across all accumulated research, finds connections across conversations,
    and provides complete knowledge context.
    """
    
    def __init__(self, accumulator: Optional[UniversalKnowledgeAccumulator] = None):
        """
        Initialize gnosis query system.
        
        Args:
            accumulator: UniversalKnowledgeAccumulator instance (creates new if None)
        """
        self.accumulator = accumulator or UniversalKnowledgeAccumulator()
        self.memory = self.accumulator.memory
        self.knowledge_graph = self.accumulator.knowledge_graph
        
        logger.info("Gnosis Query System initialized")
    
    def query_complete_knowledge(self, query: str) -> List[Knowledge]:
        """
        Query complete knowledge base.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant knowledge items
        """
        # Query gnosis knowledge base
        knowledge_items = self.accumulator.query_gnosis(query)
        
        # Also search in memory
        memory_results = self.memory.search(namespace="gnosis", query=query, k=10)
        
        # Convert memory results to knowledge
        for result in memory_results:
            # Check if already in knowledge_items
            result_id = result.get("id", "")
            if not any(k.knowledge_id == result_id for k in knowledge_items):
                knowledge = Knowledge(
                    knowledge_id=result_id,
                    content=result.get("document", ""),
                    knowledge_type=result.get("metadata", {}).get("insight_type", "fact"),
                    domains=result.get("metadata", {}).get("domains", []),
                    sources=[result.get("metadata", {}).get("insight_id", "")],
                    confidence=result.get("metadata", {}).get("confidence", 0.5),
                    metadata=result.get("metadata", {}),
                    timestamp=result.get("metadata", {}).get("timestamp", "")
                )
                knowledge_items.append(knowledge)
        
        logger.info(f"Found {len(knowledge_items)} knowledge items for query: {query[:50]}...")
        return knowledge_items
    
    def search_across_all_research(self, query: str) -> List[Research]:
        """
        Search across all accumulated research.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant research items
        """
        research_items = []
        
        # Search in gnosis knowledge base
        knowledge_items = self.query_complete_knowledge(query)
        
        # Convert knowledge to research
        for knowledge in knowledge_items:
            research = Research(
                research_id=knowledge.knowledge_id,
                title=knowledge.content[:100],
                content=knowledge.content,
                domains=knowledge.domains,
                sources=knowledge.sources,
                timestamp=knowledge.timestamp,
                metadata=knowledge.metadata
            )
            research_items.append(research)
        
        # Also search in knowledge graph
        graph_results = self.knowledge_graph.search_nodes(query)
        for result in graph_results:
            node = result.get("node", "")
            data = result.get("data", {})
            
            research = Research(
                research_id=f"graph_{node}",
                title=node,
                content=data.get("summary", node),
                domains=[data.get("type", "unknown")],
                sources=[],
                timestamp="",
                metadata=data
            )
            research_items.append(research)
        
        logger.info(f"Found {len(research_items)} research items for query: {query[:50]}...")
        return research_items
    
    def find_cross_conversation_connections(self, query: str) -> List[Connection]:
        """
        Find connections across conversations.
        
        DISABLED: Cross-conversation connections contain contaminated data from old conversations.
        Old conversations have pseudo-profound language and forced connections.
        Each query should be answered independently based on actual research, not old contaminated conversations.
        
        Args:
            query: Query string
            
        Returns:
            Empty list (connections disabled to prevent contamination)
        """
        # DISABLED: Cross-conversation connections contain contaminated data
        # Return empty list immediately to prevent contamination from old conversations
        logger.warning(f"⚠️ Cross-conversation connections DISABLED - returning empty list to prevent contamination")
        return []
    
    def get_complete_knowledge_context(self, query: str) -> Dict[str, Any]:
        """
        Get complete knowledge context for query.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with complete knowledge context
        """
        context = {
            "query": query,
            "knowledge_items": [],
            "research_items": [],
            "connections": [],
            "domains": [],
            "total_items": 0
        }
        
        # Get knowledge items
        knowledge_items = self.query_complete_knowledge(query)
        context["knowledge_items"] = [
            {
                "knowledge_id": k.knowledge_id,
                "content": k.content,
                "knowledge_type": k.knowledge_type,
                "domains": k.domains,
                "confidence": k.confidence
            }
            for k in knowledge_items
        ]
        
        # Get research items
        research_items = self.search_across_all_research(query)
        context["research_items"] = [
            {
                "research_id": r.research_id,
                "title": r.title,
                "domains": r.domains,
                "sources": r.sources
            }
            for r in research_items
        ]
        
        # DISABLED: Cross-conversation connections contain contaminated data from old conversations
        # Old conversations have pseudo-profound language and forced connections
        # Each query should be answered independently based on actual research, not old contaminated conversations
        connections = []  # Disabled to prevent contamination
        context["connections"] = []
        logger.warning(f"⚠️ Cross-conversation connections DISABLED to prevent contamination from old conversations")
        
        # Extract domains
        all_domains = set()
        for k in knowledge_items:
            all_domains.update(k.domains)
        for r in research_items:
            all_domains.update(r.domains)
        context["domains"] = list(all_domains)
        
        context["total_items"] = len(context["knowledge_items"]) + len(context["research_items"])
        
        logger.info(f"Generated complete knowledge context: {context['total_items']} items, "
                   f"{len(context['domains'])} domains, {len(context['connections'])} connections")
        return context

