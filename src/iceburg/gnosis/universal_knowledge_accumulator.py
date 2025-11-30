"""
Universal Knowledge Accumulator
Accumulates all research from all conversations into gnosis knowledge base
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

from ..memory.unified_memory import UnifiedMemory
from ..graph_store import KnowledgeGraph
from ..config import IceburgConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """Represents an extracted insight"""
    insight_id: str
    source_conversation_id: str
    insight_type: str  # "pattern", "connection", "principle", "fact", "prediction"
    content: str
    confidence: float
    domains: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Knowledge:
    """Represents a piece of knowledge in gnosis"""
    knowledge_id: str
    content: str
    knowledge_type: str  # "fact", "principle", "pattern", "connection", "prediction"
    domains: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class GnosisKnowledgeBase:
    """Complete gnosis knowledge base"""
    knowledge_items: Dict[str, Knowledge] = field(default_factory=dict)
    insights: List[Insight] = field(default_factory=list)
    total_conversations: int = 0
    total_insights: int = 0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class UniversalKnowledgeAccumulator:
    """
    Accumulates all research from all conversations into gnosis knowledge base.
    
    Extracts insights from every conversation and stores in unified knowledge structure.
    """
    
    def __init__(self, cfg: Optional[IceburgConfig] = None):
        """
        Initialize universal knowledge accumulator.
        
        Args:
            cfg: ICEBURG config (loads if None)
        """
        self.cfg = cfg or load_config()
        self.memory = UnifiedMemory(self.cfg)
        self.knowledge_graph = KnowledgeGraph(self.cfg)
        self.gnosis_base: Optional[GnosisKnowledgeBase] = None
        self._gnosis_path = Path(self.cfg.data_dir) / "gnosis" / "knowledge_base.json"
        self._gnosis_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing gnosis base
        self._load_gnosis_base()
        
        logger.info("Universal Knowledge Accumulator initialized")
    
    def extract_insights_from_conversation(self, conversation: Dict[str, Any]) -> List[Insight]:
        """
        Extract insights from a conversation.
        
        Args:
            conversation: Conversation dictionary with query, response, metadata
            
        Returns:
            List of extracted insights
        """
        insights = []
        conversation_id = conversation.get("conversation_id", f"conv_{datetime.utcnow().timestamp()}")
        
        # Extract query and response
        query = conversation.get("query", "")
        response = conversation.get("response", "")
        metadata = conversation.get("metadata", {})
        
        # Extract patterns
        patterns = self._extract_patterns(query, response)
        for pattern in patterns:
            insight = Insight(
                insight_id=f"insight_{len(insights)}_{datetime.utcnow().timestamp()}",
                source_conversation_id=conversation_id,
                insight_type="pattern",
                content=pattern,
                confidence=0.7,
                domains=metadata.get("domains", []),
                evidence=[query, response[:200]]
            )
            insights.append(insight)
        
        # Extract connections
        connections = self._extract_connections(query, response)
        for connection in connections:
            insight = Insight(
                insight_id=f"insight_{len(insights)}_{datetime.utcnow().timestamp()}",
                source_conversation_id=conversation_id,
                insight_type="connection",
                content=connection,
                confidence=0.6,
                domains=metadata.get("domains", []),
                evidence=[query, response[:200]]
            )
            insights.append(insight)
        
        # Extract principles
        principles = self._extract_principles(query, response)
        for principle in principles:
            insight = Insight(
                insight_id=f"insight_{len(insights)}_{datetime.utcnow().timestamp()}",
                source_conversation_id=conversation_id,
                insight_type="principle",
                content=principle,
                confidence=0.8,
                domains=metadata.get("domains", []),
                evidence=[query, response[:200]]
            )
            insights.append(insight)
        
        # Extract facts
        facts = self._extract_facts(query, response)
        for fact in facts:
            insight = Insight(
                insight_id=f"insight_{len(insights)}_{datetime.utcnow().timestamp()}",
                source_conversation_id=conversation_id,
                insight_type="fact",
                content=fact,
                confidence=0.9,
                domains=metadata.get("domains", []),
                evidence=[query, response[:200]]
            )
            insights.append(insight)
        
        logger.info(f"Extracted {len(insights)} insights from conversation {conversation_id}")
        return insights
    
    def accumulate_to_gnosis(self, insights: List[Insight]) -> None:
        """
        Accumulate insights into gnosis knowledge base.
        
        Args:
            insights: List of insights to accumulate
        """
        if self.gnosis_base is None:
            self.gnosis_base = GnosisKnowledgeBase()
        
        # Add insights to gnosis base
        for insight in insights:
            self.gnosis_base.insights.append(insight)
            self.gnosis_base.total_insights += 1
            
            # Convert insight to knowledge
            knowledge = Knowledge(
                knowledge_id=f"knowledge_{len(self.gnosis_base.knowledge_items)}",
                content=insight.content,
                knowledge_type=insight.insight_type,
                domains=insight.domains,
                sources=[insight.source_conversation_id],
                confidence=insight.confidence,
                metadata=insight.metadata,
                timestamp=insight.timestamp
            )
            
            self.gnosis_base.knowledge_items[knowledge.knowledge_id] = knowledge
            
            # Index in memory
            # Convert domains list to string for ChromaDB compatibility
            domains_str = ", ".join(insight.domains) if insight.domains else ""
            self.memory.index_texts(
                namespace="gnosis",
                texts=[insight.content],
                metadatas=[{
                    "insight_id": insight.insight_id,
                    "insight_type": insight.insight_type,
                    "domains": domains_str,
                    "confidence": str(insight.confidence),
                    "timestamp": insight.timestamp
                }]
            )
            
            # Add to knowledge graph
            for domain in insight.domains:
                self.knowledge_graph.add_synthesis(
                    title=insight.content[:100],
                    domains=[domain],
                    principle=insight.content if insight.insight_type == "principle" else "",
                    evidence=[(insight.content[:50], insight.source_conversation_id)]
                )
        
        # Update timestamp
        self.gnosis_base.last_updated = datetime.utcnow().isoformat()
        self.gnosis_base.total_conversations += 1
        
        # Save gnosis base
        self._save_gnosis_base()
        
        logger.info(f"Accumulated {len(insights)} insights to gnosis knowledge base")
    
    def query_gnosis(self, query: str) -> List[Knowledge]:
        """
        Query gnosis knowledge base.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant knowledge items
        """
        if self.gnosis_base is None:
            return []
        
        # Search in memory
        memory_results = self.memory.search(namespace="gnosis", query=query, k=10)
        
        # Convert to knowledge items
        knowledge_items = []
        for result in memory_results:
            insight_id = result.get("metadata", {}).get("insight_id")
            if insight_id:
                # Find corresponding knowledge
                for knowledge in self.gnosis_base.knowledge_items.values():
                    if insight_id in knowledge.sources:
                        knowledge_items.append(knowledge)
                        break
        
        # Also search knowledge base directly
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for knowledge in self.gnosis_base.knowledge_items.values():
            content_lower = knowledge.content.lower()
            if any(word in content_lower for word in query_words):
                if knowledge not in knowledge_items:
                    knowledge_items.append(knowledge)
        
        logger.info(f"Found {len(knowledge_items)} knowledge items for query: {query[:50]}...")
        return knowledge_items
    
    def build_complete_knowledge_base(self) -> GnosisKnowledgeBase:
        """
        Build complete gnosis knowledge base from all conversations.
        
        Returns:
            Complete gnosis knowledge base
        """
        if self.gnosis_base is None:
            self.gnosis_base = GnosisKnowledgeBase()
        
        # Load all conversations from memory
        events_dir = Path(self.cfg.data_dir) / "memory" / "events"
        if events_dir.exists():
            for event_file in events_dir.glob("*.jsonl"):
                try:
                    with open(event_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            event = json.loads(line)
                            if event.get("event_type") == "conversation":
                                conversation = {
                                    "conversation_id": event.get("run_id", ""),
                                    "query": event.get("payload", {}).get("query", ""),
                                    "response": event.get("payload", {}).get("response", ""),
                                    "metadata": event.get("payload", {}).get("metadata", {})
                                }
                                
                                # Extract insights
                                insights = self.extract_insights_from_conversation(conversation)
                                
                                # Accumulate to gnosis
                                self.accumulate_to_gnosis(insights)
                except Exception as e:
                    logger.warning(f"Error processing event file {event_file}: {e}")
        
        logger.info(f"Built complete knowledge base: {len(self.gnosis_base.knowledge_items)} knowledge items, "
                   f"{len(self.gnosis_base.insights)} insights, {self.gnosis_base.total_conversations} conversations")
        return self.gnosis_base
    
    def _extract_patterns(self, query: str, response: str) -> List[str]:
        """Extract patterns from query and response."""
        patterns = []
        
        # Simple pattern extraction (can be enhanced with NLP)
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Look for pattern indicators
        pattern_indicators = ["pattern", "trend", "correlation", "relationship", "connection", "link"]
        
        for indicator in pattern_indicators:
            if indicator in query_lower or indicator in response_lower:
                # Extract surrounding context
                if indicator in response_lower:
                    idx = response_lower.find(indicator)
                    pattern = response[max(0, idx-50):idx+100]
                    if pattern not in patterns:
                        patterns.append(pattern.strip())
        
        return patterns
    
    def _extract_connections(self, query: str, response: str) -> List[str]:
        """Extract connections from query and response."""
        connections = []
        
        # Simple connection extraction
        connection_indicators = ["connects", "relates", "links", "associates", "correlates", "relates to"]
        
        for indicator in connection_indicators:
            if indicator in response.lower():
                idx = response.lower().find(indicator)
                connection = response[max(0, idx-50):idx+100]
                if connection not in connections:
                    connections.append(connection.strip())
        
        return connections
    
    def _extract_principles(self, query: str, response: str) -> List[str]:
        """Extract principles from query and response."""
        principles = []
        
        # Simple principle extraction
        principle_indicators = ["principle", "rule", "law", "theorem", "axiom", "fundamental"]
        
        for indicator in principle_indicators:
            if indicator in response.lower():
                idx = response.lower().find(indicator)
                principle = response[max(0, idx-50):idx+150]
                if principle not in principles:
                    principles.append(principle.strip())
        
        return principles
    
    def _extract_facts(self, query: str, response: str) -> List[str]:
        """Extract facts from query and response."""
        facts = []
        
        # Simple fact extraction (statements that look like facts)
        fact_indicators = ["is", "are", "was", "were", "has", "have", "contains", "includes"]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in fact_indicators):
                if len(sentence.strip()) > 20:  # Filter out very short sentences
                    facts.append(sentence.strip())
        
        return facts[:5]  # Limit to 5 facts
    
    def _load_gnosis_base(self) -> None:
        """Load gnosis base from disk."""
        if self._gnosis_path.exists():
            try:
                with open(self._gnosis_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Reconstruct gnosis base
                    self.gnosis_base = GnosisKnowledgeBase(
                        knowledge_items={
                            kid: Knowledge(**kdata) for kid, kdata in data.get("knowledge_items", {}).items()
                        },
                        insights=[
                            Insight(**idata) for idata in data.get("insights", [])
                        ],
                        total_conversations=data.get("total_conversations", 0),
                        total_insights=data.get("total_insights", 0),
                        last_updated=data.get("last_updated", datetime.utcnow().isoformat())
                    )
            except Exception as e:
                logger.warning(f"Error loading gnosis base: {e}")
                self.gnosis_base = GnosisKnowledgeBase()
        else:
            self.gnosis_base = GnosisKnowledgeBase()
    
    def _save_gnosis_base(self) -> None:
        """Save gnosis base to disk."""
        if self.gnosis_base is None:
            return
        
        try:
            data = {
                "knowledge_items": {
                    kid: {
                        "knowledge_id": k.knowledge_id,
                        "content": k.content,
                        "knowledge_type": k.knowledge_type,
                        "domains": k.domains,
                        "sources": k.sources,
                        "confidence": k.confidence,
                        "metadata": k.metadata,
                        "timestamp": k.timestamp
                    }
                    for kid, k in self.gnosis_base.knowledge_items.items()
                },
                "insights": [
                    {
                        "insight_id": i.insight_id,
                        "source_conversation_id": i.source_conversation_id,
                        "insight_type": i.insight_type,
                        "content": i.content,
                        "confidence": i.confidence,
                        "domains": i.domains,
                        "evidence": i.evidence,
                        "metadata": i.metadata,
                        "timestamp": i.timestamp
                    }
                    for i in self.gnosis_base.insights
                ],
                "total_conversations": self.gnosis_base.total_conversations,
                "total_insights": self.gnosis_base.total_insights,
                "last_updated": self.gnosis_base.last_updated
            }
            
            with open(self._gnosis_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving gnosis base: {e}")

