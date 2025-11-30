"""
Total Knowledge Accumulator
Complete knowledge graph, origins-to-present mapping, limitless totality
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..gnosis.universal_knowledge_accumulator import UniversalKnowledgeAccumulator
from ..graph_store import KnowledgeGraph
from .deep_etymology_tracing import DeepEtymologyTracing
from .occult_knowledge_database import OccultKnowledgeDatabase
from .predictive_history import PredictiveHistorySystem
from ..config import IceburgConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class TotalKnowledgeItem:
    """Represents a total knowledge item"""
    knowledge_id: str
    term: str
    etymology_trace: Optional[Any] = None
    occult_connections: List[Any] = field(default_factory=list)
    secret_society_connections: List[Any] = field(default_factory=list)
    suppressed_knowledge: List[str] = field(default_factory=list)
    historical_patterns: List[Any] = field(default_factory=list)
    origins_to_present: str = ""
    complete_knowledge_chain: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TotalKnowledgeAccumulator:
    """
    Total knowledge accumulator - limitless totality of all knowledge.
    
    Builds complete knowledge graph from origins to present, including:
    - Etymology traces
    - Occult connections
    - Secret society patterns
    - Suppressed knowledge
    - Historical patterns
    - Origins-to-present mapping
    """
    
    def __init__(self, cfg: Optional[IceburgConfig] = None):
        """
        Initialize total knowledge accumulator.
        
        Args:
            cfg: ICEBURG config (loads if None)
        """
        self.cfg = cfg or load_config()
        self.gnosis_accumulator = UniversalKnowledgeAccumulator(self.cfg)
        self.knowledge_graph = KnowledgeGraph(self.cfg)
        self.etymology_tracing = DeepEtymologyTracing()
        self.occult_database = OccultKnowledgeDatabase()
        self.predictive_history = PredictiveHistorySystem()
        
        self.total_knowledge: Dict[str, TotalKnowledgeItem] = {}
        
        logger.info("Total Knowledge Accumulator initialized")
    
    def accumulate_total_knowledge(self, term: str) -> TotalKnowledgeItem:
        """
        Accumulate total knowledge for a term.
        
        Args:
            term: Term to accumulate knowledge for
            
        Returns:
            Total knowledge item with complete knowledge chain
        """
        # Check cache
        if term.lower() in self.total_knowledge:
            return self.total_knowledge[term.lower()]
        
        knowledge_item = TotalKnowledgeItem(
            knowledge_id=f"total_{term.lower().replace(' ', '_')}",
            term=term
        )
        
        # Trace etymology
        etymology_trace = self.etymology_tracing.trace_deep_etymology(term)
        knowledge_item.etymology_trace = etymology_trace
        knowledge_item.origins_to_present = etymology_trace.complete_chain
        
        # Find occult connections
        occult_conns = self.occult_database.find_occult_connections(term)
        knowledge_item.occult_connections = occult_conns
        
        # Find secret society connections
        secret_societies = self.occult_database.find_secret_society_connections(term)
        knowledge_item.secret_society_connections = secret_societies
        
        # Find suppressed knowledge
        suppressed = self.occult_database.find_suppressed_knowledge(term)
        knowledge_item.suppressed_knowledge = suppressed
        
        # Match historical patterns
        historical_patterns = self.predictive_history.match_historical_patterns(term)
        knowledge_item.historical_patterns = historical_patterns
        
        # Build complete knowledge chain
        knowledge_item.complete_knowledge_chain = self._build_complete_knowledge_chain(
            term, etymology_trace, occult_conns, secret_societies, suppressed, historical_patterns
        )
        
        # Calculate confidence
        knowledge_item.confidence = self._calculate_total_confidence(
            etymology_trace, occult_conns, secret_societies, suppressed, historical_patterns
        )
        
        # Store in gnosis
        self._store_in_gnosis(knowledge_item)
        
        # Add to knowledge graph
        self._add_to_knowledge_graph(knowledge_item)
        
        # Cache result
        self.total_knowledge[term.lower()] = knowledge_item
        
        logger.info(f"Accumulated total knowledge for '{term}': "
                   f"{len(occult_conns)} occult connections, "
                   f"{len(secret_societies)} secret societies, "
                   f"{len(suppressed)} suppressed knowledge items, "
                   f"{len(historical_patterns)} historical patterns")
        
        return knowledge_item
    
    def decode_complete_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Decode complete knowledge for a query.
        
        Args:
            query: Query to decode
            
        Returns:
            Dictionary with complete knowledge decoding
        """
        decoding = {
            "query": query,
            "terms": [],
            "etymology_traces": [],
            "occult_connections": [],
            "secret_society_connections": [],
            "suppressed_knowledge": [],
            "historical_patterns": [],
            "complete_knowledge_chains": [],
            "origins_to_present": []
        }
        
        # Extract terms from query
        terms = self._extract_terms(query)
        decoding["terms"] = terms
        
        # Accumulate total knowledge for each term
        for term in terms:
            knowledge_item = self.accumulate_total_knowledge(term)
            
            # Add to decoding
            if knowledge_item.etymology_trace:
                decoding["etymology_traces"].append({
                    "term": term,
                    "trace": knowledge_item.etymology_trace.complete_chain,
                    "occult_path": knowledge_item.etymology_trace.occult_path,
                    "secret_society_path": knowledge_item.etymology_trace.secret_society_path
                })
            
            decoding["occult_connections"].extend([
                {
                    "term": term,
                    "connection": {
                        "source": c.source_term,
                        "target": c.target_term,
                        "type": c.connection_type,
                        "description": c.description
                    }
                }
                for c in knowledge_item.occult_connections
            ])
            
            decoding["secret_society_connections"].extend([
                {
                    "term": term,
                    "society": {
                        "name": s.name,
                        "description": s.description
                    }
                }
                for s in knowledge_item.secret_society_connections
            ])
            
            decoding["suppressed_knowledge"].extend([
                {"term": term, "knowledge": supp}
                for supp in knowledge_item.suppressed_knowledge
            ])
            
            decoding["historical_patterns"].extend([
                {
                    "term": term,
                    "pattern": {
                        "type": p.pattern_type,
                        "description": p.description,
                        "time_period": p.time_period
                    }
                }
                for p in knowledge_item.historical_patterns
            ])
            
            decoding["complete_knowledge_chains"].append(knowledge_item.complete_knowledge_chain)
            decoding["origins_to_present"].append(knowledge_item.origins_to_present)
        
        logger.info(f"Decoded complete knowledge for query: {len(terms)} terms, "
                   f"{len(decoding['etymology_traces'])} etymology traces, "
                   f"{len(decoding['occult_connections'])} occult connections")
        
        return decoding
    
    def _extract_terms(self, query: str) -> List[str]:
        """Extract terms from query."""
        # Simple term extraction - in production, use NLP
        terms = []
        
        # Common terms
        common_terms = [
            "astrology", "astronomy", "star", "constellation", "zodiac",
            "planet", "sun", "moon", "celestial", "occult", "secret",
            "society", "suppressed", "knowledge", "history", "pattern"
        ]
        
        query_lower = query.lower()
        for term in common_terms:
            if term in query_lower:
                terms.append(term)
        
        # If no common terms found, extract key words
        if not terms:
            words = query_lower.split()
            # Take first 3 significant words (length > 3)
            terms = [w for w in words if len(w) > 3][:3]
        
        return terms
    
    def _build_complete_knowledge_chain(self, term: str, etymology_trace: Any,
                                      occult_conns: List[Any], secret_societies: List[Any],
                                      suppressed: List[str], historical_patterns: List[Any]) -> str:
        """Build complete knowledge chain."""
        chain_parts = [f"Term: {term}"]
        
        # Add etymology
        if etymology_trace and etymology_trace.complete_chain:
            chain_parts.append(f"Etymology: {etymology_trace.complete_chain}")
        
        # Add occult connections
        if occult_conns:
            occult_targets = [c.target_term for c in occult_conns[:3]]
            chain_parts.append(f"Occult: {' → '.join(occult_targets)}")
        
        # Add secret societies
        if secret_societies:
            society_names = [s.name for s in secret_societies[:2]]
            chain_parts.append(f"Secret Societies: {' → '.join(society_names)}")
        
        # Add suppressed knowledge
        if suppressed:
            chain_parts.append(f"Suppressed: {' → '.join(suppressed[:2])}")
        
        # Add historical patterns
        if historical_patterns:
            pattern_descs = [p.description[:50] for p in historical_patterns[:2]]
            chain_parts.append(f"Historical: {' → '.join(pattern_descs)}")
        
        return " | ".join(chain_parts)
    
    def _calculate_total_confidence(self, etymology_trace: Any, occult_conns: List[Any],
                                   secret_societies: List[Any], suppressed: List[str],
                                   historical_patterns: List[Any]) -> float:
        """Calculate total confidence."""
        confidence = 0.0
        
        # Etymology confidence
        if etymology_trace:
            confidence += etymology_trace.confidence * 0.3
        
        # Occult connections confidence
        if occult_conns:
            avg_occult_conf = sum(c.confidence for c in occult_conns) / len(occult_conns)
            confidence += avg_occult_conf * 0.2
        
        # Secret society confidence
        if secret_societies:
            confidence += 0.2
        
        # Suppressed knowledge confidence
        if suppressed:
            confidence += 0.15
        
        # Historical patterns confidence
        if historical_patterns:
            avg_pattern_conf = sum(p.confidence for p in historical_patterns) / len(historical_patterns)
            confidence += avg_pattern_conf * 0.15
        
        return min(confidence, 1.0)
    
    def _store_in_gnosis(self, knowledge_item: TotalKnowledgeItem) -> None:
        """Store in gnosis knowledge base."""
        # Create conversation for gnosis
        conversation = {
            "conversation_id": f"total_knowledge_{knowledge_item.knowledge_id}",
            "query": f"What is {knowledge_item.term}?",
            "response": knowledge_item.complete_knowledge_chain,
            "metadata": {
                "domains": ["etymology", "occult", "history", "suppressed_knowledge"],
                "total_knowledge": True,
                "etymology_trace": str(knowledge_item.etymology_trace) if knowledge_item.etymology_trace else None,
                "occult_connections": len(knowledge_item.occult_connections),
                "secret_society_connections": len(knowledge_item.secret_society_connections),
                "suppressed_knowledge": len(knowledge_item.suppressed_knowledge),
                "historical_patterns": len(knowledge_item.historical_patterns)
            }
        }
        
        # Extract insights and accumulate
        insights = self.gnosis_accumulator.extract_insights_from_conversation(conversation)
        self.gnosis_accumulator.accumulate_to_gnosis(insights)
    
    def _add_to_knowledge_graph(self, knowledge_item: TotalKnowledgeItem) -> None:
        """Add to knowledge graph."""
        # Add term as node
        domains = ["etymology", "occult", "history"]
        if knowledge_item.occult_connections:
            domains.append("occult")
        if knowledge_item.secret_society_connections:
            domains.append("secret_society")
        if knowledge_item.suppressed_knowledge:
            domains.append("suppressed_knowledge")
        
        principle = knowledge_item.complete_knowledge_chain[:200]
        
        evidence = []
        if knowledge_item.etymology_trace:
            evidence.append((knowledge_item.term, knowledge_item.etymology_trace.complete_chain[:100]))
        if knowledge_item.occult_connections:
            evidence.append((knowledge_item.term, knowledge_item.occult_connections[0].description[:100]))
        
        self.knowledge_graph.add_synthesis(
            title=knowledge_item.term,
            domains=domains,
            principle=principle,
            evidence=evidence
        )

