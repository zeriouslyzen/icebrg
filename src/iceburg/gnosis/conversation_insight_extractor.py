"""
Conversation Insight Extractor
Extracts insights from each conversation automatically
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .universal_knowledge_accumulator import Insight
from ..emergence_detector import EmergenceDetector

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Represents a cross-domain connection"""
    connection_id: str
    domain1: str
    domain2: str
    connection_type: str
    description: str
    confidence: float
    evidence: List[str] = None


class ConversationInsightExtractor:
    """
    Extracts insights from each conversation automatically.
    
    Identifies novel patterns and connections, maps cross-domain relationships,
    and stores in gnosis knowledge base.
    """
    
    def __init__(self):
        """Initialize conversation insight extractor."""
        self.emergence_detector = EmergenceDetector()
        logger.info("Conversation Insight Extractor initialized")
    
    def extract_insights(self, conversation: Dict[str, Any]) -> List[Insight]:
        """
        Extract insights from a conversation.
        
        Args:
            conversation: Conversation dictionary with query, response, metadata
            
        Returns:
            List of extracted insights
        """
        insights = []
        conversation_id = conversation.get("conversation_id", f"conv_{datetime.utcnow().timestamp()}")
        
        query = conversation.get("query", "")
        response = conversation.get("response", "")
        metadata = conversation.get("metadata", {})
        
        # Extract novel patterns
        novel_patterns = self.identify_novel_patterns(query, response, metadata)
        for pattern in novel_patterns:
            insight = Insight(
                insight_id=f"insight_{len(insights)}_{datetime.utcnow().timestamp()}",
                source_conversation_id=conversation_id,
                insight_type="pattern",
                content=pattern,
                confidence=0.7,
                domains=metadata.get("domains", []),
                evidence=[query, response[:200]],
                metadata={"novel": True}
            )
            insights.append(insight)
        
        # Extract cross-domain connections
        connections = self.map_cross_domain_connections(query, response, metadata)
        for connection in connections:
            insight = Insight(
                insight_id=f"insight_{len(insights)}_{datetime.utcnow().timestamp()}",
                source_conversation_id=conversation_id,
                insight_type="connection",
                content=connection.description,
                confidence=connection.confidence,
                domains=[connection.domain1, connection.domain2],
                evidence=connection.evidence or [],
                metadata={"connection_type": connection.connection_type}
            )
            insights.append(insight)
        
        # Use emergence detector
        try:
            emergence_result = self.emergence_detector.process(
                oracle_output=response,
                claims=[{"claim": query}],
                evidence_level="B"  # Default evidence level
            )
            
            if emergence_result:
                emergence_insight = Insight(
                    insight_id=f"insight_{len(insights)}_{datetime.utcnow().timestamp()}",
                    source_conversation_id=conversation_id,
                    insight_type="emergence",
                    content=emergence_result.get("core_principle", ""),
                    confidence=emergence_result.get("emergence_score", 0.5),
                    domains=metadata.get("domains", []),
                    evidence=[query, response[:200]],
                    metadata=emergence_result
                )
                insights.append(emergence_insight)
        except Exception as e:
            logger.warning(f"Error using emergence detector: {e}")
        
        logger.info(f"Extracted {len(insights)} insights from conversation {conversation_id}")
        return insights
    
    def identify_novel_patterns(self, query: str, response: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Identify novel patterns in conversation.
        
        Args:
            query: User query
            response: System response
            metadata: Conversation metadata
            
        Returns:
            List of novel patterns
        """
        patterns = []
        
        # Look for novel pattern indicators
        novel_indicators = [
            "novel", "new", "unprecedented", "unusual", "unique",
            "first time", "never seen", "unexpected", "surprising"
        ]
        
        response_lower = response.lower()
        
        for indicator in novel_indicators:
            if indicator in response_lower:
                # Extract surrounding context
                idx = response_lower.find(indicator)
                pattern = response[max(0, idx-100):idx+200]
                
                # Check if pattern is actually novel (not just mentioning the word)
                if len(pattern) > 50:
                    patterns.append(pattern.strip())
        
        # Look for pattern structures
        pattern_structures = [
            "pattern of", "trend of", "correlation between", "relationship between",
            "connection between", "link between"
        ]
        
        for structure in pattern_structures:
            if structure in response_lower:
                idx = response_lower.find(structure)
                pattern = response[max(0, idx-50):idx+200]
                if len(pattern) > 50:
                    patterns.append(pattern.strip())
        
        return patterns[:5]  # Limit to 5 patterns
    
    def map_cross_domain_connections(self, query: str, response: str, metadata: Dict[str, Any]) -> List[Connection]:
        """
        Map cross-domain connections in conversation.
        
        Args:
            query: User query
            response: System response
            metadata: Conversation metadata
            
        Returns:
            List of cross-domain connections
        """
        connections = []
        
        # Extract domains from metadata
        domains = metadata.get("domains", [])
        
        if len(domains) >= 2:
            # Create connections between domains
            for i, domain1 in enumerate(domains):
                for domain2 in domains[i+1:]:
                    connection = Connection(
                        connection_id=f"conn_{len(connections)}_{datetime.utcnow().timestamp()}",
                        domain1=domain1,
                        domain2=domain2,
                        connection_type="cross_domain",
                        description=f"Connection between {domain1} and {domain2}",
                        confidence=0.6,
                        evidence=[query, response[:200]]
                    )
                    connections.append(connection)
        
        # Look for explicit cross-domain mentions
        cross_domain_indicators = [
            "connects", "relates", "links", "bridges", "combines",
            "integrates", "synthesizes", "unifies"
        ]
        
        response_lower = response.lower()
        
        for indicator in cross_domain_indicators:
            if indicator in response_lower:
                idx = response_lower.find(indicator)
                context = response[max(0, idx-100):idx+200]
                
                # Try to extract domains from context
                found_domains = []
                for domain in ["physics", "chemistry", "biology", "mathematics", "computer science",
                              "astronomy", "psychology", "sociology", "economics", "philosophy"]:
                    if domain in context.lower():
                        found_domains.append(domain)
                
                if len(found_domains) >= 2:
                    connection = Connection(
                        connection_id=f"conn_{len(connections)}_{datetime.utcnow().timestamp()}",
                        domain1=found_domains[0],
                        domain2=found_domains[1],
                        connection_type="explicit",
                        description=f"{indicator.capitalize()} {found_domains[0]} and {found_domains[1]}",
                        confidence=0.7,
                        evidence=[context]
                    )
                    connections.append(connection)
        
        return connections
    
    def store_in_gnosis(self, insights: List[Insight], accumulator) -> None:
        """
        Store insights in gnosis knowledge base.
        
        Args:
            insights: List of insights to store
            accumulator: UniversalKnowledgeAccumulator instance
        """
        accumulator.accumulate_to_gnosis(insights)
        logger.info(f"Stored {len(insights)} insights in gnosis knowledge base")

