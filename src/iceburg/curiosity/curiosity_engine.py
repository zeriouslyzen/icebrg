"""Curiosity Engine - Drives autonomous curiosity-driven research and exploration."""

from __future__ import annotations

import json
import time
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_memory_persistence = None

def _get_memory_persistence():
    """Get or create MemoryPersistenceLayer instance"""
    global _memory_persistence
    if _memory_persistence is None:
        try:
            from ..database.memory_persistence import MemoryPersistenceLayer
            from ..database.unified_database import UnifiedDatabase
            from ..config import IceburgConfig
            
            cfg = IceburgConfig()
            db = UnifiedDatabase(cfg)
            _memory_persistence = MemoryPersistenceLayer(db)
        except Exception as e:
            logger.warning(f"Could not initialize MemoryPersistenceLayer: {e}")
            _memory_persistence = None
    return _memory_persistence


@dataclass
class CuriosityQuery:
    """Represents a curiosity-driven query."""
    id: str
    query_text: str
    uncertainty_score: float
    novelty_score: float
    exploration_type: str
    domains: List[str]
    priority: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class KnowledgeGap:
    """Represents a detected knowledge gap."""
    id: str
    gap_description: str
    uncertainty_level: float
    related_domains: List[str]
    exploration_priority: float
    suggested_queries: List[str]
    timestamp: float
    metadata: Dict[str, Any]


class CuriosityEngine:
    """Drives autonomous curiosity-driven research and exploration."""
    
    def __init__(self, cfg: Any = None, enable_persistence: bool = True):
        self.cfg = cfg
        self.uncertainty_threshold = 0.6
        self.novelty_threshold = 0.7
        self.exploration_history: List[CuriosityQuery] = []
        self.knowledge_gaps: List[KnowledgeGap] = []
        
        # Persistence layer
        self.enable_persistence = enable_persistence
        self.persistence = _get_memory_persistence() if enable_persistence else None
        
        # Load existing data from database if persistence is enabled
        if self.persistence and enable_persistence:
            try:
                self._load_from_database()
            except Exception as e:
                logger.warning(f"Could not load existing curiosity data: {e}")
        
        # Uncertainty indicators
        self.uncertainty_indicators = {
            "unknown", "unclear", "uncertain", "mystery", "puzzle", "question",
            "unexplained", "controversial", "debated", "speculation", "hypothesis",
            "gap", "missing", "incomplete", "partial", "limited", "insufficient"
        }
        
        # Novelty indicators
        self.novelty_indicators = {
            "novel", "new", "unprecedented", "breakthrough", "discovery",
            "first_time", "never_before", "revolutionary", "innovative",
            "cutting_edge", "emerging", "developing", "evolving"
        }
        
        # Exploration types
        self.exploration_types = {
            "uncertainty_driven": "Exploring areas of high uncertainty",
            "novelty_seeking": "Seeking novel connections and patterns",
            "gap_filling": "Filling identified knowledge gaps",
            "cross_domain": "Exploring cross-domain connections",
            "paradigm_challenging": "Challenging existing paradigms"
        }
    
    def analyze_uncertainty(self, text: str, context: Dict[str, Any] = None) -> float:
        """Analyze uncertainty level in the given text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        uncertainty_count = sum(1 for indicator in self.uncertainty_indicators 
                              if indicator in text_lower)
        
        # Normalize by text length
        text_length = len(text.split())
        if text_length == 0:
            return 0.0
        
        uncertainty_score = min(1.0, uncertainty_count / (text_length / 100))
        
        # Boost for question marks and uncertainty patterns
        question_count = text.count('?')
        uncertainty_score += min(0.3, question_count * 0.1)
        
        # Boost for conditional language
        conditional_terms = ["if", "might", "could", "possibly", "perhaps", "maybe"]
        conditional_count = sum(1 for term in conditional_terms if term in text_lower)
        uncertainty_score += min(0.2, conditional_count * 0.05)
        
        return min(1.0, uncertainty_score)
    
    def analyze_novelty(self, text: str, context: Dict[str, Any] = None) -> float:
        """Analyze novelty level in the given text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        novelty_count = sum(1 for indicator in self.novelty_indicators 
                           if indicator in text_lower)
        
        # Normalize by text length
        text_length = len(text.split())
        if text_length == 0:
            return 0.0
        
        novelty_score = min(1.0, novelty_count / (text_length / 100))
        
        # Boost for cross-domain terms
        cross_domain_terms = ["synthesis", "integration", "bridge", "connection", "unification"]
        cross_domain_count = sum(1 for term in cross_domain_terms if term in text_lower)
        novelty_score += min(0.3, cross_domain_count * 0.1)
        
        # Boost for prediction terms
        prediction_terms = ["predict", "hypothesis", "testable", "falsifiable", "experiment"]
        prediction_count = sum(1 for term in prediction_terms if term in text_lower)
        novelty_score += min(0.2, prediction_count * 0.1)
        
        return min(1.0, novelty_score)
    
    def detect_knowledge_gaps(self, text: str, context: Dict[str, Any] = None) -> List[KnowledgeGap]:
        """Detect knowledge gaps in the given text."""
        gaps = []
        text_lower = text.lower()
        
        # Look for gap indicators
        gap_indicators = [
            "not well understood", "poorly understood", "unclear", "unknown",
            "mystery", "puzzle", "controversial", "debated", "speculation",
            "limited understanding", "incomplete", "missing", "gap"
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains gap indicators
            if any(indicator in sentence_lower for indicator in gap_indicators):
                uncertainty_level = self.analyze_uncertainty(sentence)
                
                if uncertainty_level > 0.5:  # Threshold for knowledge gap
                    gap_id = self._generate_gap_id(sentence, uncertainty_level)
                    
                    # Extract domains from context or sentence
                    domains = self._extract_domains(sentence, context)
                    
                    # Generate suggested queries
                    suggested_queries = self._generate_exploration_queries(sentence, domains)
                    
                    gap = KnowledgeGap(
                        id=gap_id,
                        gap_description=sentence.strip(),
                        uncertainty_level=uncertainty_level,
                        related_domains=domains,
                        exploration_priority=uncertainty_level * 0.8,
                        suggested_queries=suggested_queries,
                        timestamp=time.time(),
                        metadata={
                            "source_text": sentence,
                            "context": context or {},
                            "gap_indicators": [ind for ind in gap_indicators if ind in sentence_lower]
                        }
                    )
                    
                    gaps.append(gap)
        
        # Persist knowledge gaps to database
        if self.enable_persistence and self.persistence and gaps:
            try:
                self._persist_knowledge_gaps(gaps)
            except Exception as e:
                logger.warning(f"Could not persist knowledge gaps: {e}")
        
        # Add to in-memory list
        self.knowledge_gaps.extend(gaps)
        
        return gaps
    
    def generate_curiosity_queries(
        self, 
        text: str, 
        context: Dict[str, Any] = None,
        max_queries: int = 5
    ) -> List[CuriosityQuery]:
        """Generate curiosity-driven queries based on uncertainty and novelty analysis."""
        queries = []
        
        # Analyze uncertainty and novelty
        uncertainty_score = self.analyze_uncertainty(text, context)
        novelty_score = self.analyze_novelty(text, context)
        
        # Generate queries based on uncertainty
        if uncertainty_score > self.uncertainty_threshold:
            uncertainty_queries = self._generate_uncertainty_queries(text, uncertainty_score, context)
            queries.extend(uncertainty_queries)
        
        # Generate queries based on novelty
        if novelty_score > self.novelty_threshold:
            novelty_queries = self._generate_novelty_queries(text, novelty_score, context)
            queries.extend(novelty_queries)
        
        # Detect knowledge gaps and generate gap-filling queries
        knowledge_gaps = self.detect_knowledge_gaps(text, context)
        for gap in knowledge_gaps:
            gap_queries = self._generate_gap_queries(gap)
            queries.extend(gap_queries)
        
        # Sort by priority and limit
        queries.sort(key=lambda q: q.priority, reverse=True)
        queries = queries[:max_queries]
        
        # Persist queries to database
        if self.enable_persistence and self.persistence:
            try:
                self._persist_queries(queries)
            except Exception as e:
                logger.warning(f"Could not persist curiosity queries: {e}")
        
        # Add to in-memory history
        self.exploration_history.extend(queries)
        
        return queries
    
    def _generate_uncertainty_queries(
        self, 
        text: str, 
        uncertainty_score: float, 
        context: Dict[str, Any] = None
    ) -> List[CuriosityQuery]:
        """Generate queries to explore areas of uncertainty."""
        queries = []
        domains = self._extract_domains(text, context)
        
        # Template queries for uncertainty exploration
        uncertainty_templates = [
            "What is the current understanding of {topic}?",
            "What are the main uncertainties surrounding {topic}?",
            "What evidence exists for {topic}?",
            "What are the competing theories about {topic}?",
            "What research is needed to resolve {topic}?"
        ]
        
        # Extract key topics from text
        topics = self._extract_topics(text)
        
        for topic in topics[:3]:  # Limit to 3 topics
            for template in uncertainty_templates[:2]:  # Limit to 2 templates per topic
                query_text = template.format(topic=topic)
                
                query = CuriosityQuery(
                    id=self._generate_query_id(query_text, uncertainty_score),
                    query_text=query_text,
                    uncertainty_score=uncertainty_score,
                    novelty_score=0.3,  # Lower novelty for uncertainty queries
                    exploration_type="uncertainty_driven",
                    domains=domains,
                    priority=uncertainty_score * 0.8,
                    timestamp=time.time(),
                    metadata={
                        "source_text": text,
                        "topic": topic,
                        "template": template,
                        "context": context or {}
                    }
                )
                
                queries.append(query)
        
        return queries
    
    def _generate_novelty_queries(
        self, 
        text: str, 
        novelty_score: float, 
        context: Dict[str, Any] = None
    ) -> List[CuriosityQuery]:
        """Generate queries to explore novel connections."""
        queries = []
        domains = self._extract_domains(text, context)
        
        # Template queries for novelty exploration
        novelty_templates = [
            "How does {topic} connect to other domains?",
            "What are the novel implications of {topic}?",
            "What breakthrough discoveries relate to {topic}?",
            "How might {topic} revolutionize our understanding?",
            "What are the cutting-edge developments in {topic}?"
        ]
        
        # Extract key topics from text
        topics = self._extract_topics(text)
        
        for topic in topics[:2]:  # Limit to 2 topics
            for template in novelty_templates[:2]:  # Limit to 2 templates per topic
                query_text = template.format(topic=topic)
                
                query = CuriosityQuery(
                    id=self._generate_query_id(query_text, novelty_score),
                    query_text=query_text,
                    uncertainty_score=0.3,  # Lower uncertainty for novelty queries
                    novelty_score=novelty_score,
                    exploration_type="novelty_seeking",
                    domains=domains,
                    priority=novelty_score * 0.9,
                    timestamp=time.time(),
                    metadata={
                        "source_text": text,
                        "topic": topic,
                        "template": template,
                        "context": context or {}
                    }
                )
                
                queries.append(query)
        
        return queries
    
    def _generate_gap_queries(self, gap: KnowledgeGap) -> List[CuriosityQuery]:
        """Generate queries to fill knowledge gaps."""
        queries = []
        
        for suggested_query in gap.suggested_queries[:3]:  # Limit to 3 queries per gap
            query = CuriosityQuery(
                id=self._generate_query_id(suggested_query, gap.uncertainty_level),
                query_text=suggested_query,
                uncertainty_score=gap.uncertainty_level,
                novelty_score=0.5,  # Medium novelty for gap-filling
                exploration_type="gap_filling",
                domains=gap.related_domains,
                priority=gap.exploration_priority,
                timestamp=time.time(),
                metadata={
                    "gap_id": gap.id,
                    "gap_description": gap.gap_description,
                    "source_gap": gap
                }
            )
            
            queries.append(query)
        
        return queries
    
    def _generate_exploration_queries(self, text: str, domains: List[str]) -> List[str]:
        """Generate suggested exploration queries for a knowledge gap."""
        queries = []
        
        # Extract key concepts
        concepts = self._extract_concepts(text)
        
        # Generate queries based on concepts and domains
        for concept in concepts[:2]:  # Limit to 2 concepts
            for domain in domains[:2]:  # Limit to 2 domains
                queries.extend([
                    f"What is the current state of research on {concept} in {domain}?",
                    f"How does {concept} relate to {domain}?",
                    f"What are the key questions about {concept} in {domain}?",
                    f"What evidence exists for {concept} in {domain}?"
                ])
        
        return queries[:5]  # Limit to 5 queries total
    
    def _extract_domains(self, text: str, context: Dict[str, Any] = None) -> List[str]:
        """Extract relevant domains from text and context."""
        domains = []
        text_lower = text.lower()
        
        domain_keywords = {
            "physics": ["quantum", "particle", "field", "energy", "wave", "physics"],
            "biology": ["cell", "organism", "evolution", "genetic", "protein", "biology"],
            "consciousness": ["mind", "awareness", "experience", "perception", "consciousness"],
            "information": ["data", "signal", "processing", "computation", "information"],
            "mathematics": ["equation", "function", "topology", "geometry", "mathematics"],
            "chemistry": ["molecule", "reaction", "bond", "chemical", "chemistry"],
            "psychology": ["behavior", "cognitive", "mental", "psychological", "psychology"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)
        
        # Add domains from context if available
        if context and "domains" in context:
            domains.extend(context["domains"])
        
        return list(set(domains)) if domains else ["general"]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        # Simple topic extraction - look for noun phrases
        words = text.split()
        topics = []
        
        # Look for capitalized words (potential proper nouns)
        for word in words:
            if word[0].isupper() and len(word) > 3:
                topics.append(word.strip('.,!?'))
        
        # Look for quoted terms
        import re
        quoted_terms = re.findall(r'"([^"]*)"', text)
        topics.extend(quoted_terms)
        
        return topics[:5]  # Limit to 5 topics
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple concept extraction
        concepts = []
        text_lower = text.lower()
        
        # Look for concept indicators
        concept_indicators = [
            "concept", "theory", "principle", "law", "model", "framework",
            "approach", "method", "technique", "process", "mechanism"
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for indicator in concept_indicators:
                if indicator in sentence_lower:
                    # Extract the concept (simplified)
                    words = sentence.split()
                    for i, word in enumerate(words):
                        if indicator in word.lower():
                            # Take surrounding words as concept
                            start = max(0, i-2)
                            end = min(len(words), i+3)
                            concept = ' '.join(words[start:end])
                            concepts.append(concept.strip('.,!?'))
                            break
        
        return concepts[:3]  # Limit to 3 concepts
    
    def _generate_query_id(self, query_text: str, score: float) -> str:
        """Generate a unique ID for a query."""
        content = f"{query_text}:{score}"
        hash_obj = hashlib.md5(content.encode())
        return f"curiosity_{hash_obj.hexdigest()[:12]}"
    
    def _generate_gap_id(self, text: str, uncertainty_level: float) -> str:
        """Generate a unique ID for a knowledge gap."""
        content = f"{text}:{uncertainty_level}"
        hash_obj = hashlib.md5(content.encode())
        return f"gap_{hash_obj.hexdigest()[:12]}"
    
    def get_exploration_history(self, limit: int = 10) -> List[CuriosityQuery]:
        """Get recent exploration history."""
        return sorted(self.exploration_history, key=lambda q: q.timestamp, reverse=True)[:limit]
    
    def get_knowledge_gaps(self, min_priority: float = 0.5) -> List[KnowledgeGap]:
        """Get knowledge gaps above minimum priority."""
        return [gap for gap in self.knowledge_gaps if gap.exploration_priority >= min_priority]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curiosity engine statistics."""
        # Get stats from in-memory data
        stats = {
            "total_queries_generated": len(self.exploration_history),
            "total_knowledge_gaps": len(self.knowledge_gaps),
            "avg_uncertainty": sum(q.uncertainty_score for q in self.exploration_history) / max(1, len(self.exploration_history)),
            "avg_novelty": sum(q.novelty_score for q in self.exploration_history) / max(1, len(self.exploration_history)),
            "exploration_types": list(set(q.exploration_type for q in self.exploration_history)),
            "domains_explored": list(set(domain for q in self.exploration_history for domain in q.domains))
        }
        
        # If persistence is enabled, get database stats too
        if self.enable_persistence and self.persistence:
            try:
                # Get database query count
                db_queries = asyncio.run(self.persistence.get_curiosity_queries(limit=1, offset=0))
                # This is approximate - would need COUNT query for exact
                stats["database_queries"] = len(db_queries) if db_queries else 0
                
                # Get database gap count
                db_gaps = asyncio.run(self.persistence.get_knowledge_gaps(limit=1, offset=0))
                stats["database_gaps"] = len(db_gaps) if db_gaps else 0
            except Exception as e:
                logger.warning(f"Could not get database statistics: {e}")
        
        return stats
    
    # ========== Persistence Methods ==========
    
    def _load_from_database(self):
        """Load existing curiosity queries and knowledge gaps from database"""
        if not self.persistence:
            return
        
        try:
            import asyncio
            
            # Load recent queries (last 100)
            queries_data = asyncio.run(self.persistence.get_curiosity_queries(limit=100))
            
            for q_data in queries_data:
                query = CuriosityQuery(
                    id=q_data.get('query_id', ''),
                    query_text=q_data.get('query_text', ''),
                    uncertainty_score=q_data.get('uncertainty_score', 0.0),
                    novelty_score=q_data.get('novelty_score', 0.0),
                    exploration_type=q_data.get('exploration_type', ''),
                    domains=q_data.get('domains', []) if isinstance(q_data.get('domains'), list) else [],
                    priority=q_data.get('priority', 0.0),
                    timestamp=q_data.get('timestamp', time.time()),
                    metadata=q_data.get('metadata', {}) if isinstance(q_data.get('metadata'), dict) else {}
                )
                if query.id not in [q.id for q in self.exploration_history]:
                    self.exploration_history.append(query)
            
            # Load unresolved knowledge gaps
            gaps_data = asyncio.run(self.persistence.get_knowledge_gaps(resolved=False, limit=100))
            
            for g_data in gaps_data:
                gap = KnowledgeGap(
                    id=g_data.get('gap_id', ''),
                    gap_description=g_data.get('gap_description', ''),
                    uncertainty_level=g_data.get('uncertainty_level', 0.0),
                    related_domains=g_data.get('related_domains', []) if isinstance(g_data.get('related_domains'), list) else [],
                    exploration_priority=g_data.get('exploration_priority', 0.0),
                    suggested_queries=g_data.get('suggested_queries', []) if isinstance(g_data.get('suggested_queries'), list) else [],
                    timestamp=g_data.get('timestamp', time.time()),
                    metadata=g_data.get('metadata', {}) if isinstance(g_data.get('metadata'), dict) else {}
                )
                if gap.id not in [g.id for g in self.knowledge_gaps]:
                    self.knowledge_gaps.append(gap)
            
            logger.info(f"Loaded {len(queries_data)} queries and {len(gaps_data)} gaps from database")
            
        except Exception as e:
            logger.warning(f"Could not load from database: {e}")
    
    def _persist_queries(self, queries: List[CuriosityQuery]):
        """Persist curiosity queries to database"""
        if not self.persistence:
            return
        
        try:
            import asyncio
            
            for query in queries:
                asyncio.run(self.persistence.store_curiosity_query(
                    query_id=query.id,
                    query_text=query.query_text,
                    uncertainty_score=query.uncertainty_score,
                    novelty_score=query.novelty_score,
                    exploration_type=query.exploration_type,
                    domains=query.domains,
                    priority=query.priority,
                    timestamp=query.timestamp,
                    answered=False,
                    answer_quality=None,
                    metadata=query.metadata
                ))
            
            logger.debug(f"Persisted {len(queries)} curiosity queries to database")
            
        except Exception as e:
            logger.warning(f"Could not persist queries: {e}")
    
    def _persist_knowledge_gaps(self, gaps: List[KnowledgeGap]):
        """Persist knowledge gaps to database"""
        if not self.persistence:
            return
        
        try:
            import asyncio
            
            for gap in gaps:
                asyncio.run(self.persistence.store_knowledge_gap(
                    gap_id=gap.id,
                    gap_description=gap.gap_description,
                    uncertainty_level=gap.uncertainty_level,
                    related_domains=gap.related_domains,
                    exploration_priority=gap.exploration_priority,
                    suggested_queries=gap.suggested_queries,
                    timestamp=gap.timestamp,
                    resolved=False,
                    resolution_timestamp=None,
                    resolution_quality=None,
                    metadata=gap.metadata
                ))
            
            logger.debug(f"Persisted {len(gaps)} knowledge gaps to database")
            
        except Exception as e:
            logger.warning(f"Could not persist knowledge gaps: {e}")
