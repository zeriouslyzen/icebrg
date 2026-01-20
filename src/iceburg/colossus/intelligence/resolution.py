"""
COLOSSUS Entity Resolution

Deduplicate and link entities across sources using ML and embeddings.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


@dataclass
class ResolvedEntity:
    """Unified entity after resolution."""
    canonical_id: str
    canonical_name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    merged_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchCandidate:
    """Candidate match for resolution."""
    entity_id: str
    name: str
    entity_type: str
    score: float
    match_reasons: List[str] = field(default_factory=list)


class EntityResolver:
    """
    Entity resolution and deduplication.
    
    Uses multiple matching strategies:
    1. Exact name matching
    2. Fuzzy string matching (Levenshtein, Jaro-Winkler)
    3. Embedding similarity (sentence transformers)
    4. Rule-based matching (DOB, nationality, IDs)
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        use_embeddings: bool = True,
    ):
        """
        Initialize resolver.
        
        Args:
            embedding_model: Sentence transformer model
            similarity_threshold: Min similarity for match
            use_embeddings: Use embedding similarity
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        
        self._encoder = None
        
        if use_embeddings:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model)
            logger.info(f"✅ Loaded embedding model: {self.embedding_model}")
        except Exception as e:
            logger.warning(f"⚠️ Embeddings unavailable: {e}")
    
    def find_matches(
        self,
        entity: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        strategies: List[str] = None
    ) -> List[MatchCandidate]:
        """
        Find matching entities from candidates.
        
        Args:
            entity: Source entity to match
            candidates: Potential matches
            strategies: Matching strategies to use
            
        Returns:
            Ranked match candidates
        """
        strategies = strategies or ["exact", "fuzzy", "embedding"]
        matches = []
        
        name = entity.get("name", "")
        entity_type = entity.get("entity_type", "")
        
        for candidate in candidates:
            cand_name = candidate.get("name", "")
            cand_type = candidate.get("entity_type", "")
            
            # Skip if types don't match
            if entity_type and cand_type and entity_type != cand_type:
                continue
            
            score = 0.0
            reasons = []
            
            # Exact matching
            if "exact" in strategies:
                if self._normalize_name(name) == self._normalize_name(cand_name):
                    score = 1.0
                    reasons.append("exact_match")
            
            # Fuzzy matching
            if "fuzzy" in strategies and score < 1.0:
                fuzzy_score = self._fuzzy_match(name, cand_name)
                if fuzzy_score > score:
                    score = fuzzy_score
                    reasons.append(f"fuzzy:{fuzzy_score:.2f}")
            
            # Embedding similarity
            if "embedding" in strategies and self._encoder and score < self.similarity_threshold:
                emb_score = self._embedding_similarity(name, cand_name)
                if emb_score > score:
                    score = emb_score
                    reasons.append(f"embedding:{emb_score:.2f}")
            
            # Property matching (DOB, nationality, etc.)
            prop_boost = self._property_match_score(entity, candidate)
            if prop_boost > 0:
                score = min(1.0, score + prop_boost * 0.2)
                reasons.append("property_match")
            
            if score >= self.similarity_threshold:
                matches.append(MatchCandidate(
                    entity_id=candidate.get("entity_id", ""),
                    name=cand_name,
                    entity_type=cand_type,
                    score=score,
                    match_reasons=reasons,
                ))
        
        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches
    
    def resolve(
        self,
        entities: List[Dict[str, Any]],
        cluster_threshold: float = 0.9
    ) -> List[ResolvedEntity]:
        """
        Cluster and resolve entities.
        
        Args:
            entities: Entities to resolve
            cluster_threshold: Min similarity to cluster
            
        Returns:
            Resolved unified entities
        """
        if not entities:
            return []
        
        # Build clusters using union-find
        n = len(entities)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Compare all pairs (O(n^2) - use blocking for large datasets)
        for i in range(n):
            for j in range(i + 1, n):
                matches = self.find_matches(entities[i], [entities[j]])
                if matches and matches[0].score >= cluster_threshold:
                    union(i, j)
        
        # Group by cluster
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(entities[i])
        
        # Create resolved entities
        resolved = []
        for cluster_entities in clusters.values():
            resolved.append(self._merge_cluster(cluster_entities))
        
        return resolved
    
    def _merge_cluster(self, entities: List[Dict[str, Any]]) -> ResolvedEntity:
        """Merge cluster of entities into one."""
        if len(entities) == 1:
            e = entities[0]
            return ResolvedEntity(
                canonical_id=e.get("entity_id", ""),
                canonical_name=e.get("name", ""),
                entity_type=e.get("entity_type", "unknown"),
                aliases=[],
                source_ids=[e.get("entity_id", "")],
                sources=e.get("sources", []),
                confidence=1.0,
                merged_properties=e.get("properties", {}),
            )
        
        # Find best canonical name (longest, most common)
        names = [e.get("name", "") for e in entities]
        canonical_name = max(set(names), key=lambda n: (names.count(n), len(n)))
        
        # Collect all aliases
        aliases = list(set(n for n in names if n != canonical_name))
        
        # Merge properties
        merged_props = {}
        for e in entities:
            for k, v in e.get("properties", {}).items():
                if k not in merged_props:
                    merged_props[k] = v
        
        # Collect sources
        all_sources = []
        source_ids = []
        for e in entities:
            source_ids.append(e.get("entity_id", ""))
            all_sources.extend(e.get("sources", []))
        
        return ResolvedEntity(
            canonical_id=f"resolved_{hash(canonical_name) & 0xFFFFFFFF:08x}",
            canonical_name=canonical_name,
            entity_type=entities[0].get("entity_type", "unknown"),
            aliases=aliases,
            source_ids=source_ids,
            sources=list(set(all_sources)),
            confidence=0.9,  # Slightly lower for merged entities
            merged_properties=merged_props,
        )
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison."""
        name = name.lower()
        name = re.sub(r'[^\w\s]', '', name)
        name = ' '.join(name.split())
        return name
    
    def _fuzzy_match(self, name1: str, name2: str) -> float:
        """Calculate fuzzy string similarity."""
        # Jaro-Winkler similarity
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)
        
        if n1 == n2:
            return 1.0
        
        len1, len2 = len(n1), len(n2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Simple ratio for now (TODO: use jellyfish for Jaro-Winkler)
        common = sum(1 for c in n1 if c in n2)
        return common / max(len1, len2)
    
    def _embedding_similarity(self, name1: str, name2: str) -> float:
        """Calculate embedding cosine similarity."""
        if not self._encoder:
            return 0.0
        
        try:
            emb1 = self._encoder.encode(name1)
            emb2 = self._encoder.encode(name2)
            
            # Cosine similarity
            import numpy as np
            dot = np.dot(emb1, emb2)
            norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
            return float(dot / norm) if norm > 0 else 0.0
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}")
            return 0.0
    
    def _property_match_score(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> float:
        """Score based on matching properties."""
        props1 = entity1.get("properties", {})
        props2 = entity2.get("properties", {})
        
        matches = 0
        total = 0
        
        # Key properties that strongly indicate a match
        key_props = ["birth_date", "nationality", "registration_number", "tax_id"]
        
        for prop in key_props:
            if prop in props1 and prop in props2:
                total += 1
                if props1[prop] == props2[prop]:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
