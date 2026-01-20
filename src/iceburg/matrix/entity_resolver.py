"""
Entity Resolver - Deduplication and entity matching using embeddings.
Detects same entity with different spellings and merges duplicates.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ResolvedEntity:
    """An entity after resolution."""
    canonical_id: str
    canonical_name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    merged_from: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0


class EntityResolver:
    """
    Entity resolution using string similarity and embeddings.
    
    Features:
    - Fuzzy name matching
    - Alias detection
    - Merge duplicate entities
    - Embedding-based similarity (optional)
    """
    
    # Common suffixes to normalize
    COMPANY_SUFFIXES = [
        " inc.", " inc", " llc", " corp.", " corp", " corporation",
        " company", " co.", " co", " ltd.", " ltd", " limited",
        " plc", " lp", " llp", " pllc", " pc"
    ]
    
    # Common title prefixes
    PERSON_PREFIXES = [
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sen.", "rep.",
        "gov.", "hon.", "rev.", "gen.", "col.", "capt."
    ]
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize the resolver.
        
        Args:
            data_dir: Directory for persisting resolution data
            similarity_threshold: Minimum similarity for matching (0-1)
        """
        self.data_dir = data_dir or Path.home() / "Documents" / "iceburg_matrix"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.aliases_path = self.data_dir / "aliases.json"
        
        # Known aliases: alias -> canonical_id
        self.known_aliases: Dict[str, str] = {}
        # Canonical entities: canonical_id -> ResolvedEntity
        self.canonical_entities: Dict[str, ResolvedEntity] = {}
        
        self._load_aliases()
        logger.info(f"🔗 Entity Resolver initialized ({len(self.known_aliases)} known aliases)")
    
    def _load_aliases(self):
        """Load known aliases from disk."""
        try:
            if self.aliases_path.exists():
                with open(self.aliases_path, "r") as f:
                    data = json.load(f)
                    self.known_aliases = data.get("aliases", {})
        except Exception as e:
            logger.warning(f"Could not load aliases: {e}")
    
    def _save_aliases(self):
        """Persist aliases to disk."""
        try:
            with open(self.aliases_path, "w") as f:
                json.dump({"aliases": self.known_aliases}, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save aliases: {e}")
    
    def normalize_name(self, name: str, entity_type: str = "unknown") -> str:
        """
        Normalize an entity name for comparison.
        
        Args:
            name: Raw entity name
            entity_type: Type of entity
            
        Returns:
            Normalized name
        """
        normalized = name.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove company suffixes
        if entity_type in ("company", "organization", "unknown"):
            for suffix in self.COMPANY_SUFFIXES:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)].strip()
        
        # Remove person prefixes
        if entity_type in ("person", "unknown"):
            for prefix in self.PERSON_PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
        
        # Remove punctuation except hyphens
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        
        return normalized
    
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using multiple methods.
        
        Args:
            name1: First name
            name2: Second name
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize both names
        n1 = self.normalize_name(name1)
        n2 = self.normalize_name(name2)
        
        # Exact match after normalization
        if n1 == n2:
            return 1.0
        
        # One contains the other
        if n1 in n2 or n2 in n1:
            return 0.9
        
        # Token-based similarity (Jaccard)
        tokens1 = set(n1.split())
        tokens2 = set(n2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Levenshtein-like similarity
        edit_sim = self._edit_similarity(n1, n2)
        
        # Combined score
        return max(jaccard, edit_sim)
    
    def _edit_similarity(self, s1: str, s2: str) -> float:
        """Calculate edit distance-based similarity."""
        if not s1 or not s2:
            return 0.0
        
        # Simple ratio based on length
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        # Count matching characters
        matches = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                matches += 1
        
        return matches / max_len
    
    def resolve(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        source: str = "unknown"
    ) -> Tuple[str, bool]:
        """
        Resolve an entity name to a canonical ID.
        
        Args:
            name: Entity name to resolve
            entity_type: Type of entity
            properties: Additional properties
            source: Source of the entity
            
        Returns:
            (canonical_id, is_new) tuple
        """
        normalized = self.normalize_name(name, entity_type)
        
        # Check known aliases
        if normalized in self.known_aliases:
            return self.known_aliases[normalized], False
        
        # Check for similar existing entities
        best_match = None
        best_score = 0.0
        
        for canonical_id, entity in self.canonical_entities.items():
            if entity.entity_type != entity_type:
                continue
            
            # Check against canonical name
            score = self.calculate_similarity(name, entity.canonical_name)
            if score > best_score:
                best_score = score
                best_match = canonical_id
            
            # Check against aliases
            for alias in entity.aliases:
                score = self.calculate_similarity(name, alias)
                if score > best_score:
                    best_score = score
                    best_match = canonical_id
        
        # Found a match above threshold?
        if best_match and best_score >= self.similarity_threshold:
            # Add as alias
            self.known_aliases[normalized] = best_match
            self.canonical_entities[best_match].aliases.add(name)
            if source:
                self.canonical_entities[best_match].sources.append(source)
            self._save_aliases()
            return best_match, False
        
        # Create new canonical entity
        import hashlib
        canonical_id = f"{entity_type}_{hashlib.md5(normalized.encode()).hexdigest()[:12]}"
        
        self.canonical_entities[canonical_id] = ResolvedEntity(
            canonical_id=canonical_id,
            canonical_name=name,
            entity_type=entity_type,
            aliases={name, normalized},
            properties=properties or {},
            sources=[source] if source else [],
        )
        self.known_aliases[normalized] = canonical_id
        self._save_aliases()
        
        return canonical_id, True
    
    def add_alias(self, canonical_id: str, alias: str) -> bool:
        """
        Add an alias for a canonical entity.
        
        Args:
            canonical_id: Canonical entity ID
            alias: Alias to add
            
        Returns:
            True if successful
        """
        if canonical_id not in self.canonical_entities:
            return False
        
        normalized = self.normalize_name(alias)
        self.known_aliases[normalized] = canonical_id
        self.canonical_entities[canonical_id].aliases.add(alias)
        self._save_aliases()
        
        return True
    
    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """
        Merge one entity into another.
        
        Args:
            keep_id: Entity to keep
            merge_id: Entity to merge into keep_id
            
        Returns:
            True if successful
        """
        if keep_id not in self.canonical_entities:
            return False
        if merge_id not in self.canonical_entities:
            return False
        
        keep = self.canonical_entities[keep_id]
        merge = self.canonical_entities[merge_id]
        
        # Transfer aliases
        keep.aliases.update(merge.aliases)
        keep.aliases.add(merge.canonical_name)
        keep.merged_from.append(merge_id)
        keep.sources.extend(merge.sources)
        
        # Update alias mappings
        for alias in merge.aliases:
            normalized = self.normalize_name(alias)
            self.known_aliases[normalized] = keep_id
        
        # Remove merged entity
        del self.canonical_entities[merge_id]
        self._save_aliases()
        
        logger.info(f"🔗 Merged entity {merge_id} into {keep_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics."""
        type_counts = {}
        for entity in self.canonical_entities.values():
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
        
        return {
            "total_canonical_entities": len(self.canonical_entities),
            "total_aliases": len(self.known_aliases),
            "entity_types": type_counts,
        }
