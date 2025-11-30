"""
Semantic Cache
L3: Semantic cache for similar queries
"""

from typing import Any, Dict, Optional, List
import time
import hashlib
import json


class SemanticCache:
    """L3: Semantic cache for similar queries"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0
        }
    
    def get(self, query: str, embedding: Optional[List[float]] = None) -> Optional[Any]:
        """Get value from semantic cache"""
        # Try exact match first
        query_hash = self._hash_query(query)
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            if time.time() <= entry["expires_at"]:
                self.stats["hits"] += 1
                return entry["value"]
        
        # Try semantic similarity
        if embedding:
            similar_key = self._find_similar(embedding)
            if similar_key:
                entry = self.cache[similar_key]
                if time.time() <= entry["expires_at"]:
                    self.stats["hits"] += 1
                    return entry["value"]
        
        self.stats["misses"] += 1
        return None
    
    def set(
        self,
        query: str,
        value: Any,
        embedding: Optional[List[float]] = None,
        ttl: int = 3600
    ) -> bool:
        """Set value in semantic cache"""
        query_hash = self._hash_query(query)
        
        self.cache[query_hash] = {
            "value": value,
            "query": query,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "ttl": ttl
        }
        
        if embedding:
            self.embeddings[query_hash] = embedding
        
        self.stats["sets"] += 1
        return True
    
    def _hash_query(self, query: str) -> str:
        """Hash query for exact matching"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def _find_similar(self, embedding: List[float]) -> Optional[str]:
        """Find similar query by embedding"""
        if not embedding or not self.embeddings:
            return None
        
        best_similarity = 0.0
        best_key = None
        
        for key, cached_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(embedding, cached_embedding)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_key = key
        
        return best_key
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry["expires_at"]
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.embeddings:
                del self.embeddings[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "similarity_threshold": self.similarity_threshold
        }

