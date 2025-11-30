"""
Multi-Level Cache
L1: In-memory cache (fast)
L2: Redis cache (medium)
L3: Semantic cache (similar queries)
"""

from typing import Any, Dict, Optional, List
import time
from .memory_cache import MemoryCache
from .semantic_cache import SemanticCache
from .redis_intelligence import IntelligentCache


class MultiLevelCache:
    """Multi-level caching strategy"""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        similarity_threshold: float = 0.8
    ):
        # L1: In-memory cache (fast)
        self.l1_cache = MemoryCache(max_size=1000, default_ttl=300)
        
        # L2: Redis cache (medium)
        try:
            self.l2_cache = IntelligentCache(
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=redis_db
            )
            self.l2_available = True
        except Exception:
            self.l2_cache = None
            self.l2_available = False
        
        # L3: Semantic cache (similar queries)
        self.l3_cache = SemanticCache(similarity_threshold=similarity_threshold)
        
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "sets": 0
        }
    
    def get(
        self,
        key: str,
        embedding: Optional[List[float]] = None
    ) -> Optional[Any]:
        """Get value from multi-level cache"""
        # Try L1 first (fastest)
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats["l1_hits"] += 1
            return value
        
        # Try L2 (Redis)
        if self.l2_available and self.l2_cache:
            try:
                value = self.l2_cache.get(key)
                if value is not None:
                    self.stats["l2_hits"] += 1
                    # Promote to L1
                    self.l1_cache.set(key, value, ttl=300)
                    return value
            except Exception:
                pass
        
        # Try L3 (semantic)
        if embedding:
            value = self.l3_cache.get(key, embedding)
            if value is not None:
                self.stats["l3_hits"] += 1
                # Promote to L1 and L2
                self.l1_cache.set(key, value, ttl=300)
                if self.l2_available and self.l2_cache:
                    try:
                        self.l2_cache.set(key, value, ttl=3600)
                    except Exception:
                        pass
                return value
        
        self.stats["misses"] += 1
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        embedding: Optional[List[float]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in multi-level cache"""
        # Set in all levels
        self.l1_cache.set(key, value, ttl=ttl or 300)
        
        if self.l2_available and self.l2_cache:
            try:
                self.l2_cache.set(key, value, ttl=ttl or 3600)
            except Exception:
                pass
        
        if embedding:
            self.l3_cache.set(key, value, embedding, ttl=ttl or 3600)
        
        self.stats["sets"] += 1
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from all cache levels"""
        deleted = False
        
        if self.l1_cache.delete(key):
            deleted = True
        
        if self.l2_available and self.l2_cache:
            try:
                if self.l2_cache.delete(key):
                    deleted = True
            except Exception:
                pass
        
        if key in self.l3_cache.cache:
            del self.l3_cache.cache[key]
            if key in self.l3_cache.embeddings:
                del self.l3_cache.embeddings[key]
            deleted = True
        
        return deleted
    
    def clear(self) -> int:
        """Clear all cache levels"""
        count = 0
        
        count += self.l1_cache.clear()
        
        if self.l2_available and self.l2_cache:
            try:
                count += self.l2_cache.clear()
            except Exception:
                pass
        
        count += len(self.l3_cache.cache)
        self.l3_cache.cache.clear()
        self.l3_cache.embeddings.clear()
        
        return count
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries from all levels"""
        count = 0
        
        count += self.l1_cache.cleanup_expired()
        
        if self.l2_available and self.l2_cache:
            try:
                count += self.l2_cache.cleanup_expired()
            except Exception:
                pass
        
        count += self.l3_cache.cleanup_expired()
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = (
            self.stats["l1_hits"] +
            self.stats["l2_hits"] +
            self.stats["l3_hits"]
        )
        total_requests = total_hits + self.stats["misses"]
        hit_rate = (
            total_hits / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            **self.stats,
            "total_hits": total_hits,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "l1_stats": self.l1_cache.get_stats(),
            "l2_available": self.l2_available,
            "l2_stats": self.l2_cache.get_stats() if self.l2_available and self.l2_cache else None,
            "l3_stats": self.l3_cache.get_stats()
        }
    
    def warm_cache(self, queries: List[Dict[str, Any]]) -> int:
        """Warm cache with common queries"""
        warmed = 0
        
        for query_data in queries:
            key = query_data.get("key")
            value = query_data.get("value")
            embedding = query_data.get("embedding")
            
            if key and value:
                self.set(key, value, embedding)
                warmed += 1
        
        return warmed

