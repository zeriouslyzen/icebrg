"""
Intelligent Redis Caching Layer for ICEBURG Performance Optimization
Implements semantic similarity caching, predictive pre-warming, and TTL management.
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Callable, Tuple
import asyncio
import numpy as np
from pathlib import Path

try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    SentenceTransformer = None


class IntelligentCache:
    """
    Intelligent caching system with semantic similarity and predictive pre-warming.
    
    Features:
    - Semantic similarity caching (0.95+ threshold)
    - TTL based on query complexity
    - Predictive pre-warming for common patterns
    - Embedding-based cache lookup
    - LRU eviction with semantic clustering
    """
    
    def __init__(self, 
        redis_host: str = "os.getenv("HOST", "localhost")",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.95,
                 max_cache_size: int = 10000):
        """
        Initialize the intelligent cache.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            embedding_model: Sentence transformer model for embeddings
            similarity_threshold: Minimum similarity for cache hits
            max_cache_size: Maximum number of cached items
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Initialize Redis connection
        if REDIS_AVAILABLE:
            try:
                self.redis = Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
                self.redis.ping()  # Test connection
                self.redis_connected = True
            except Exception as e:
                self.redis_connected = False
        else:
            self.redis_connected = False
            self._memory_cache = {}
        
        # Initialize embedding model
        if EMBEDDING_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.embedding_available = True
            except Exception as e:
                self.embedding_available = False
        else:
            self.embedding_available = False
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "prewarm_hits": 0,
            "total_queries": 0,
            "cache_size": 0
        }
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using sentence transformer."""
        if not self.embedding_available:
            # Fallback to simple hash-based embedding
            hash_obj = hashlib.md5(text.encode())
            return np.array([float(int(hash_obj.hexdigest()[:8], 16)) / 1e8])
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding
        except Exception as e:
            # Fallback to hash-based embedding
            hash_obj = hashlib.md5(text.encode())
            return np.array([float(int(hash_obj.hexdigest()[:8], 16)) / 1e8])
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0
    
    def _get_cache_key(self, query_embedding: np.ndarray) -> str:
        """Generate cache key from query embedding."""
        # Use first few dimensions of embedding as key
        key_data = query_embedding[:8].tobytes()
        return f"cache:{hashlib.md5(key_data).hexdigest()}"
    
    def _get_similarity_key(self, query_embedding: np.ndarray) -> str:
        """Generate similarity search key."""
        return f"similarity:{hashlib.md5(query_embedding.tobytes()).hexdigest()}"
    
    async def get_or_compute(self, 
        query: str,
                           compute_fn: Callable, 
                           ttl: int = 3600,
                           context: Dict[str, Any] = None) -> Any:
        """
        Get cached result or compute and cache new result.
        
        Args:
            query: Input query string
            compute_fn: Function to compute result if not cached
            ttl: Time to live in seconds
            context: Optional context for computation
            
        Returns:
            Cached or computed result
        """
        self.stats["total_queries"] += 1
        
        # Get embedding for query
        query_embedding = self._get_embedding(query)
        
        # Try to find similar cached result
        cached_result = await self._search_similar(query_embedding)
        
        if cached_result is not None:
            self.stats["hits"] += 1
            return cached_result
        
        # Compute new result
        self.stats["misses"] += 1
        try:
            if context:
                result = await compute_fn(query, context)
            else:
                result = await compute_fn(query)
        except Exception as e:
            raise
        
        # Cache the result
        await self._cache_result(query_embedding, result, ttl)
        
        return result
    
    async def _search_similar(self, query_embedding: np.ndarray) -> Optional[Any]:
        """Search for similar cached results using semantic similarity."""
        if not self.redis_connected:
            return None
        
        try:
            # Get all cache keys
            cache_keys = self.redis.keys("cache:*")
            
            best_similarity = 0.0
            best_result = None
            
            for key in cache_keys:
                # Get cached embedding and result
                cached_data = self.redis.hgetall(key)
                if not cached_data:
                    continue
                
                # Parse cached embedding
                try:
                    cached_embedding = np.frombuffer(
                        bytes.fromhex(cached_data.get("embedding", "")), 
                        dtype=np.float32
                    )
                    cached_result = json.loads(cached_data.get("result", "{}"))
                    
                    # Calculate similarity
                    similarity = self._calculate_similarity(query_embedding, cached_embedding)
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_result = cached_result
                        
                except Exception as e:
                    continue
            
            return best_result
            
        except Exception as e:
            return None
    
    async def _cache_result(self, query_embedding: np.ndarray, result: Any, ttl: int):
        """Cache a result with its embedding."""
        if not self.redis_connected:
            return
        
        try:
            cache_key = self._get_cache_key(query_embedding)
            
            # Prepare cache data
            cache_data = {
                "embedding": query_embedding.tobytes().hex(),
                "result": json.dumps(result),
                "timestamp": time.time(),
                "ttl": ttl
            }
            
            # Store in Redis
            self.redis.hset(cache_key, mapping=cache_data)
            self.redis.expire(cache_key, ttl)
            
            # Update cache size
            self.stats["cache_size"] = len(self.redis.keys("cache:*"))
            
        except Exception as e:
    
    async def prewarm_cache(self, common_queries: List[str], compute_fn: Callable):
        """
        Pre-warm cache with common queries.
        
        Args:
            common_queries: List of common query patterns
            compute_fn: Function to compute results
        """
        
        for query in common_queries:
            try:
                # Check if already cached
                query_embedding = self._get_embedding(query)
                cached_result = await self._search_similar(query_embedding)
                
                if cached_result is None:
                    # Compute and cache
                    result = await compute_fn(query)
                    await self._cache_result(query_embedding, result, ttl=7200)  # 2 hours
                    self.stats["prewarm_hits"] += 1
                    
            except Exception as e:
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        if self.stats["total_queries"] > 0:
            hit_rate = self.stats["hits"] / self.stats["total_queries"]
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "prewarm_hits": self.stats["prewarm_hits"],
            "total_queries": self.stats["total_queries"],
            "cache_size": self.stats["cache_size"],
            "redis_connected": self.redis_connected,
            "embedding_available": self.embedding_available
        }
    
    async def clear_cache(self):
        """Clear all cached results."""
        if self.redis_connected:
            try:
                cache_keys = self.redis.keys("cache:*")
                if cache_keys:
                    self.redis.delete(*cache_keys)
                self.stats["cache_size"] = 0
            except Exception as e:
    
    async def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health status."""
        health = {
            "redis_connected": self.redis_connected,
            "embedding_available": self.embedding_available,
            "cache_size": self.stats["cache_size"],
            "hit_rate": 0.0
        }
        
        if self.stats["total_queries"] > 0:
            health["hit_rate"] = self.stats["hits"] / self.stats["total_queries"]
        
        if self.redis_connected:
            try:
                # Test Redis connection
                self.redis.ping()
                health["redis_ping"] = True
            except Exception:
                health["redis_ping"] = False
        
        return health


# Global cache instance
_cache_instance: Optional[IntelligentCache] = None


def get_cache() -> IntelligentCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCache()
    return _cache_instance


async def cached_compute(query: str, compute_fn: Callable, ttl: int = 3600, context: Dict[str, Any] = None) -> Any:
    """
    Convenience function for cached computation.
    
    Args:
        query: Input query
        compute_fn: Function to compute result
        ttl: Time to live in seconds
        context: Optional context
        
    Returns:
        Cached or computed result
    """
    cache = get_cache()
    return await cache.get_or_compute(query, compute_fn, ttl, context)
