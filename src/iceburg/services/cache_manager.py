"""
Cache Manager
Response caching with TTL and invalidation
"""

from typing import Any, Dict, Optional, List
import hashlib
import json
import time
from datetime import datetime, timedelta


class CacheManager:
    """Manages response caching"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    def _generate_key(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from query and context"""
        key_data = {"query": query}
        if context:
            key_data["context"] = context
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get cached response"""
        key = self._generate_key(query, context)
        
        if key not in self.cache:
            self.cache_stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() > entry["expires_at"]:
            del self.cache[key]
            self.cache_stats["misses"] += 1
            return None
        
        self.cache_stats["hits"] += 1
        return entry["value"]
    
    def set(
        self,
        query: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Set cached response"""
        key = self._generate_key(query, context)
        ttl = ttl or self.default_ttl
        
        self.cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "ttl": ttl
        }
        
        self.cache_stats["sets"] += 1
        return True
    
    def delete(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Delete cached response"""
        key = self._generate_key(query, context)
        
        if key in self.cache:
            del self.cache[key]
            self.cache_stats["deletes"] += 1
            return True
        
        return False
    
    def clear(self) -> int:
        """Clear all cache"""
        count = len(self.cache)
        self.cache.clear()
        return count
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry["expires_at"]
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "default_ttl": self.default_ttl
        }
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        count = 0
        
        for key in list(self.cache.keys()):
            entry = self.cache[key]
            query = entry.get("value", {}).get("query", "")
            
            if pattern.lower() in query.lower():
                del self.cache[key]
                count += 1
        
        return count

