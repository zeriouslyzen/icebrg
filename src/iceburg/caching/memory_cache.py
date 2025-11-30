"""
Memory Cache
L1: In-memory cache (fast)
"""

from typing import Any, Dict, Optional, List
import time
from collections import OrderedDict


class MemoryCache:
    """L1: In-memory cache for fast access"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() > entry["expires_at"]:
            del self.cache[key]
            self.stats["misses"] += 1
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        self.stats["hits"] += 1
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Add new entry
        self.cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "ttl": ttl
        }
        
        # Evict if over size limit
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        self.stats["sets"] += 1
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            self.stats["deletes"] += 1
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
            "max_size": self.max_size,
            "default_ttl": self.default_ttl
        }

