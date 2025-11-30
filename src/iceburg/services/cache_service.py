"""
Cache Service
Microservice for caching with dependency injection
"""

from typing import Any, Dict, Optional, List
from ..interfaces import IService
from ..config import IceburgConfig
import time


class CacheService(IService):
    """Service for caching"""
    
    def __init__(self, config: IceburgConfig):
        self.name = "CacheService"
        self.config = config
        self.initialized = False
        self.running = False
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = 3600  # 1 hour
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the cache service"""
        self.config = config
        self.default_ttl = config.get("cache_ttl", 3600)
        self.initialized = True
        return True
    
    def start(self) -> bool:
        """Start the cache service"""
        if not self.initialized:
            return False
        self.running = True
        return True
    
    def stop(self) -> bool:
        """Stop the cache service"""
        self.running = False
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health status"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "running": self.running,
            "healthy": self.initialized and self.running,
            "cache_size": len(self.cache)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return self.health_check()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time()
        }
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        self.cache.clear()
        return True
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        now = time.time()
        expired_keys = [key for key, entry in self.cache.items() if now > entry["expires_at"]]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)

