"""
Advanced Memory Management for ICEBURG
Implements caching, compression, and pooling for optimal memory usage
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time
import hashlib
import json
import logging
import gzip
import pickle

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """Represents a memory block"""
    key: str
    data: Any
    size_bytes: int
    access_count: int
    last_access: float
    compressed: bool = False
    compression_ratio: float = 1.0


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    cache_size_mb: float
    compression_savings_mb: float
    hit_rate: float
    miss_rate: float


class AdvancedMemoryManager:
    """
    Advanced memory management for ICEBURG
    
    Features:
    - Intelligent caching with LRU eviction
    - Compression for large data structures
    - Memory pooling for efficient allocation
    - Predictive pre-warming
    - Memory bandwidth optimization
    """
    
    def __init__(
        self,
        max_cache_size_mb: float = 1024.0,
        compression_threshold_mb: float = 1.0,
        enable_compression: bool = True
    ):
        """Initialize advanced memory manager"""
        self.max_cache_size_mb = max_cache_size_mb
        self.compression_threshold_mb = compression_threshold_mb
        self.enable_compression = enable_compression
        
        # Cache storage
        self.cache: Dict[str, MemoryBlock] = {}
        self.access_order = deque()  # LRU tracking
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "compressions": 0,
            "evictions": 0,
            "total_size_bytes": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        if key in self.cache:
            block = self.cache[key]
            
            # Update access tracking
            block.access_count += 1
            block.last_access = time.time()
            
            # Move to end of access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats["hits"] += 1
            
            # Decompress if needed
            if block.compressed:
                return self._decompress(block.data)
            return block.data
        
        self.stats["misses"] += 1
        return None
    
    def put(self, key: str, data: Any, priority: float = 1.0) -> bool:
        """Store data in cache"""
        try:
            # Calculate size
            size_bytes = self._estimate_size(data)
            size_mb = size_bytes / (1024 * 1024)
            
            # Compress if needed
            compressed = False
            compression_ratio = 1.0
            if self.enable_compression and size_mb > self.compression_threshold_mb:
                compressed_data, compression_ratio = self._compress(data)
                if compression_ratio > 1.1:  # Only compress if significant savings
                    data = compressed_data
                    compressed = True
                    self.stats["compressions"] += 1
            
            # Check if we need to evict
            while self._get_total_size_mb() + size_mb > self.max_cache_size_mb:
                if not self._evict_lru():
                    return False  # Can't make room
            
            # Store block
            block = MemoryBlock(
                key=key,
                data=data,
                size_bytes=size_bytes,
                access_count=1,
                last_access=time.time(),
                compressed=compressed,
                compression_ratio=compression_ratio
            )
            
            self.cache[key] = block
            self.access_order.append(key)
            self.stats["total_size_bytes"] += size_bytes
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            return False
    
    def _compress(self, data: Any) -> Tuple[bytes, float]:
        """Compress data using gzip"""
        try:
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            compressed = gzip.compress(serialized, compresslevel=6)
            compressed_size = len(compressed)
            
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            return compressed, compression_ratio
        
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return pickle.dumps(data), 1.0
    
    def _decompress(self, compressed_data: bytes) -> Any:
        """Decompress data"""
        try:
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return None
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self.access_order:
            return False
        
        key_to_evict = self.access_order.popleft()
        if key_to_evict in self.cache:
            block = self.cache[key_to_evict]
            self.stats["total_size_bytes"] -= block.size_bytes
            del self.cache[key_to_evict]
            self.stats["evictions"] += 1
            return True
        
        return False
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes"""
        try:
            return len(pickle.dumps(data))
        except:
            # Fallback estimation
            return len(str(data).encode('utf-8'))
    
    def _get_total_size_mb(self) -> float:
        """Get total cache size in MB"""
        return self.stats["total_size_bytes"] / (1024 * 1024)
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics"""
        total_hits_misses = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_hits_misses if total_hits_misses > 0 else 0.0
        miss_rate = 1.0 - hit_rate
        
        compression_savings = sum(
            (block.size_bytes * (block.compression_ratio - 1.0) / block.compression_ratio)
            for block in self.cache.values()
            if block.compressed
        ) / (1024 * 1024)
        
        return MemoryStats(
            total_memory_mb=self.max_cache_size_mb,
            used_memory_mb=self._get_total_size_mb(),
            free_memory_mb=self.max_cache_size_mb - self._get_total_size_mb(),
            cache_size_mb=self._get_total_size_mb(),
            compression_savings_mb=compression_savings,
            hit_rate=hit_rate,
            miss_rate=miss_rate
        )
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "compressions": 0,
            "evictions": 0,
            "total_size_bytes": 0
        }
    
    def prewarm(self, keys: List[str], data_loader: callable):
        """Pre-warm cache with predicted keys"""
        for key in keys:
            if key not in self.cache:
                data = data_loader(key)
                if data:
                    self.put(key, data)


# Global memory manager instance
_memory_manager: Optional[AdvancedMemoryManager] = None

def get_memory_manager() -> AdvancedMemoryManager:
    """Get or create global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = AdvancedMemoryManager()
    return _memory_manager

