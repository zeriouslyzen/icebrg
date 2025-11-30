"""
ICEBURG Performance Optimizations

Provides:
- Caching
- Connection pooling
- Graph optimization
- Performance monitoring
"""

import time
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from collections import OrderedDict

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: float = 3600.0  # 1 hour default
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_access = datetime.now()


class LRUCache:
    """
    LRU (Least Recently Used) cache implementation.
    
    Features:
    - Size-limited cache
    - TTL (Time To Live) support
    - Access tracking
    - Automatic eviction
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired():
            del self.cache[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.touch()
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache"""
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Add new entry
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl or self.default_ttl
        )
        self.cache[key] = entry
        
        # Evict if over size limit
        if len(self.cache) > self.max_size:
            # Remove oldest entry
            self.cache.popitem(last=False)
    
    def delete(self, key: str):
        """Delete entry from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self.cleanup_expired()
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": sum(1 for entry in self.cache.values() if entry.access_count > 0) / max(1, len(self.cache))
        }


class ConnectionPool:
    """
    Connection pooling for resource management.
    
    Features:
    - Connection reuse
    - Connection limits
    - Connection health checks
    - Automatic cleanup
    """
    
    def __init__(self, max_connections: int = 10, connection_timeout: float = 30.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.pool: Dict[str, Any] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    def get_connection(self, connection_id: str, factory: Callable[[], Any]) -> Any:
        """
        Get connection from pool or create new one.
        
        Args:
            connection_id: Unique connection identifier
            factory: Factory function to create new connection
            
        Returns:
            Connection object
        """
        # Check if connection exists and is valid
        if connection_id in self.pool:
            connection = self.pool[connection_id]
            metadata = self.connection_metadata.get(connection_id, {})
            
            # Check if connection is still valid
            last_used = metadata.get("last_used", 0)
            if time.time() - last_used < self.connection_timeout:
                metadata["last_used"] = time.time()
                metadata["use_count"] = metadata.get("use_count", 0) + 1
                return connection
        
        # Create new connection if pool not full
        if len(self.pool) >= self.max_connections:
            # Evict oldest connection
            oldest_id = min(
                self.connection_metadata.keys(),
                key=lambda k: self.connection_metadata[k].get("last_used", 0)
            )
            self.release_connection(oldest_id)
        
        # Create new connection
        connection = factory()
        self.pool[connection_id] = connection
        self.connection_metadata[connection_id] = {
            "created": time.time(),
            "last_used": time.time(),
            "use_count": 1
        }
        
        return connection
    
    def release_connection(self, connection_id: str):
        """Release connection from pool"""
        if connection_id in self.pool:
            connection = self.pool[connection_id]
            
            # Close connection if it has close method
            if hasattr(connection, 'close'):
                try:
                    connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {connection_id}: {e}")
            
            del self.pool[connection_id]
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
    
    def cleanup_idle_connections(self):
        """Remove idle connections"""
        current_time = time.time()
        idle_connections = [
            conn_id for conn_id, metadata in self.connection_metadata.items()
            if current_time - metadata.get("last_used", 0) > self.connection_timeout
        ]
        
        for conn_id in idle_connections:
            self.release_connection(conn_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        self.cleanup_idle_connections()
        return {
            "active_connections": len(self.pool),
            "max_connections": self.max_connections,
            "connections": {
                conn_id: {
                    "age": time.time() - metadata.get("created", 0),
                    "use_count": metadata.get("use_count", 0),
                    "last_used": time.time() - metadata.get("last_used", 0)
                }
                for conn_id, metadata in self.connection_metadata.items()
            }
        }


class GraphOptimizer:
    """
    Graph optimization for dependency resolution and execution planning.
    
    Features:
    - Topological sort optimization
    - Parallel execution grouping
    - Dependency graph caching
    - Execution path optimization
    """
    
    def __init__(self):
        self.graph_cache: Dict[str, Any] = {}
        self.execution_plans: Dict[str, List[Any]] = {}
    
    def optimize_dependency_graph(self, graph: Dict[str, set]) -> List[str]:
        """
        Optimize dependency graph using topological sort.
        
        Args:
            graph: Dependency graph {node: {dependencies}}
            
        Returns:
            Optimized execution order
        """
        # Generate cache key
        cache_key = self._graph_cache_key(graph)
        
        # Check cache
        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key]
        
        # Topological sort
        ordered = self._topological_sort(graph)
        
        # Cache result
        self.graph_cache[cache_key] = ordered
        
        return ordered
    
    def group_parallel_execution(self, graph: Dict[str, set]) -> List[List[str]]:
        """
        Group nodes for parallel execution.
        
        Args:
            graph: Dependency graph {node: {dependencies}}
            
        Returns:
            List of groups, where nodes in each group can execute in parallel
        """
        ordered = self.optimize_dependency_graph(graph)
        
        # Group by dependency level
        groups: List[List[str]] = []
        completed: set = set()
        
        while len(completed) < len(ordered):
            # Find nodes ready to execute (all dependencies completed)
            ready = [
                node for node in ordered
                if node not in completed
                and graph.get(node, set()).issubset(completed)
            ]
            
            if ready:
                groups.append(ready)
                completed.update(ready)
            else:
                # No nodes ready - might be circular dependency
                remaining = [node for node in ordered if node not in completed]
                if remaining:
                    logger.warning(f"Could not resolve dependencies for: {remaining}")
                    groups.append(remaining)
                    completed.update(remaining)
                break
        
        return groups
    
    def _topological_sort(self, graph: Dict[str, set]) -> List[str]:
        """Perform topological sort"""
        ordered = []
        visited = set()
        temp_visited = set()
        
        def visit(node: str):
            if node in temp_visited:
                logger.warning(f"Circular dependency detected involving {node}")
                return
            if node in visited:
                return
            
            temp_visited.add(node)
            
            # Visit dependencies first
            for dep in graph.get(node, set()):
                if dep in graph:
                    visit(dep)
            
            temp_visited.remove(node)
            visited.add(node)
            ordered.append(node)
        
        for node in graph:
            if node not in visited:
                visit(node)
        
        return ordered
    
    def _graph_cache_key(self, graph: Dict[str, set]) -> str:
        """Generate cache key for graph"""
        graph_str = json.dumps({k: sorted(v) for k, v in graph.items()}, sort_keys=True)
        return hashlib.md5(graph_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear graph optimization cache"""
        self.graph_cache.clear()
        self.execution_plans.clear()


class PerformanceOptimizer:
    """
    Centralized performance optimization manager.
    
    Provides:
    - Caching
    - Connection pooling
    - Graph optimization
    - Performance monitoring
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl: float = 3600.0,
        max_connections: int = 10,
        connection_timeout: float = 30.0
    ):
        self.cache = LRUCache(max_size=cache_size, default_ttl=cache_ttl)
        self.connection_pool = ConnectionPool(
            max_connections=max_connections,
            connection_timeout=connection_timeout
        )
        self.graph_optimizer = GraphOptimizer()
        self.performance_metrics: Dict[str, Any] = {}
    
    def cached(self, ttl: Optional[float] = None):
        """
        Decorator for caching function results.
        
        Usage:
            @optimizer.cached(ttl=3600)
            def expensive_function(arg1, arg2):
                # expensive computation
                return result
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Check cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.cache.set(cache_key, result, ttl=ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_connection(self, connection_id: str, factory: Callable[[], Any]) -> Any:
        """Get connection from pool"""
        return self.connection_pool.get_connection(connection_id, factory)
    
    def release_connection(self, connection_id: str):
        """Release connection to pool"""
        self.connection_pool.release_connection(connection_id)
    
    def optimize_graph(self, graph: Dict[str, set]) -> List[str]:
        """Optimize dependency graph"""
        return self.graph_optimizer.optimize_dependency_graph(graph)
    
    def group_parallel(self, graph: Dict[str, set]) -> List[List[str]]:
        """Group nodes for parallel execution"""
        return self.graph_optimizer.group_parallel_execution(graph)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "cache": self.cache.get_stats(),
            "connection_pool": self.connection_pool.get_stats(),
            "graph_optimizer": {
                "cached_graphs": len(self.graph_optimizer.graph_cache),
                "cached_plans": len(self.graph_optimizer.execution_plans)
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.cache.cleanup_expired()
        self.connection_pool.cleanup_idle_connections()


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


# Import wraps for decorator
from functools import wraps

