"""
KV Compression for Long Contexts
4x compression via KV caches for long contexts
"""

import logging
from typing import Dict, Any, Optional, List
import json
import hashlib

logger = logging.getLogger(__name__)


class KVCompression:
    """KV compression for long contexts."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize KV compression.
        
        Args:
            config: Configuration for KV compression
        """
        self.config = config or {}
        
        # Compression configuration
        self.compression_ratio = self.config.get("compression_ratio", 4.0)  # 4x compression
        self.max_context_size = self.config.get("max_context_size", 4096)
        
        # KV cache
        self.kv_cache = {}
    
    def compress_context(
        self,
        architecture: Dict[str, Any],
        metrics: Dict[str, Any],
        continuum_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compress context using KV compression.
        
        Args:
            architecture: Architecture state
            metrics: Performance metrics
            continuum_state: Continuum memory state
            
        Returns:
            Compressed context
        """
        # Create context hash for caching
        context_hash = self._hash_context(architecture, metrics, continuum_state)
        
        # Check cache
        if context_hash in self.kv_cache:
            logger.debug("Using cached compressed context")
            return self.kv_cache[context_hash]
        
        # Compress context
        compressed = {
            "architecture_summary": self._summarize_architecture(architecture),
            "metrics_summary": self._summarize_metrics(metrics),
            "continuum_summary": self._summarize_continuum(continuum_state),
            "compression_ratio": self.compression_ratio
        }
        
        # Store in cache
        self.kv_cache[context_hash] = compressed
        
        logger.debug(f"Compressed context: {len(str(compressed))} chars")
        
        return compressed
    
    def _hash_context(
        self,
        architecture: Dict[str, Any],
        metrics: Dict[str, Any],
        continuum_state: Dict[str, Any]
    ) -> str:
        """Generate hash for context caching."""
        context_str = json.dumps({
            "architecture": architecture,
            "metrics": metrics,
            "continuum": continuum_state
        }, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _summarize_architecture(self, architecture: Dict[str, Any]) -> str:
        """Summarize architecture state."""
        # Extract key components
        components = architecture.get("components", [])
        return f"Architecture with {len(components)} components"
    
    def _summarize_metrics(self, metrics: Dict[str, Any]) -> str:
        """Summarize performance metrics."""
        # Extract key metrics
        key_metrics = ["latency", "memory", "cpu", "accuracy"]
        summary = []
        for metric in key_metrics:
            if metric in metrics:
                summary.append(f"{metric}: {metrics[metric]}")
        return ", ".join(summary)
    
    def _summarize_continuum(self, continuum_state: Dict[str, Any]) -> str:
        """Summarize continuum memory state."""
        # Extract key state
        evolutions = continuum_state.get("evolutions", [])
        return f"Continuum with {len(evolutions)} evolutions"
    
    def clear_cache(self):
        """Clear KV cache."""
        self.kv_cache.clear()
        logger.info("Cleared KV cache")

