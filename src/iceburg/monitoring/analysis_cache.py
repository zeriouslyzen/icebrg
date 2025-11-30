"""
Analysis Cache for LLM Bottleneck Analysis
Caches common LLM analyses to reduce latency (avoid 10-20s delays)
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Cache for LLM bottleneck analyses to reduce latency."""
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_seconds: int = 3600):
        """
        Initialize analysis cache.
        
        Args:
            cache_dir: Cache directory path
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        if cache_dir is None:
            cache_dir = Path("data/monitoring/analysis_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "analysis_cache.json"
        self.ttl_seconds = ttl_seconds
        
        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        
        # Load cache from disk
        self._load_cache()
    
    def _get_cache_key(self, alert: Dict[str, Any]) -> str:
        """Generate cache key from alert."""
        # Create key from bottleneck type, severity, and threshold
        key_data = {
            "bottleneck_type": alert.get("bottleneck_type", ""),
            "severity": alert.get("severity", ""),
            "threshold": alert.get("threshold", 0),
            "current_value": round(alert.get("current_value", 0), 2)  # Round to avoid exact match issues
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk."""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Filter expired entries
                current_time = time.time()
                for key, entry in data.items():
                    if current_time - entry.get("timestamp", 0) < self.ttl_seconds:
                        self._cache[key] = entry
                logger.info(f"Loaded {len(self._cache)} cached analyses")
        except Exception as e:
            logger.warning(f"Failed to load analysis cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save analysis cache: {e}")
    
    def get(self, alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis for alert.
        
        Args:
            alert: Bottleneck alert dictionary
            
        Returns:
            Cached analysis or None if not found/expired
        """
        key = self._get_cache_key(alert)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                # Check if expired
                if time.time() - entry.get("timestamp", 0) < self.ttl_seconds:
                    logger.debug(f"Cache hit for alert: {alert.get('alert_id', 'unknown')}")
                    return entry.get("analysis")
                else:
                    # Remove expired entry
                    del self._cache[key]
                    logger.debug(f"Cache expired for alert: {alert.get('alert_id', 'unknown')}")
        
        return None
    
    def set(self, alert: Dict[str, Any], analysis: Dict[str, Any]):
        """
        Cache analysis for alert.
        
        Args:
            alert: Bottleneck alert dictionary
            analysis: LLM analysis result
        """
        key = self._get_cache_key(alert)
        
        with self._lock:
            self._cache[key] = {
                "analysis": analysis,
                "timestamp": time.time(),
                "alert_id": alert.get("alert_id", "unknown")
            }
            
            # Save to disk periodically (every 10 entries)
            if len(self._cache) % 10 == 0:
                self._save_cache()
        
        logger.debug(f"Cached analysis for alert: {alert.get('alert_id', 'unknown')}")
    
    def clear(self):
        """Clear all cached analyses."""
        with self._lock:
            self._cache.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()
            logger.info("Cleared analysis cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            valid_entries = sum(
                1 for entry in self._cache.values()
                if current_time - entry.get("timestamp", 0) < self.ttl_seconds
            )
            
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_entries,
                "expired_entries": len(self._cache) - valid_entries,
                "cache_dir": str(self.cache_dir)
            }

