"""
ICEBURG Prompt Interpreter Engine
Persistent caching engine that tracks and reuses previous word analyses
Works like a Small Language Model (SML) that learns from previous runs
"""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from threading import Lock
import time

from ..config import IceburgConfig
from .word_breakdown import WordBreakdownAnalyzer, WordBreakdown


class PromptInterpreterEngine:
    """
    Shared persistent caching engine for prompt interpretation.
    Tracks and quickly retrieves previously analyzed words and queries.
    SHARED ACROSS ALL USERS - Each user's analysis benefits everyone!
    """
    
    # Class-level shared cache (shared across all instances/users)
    _shared_word_cache: Dict[str, WordBreakdown] = {}
    _shared_query_cache: Dict[str, Dict[str, Any]] = {}
    _shared_stats: Dict[str, Any] = {
        "word_hits": 0,
        "word_misses": 0,
        "query_hits": 0,
        "query_misses": 0,
        "total_words_cached": 0,
        "total_queries_cached": 0,
        "total_users_contributed": 0,
        "last_updated": datetime.now().isoformat(),
        "cache_loaded": False
    }
    _shared_lock = Lock()
    
    def __init__(self, cfg: IceburgConfig, cache_dir: Optional[Path] = None):
        self.cfg = cfg
        self.word_analyzer = WordBreakdownAnalyzer()
        
        # Shared persistent cache directory (shared across all users)
        if cache_dir is None:
            cache_dir = Path(cfg.data_dir) / "prompt_interpreter_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.word_cache_file = self.cache_dir / "word_breakdowns.json"
        self.query_cache_file = self.cache_dir / "query_analyses.json"
        self.stats_file = self.cache_dir / "cache_stats.json"
        
        # Use shared class-level caches (shared across all users)
        # Load from disk only once (first instance)
        if not PromptInterpreterEngine._shared_stats.get("cache_loaded", False):
            self._load_caches()
            PromptInterpreterEngine._shared_stats["cache_loaded"] = True
            PromptInterpreterEngine._shared_stats["total_users_contributed"] = 1
        else:
            # Increment user count (each user contributes)
            PromptInterpreterEngine._shared_stats["total_users_contributed"] += 1
    
    def _load_caches(self):
        """Load word and query caches from disk (shared across all users)"""
        with PromptInterpreterEngine._shared_lock:
            # Load word breakdown cache
            if self.word_cache_file.exists():
                try:
                    with open(self.word_cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for word, breakdown_data in data.items():
                            # Reconstruct WordBreakdown from dict
                            PromptInterpreterEngine._shared_word_cache[word] = WordBreakdown(
                                word=breakdown_data.get('word', word),
                                morphological=breakdown_data.get('morphological', {}),
                                etymology=breakdown_data.get('etymology', {}),
                                semantic=breakdown_data.get('semantic', {}),
                                compression_hints=breakdown_data.get('compression_hints', []),
                                timestamp=breakdown_data.get('timestamp', time.time())
                            )
                    print(f"[PROMPT_INTERPRETER_ENGINE] Loaded {len(PromptInterpreterEngine._shared_word_cache)} cached words (shared across all users)")
                except Exception as e:
                    print(f"[PROMPT_INTERPRETER_ENGINE] Error loading word cache: {e}")
            
            # Load query analysis cache
            if self.query_cache_file.exists():
                try:
                    with open(self.query_cache_file, 'r', encoding='utf-8') as f:
                        PromptInterpreterEngine._shared_query_cache = json.load(f)
                    print(f"[PROMPT_INTERPRETER_ENGINE] Loaded {len(PromptInterpreterEngine._shared_query_cache)} cached queries (shared across all users)")
                except Exception as e:
                    print(f"[PROMPT_INTERPRETER_ENGINE] Error loading query cache: {e}")
            
            # Load cache stats
            if self.stats_file.exists():
                try:
                    with open(self.stats_file, 'r', encoding='utf-8') as f:
                        PromptInterpreterEngine._shared_stats = json.load(f)
                except Exception as e:
                    print(f"[PROMPT_INTERPRETER_ENGINE] Error loading cache stats: {e}")
    
    def _save_caches(self):
        """Save word and query caches to disk (shared across all users)"""
        with PromptInterpreterEngine._shared_lock:
            try:
                # Save word breakdown cache (shared)
                word_data = {}
                for word, breakdown in PromptInterpreterEngine._shared_word_cache.items():
                    word_data[word] = {
                        'word': breakdown.word,
                        'morphological': breakdown.morphological,
                        'etymology': breakdown.etymology,
                        'semantic': breakdown.semantic,
                        'compression_hints': breakdown.compression_hints,
                        'timestamp': breakdown.timestamp
                    }
                
                with open(self.word_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(word_data, f, indent=2, ensure_ascii=False)
                
                # Save query analysis cache (shared)
                with open(self.query_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(PromptInterpreterEngine._shared_query_cache, f, indent=2, ensure_ascii=False)
                
                # Save cache stats (shared)
                PromptInterpreterEngine._shared_stats['last_updated'] = datetime.now().isoformat()
                with open(self.stats_file, 'w', encoding='utf-8') as f:
                    json.dump(PromptInterpreterEngine._shared_stats, f, indent=2)
            
            except Exception as e:
                print(f"[PROMPT_INTERPRETER_ENGINE] Error saving caches: {e}")
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query caching"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_cached_word(self, word: str) -> Optional[WordBreakdown]:
        """Get cached word breakdown if available (shared across all users)"""
        word_key = word.lower()
        with PromptInterpreterEngine._shared_lock:
            if word_key in PromptInterpreterEngine._shared_word_cache:
                PromptInterpreterEngine._shared_stats['word_hits'] += 1
                return PromptInterpreterEngine._shared_word_cache[word_key]
            PromptInterpreterEngine._shared_stats['word_misses'] += 1
            return None
    
    def cache_word(self, word: str, breakdown: WordBreakdown):
        """Cache word breakdown for future use (shared across all users - benefits everyone!)"""
        word_key = word.lower()
        with PromptInterpreterEngine._shared_lock:
            if word_key not in PromptInterpreterEngine._shared_word_cache:
                # New word - cache it for all users
                PromptInterpreterEngine._shared_word_cache[word_key] = breakdown
                PromptInterpreterEngine._shared_stats['total_words_cached'] = len(PromptInterpreterEngine._shared_word_cache)
                # Save periodically (every 10 words)
                if len(PromptInterpreterEngine._shared_word_cache) % 10 == 0:
                    self._save_caches()
    
    def get_cached_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached query analysis if available (shared across all users)"""
        query_hash = self._get_query_hash(query)
        with PromptInterpreterEngine._shared_lock:
            if query_hash in PromptInterpreterEngine._shared_query_cache:
                PromptInterpreterEngine._shared_stats['query_hits'] += 1
                return PromptInterpreterEngine._shared_query_cache[query_hash]
            PromptInterpreterEngine._shared_stats['query_misses'] += 1
            return None
    
    def cache_query(self, query: str, analysis: Dict[str, Any]):
        """Cache query analysis for future use (shared across all users - benefits everyone!)"""
        query_hash = self._get_query_hash(query)
        with PromptInterpreterEngine._shared_lock:
            if query_hash not in PromptInterpreterEngine._shared_query_cache:
                # New query - cache it for all users
                PromptInterpreterEngine._shared_query_cache[query_hash] = {
                    **analysis,
                    'cached_at': datetime.now().isoformat(),
                    'query': query
                }
                PromptInterpreterEngine._shared_stats['total_queries_cached'] = len(PromptInterpreterEngine._shared_query_cache)
                self._save_caches()
    
    def analyze_word_fast(self, word: str) -> WordBreakdown:
        """
        Analyze word with caching - quickly retrieves from cache if available.
        This is the engine's core function - fast word analysis with tracking.
        """
        # Check cache first
        cached = self.get_cached_word(word)
        if cached:
            return cached
        
        # Not in cache - analyze and cache
        breakdown = self.word_analyzer.analyze_word(word)
        self.cache_word(word, breakdown)
        
        return breakdown
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (shared across all users)"""
        with PromptInterpreterEngine._shared_lock:
            stats = PromptInterpreterEngine._shared_stats.copy()
            total_words = stats['word_hits'] + stats['word_misses']
            total_queries = stats['query_hits'] + stats['query_misses']
            
            word_hit_rate = (stats['word_hits'] / total_words * 100) if total_words > 0 else 0
            query_hit_rate = (stats['query_hits'] / total_queries * 100) if total_queries > 0 else 0
            
            return {
                **stats,
                'word_hit_rate': f"{word_hit_rate:.1f}%",
                'query_hit_rate': f"{query_hit_rate:.1f}%",
                'cached_words_count': len(PromptInterpreterEngine._shared_word_cache),
                'cached_queries_count': len(PromptInterpreterEngine._shared_query_cache),
                'shared_across_users': True,
                'benefit_message': f"Each user's analysis benefits all {stats.get('total_users_contributed', 0)} users!"
            }
    
    def clear_cache(self):
        """Clear all caches (for testing/debugging) - affects all users"""
        with PromptInterpreterEngine._shared_lock:
            PromptInterpreterEngine._shared_word_cache.clear()
            PromptInterpreterEngine._shared_query_cache.clear()
            PromptInterpreterEngine._shared_stats = {
                "word_hits": 0,
                "word_misses": 0,
                "query_hits": 0,
                "query_misses": 0,
                "total_words_cached": 0,
                "total_queries_cached": 0,
                "total_users_contributed": 0,
                "last_updated": datetime.now().isoformat(),
                "cache_loaded": True
            }
            self._save_caches()

