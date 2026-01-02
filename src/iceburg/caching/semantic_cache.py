"""
Semantic Cache
L3: Semantic cache for similar queries
"""

from typing import Any, Dict, Optional, List
import time
import hashlib
import json


class SemanticCache:
    """L3: Semantic cache for similar queries"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0
        }
    
    def get(self, query: str, embedding: Optional[List[float]] = None) -> Optional[Any]:
        """Get value from semantic cache"""
        # Try exact match first
        query_hash = self._hash_query(query)
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            if time.time() <= entry["expires_at"]:
                self.stats["hits"] += 1
                return entry["value"]
        
        # Try semantic similarity
        if embedding:
            similar_key = self._find_similar(embedding)
            if similar_key:
                entry = self.cache[similar_key]
                if time.time() <= entry["expires_at"]:
                    self.stats["hits"] += 1
                    return entry["value"]
        
        self.stats["misses"] += 1
        return None
    
    def set(
        self,
        query: str,
        value: Any,
        embedding: Optional[List[float]] = None,
        ttl: int = 3600
    ) -> bool:
        """Set value in semantic cache"""
        query_hash = self._hash_query(query)
        
        self.cache[query_hash] = {
            "value": value,
            "query": query,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "ttl": ttl
        }
        
        if embedding:
            self.embeddings[query_hash] = embedding
        
        self.stats["sets"] += 1
        return True
    
    def _hash_query(self, query: str) -> str:
        """Hash query for exact matching"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def _find_similar(self, embedding: List[float]) -> Optional[str]:
        """Find similar query by embedding"""
        if not embedding or not self.embeddings:
            return None
        
        best_similarity = 0.0
        best_key = None
        
        for key, cached_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(embedding, cached_embedding)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_key = key
        
        return best_key
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    
    def save(self, path: str) -> bool:
        """Save cache to disk"""
        try:
            data = {
                "cache": self.cache,
                "embeddings": self.embeddings,
                "stats": self.stats
            }
            with open(path, 'w') as f:
                json.dump(data, f)
            return True
        except Exception as e:
            print(f"Failed to save cache: {e}")
            return False
            
    def load(self, path: str) -> bool:
        """Load cache from disk"""
        try:
            import os
            if not os.path.exists(path):
                return False
                
            with open(path, 'r') as f:
                data = json.load(f)
                
            self.cache = data.get("cache", {})
            self.embeddings = data.get("embeddings", {})
            self.stats = data.get("stats", {
                "hits": 0,
                "misses": 0,
                "sets": 0
            })
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def warm_from_transcript(self, transcript_path: str):
        """Warm cache from transcript markdown"""
        try:
            import os
            import re
            from ..llm import embed_texts
            
            if not os.path.exists(transcript_path):
                print(f"Transcript not found: {transcript_path}")
                return
                
            print(f"Warming cache from {transcript_path}...")
            
            with open(transcript_path, 'r') as f:
                content = f.read()
                
            # Split by queries
            # Format: ## Query: <query>\n...### Response:\n<response>\n\n---
            sections = content.split("## Query: ")
            count = 0
            
            for section in sections[1:]: # Skip header
                try:
                    # Extract query
                    query_end = section.find("\n")
                    if query_end == -1: continue
                    query = section[:query_end].strip()
                    
                    # Extract response
                    resp_start = section.find("### Response:\n")
                    if resp_start == -1: continue
                    resp_start += len("### Response:\n")
                    
                    resp_end = section.find("\n\n---")
                    if resp_end == -1: 
                        resp_end = len(section)
                        
                    response = section[resp_start:resp_end].strip()
                    
                    if query and response:
                        # Skip if already cached
                        if self.get(query):
                            continue
                            
                        # Generate embedding
                        embeddings = embed_texts("nomic-embed-text", [query])
                        if embeddings:
                            self.set(query, response, embedding=embeddings[0], ttl=86400*7) # 1 week TTL
                            count += 1
                except Exception as e:
                    print(f"Error parsing section: {e}")
                    continue
            
            print(f"✅ Warmed cache with {count} entries")
            
        except Exception as e:
            print(f"Failed to warm cache: {e}")

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
            "similarity_threshold": self.similarity_threshold
        }

