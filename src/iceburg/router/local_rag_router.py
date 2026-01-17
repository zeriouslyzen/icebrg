"""
Local RAG Router for ICEBURG v5
Routes queries to appropriate local RAG backends.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Literal

from ..memory.rag_gateway import get_rag_gateway, LocalSearchResponse

logger = logging.getLogger(__name__)


class LocalRAGRouter:
    """
    Routes queries to appropriate local RAG backends.
    
    Chooses between:
    - UnifiedMemory (persistent memories)
    - RAGMemoryIntegration (multi-layer memories)
    - Code/document embeddings
    """
    
    def __init__(self, cfg=None):
        """
        Initialize local RAG router.
        
        Args:
            cfg: ICEBURG configuration
        """
        self.cfg = cfg
        self.rag_gateway = get_rag_gateway(cfg)
        logger.info("LocalRAGRouter initialized")
    
    async def route_and_search(
        self,
        query: str,
        scope: Optional[Literal["code", "docs", "memory", "all"]] = None
    ) -> LocalSearchResponse:
        """
        Route query to appropriate local RAG backend and search.
        
        Args:
            query: Search query
            scope: Optional scope override (auto-detect if None)
            
        Returns:
            LocalSearchResponse with results
        """
        # Auto-detect scope if not provided
        if scope is None:
            scope = self._detect_scope(query)
        
        # Search via gateway
        response = await self.rag_gateway.search_local_knowledge(
            query=query,
            k=10,
            scope=scope
        )
        
        logger.debug(f"Local RAG search: {len(response.results)} results for scope={scope}")
        return response
    
    def _detect_scope(self, query: str) -> Literal["code", "docs", "memory", "all"]:
        """
        Auto-detect search scope from query.
        
        Args:
            query: Search query
            
        Returns:
            Detected scope
        """
        query_lower = query.lower()
        
        # Code-related keywords
        code_keywords = ["code", "function", "class", "module", "implementation", "source", "api", "endpoint"]
        if any(kw in query_lower for kw in code_keywords):
            return "code"
        
        # Docs-related keywords
        docs_keywords = ["documentation", "docs", "readme", "guide", "tutorial", "how to"]
        if any(kw in query_lower for kw in docs_keywords):
            return "docs"
        
        # Memory-related keywords
        memory_keywords = ["remember", "previous", "earlier", "conversation", "history"]
        if any(kw in query_lower for kw in memory_keywords):
            return "memory"
        
        # Default to all
        return "all"


# Global instance
_local_router: Optional[LocalRAGRouter] = None


def get_local_rag_router(cfg=None) -> LocalRAGRouter:
    """Get or create global local RAG router instance"""
    global _local_router
    if _local_router is None:
        _local_router = LocalRAGRouter(cfg)
    return _local_router





