"""
RAG Gateway Stub for ICEBURG.
Provides a temporary interface for local RAG operations while the full implementation is pending.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class LocalSearchResult:
    """A single result from local RAG search"""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

@dataclass
class LocalSearchResponse:
    """Response from local RAG search"""
    query: str
    results: List[LocalSearchResult] = field(default_factory=list)
    scope: str = "all"
    metadata: Dict[str, Any] = field(default_factory=dict)

class RAGGateway:
    """Stub for RAG Gateway"""
    def __init__(self, cfg=None):
        self.cfg = cfg

    async def search_local_knowledge(
        self,
        query: str,
        k: int = 10,
        scope: str = "all"
    ) -> LocalSearchResponse:
        """Stub for searching local knowledge"""
        return LocalSearchResponse(query=query, results=[], scope=scope)

_gateway = None

def get_rag_gateway(cfg=None) -> RAGGateway:
    """Get global RAG gateway instance"""
    global _gateway
    if _gateway is None:
        _gateway = RAGGateway(cfg)
    return _gateway

