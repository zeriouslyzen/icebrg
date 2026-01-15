"""
Request Router for ICEBURG v5
Classifies incoming queries and routes them to appropriate handlers:
- web_research: Search-first mode (default for factual/current/technical queries)
- local_rag: Local codebase/docs/internal data
- pure_reasoning: Philosophical/meta/creative queries
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Routing decision for a query"""
    mode: Literal["web_research", "local_rag", "pure_reasoning"]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class RequestRouter:
    """
    Routes queries to appropriate handlers based on content and intent.
    
    Uses rule-based classification with optional LLM fallback for ambiguous cases.
    """
    
    def __init__(self, use_llm_fallback: bool = False):
        """
        Initialize request router.
        
        Args:
            use_llm_fallback: Use LLM for ambiguous queries (default: False, rules-only)
        """
        self.use_llm_fallback = use_llm_fallback
        
        # Patterns for web_research (default)
        self.web_research_patterns = [
            r"\b(what|when|where|who|how|why)\s+(is|are|was|were|did|does|will|can|could)\s+",
            r"\b(current|latest|recent|new|today|2025|2024)\b",
            r"\b(explain|describe|tell me about|what's|what is)\s+",
            r"\b(search|find|look up|research|information about)\b",
            r"\b(definition|meaning|examples?|comparison|difference)\b",
            r"\b(how to|how do|tutorial|guide|steps?)\b",
            r"\b(price|cost|where to buy|availability)\b",
            r"\b(news|article|report|study|paper|research)\b",
        ]
        
        # Patterns for local_rag
        self.local_rag_patterns = [
            r"\b(code|function|class|module|file|implementation|source)\b",
            r"\b(iceburg|iceberg|this project|our codebase|repository)\b",
            r"\b(how does.*work|where is.*defined|show me.*code)\b",
            r"\b(docs?|documentation|readme|guide|tutorial)\b",
            r"\b(config|configuration|settings?|options?)\b",
            r"\b(api|endpoint|route|handler)\b",
        ]
        
        # Patterns for pure_reasoning
        self.pure_reasoning_patterns = [
            r"\b(philosophy|philosophical|meaning of life|ethics|morality)\b",
            r"\b(creative|imagine|hypothetical|what if|scenario)\b",
            r"\b(opinion|think|believe|perspective|viewpoint)\b",
            r"\b(abstract|conceptual|theoretical|metaphysical)\b",
            r"\b(art|poetry|story|narrative|fiction)\b",
        ]
        
        logger.info("RequestRouter initialized")
    
    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route a query to appropriate handler.
        
        Args:
            query: User query
            context: Optional context (mode preference, user history, etc.)
            
        Returns:
            RoutingDecision with mode, confidence, and reasoning
        """
        query_lower = query.lower().strip()
        
        # Check for explicit mode in context
        if context and "mode" in context:
            explicit_mode = context["mode"]
            if explicit_mode in ["web_research", "local_rag", "pure_reasoning"]:
                return RoutingDecision(
                    mode=explicit_mode,
                    confidence=1.0,
                    reasoning=f"Explicit mode specified: {explicit_mode}",
                    metadata={"explicit": True}
                )
        
        # Score each mode
        web_score = self._score_web_research(query_lower)
        local_score = self._score_local_rag(query_lower)
        reasoning_score = self._score_pure_reasoning(query_lower)
        
        # Determine winner
        scores = {
            "web_research": web_score,
            "local_rag": local_score,
            "pure_reasoning": reasoning_score
        }
        
        best_mode = max(scores, key=scores.get)
        best_score = scores[best_mode]
        
        # Normalize confidence (0.0 to 1.0)
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5
        
        # Default to web_research if ambiguous (search-first philosophy)
        if confidence < 0.4:
            best_mode = "web_research"
            confidence = 0.6
            reasoning = "Ambiguous query, defaulting to web_research (search-first)"
        else:
            reasoning = f"Matched {best_mode} patterns (score: {best_score:.2f})"
        
        # Use LLM fallback for very ambiguous cases if enabled
        if self.use_llm_fallback and confidence < 0.5:
            llm_decision = self._llm_classify(query, context)
            if llm_decision:
                return llm_decision
        
        return RoutingDecision(
            mode=best_mode,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "scores": scores,
                "query_length": len(query)
            }
        )
    
    def _score_web_research(self, query_lower: str) -> float:
        """Score query for web_research mode"""
        score = 0.0
        for pattern in self.web_research_patterns:
            if re.search(pattern, query_lower):
                score += 1.0
        return score
    
    def _score_local_rag(self, query_lower: str) -> float:
        """Score query for local_rag mode"""
        score = 0.0
        for pattern in self.local_rag_patterns:
            if re.search(pattern, query_lower):
                score += 1.5  # Higher weight for local patterns (more specific)
        return score
    
    def _score_pure_reasoning(self, query_lower: str) -> float:
        """Score query for pure_reasoning mode"""
        score = 0.0
        for pattern in self.pure_reasoning_patterns:
            if re.search(pattern, query_lower):
                score += 1.0
        return score
    
    def _llm_classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[RoutingDecision]:
        """
        Use LLM to classify ambiguous queries.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            RoutingDecision or None if LLM unavailable
        """
        # For now, return None (LLM fallback not implemented)
        # Could use a small model or Secretary in classifier mode
        return None


# Global instance
_router: Optional[RequestRouter] = None


def get_request_router(use_llm_fallback: bool = False) -> RequestRouter:
    """Get or create global request router instance"""
    global _router
    if _router is None:
        _router = RequestRouter(use_llm_fallback=use_llm_fallback)
    return _router



