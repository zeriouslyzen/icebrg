"""
Unified LLM Interface - Multi-ASI System That Mimics Single LLM
Like ICEBURG, OpenAI, Anthropic - Instant responses + Deep research

This creates a unified interface that routes queries intelligently:
- Simple queries → Instant responses (<2 seconds)
- Complex queries → Deep research (2-5 minutes)
- All queries feel like talking to a single LLM
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import json

from .config import load_config
from .llm import chat_complete, get_llm_cache_stats
from .protocol.legacy.protocol_legacy import iceberg_protocol
from .integration.reflexive_routing import ReflexiveRoutingSystem
from .memory.unified_memory import UnifiedMemory
from .caching.semantic_cache import SemanticCache
from .tracking.source_citation_tracker import SourceCitationTracker


class ResponseMode(Enum):
    """Response modes for unified LLM interface"""
    INSTANT = "instant"  # <2 seconds, single LLM call
    FAST = "fast"  # 2-10 seconds, lightweight processing
    BALANCED = "balanced"  # 10-30 seconds, hybrid processing
    DEEP = "deep"  # 30-300 seconds, full protocol


@dataclass
class QueryComplexity:
    """Query complexity analysis"""
    score: float  # 0.0 (simple) to 1.0 (complex)
    confidence: float  # 0.0 to 1.0
    requires_research: bool
    requires_agents: bool
    estimated_time: float  # seconds
    recommended_mode: ResponseMode


@dataclass
class UnifiedResponse:
    """Unified response from multi-ASI system"""
    content: str
    mode: ResponseMode
    processing_time: float
    complexity_score: float
    confidence: float
    sources: List[str]
    metadata: Dict[str, Any]


class UnifiedLLMInterface:
    """
    Unified LLM Interface - Multi-ASI System That Mimics Single LLM
    
    Architecture:
    - Single API endpoint that feels like talking to one LLM
    - Intelligent routing based on query complexity
        - Instant responses for simple queries (like ICEBURG)
    - Deep research for complex queries (full ICEBURG protocol)
    - Transparent switching between modes
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize unified LLM interface"""
        self.config = load_config()  # load_config doesn't take arguments
        self.memory = UnifiedMemory(self.config)
        self.reflexive_router = ReflexiveRoutingSystem(self.config)
        self.semantic_cache = SemanticCache(similarity_threshold=0.8)
        self.source_tracker = SourceCitationTracker(config=self.config)
        
        # Model pool configuration (like big companies)
        self.model_pool = {
            "instant": {
                "model": "qwen2.5:1.5b",  # Fastest model for instant responses
                "max_tokens": 512,
                "temperature": 0.7,
                "target_time": 1.0  # <1 second target
            },
            "fast": {
                "model": "llama3.1:3b",  # Balanced fast model
                "max_tokens": 1024,
                "temperature": 0.7,
                "target_time": 5.0  # <5 seconds target
            },
            "balanced": {
                "model": "llama3.1:8b",  # Balanced model
                "max_tokens": 2048,
                "temperature": 0.7,
                "target_time": 20.0  # <20 seconds target
            },
            "deep": {
                "model": "llama3.1:70b",  # Deep research model
                "max_tokens": 4096,
                "temperature": 0.7,
                "target_time": 300.0  # <5 minutes target
            }
        }
        
        # Complexity thresholds
        self.complexity_thresholds = {
            ResponseMode.INSTANT: 0.2,  # <20% complexity → instant
            ResponseMode.FAST: 0.4,  # <40% complexity → fast
            ResponseMode.BALANCED: 0.7,  # <70% complexity → balanced
            ResponseMode.DEEP: 1.0  # >70% complexity → deep
        }
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "instant_responses": 0,
            "fast_responses": 0,
            "balanced_responses": 0,
            "deep_responses": 0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    def analyze_complexity(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryComplexity:
        """
        Analyze query complexity to determine routing strategy.
        
        This is how big companies like ICEBURG route queries:
        - Fast pattern matching for instant routing
        - Confidence scoring for reliability
        - Time estimation for user expectations
        """
        if context is None:
            context = {}
        
        # Simple heuristics (fast, <10ms)
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Complexity indicators
        simple_patterns = [
            "what is", "define", "explain briefly", "tell me",
            "hello", "hi", "hey", "thanks", "thank you"
        ]
        complex_patterns = [
            "analyze", "research", "investigate", "comprehensive",
            "compare", "evaluate", "design", "create", "build",
            "breakdown", "detailed analysis", "deep dive"
        ]
        
        # Calculate complexity score (0.0 to 1.0)
        complexity_score = 0.0
        
        # Length factor
        if word_count > 50:
            complexity_score += 0.3
        elif word_count > 20:
            complexity_score += 0.2
        
        # Pattern matching
        if any(pattern in query_lower for pattern in complex_patterns):
            complexity_score += 0.4
        if any(pattern in query_lower for pattern in simple_patterns):
            complexity_score -= 0.2
        
        # Question count
        question_count = query.count("?")
        if question_count > 2:
            complexity_score += 0.2
        
        # Technical terms
        technical_terms = [
            "quantum", "neural", "algorithm", "architecture", "framework",
            "protocol", "system", "implementation", "optimization"
        ]
        if any(term in query_lower for term in technical_terms):
            complexity_score += 0.2
        
        # Clamp to [0, 1]
        complexity_score = max(0.0, min(1.0, complexity_score))
        
        # Determine recommended mode
        if complexity_score < self.complexity_thresholds[ResponseMode.INSTANT]:
            recommended_mode = ResponseMode.INSTANT
            estimated_time = 1.0
        elif complexity_score < self.complexity_thresholds[ResponseMode.FAST]:
            recommended_mode = ResponseMode.FAST
            estimated_time = 5.0
        elif complexity_score < self.complexity_thresholds[ResponseMode.BALANCED]:
            recommended_mode = ResponseMode.BALANCED
            estimated_time = 20.0
        else:
            recommended_mode = ResponseMode.DEEP
            estimated_time = 180.0
        
        # Calculate confidence (higher for clear patterns)
        confidence = 0.8  # Base confidence
        if any(pattern in query_lower for pattern in simple_patterns):
            confidence = 0.95  # Very confident for simple queries
        elif any(pattern in query_lower for pattern in complex_patterns):
            confidence = 0.85  # High confidence for complex queries
        
        return QueryComplexity(
            score=complexity_score,
            confidence=confidence,
            requires_research=complexity_score > 0.6,
            requires_agents=complexity_score > 0.7,
            estimated_time=estimated_time,
            recommended_mode=recommended_mode
        )
    
    def process_instant(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process query with instant response (<2 seconds).
        
        Like ICEBURG's instant mode - single LLM call, no agents, no deliberation.
        """
        model_config = self.model_pool["instant"]
        
        # Single LLM call with optimized prompt
        prompt = f"Provide a concise, accurate response to this query: {query}"
        system = "You are ICEBURG, an advanced AI assistant. Provide helpful, accurate responses."
        
        start_time = time.time()
        response = chat_complete(
            model=model_config["model"],
            prompt=prompt,
            system=system,
            temperature=model_config["temperature"],
            options={"num_predict": model_config["max_tokens"]}
        )
        processing_time = time.time() - start_time
        
        if processing_time > 2.0:
            # Log slow instant response
            print(f"[WARNING] Instant response took {processing_time:.2f}s (target: <2s)")
        
        return response
    
    async def process_fast(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process query with fast response (2-10 seconds).
        
        Lightweight processing with minimal agents, no full protocol.
        """
        model_config = self.model_pool["fast"]
        
        # Use reflexive routing for fast responses
        routing_decision = self.reflexive_router.route_query(query)
        
        if routing_decision.route_type == "reflexive" and routing_decision.confidence > 0.7:
            # Use reflexive response
            reflexive_response = await self.reflexive_router.process_reflexive(query)
            if not reflexive_response.escalation_recommended:
                return reflexive_response.response
        
        # Fallback to single LLM call with more tokens
        prompt = f"Provide a thoughtful, accurate response to this query: {query}"
        system = "You are ICEBURG, an advanced AI assistant. Provide helpful, accurate responses."
        
        response = chat_complete(
            model=model_config["model"],
            prompt=prompt,
            system=system,
            temperature=model_config["temperature"],
            options={"num_predict": model_config["max_tokens"]}
        )
        
        return response
    
    async def process_balanced(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process query with balanced response (10-30 seconds).
        
        Hybrid processing with some agents, but not full protocol.
        """
        # Use fast mode of ICEBURG protocol
        result = iceberg_protocol(
            query,
            fast=True,  # Use fast mode
            verbose=False
        )
        
        return result if isinstance(result, str) else str(result)
    
    async def process_deep(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process query with deep research (30-300 seconds).
        
        Full ICEBURG protocol with all agents and emergence detection.
        """
        # Use full ICEBURG protocol
        result = iceberg_protocol(
            query,
            fast=False,  # Full protocol
            verbose=False
        )
        
        return result if isinstance(result, str) else str(result)
    
    async def query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        force_mode: Optional[ResponseMode] = None,
        stream: bool = False
    ) -> AsyncGenerator[UnifiedResponse, None]:
        """
        Main query interface - mimics single LLM behavior.
        
        This is the unified API that big companies use:
        - Single entry point for all queries
        - Automatic routing based on complexity
        - Transparent mode switching
        - Streaming support for instant feedback
        """
        if context is None:
            context = {}
        
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Analyze complexity
        complexity = self.analyze_complexity(query, context)
        
        # Determine mode
        if force_mode:
            mode = force_mode
        else:
            mode = complexity.recommended_mode
        
        # Check cache first (for instant responses)
        if mode == ResponseMode.INSTANT:
            cache_key = f"instant:{query[:100]}"
            # Check semantic cache
            cached_result = self.semantic_cache.get(query)
            if cached_result:
                yield UnifiedResponse(
                    content=cached_result.get("content", ""),
                    mode=mode,
                    processing_time=0.1,
                    complexity_score=0.0,
                    confidence=1.0,
                    sources=cached_result.get("sources", []),
                    metadata={"cached": True, **cached_result.get("metadata", {})}
                )
                return
        
        # Process query based on mode
        try:
            if mode == ResponseMode.INSTANT:
                self.stats["instant_responses"] += 1
                content = self.process_instant(query, context)
            elif mode == ResponseMode.FAST:
                self.stats["fast_responses"] += 1
                content = await self.process_fast(query, context)
            elif mode == ResponseMode.BALANCED:
                self.stats["balanced_responses"] += 1
                content = await self.process_balanced(query, context)
            else:  # DEEP
                self.stats["deep_responses"] += 1
                content = await self.process_deep(query, context)
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_queries"] - 1) + processing_time) /
                self.stats["total_queries"]
            )
            
            # Extract sources from response
            sources = self.source_tracker.extract_sources(content)
            if not sources:
                # Try to extract from metadata if available
                if isinstance(content, dict) and "sources" in content:
                    sources = content["sources"]
                elif hasattr(context, "sources"):
                    sources = context.sources
            
            # Create unified response
            response = UnifiedResponse(
                content=content if isinstance(content, str) else str(content),
                mode=mode,
                processing_time=processing_time,
                complexity_score=complexity.score,
                confidence=complexity.confidence,
                sources=sources,
                metadata={
                    "estimated_time": complexity.estimated_time,
                    "requires_research": complexity.requires_research,
                    "requires_agents": complexity.requires_agents
                }
            )
            
            # Cache response for future similar queries
            if mode == ResponseMode.INSTANT:
                self.semantic_cache.set(
                    query,
                    {
                        "content": response.content,
                        "sources": response.sources,
                        "metadata": response.metadata
                    },
                    ttl=3600
                )
            
            if stream:
                # Stream response in chunks
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size]
                    partial_response = UnifiedResponse(
                        content=chunk,
                        mode=mode,
                        processing_time=processing_time,
                        complexity_score=complexity.score,
                        confidence=complexity.confidence,
                        sources=[],
                        metadata={"partial": True, "chunk_index": i // chunk_size}
                    )
                    yield partial_response
            else:
                yield response
        
        except Exception as e:
            # Return error response
            error_response = UnifiedResponse(
                content=f"Error processing query: {str(e)}",
                mode=mode,
                processing_time=time.time() - start_time,
                complexity_score=complexity.score,
                confidence=0.0,
                sources=[],
                metadata={"error": str(e)}
            )
            yield error_response
    
    async def query_sync(self, query: str, context: Optional[Dict[str, Any]] = None, 
                        force_mode: Optional[ResponseMode] = None) -> UnifiedResponse:
        """
        Synchronous query interface (for backward compatibility).
        """
        async for response in self.query(query, context, force_mode):
            return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about unified LLM interface"""
        cache_stats = get_llm_cache_stats()
        
        return {
            **self.stats,
            "cache_stats": cache_stats,
            "mode_distribution": {
                "instant": self.stats["instant_responses"] / max(self.stats["total_queries"], 1),
                "fast": self.stats["fast_responses"] / max(self.stats["total_queries"], 1),
                "balanced": self.stats["balanced_responses"] / max(self.stats["total_queries"], 1),
                "deep": self.stats["deep_responses"] / max(self.stats["total_queries"], 1)
            }
        }


# Global instance for easy access
_unified_llm: Optional[UnifiedLLMInterface] = None


def get_unified_llm(config_path: Optional[str] = None) -> UnifiedLLMInterface:
    """Get or create global unified LLM interface instance"""
    global _unified_llm
    if _unified_llm is None:
        _unified_llm = UnifiedLLMInterface()  # config_path not used
    return _unified_llm


async def unified_query(query: str, context: Optional[Dict[str, Any]] = None, 
                       force_mode: Optional[str] = None) -> UnifiedResponse:
    """
    Convenience function for unified query interface.
    
    Usage:
        response = await unified_query("What is ICEBURG?")
        print(response.content)
    """
    llm = get_unified_llm()
    
    mode = None
    if force_mode:
        mode = ResponseMode[force_mode.upper()]
    
    return await llm.query_sync(query, context, mode)

