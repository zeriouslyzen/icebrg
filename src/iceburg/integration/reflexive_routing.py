"""Reflexive Routing System - Fast 30s responses with 6.5min escalation for complex queries."""

from __future__ import annotations

import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Represents a routing decision."""
    route_type: str  # "reflexive", "escalated", "hybrid"
    estimated_time: str
    confidence: float
    reasoning: str
    complexity_score: float
    metadata: Dict[str, Any]


@dataclass
class ReflexiveResponse:
    """Represents a reflexive response."""
    response: str
    confidence: float
    escalation_recommended: bool
    escalation_reason: Optional[str]
    processing_time: float
    metadata: Dict[str, Any]


class ReflexiveRoutingSystem:
    """Routes queries between fast reflexive responses and full ICEBURG analysis."""
    
    def __init__(self, cfg: Any = None):
        self.cfg = cfg
        self.reflexive_timeout = 30  # 30 seconds for reflexive responses
        self.escalation_threshold = 0.7  # Complexity threshold for escalation
        self._latency_ema: float = 1.0
        self._success_ema: float = 0.9
        
        # Simple query patterns that can be handled reflexively
        self.simple_patterns = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "definition": ["what is", "define", "explain", "meaning of"],
            "factual": ["when", "where", "who", "how many", "how much"],
            "simple_question": ["can you", "do you", "are you", "will you"]
        }
        
        # Complex patterns that require escalation
        self.complex_patterns = {
            "research": ["research", "study", "investigate", "analyze", "breakthrough"],
            "philosophical": ["consciousness", "reality", "existence", "meaning", "purpose"],
            "scientific": ["quantum", "physics", "biology", "chemistry", "mathematics"],
            "creative": ["create", "design", "invent", "imagine", "generate"],
            "analysis": ["compare", "contrast", "evaluate", "assess", "critique"],
            "ide": ["ide", "editor", "monaco", "vscode", "code editor", "terminal", "file explorer", "lsp", "language server"],
            "complex_apps": ["vs code", "visual studio", "intellij", "eclipse", "sublime", "atom", "brackets"]
        }
        
        # LRU Cache for common queries (max 100 entries)
        self.response_cache: OrderedDict[str, ReflexiveResponse] = OrderedDict()
        self.max_cache_size = 100
        
        # Escalation history
        self.escalation_history: List[Dict[str, Any]] = []
        
        # Complexity scoring weights
        self.complexity_weights = {
            "length": 0.2,
            "technical_terms": 0.3,
            "question_count": 0.1,
            "pattern_complexity": 0.4
        }
    
    def route_query(self, query: str, context: Dict[str, Any] = None) -> RoutingDecision:
        """Route a query to appropriate processing path."""
        try:
            # Analyze query complexity
            complexity_score = self._analyze_complexity(query, context)
            
            # Check for cached response
            cached_response = self._check_cache(query)
            if cached_response and complexity_score < 0.5:
                return RoutingDecision(
                    route_type="reflexive",
                    estimated_time="< 1 second",
                    confidence=0.9,
                    reasoning="Cached response available",
                    complexity_score=complexity_score,
                    metadata={"cached": True, "cache_hit": True}
                )
            
            # Make routing decision
            if complexity_score < 0.3:
                return RoutingDecision(
                    route_type="reflexive",
                    estimated_time="10-30 seconds",
                    confidence=0.8,
                    reasoning="Simple query suitable for reflexive response",
                    complexity_score=complexity_score,
                    metadata={"cached": False}
                )
            elif complexity_score < self.escalation_threshold:
                return RoutingDecision(
                    route_type="hybrid",
                    estimated_time="1-2 minutes",
                    confidence=0.7,
                    reasoning="Medium complexity, hybrid approach",
                    complexity_score=complexity_score,
                    metadata={"cached": False}
                )
            else:
                return RoutingDecision(
                    route_type="escalated",
                    estimated_time="4-6 minutes",
                    confidence=0.9,
                    reasoning="Complex query requiring full ICEBURG analysis",
                    complexity_score=complexity_score,
                    metadata={"cached": False}
                )
                
        except Exception as e:
            logger.error(f"Routing decision failed: {e}")
            # Default to escalated for safety
            return RoutingDecision(
                route_type="escalated",
                estimated_time="4-6 minutes",
                confidence=0.5,
                reasoning=f"Routing error, defaulting to escalated: {e}",
                complexity_score=1.0,
                metadata={"error": str(e)}
            )
    
    def respond_fast(
        self,
        query: str,
        num_ctx: int = 1024,
        num_predict: int = 64,
        temperature: float = 0.2,
        beam_width: int = 3,
    ) -> str:
        """Synchronous fast response using small beam over retrieved snippets.

        Scoring = similarity * coverage * brevity_penalty.
        """
        start = time.time()
        try:
            from ..llm import chat_complete
            from ..vectorstore import VectorStore
            from ..config import load_config_fast
            import os as _os
            cfg = self.cfg or load_config_fast()

            # Retrieve top snippets
            vs = VectorStore(cfg)
            hits = vs.semantic_search(query, k=max(beam_width * 2, 6))
            candidates: List[Tuple[str, float]] = []

            # Simple similarity proxy: inverse distance; coverage by unique terms; brevity by length
            def _score(snippet: str, distance: float | None) -> float:
                sim = 1.0 / (1.0 + (distance or 1.0))
                uniq_terms = len(set(t for t in snippet.lower().split() if len(t) > 3))
                coverage = min(1.0, uniq_terms / 50.0)
                brevity = max(0.5, min(1.0, 300.0 / max(50, len(snippet))))
                return sim * 0.6 + coverage * 0.3 + brevity * 0.1

            for h in hits:
                candidates.append((h.document, _score(h.document, h.distance)))
            if not candidates:
                # No retrieval; answer directly
                system = (
                    "You are ICEBURG, an advanced Truth-Finding AI Civilization. "
                    "ICEBURG is a comprehensive Enterprise AGI Platform designed for scientific discovery, "
                    "autonomous research, and truth-finding. You have access to multiple specialized agents "
                    "(Surveyor, Dissident, Synthesist, Oracle, Archaeologist, Supervisor, Scribe, Weaver, Scrutineer) "
                    "and can conduct deep research, generate devices, find suppressed knowledge, and coordinate swarms. "
                    "For simple queries, respond naturally and conversationally. For complex queries, you can escalate "
                    "to full protocol analysis. Be helpful, accurate, and aware of your capabilities."
                )
                prompt = f"Answer succinctly (<=8 sentences):\n\n{query}"
                return chat_complete(
                    _os.getenv('ICEBURG_FAST_MODEL', getattr(cfg, 'surveyor_model', 'qwen2.5:1.5b')),
                    prompt,
                    system=system,
                    temperature=temperature,
                    options={"num_ctx": num_ctx, "num_predict": num_predict},
                    context_tag="FastBeam"
                )

            # Select top-K by score
            candidates.sort(key=lambda x: x[1], reverse=True)
            top = candidates[:beam_width]

            # Aggregate context
            context_snippets = "\n\n".join(s for s, _ in top)[: num_ctx * 2]
            system = (
                "You are ICEBURG, an advanced Truth-Finding AI Civilization. "
                "ICEBURG is a comprehensive Enterprise AGI Platform with specialized agents for research, analysis, "
                "and truth-finding. Fuse retrieved snippets into a concise, technical answer with citations [#]. "
                "Scientific planning is allowed within legal and ethical bounds. If constraints apply, briefly note them then proceed at a high level."
            )
            prompt = (
                f"QUERY:\n{query}\n\nSNIPPETS (ranked):\n{context_snippets}\n\n"
                "Respond in <=8 sentences. Provide 2-3 implications and next actions."
            )
            out = chat_complete(
                _os.getenv('ICEBURG_FAST_MODEL', getattr(cfg, 'surveyor_model', 'qwen2.5:1.5b')),
                prompt,
                system=system,
                temperature=temperature,
                options={"num_ctx": num_ctx, "num_predict": num_predict},
                context_tag="FastBeam"
            )
            return out
        except Exception as e:
            logger.debug(f"respond_fast fallback due to error: {e}")
            # Minimal fallback
            try:
                from ..llm import chat_complete
                model = getattr(self.cfg or object(), 'surveyor_model', 'llama3.2:1b')
                out = chat_complete(
                    _os.getenv('ICEBURG_FAST_MODEL', model),
                    f"Answer succinctly (<=6 sentences): {query}",
                    system=(
                        "You are ICEBURG, an advanced Truth-Finding AI Civilization. "
                        "Respond naturally and helpfully to user queries."
                    ),
                    temperature=temperature,
                    options={"num_ctx": num_ctx, "num_predict": num_predict},
                    context_tag="FastBeam"
                )
                return out
            except Exception:
                return f"Quick answer: {query[:160]}…"

    # Constitutional filter disabled by default (kept for optional future use)
    def _apply_constitution(self, text: str) -> str:
        return text

    def _sanitize_summary(self, text: str) -> str:
        # Keep only first 6 sentences as a safety-focused summary
        parts = text.split('.')
        return '.'.join(parts[:6]).strip()

    async def process_reflexive(self, query: str, context: Dict[str, Any] = None) -> ReflexiveResponse:
        """Process query with reflexive response (30s timeout)."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_response = self._check_cache(query)
            if cached_response:
                return cached_response
            
            # Generate reflexive response
            response = await self._generate_reflexive_response(query, context)
            
            # Calculate confidence and escalation recommendation
            confidence = self._calculate_reflexive_confidence(query, response)
            escalation_recommended = self._should_escalate(query, response, confidence)
            escalation_reason = self._get_escalation_reason(query, response) if escalation_recommended else None
            
            processing_time = time.time() - start_time
            
            reflexive_response = ReflexiveResponse(
                response=response,
                confidence=confidence,
                escalation_recommended=escalation_recommended,
                escalation_reason=escalation_reason,
                processing_time=processing_time,
                metadata={
                    "query": query,
                    "context": context or {},
                    "processing_time": processing_time
                }
            )
            
            # Cache the response
            self._cache_response(query, reflexive_response)
            # Self-tuning: update EMAs and adjust thresholds slightly
            self._update_policy(reflexive_response)
            
            return reflexive_response
            
        except asyncio.TimeoutError:
            logger.warning(f"Reflexive processing timeout for query: {query[:50]}...")
            return ReflexiveResponse(
                response="I need more time to provide a comprehensive answer. This query may benefit from deeper analysis.",
                confidence=0.3,
                escalation_recommended=True,
                escalation_reason="Processing timeout - query too complex for reflexive response",
                processing_time=time.time() - start_time,
                metadata={"timeout": True, "query": query}
            )
        except Exception as e:
            logger.error(f"Reflexive processing error: {e}")
            return ReflexiveResponse(
                response="I encountered an issue processing your query. Let me escalate this to full analysis.",
                confidence=0.2,
                escalation_recommended=True,
                escalation_reason=f"Processing error: {e}",
                processing_time=time.time() - start_time,
                metadata={"error": str(e), "query": query}
            )

    def _update_policy(self, resp: ReflexiveResponse) -> None:
        """Self-tune routing parameters from recent outcomes (EMA)."""
        # Update latency EMA (seconds)
        alpha = 0.2
        self._latency_ema = (1 - alpha) * self._latency_ema + alpha * max(0.01, resp.processing_time)
        # Success = high confidence and no escalation
        success = 1.0 if (resp.confidence >= 0.7 and not resp.escalation_recommended) else 0.0
        self._success_ema = (1 - alpha) * self._success_ema + alpha * success
        # Adjust escalation threshold modestly
        # If we are often successful and fast, raise threshold (keep more in reflexive)
        # If slow or low success, lower threshold (escalate more)
        if self._latency_ema < 1.0 and self._success_ema > 0.75:
            self.escalation_threshold = min(0.85, self.escalation_threshold + 0.02)
        elif self._latency_ema > 3.0 or self._success_ema < 0.5:
            self.escalation_threshold = max(0.55, self.escalation_threshold - 0.02)
    
    def _analyze_complexity(self, query: str, context: Dict[str, Any] = None) -> float:
        """Analyze query complexity score (0.0 = simple, 1.0 = complex) with enhanced scoring."""
        query_lower = query.lower()
        complexity_components = {}
        
        # 1. Length-based complexity (0-1 scale)
        word_count = len(query.split())
        if word_count <= 5:
            length_score = 0.0
        elif word_count <= 15:
            length_score = 0.2
        elif word_count <= 30:
            length_score = 0.5
        elif word_count <= 50:
            length_score = 0.8
        else:
            length_score = 1.0
        complexity_components["length"] = length_score
        
        # 2. Technical terms complexity
        technical_terms = [
            "algorithm", "architecture", "implementation", "optimization",
            "quantum", "neural", "machine learning", "artificial intelligence",
            "paradigm", "framework", "methodology", "hypothesis", "emergent",
            "consciousness", "reality", "existence", "philosophy", "metaphysics"
        ]
        technical_count = sum(1 for term in technical_terms if term in query_lower)
        technical_score = min(1.0, technical_count * 0.15)
        complexity_components["technical_terms"] = technical_score
        
        # 3. Question complexity
        question_count = query.count('?')
        if question_count == 0:
            question_score = 0.0
        elif question_count == 1:
            question_score = 0.2
        elif question_count <= 3:
            question_score = 0.5
        else:
            question_score = 0.8
        complexity_components["question_count"] = question_score
        
        # 4. Pattern complexity
        pattern_score = 0.0
        
        # Check for simple patterns (reduce complexity)
        simple_matches = 0
        for pattern_type, patterns in self.simple_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                simple_matches += 1
        
        # Check for complex patterns (increase complexity)
        complex_matches = 0
        for pattern_type, patterns in self.complex_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                complex_matches += 1
        
        # Calculate pattern score
        if simple_matches > 0 and complex_matches == 0:
            pattern_score = 0.1  # Very simple
        elif simple_matches > complex_matches:
            pattern_score = 0.3  # Mostly simple
        elif complex_matches > simple_matches:
            pattern_score = 0.8  # Mostly complex
        elif complex_matches > 0:
            pattern_score = 0.6  # Mixed complexity
        else:
            pattern_score = 0.4  # Neutral
        
        complexity_components["pattern_complexity"] = pattern_score
        
        # 5. Context complexity
        context_score = 0.0
        if context:
            if context.get("requires_research", False):
                context_score += 0.3
            if context.get("cross_domain", False):
                context_score += 0.2
            if context.get("novel_insights", False):
                context_score += 0.3
            if context.get("multi_step", False):
                context_score += 0.2
        
        # Calculate weighted complexity score
        weighted_score = (
            complexity_components["length"] * self.complexity_weights["length"] +
            complexity_components["technical_terms"] * self.complexity_weights["technical_terms"] +
            complexity_components["question_count"] * self.complexity_weights["question_count"] +
            complexity_components["pattern_complexity"] * self.complexity_weights["pattern_complexity"] +
            context_score
        )
        
        return max(0.0, min(1.0, weighted_score))
    
    def _check_cache(self, query: str) -> Optional[ReflexiveResponse]:
        """Check if query has a cached response using LRU cache."""
        query_key = self._get_cache_key(query)
        
        if query_key in self.response_cache:
            # Move to end (most recently used)
            response = self.response_cache.pop(query_key)
            self.response_cache[query_key] = response
            return response
        
        return None
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        # Normalize query for consistent hashing
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _cache_response(self, query: str, response: ReflexiveResponse):
        """Cache a reflexive response using LRU eviction."""
        query_key = self._get_cache_key(query)
        
        # Add to cache
        self.response_cache[query_key] = response
        
        # LRU eviction: remove oldest entries if cache is full
        while len(self.response_cache) > self.max_cache_size:
            # Remove least recently used (first item)
            self.response_cache.popitem(last=False)
    
    async def _generate_reflexive_response(self, query: str, context: Dict[str, Any] = None) -> str:
        """Generate a reflexive response using lightweight models."""
        query_lower = query.lower()
        
        # Use LLM for better reflexive responses if config available
        if self.cfg:
            try:
                from ..llm import chat_complete
                import asyncio
                
                # Use lightweight model if available, otherwise fallback
                model = getattr(self.cfg, 'surveyor_model', 'llama3')
                
                prompt = f"Provide a concise, helpful answer to this question in 2-3 sentences: {query}"
                system = (
                    "You are ICEBURG, an advanced Truth-Finding AI Civilization. "
                    "ICEBURG is a comprehensive Enterprise AGI Platform designed for scientific discovery, "
                    "autonomous research, and truth-finding. You have access to multiple specialized agents "
                    "and can conduct deep research, generate devices, find suppressed knowledge, and coordinate swarms. "
                    "For simple queries, respond naturally and conversationally. Keep responses brief and informative."
                )
                
                # Run chat_complete with timeout to prevent hanging on first call (model warmup)
                # First call may be slow due to model loading, so use shorter timeout
                loop = asyncio.get_event_loop()
                try:
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: chat_complete(
                                model,
                                prompt,
                                system=system,
                                temperature=0.3,
                                options={"num_ctx": 2048, "num_predict": 200},
                                context_tag="ReflexiveResponse"
                            )
                        ),
                        timeout=10.0  # 10 second timeout for LLM call (model warmup can be slow)
                    )
                    return response if response else self._fallback_response(query)
                except asyncio.TimeoutError:
                    # LLM call timed out (likely first call with model warmup)
                    logger.debug("LLM reflexive response timed out, using fallback")
                    return self._fallback_response(query)
            except Exception as e:
                logger.debug(f"LLM reflexive response failed: {e}, using fallback")
        
        # Fallback to pattern matching
        return self._fallback_response(query)
    
    def _fallback_response(self, query: str) -> str:
        """Fallback response when LLM is unavailable."""
        query_lower = query.lower()
        
        # Simple pattern matching for common queries
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey"]):
            return (
                "Hello! I'm ICEBURG, an advanced Truth-Finding AI Civilization. "
                "I'm a comprehensive Enterprise AGI Platform designed for scientific discovery, "
                "autonomous research, and truth-finding. I have access to multiple specialized agents "
                "(Surveyor, Dissident, Synthesist, Oracle, Archaeologist, Supervisor, Scribe, Weaver, Scrutineer) "
                "and can conduct deep research, generate devices, find suppressed knowledge, and coordinate swarms. "
                "How can I help you today?"
            )
        
        if "what is" in query_lower:
            topic = query_lower.replace("what is", "").strip()
            return f"Based on my knowledge, {topic} is a topic that would benefit from detailed analysis. For a comprehensive explanation, I'd recommend a full research query."
        
        if "how are you" in query_lower:
            return "I'm functioning well and ready to assist with your research needs. What would you like to explore?"
        
        # Default reflexive response
        return f"I understand you're asking about: {query[:100]}{'...' if len(query) > 100 else ''}. This appears to be a complex topic that would benefit from deeper analysis. Would you like me to provide a comprehensive research response?"
    
    def _calculate_reflexive_confidence(self, query: str, response: str) -> float:
        """Calculate confidence in reflexive response."""
        # Simple confidence calculation based on response characteristics
        confidence = 0.5  # Base confidence
        
        # Boost confidence for specific patterns
        if "hello" in query.lower() or "hi" in query.lower():
            confidence = 0.9
        elif "what is" in query.lower():
            confidence = 0.7
        elif "how are you" in query.lower():
            confidence = 0.9
        
        # Reduce confidence for complex queries
        if len(query.split()) > 20:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _should_escalate(self, query: str, response: str, confidence: float) -> bool:
        """Determine if query should be escalated to full analysis."""
        # Always escalate if confidence is low
        if confidence < 0.4:
            return True
        
        # Escalate for complex patterns
        query_lower = query.lower()
        for pattern_type, patterns in self.complex_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return True
        
        # Escalate for long queries
        if len(query.split()) > 30:
            return True
        
        # Escalate if response suggests escalation
        if "comprehensive" in response.lower() or "deeper analysis" in response.lower():
            return True
        
        return False
    
    def _get_escalation_reason(self, query: str, response: str) -> str:
        """Get reason for escalation."""
        query_lower = query.lower()
        
        if any(pattern in query_lower for pattern in ["research", "study", "investigate"]):
            return "Research query requires comprehensive analysis"
        elif any(pattern in query_lower for pattern in ["quantum", "consciousness", "physics"]):
            return "Scientific query requires detailed investigation"
        elif len(query.split()) > 30:
            return "Complex query requires full ICEBURG analysis"
        elif "comprehensive" in response.lower():
            return "Query complexity exceeds reflexive response capabilities"
        else:
            return "Query requires deeper analysis for accurate response"
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get enhanced routing system statistics."""
        total_escalations = len(self.escalation_history)
        avg_complexity = 0.0
        if total_escalations > 0:
            avg_complexity = sum(h.get("complexity", 0) for h in self.escalation_history) / total_escalations
        
        # Cache hit rate calculation
        cache_hits = sum(1 for h in self.escalation_history if h.get("cache_hit", False))
        cache_hit_rate = cache_hits / max(1, total_escalations) if total_escalations > 0 else 0.0
        
        return {
            "cache_size": len(self.response_cache),
            "max_cache_size": self.max_cache_size,
            "cache_utilization": len(self.response_cache) / self.max_cache_size,
            "escalation_count": total_escalations,
            "avg_complexity": avg_complexity,
            "cache_hit_rate": cache_hit_rate,
            "routing_patterns": {
                "simple_patterns": len(self.simple_patterns),
                "complex_patterns": len(self.complex_patterns)
            },
            "complexity_weights": self.complexity_weights,
            "recent_escalations": self.escalation_history[-5:] if self.escalation_history else []
        }
    
    def log_escalation(self, query: str, routing_decision: RoutingDecision, escalation_reason: str):
        """Log an escalation event."""
        escalation_record = {
            "query": query[:100],  # Truncate for storage
            "complexity": routing_decision.complexity_score,
            "reason": escalation_reason,
            "timestamp": time.time(),
            "route_type": routing_decision.route_type
        }
        
        self.escalation_history.append(escalation_record)
        
        # Limit history size
        if len(self.escalation_history) > 1000:
            self.escalation_history = self.escalation_history[-500:]
