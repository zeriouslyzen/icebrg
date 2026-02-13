"""
ICEBURG Prompt Interpreter Agent - CIM Layer 0
Handles advanced prompt interpretation with linguistics, etymology, and intent extraction
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Optional
import time

from ..llm import chat_complete
from ..config import IceburgConfig
from .word_breakdown import WordBreakdownAnalyzer
from .prompt_interpreter_engine import PromptInterpreterEngine


class PromptInterpreter:
    """Advanced prompt interpreter with linguistics, etymology, and intent extraction"""

    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.word_analyzer = WordBreakdownAnalyzer()
        # Initialize persistent caching engine
        self.engine = PromptInterpreterEngine(cfg)

    async def run(self, query: str, verbose: bool = False, stream_breakdown: Optional[callable] = None) -> Dict[str, Any]:
        """Run advanced prompt interpreter with LLM-powered analysis"""
        return await run(self.cfg, query, verbose=verbose, stream_breakdown=stream_breakdown)


async def run(cfg: IceburgConfig, query: str, verbose: bool = False, stream_breakdown: Optional[callable] = None) -> Dict[str, Any]:
    """
    Advanced prompt interpreter with LLM-powered linguistics, etymology, and intent analysis.
    
    Returns structured analysis with:
    - Etymology: Word origins and linguistic roots
    - Intent: Primary intent and sub-intents
    - Domain: Primary and secondary domains
    - Complexity: Query complexity score (0-1)
    - Semantics: Deep meaning extraction
    - Word Breakdown: Real-time morphological, etymological, and semantic analysis
    
    Args:
        cfg: ICEBURG configuration
        query: Query string to analyze
        verbose: Enable verbose output
        stream_breakdown: Optional callback function to stream word breakdown results
    """
    if verbose:
        print(f"[PROMPT_INTERPRETER] Analyzing query with linguistics/etymology: {query[:60]}...")
    
    # Fast complexity detection for simple queries
    simple_patterns = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
    query_lower = query.lower().strip()
    
    # Check if query is simple (skip deep analysis)
    # BUT still show word breakdown for visualization (user wants to see it)
    is_simple = any(pattern in query_lower for pattern in simple_patterns) and len(query.split()) <= 3
    
    if is_simple:
        if verbose:
            print("[PROMPT_INTERPRETER] Simple query detected - using fast path but still showing word breakdown")
    
    # Check if we have a cached query analysis
    engine = PromptInterpreterEngine(cfg)
    cached_analysis = engine.get_cached_query(query)
    
    if cached_analysis and not verbose:
        # Return cached analysis immediately (very fast)
        # Add cache status to response
        cached_analysis['cache_status'] = {
            'query_cached': True,
            'cache_hit': True,
            'response_time': 'instant',
            'cache_source': 'shared_across_users'
        }
        if verbose:
            print(f"[PROMPT_INTERPRETER] Using cached analysis for query (instant)")
        return cached_analysis
    
    # Track cache status for new analysis
    cache_status = {
        'query_cached': False,
        'cache_hit': False,
        'response_time': 'analyzing',
        'cache_source': 'new_analysis'
    }
    
    # Perform word breakdown analysis using engine (with caching)
    word_breakdowns = []
    algorithm_steps = []
    
    # Stream word breakdown if callback provided
    if stream_breakdown:
        import asyncio
        
        # Visualize algorithm pipeline FIRST (so it appears first)
        word_analyzer = WordBreakdownAnalyzer()
        algorithm_steps = word_analyzer.visualize_algorithm_pipeline(query)
        for step in algorithm_steps:
            # Check if callback is async
            if asyncio.iscoroutinefunction(stream_breakdown):
                await stream_breakdown({
                    "type": "algorithm_step",
                    "step": step.step_name,
                    "status": step.status,
                    "processing_time": step.processing_time,
                    "input_size": len(str(step.input_data)),
                    "output_size": len(str(step.output_data))
                })
            else:
                stream_breakdown({
                    "type": "algorithm_step",
                    "step": step.step_name,
                    "status": step.status,
                    "processing_time": step.processing_time,
                    "input_size": len(str(step.input_data)),
                    "output_size": len(str(step.output_data))
                })
            # Small delay between steps for better visualization
            await asyncio.sleep(0.1)
        
        # Analyze words one by one using engine (with caching - fast retrieval)
        words = re.findall(r'\b\w+\b', query)
        for word in words:
            # Use engine's fast word analysis (checks cache first)
            breakdown = engine.analyze_word_fast(word)
            word_breakdowns.append(breakdown)
            
            # Stream breakdown result
            if asyncio.iscoroutinefunction(stream_breakdown):
                await stream_breakdown({
                    "type": "word_breakdown",
                    "word": word,
                    "morphological": breakdown.morphological,
                    "etymology": breakdown.etymology,
                    "semantic": breakdown.semantic,
                    "compression_hints": breakdown.compression_hints
                })
            else:
                stream_breakdown({
                    "type": "word_breakdown",
                    "word": word,
                    "morphological": breakdown.morphological,
                    "etymology": breakdown.etymology,
                    "semantic": breakdown.semantic,
                    "compression_hints": breakdown.compression_hints
                })
            # Small delay between words for better visualization
            await asyncio.sleep(0.15)
    else:
        # Perform analysis without streaming (using engine for caching)
        word_analyzer = WordBreakdownAnalyzer()
        words = re.findall(r'\b\w+\b', query)
        word_breakdowns = [engine.analyze_word_fast(word) for word in words]
        algorithm_steps = word_analyzer.visualize_algorithm_pipeline(query)
    
    # For simple queries, use fast path but still return word breakdown
    # (Word breakdown already done above, so just return fast result)
    if is_simple:
        if verbose:
            print("[PROMPT_INTERPRETER] Simple query detected - using fast path but still showing word breakdown")
        
        # Build fast path result with word breakdown (already computed above)
        result = {
            "status": "interpreted",
            "query": query,
            "intent_analysis": {"primary": "greeting", "confidence": 0.9},
            "domain_analysis": {"primary": "general"},
            "complexity_analysis": {"score": 0.1, "depth_level": "shallow"},
            "agent_routing": {"recommended_path": "simple"},
            "word_breakdown": [
                {
                    "word": b.word,
                    "morphological": b.morphological,
                    "etymology": b.etymology,
                    "semantic": b.semantic,
                    "compression_hints": b.compression_hints
                }
                for b in word_breakdowns
            ],
            "algorithm_pipeline": [
                {
                    "step_name": s.step_name,
                    "processing_time": s.processing_time,
                    "status": s.status
                }
                for s in algorithm_steps
            ],
            "fast_path": True
        }
        
        # Cache the simple query analysis
        engine.cache_query(query, result)
        
        # Add cache status to result
        result['cache_status'] = cache_status
        
        return result
    
    # Build comprehensive analysis prompt
    analysis_prompt = f"""
Perform comprehensive analysis of this query:

QUERY: "{query}"

Analyze:
1. ETYMOLOGY: Extract word origins, linguistic roots, and etymological connections
2. INTENT: Identify primary intent, sub-intents, and underlying goals
3. DOMAIN: Detect primary domain, secondary domains, and cross-domain connections
4. COMPLEXITY: Assess query complexity (0.0-1.0) based on:
   - Depth of reasoning required
   - Number of domains involved
   - Abstraction level
   - Evidence requirements
5. SEMANTICS: Extract deep meaning, relationships, and implicit concepts
6. ROUTING: Recommend optimal processing path (simple/standard/experimental)

Provide structured JSON response with:
{{
    "etymology": {{
        "key_terms": ["term1", "term2"],
        "word_origins": {{"term": "origin"}},
        "linguistic_roots": ["root1", "root2"],
        "etymological_connections": "analysis"
    }},
    "intent": {{
        "primary": "primary_intent",
        "sub_intents": ["sub1", "sub2"],
        "underlying_goals": ["goal1", "goal2"],
        "confidence": 0.0-1.0
    }},
    "domain": {{
        "primary": "domain_name",
        "secondary": ["domain1", "domain2"],
        "cross_domain": true/false,
        "field_specific": true/false
    }},
    "complexity": {{
        "score": 0.0-1.0,
        "depth_level": "shallow/medium/deep",
        "reasoning_required": "description",
        "abstraction_level": "concrete/abstract/theoretical"
    }},
    "semantics": {{
        "core_meaning": "core meaning extraction",
        "relationships": ["rel1", "rel2"],
        "implicit_concepts": ["concept1", "concept2"],
        "semantic_field": "description"
    }},
    "routing": {{
        "recommended_path": "simple/standard/experimental",
        "requires_molecular": true/false,
        "requires_bioelectric": true/false,
        "requires_hypothesis_testing": true/false,
        "reasoning": "why this routing"
    }}
}}
"""
    
    system_prompt = """You are a linguistic, etymology, and semantic analysis specialist with expertise in:
- Etymology: Word origins, linguistic roots, historical language development
- Linguistics: Morphology, phonetics, semantics, syntax
- Intent Extraction: Understanding underlying goals and motivations
- Domain Analysis: Identifying knowledge domains and cross-domain connections
- Complexity Assessment: Evaluating query depth and reasoning requirements

Analyze queries comprehensively, providing structured insights that enable optimal routing and processing."""
    
    try:
        # Use LLM for analysis
        response = chat_complete(
            cfg.surveyor_model,
            analysis_prompt,
            system=system_prompt,
            temperature=0.2,
            options={"num_ctx": 2048, "num_predict": 300},
            context_tag="PromptInterpreter"
        )
        
        # Parse JSON response
        result = _parse_analysis_response(response, query, verbose)
        
        # Add word breakdown to result
        result["word_breakdown"] = [
            {
                "word": b.word,
                "morphological": b.morphological,
                "etymology": b.etymology,
                "semantic": b.semantic,
                "compression_hints": b.compression_hints
            }
            for b in word_breakdowns
        ]
        result["algorithm_pipeline"] = [
            {
                "step_name": s.step_name,
                "processing_time": s.processing_time,
                "status": s.status
            }
            for s in algorithm_steps
        ]
        
        # Cache the comprehensive analysis
        engine.cache_query(query, result)
        
        # Add cache status to result
        result['cache_status'] = cache_status
        
        if verbose:
            print(f"[PROMPT_INTERPRETER] Analysis complete - Intent: {result.get('intent_analysis', {}).get('primary', 'unknown')}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"[PROMPT_INTERPRETER] Error in analysis: {e}")
        
        # Fallback to basic analysis
        result = _fallback_analysis(query)
        # Add word breakdown even in fallback
        word_analyzer = WordBreakdownAnalyzer()
        word_breakdowns = word_analyzer.analyze_query(query)
        result["word_breakdown"] = [
            {
                "word": b.word,
                "morphological": b.morphological,
                "etymology": b.etymology,
                "semantic": b.semantic,
                "compression_hints": b.compression_hints
            }
            for b in word_breakdowns
        ]
        return result


def _parse_analysis_response(response: str, query: str, verbose: bool = False) -> Dict[str, Any]:
    """Parse LLM response into structured format"""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "status": "interpreted",
                "query": query,
                "intent_analysis": parsed.get("intent", {}),
                "domain_analysis": parsed.get("domain", {}),
                "complexity_analysis": parsed.get("complexity", {}),
                "etymology_analysis": parsed.get("etymology", {}),
                "semantics_analysis": parsed.get("semantics", {}),
                "agent_routing": parsed.get("routing", {}),
                "requires_molecular": parsed.get("routing", {}).get("requires_molecular", False),
                "requires_bioelectric": parsed.get("routing", {}).get("requires_bioelectric", False),
                "requires_hypothesis_testing": parsed.get("routing", {}).get("requires_hypothesis_testing", False),
                "primary_domain": parsed.get("domain", {}).get("primary", "general"),
                "detail_level": parsed.get("complexity", {}).get("depth_level", "medium"),
                "cross_domain_relevance": parsed.get("domain", {}).get("cross_domain", False),
            }
    except json.JSONDecodeError:
        if verbose:
            print("[PROMPT_INTERPRETER] Could not parse JSON, using fallback")
    
    # If parsing fails, extract key information heuristically
    return _heuristic_analysis(response, query)


def _heuristic_analysis(response: str, query: str) -> Dict[str, Any]:
    """Heuristic fallback analysis"""
    # Simple keyword detection for domains
    query_lower = query.lower()
    domains = []
    if any(kw in query_lower for kw in ["molecule", "chemical", "protein", "enzyme", "compound"]):
        domains.append("chemistry")
        domains.append("biology")
    if any(kw in query_lower for kw in ["physics", "quantum", "energy", "field", "resonance"]):
        domains.append("physics")
    if any(kw in query_lower for kw in ["cancer", "disease", "health", "treatment", "therapy"]):
        domains.append("medicine")
        domains.append("biology")
    
    # Simple complexity estimate
    word_count = len(query.split())
    complexity = min(0.3 + (word_count / 100), 1.0)
    
    return {
        "status": "interpreted",
        "query": query,
        "intent_analysis": {
            "primary": "general",
            "confidence": 0.5
        },
        "domain_analysis": {
            "primary": domains[0] if domains else "general",
            "secondary": domains[1:] if len(domains) > 1 else []
        },
        "complexity_analysis": {
            "score": complexity,
            "depth_level": "medium" if complexity > 0.5 else "shallow"
        },
        "agent_routing": {
            "recommended_path": "standard",
            "requires_molecular": "molecule" in query_lower or "chemical" in query_lower,
            "requires_bioelectric": "bioelectric" in query_lower or "field" in query_lower,
            "requires_hypothesis_testing": "hypothesis" in query_lower or "test" in query_lower
        },
        "primary_domain": domains[0] if domains else "general",
        "detail_level": "medium",
        "cross_domain_relevance": len(domains) > 1,
    }


def _fallback_analysis(query: str) -> Dict[str, Any]:
    """Basic fallback when LLM analysis fails"""
    return {
        "status": "interpreted",
        "intent": "general",
        "query": query,
        "intent_analysis": {"primary": "general"},
        "domain_analysis": {"primary": "general"},
        "complexity_analysis": {"score": 0.5},
        "agent_routing": {"recommended_path": "standard"},
    }
