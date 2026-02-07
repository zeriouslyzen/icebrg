"""
Search-to-Answer Pipeline
Combines web search, hybrid ranking, and LLM answer generation with citations.
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SearchAnswerPipeline:
    """
    Full pipeline: Web Search → Hybrid Rerank → Answer Generation → Citations
    Perplexity-style answer engine.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize pipeline with web search and LLM.
        
        Args:
            llm_client: LLM client (e.g., Ollama client for dolphin-mistral)
        """
        from .web_search import get_web_search
        
        self.web_search = get_web_search()
        self.llm_client = llm_client
        logger.info("Search-to-Answer pipeline initialized")
    
    def generate_answer(
        self,
        query: str,
        search_for_web: bool = True,
        max_sources: int = 5
    ) -> Dict[str, Any]:
        """
        Full pipeline: search web → generate answer with citations.
        
        Args:
            query: User query
            search_for_web: Whether to search web (True) or use cached knowledge (False)
            max_sources: Maximum number of sources to cite
            
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Generating answer for: {query}")
        
        # Step 1: Web search if needed
        web_results = []
        if search_for_web:
            web_results = self.web_search.search_for_current_events(query)
            logger.info(f"Retrieved {len(web_results)} web results")
        
        # Step 2: Build context from search results
        context = self._build_context(web_results, max_sources)
        
        # Step 3: Generate answer with LLM
        answer = self._generate_llm_answer(query, context)
        
        # Step 4: Format sources
        sources = self._format_sources(web_results[:max_sources])
        
        return {
            'query': query,
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources),
            'timestamp': datetime.now().isoformat(),
            'search_performed': search_for_web
        }
    
    def _build_context(self, web_results: List, max_sources: int) -> str:
        """
        Build context string from web results for LLM.
        
        Args:
            web_results: List of WebSearchResult objects
            max_sources: Max sources to include
            
        Returns:
            Formatted context string
        """
        if not web_results:
            return "No web sources available."
        
        context_parts = []
        for i, result in enumerate(web_results[:max_sources], start=1):
            context_parts.append(
                f"[Source {i}] {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {result.snippet}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_llm_answer(self, query: str, context: str) -> str:
        """
        Generate answer using LLM with web context.
        
        Args:
            query: User query
            context: Web search context
            
        Returns:
            Generated answer with citations
        """
        # Uncensored, direct prompt (dolphin-mistral style)
        system_prompt = """You are a direct, unfiltered search assistant powered by real-time web data.

RULES:
1. Answer using ONLY the provided web sources
2. ALWAYS cite sources using [Source N] format
3. Be direct and factual - no hedging, no apologies
4. If sources conflict, mention both viewpoints
5. If sources don't answer the query, say so directly"""

        user_prompt = f"""WEB SOURCES:
{context}

USER QUERY: {query}

Provide a direct answer with inline citations [Source N]:"""

        # Generate with LLM
        if self.llm_client:
            try:
                # Use Ollama chat API format (more reliable than generate)
                response = self.llm_client.chat(
                    model="llama3.1:8b",  # Use available model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.3,  # Lower for factual answers
                        "num_predict": 500
                    }
                )
                return response.get('message', {}).get('content', 'Error generating answer')
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                return self._fallback_answer(context)
        else:
            # Fallback: simple extraction from sources
            logger.warning("No LLM client provided, using fallback extraction")
            return self._fallback_answer(context)
    
    def _fallback_answer(self, context: str) -> str:
        """
        Fallback answer when LLM unavailable - just return context snippets.
        
        Args:
            context: Formatted context string
            
        Returns:
            Simple answer from context
        """
        return f"Based on web sources:\n\n{context}"
    
    def _format_sources(self, web_results: List) -> List[Dict[str, str]]:
        """
        Format sources for display.
        
        Args:
            web_results: List of WebSearchResult objects
            
        Returns:
            List of formatted source dicts
        """
        sources = []
        for i, result in enumerate(web_results, start=1):
            sources.append({
                'number': i,
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet[:200] + '...' if len(result.snippet) > 200 else result.snippet,
                'source_type': result.source
            })
        
        return sources


def is_current_event_query(query: str) -> bool:
    """
    Detect if query requires current/real-time data.
    
    Args:
        query: User query
        
    Returns:
        True if query needs web search
    """
    keywords = [
        'today', 'now', 'current', 'currently', 'latest', 'recent', 'recently',
        'this week', 'this month', 'this year', '2024', '2025', '2026', '2027',
        'price', 'market', 'stock', 'crypto', 'bitcoin', 'ethereum',
        'news', 'breaking', 'update', 'happening', 'just',
        'weather', 'score', 'result', 'election',
        # Added: common current event topics
        'protest', 'protests', 'immigration', 'war', 'conflict', 'attack',
        'government', 'trump', 'biden', 'congress', 'senate',
        'ice', 'raid', 'arrests', 'deportation'
    ]
    
    query_lower = query.lower()
    
    # EXCEPTION: If asking about today's date or time specifically, let ContextService handle it
    # These are handled internally by the agent's runtime context
    internal_context_patterns = [
        "what is the date", "whats the date", "what's the date",
        "what day is it", "what is today", "what's today",
        "what time is it", "current time", "time now"
    ]
    
    if any(pattern in query_lower for pattern in internal_context_patterns):
        # Unless it asks for "news" or "weather" along with it
        if not any(k in query_lower for k in ['news', 'weather', 'happening', 'latest']):
            return False
    
    return any(keyword in query_lower for keyword in keywords)


# Example usage function
def answer_query(query: str, llm_client=None) -> Dict[str, Any]:
    """
    High-level function to answer queries with web search when needed.
    
    Args:
        query: User query
        llm_client: Optional LLM client
        
    Returns:
        Answer with sources
    """
    pipeline = SearchAnswerPipeline(llm_client)
    
    # Auto-detect if web search is needed
    needs_search = is_current_event_query(query)
    
    return pipeline.generate_answer(query, search_for_web=needs_search)
