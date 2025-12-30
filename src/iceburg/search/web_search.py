"""
Web Search API Clients
Integrates Brave Search, DuckDuckGo, and arXiv for real-time web data.
"""

from typing import List, Dict, Any, Optional
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """Web search result with rich metadata."""
    url: str
    title: str
    snippet: str
    published_date: Optional[datetime] = None
    source: str = "web"  # 'brave', 'ddg', 'arxiv', 'web'
    score: float = 1.0
    
    def to_search_result_dict(self) -> Dict[str, Any]:
        """Convert to SearchResult format for hybrid search."""
        return {
            'url': self.url,
            'title': self.title,
            'text': self.snippet,
            'timestamp': self.published_date
        }


class BraveSearchClient:
    """
    Brave Search API client for web search with privacy.
    Requires BRAVE_SEARCH_API_KEY environment variable.
    """
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        if not self.api_key:
            logger.warning("Brave Search API key not found - searches will fail")
    
    def search(self, query: str, count: int = 10) -> List[WebSearchResult]:
        """
        Search using Brave Search API.
        
        Args:
            query: Search query
            count: Number of results
            
        Returns:
            List of WebSearchResult objects
        """
        if not self.api_key:
            logger.warning("No Brave API key, skipping search")
            return []
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": count,
            "safesearch": "off",  # Uncensored as per user request
            "freshness": "24h"   # Recent results for current events
        }
        
        try:
            response = requests.get(self.BASE_URL, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('web', {}).get('results', []):
                results.append(WebSearchResult(
                    url=item.get('url', ''),
                    title=item.get('title', ''),
                    snippet=item.get('description', ''),
                    source='brave'
                ))
            
            logger.info(f"Brave Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Brave Search error: {e}")
            return []


class DuckDuckGoClient:
    """
    DuckDuckGo search client (free, no API key needed).
    Uses HTML scraping as fallback.
    """
    
    def search(self, query: str, count: int = 10) -> List[WebSearchResult]:
        """
        Search using DuckDuckGo HTML interface.
        
        Args:
            query: Search query
            count: Number of results (approximate)
            
        Returns:
            List of WebSearchResult objects
        """
        try:
            # DuckDuckGo HTML search
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            headers = {"User-Agent": "Mozilla/5.0"}
            
            response = requests.post(url, data=params, headers=headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            results = []
            
            # Parse results
            for result_div in soup.select('.result'):
                title_tag = result_div.select_one('.result__title')
                snippet_tag = result_div.select_one('.result__snippet')
                
                if title_tag and snippet_tag:
                    link = title_tag.select_one('a')
                    if link:
                        results.append(WebSearchResult(
                            url=link.get('href', ''),
                            title=title_tag.get_text(strip=True),
                            snippet=snippet_tag.get_text(strip=True),
                            source='ddg'
                        ))
                
                if len(results) >= count:
                    break
            
            logger.info(f"DuckDuckGo returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []


class ArXivClient:
    """
    arXiv API client for academic/research papers.
    """
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def search(self, query: str, max_results: int = 10) -> List[WebSearchResult]:
        """
        Search arXiv for research papers.
        
        Args:
            query: Search query
            max_results: Max number of results
            
        Returns:
            List of WebSearchResult objects
        """
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'xml')
            results = []
            
            for entry in soup.find_all('entry'):
                title = entry.find('title')
                summary = entry.find('summary')
                link = entry.find('id')
                published = entry.find('published')
                
                published_date = None
                if published:
                    try:
                        published_date = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
                    except:
                        pass
                
                if title and summary and link:
                    results.append(WebSearchResult(
                        url=link.text,
                        title=title.text.strip(),
                        snippet=summary.text.strip()[:300],
                        published_date=published_date,
                        source='arxiv'
                    ))
            
            logger.info(f"arXiv returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []


class WebSearchAggregator:
    """
    Aggregates results from multiple search APIs.
    Provides fallback and deduplication.
    """
    
    def __init__(self):
        self.brave = BraveSearchClient()
        self.ddg = DuckDuckGoClient()
        self.arxiv = ArXivClient()
    
    def search(
        self,
        query: str,
        sources: List[str] = None,
        max_results_per_source: int = 10
    ) -> List[WebSearchResult]:
        """
        Search across multiple sources and aggregate results.
        
        Args:
            query: Search query
            sources: List of sources to search ('brave', 'ddg', 'arxiv')
                    If None, uses all available
            max_results_per_source: Max results from each source
            
        Returns:
            Deduplicated list of WebSearchResult objects
        """
        if sources is None:
            sources = ['brave', 'ddg', 'arxiv']
        
        all_results = []
        
        # Search each source
        for source in sources:
            if source == 'brave':
                all_results.extend(self.brave.search(query, max_results_per_source))
            elif source == 'ddg':
                all_results.extend(self.ddg.search(query, max_results_per_source))
            elif source == 'arxiv':
                all_results.extend(self.arxiv.search(query, max_results_per_source))
        
        # Deduplicate by URL
        seen_urls = set()
        deduplicated = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                deduplicated.append(result)
        
        logger.info(f"Web search aggregated {len(deduplicated)} unique results from {len(sources)} sources")
        return deduplicated
    
    def search_for_current_events(self, query: str) -> List[WebSearchResult]:
        """
        Search optimized for current events (prioritizes Brave for freshness).
        
        Args:
            query: Query about current events
            
        Returns:
            Fresh web search results
        """
        # Try Brave first (has freshness filter)
        results = self.brave.search(query, count=15)
        
        # Fallback to DDG if Brave fails
        if len(results) < 5:
            logger.info("Brave returned few results, trying DuckDuckGo fallback")
            results.extend(self.ddg.search(query, count=10))
        
        return results[:20]  # Cap at 20 total


# Singleton
_web_search = None

def get_web_search() -> WebSearchAggregator:
    """Get or create singleton web search aggregator."""
    global _web_search
    if _web_search is None:
        _web_search = WebSearchAggregator()
    return _web_search
