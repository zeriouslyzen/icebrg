"""
Hybrid Search
Combines keyword, semantic, and temporal search with relevance ranking
"""

from typing import Any, Dict, Optional, List
from .query_priming import QueryPriming
from .context_retrieval import ContextRetrieval
from ..vectorstore import VectorStore


class HybridSearch:
    """Hybrid search combining multiple retrieval methods"""
    
    def __init__(self, vectorstore: Optional[VectorStore] = None):
        self.vectorstore = vectorstore
        self.query_priming = QueryPriming()
        self.context_retrieval = ContextRetrieval(vectorstore)
    
    def search(
        self,
        query: str,
        k: int = 10,
        context: Optional[Dict[str, Any]] = None,
        suppression_aware: bool = True
    ) -> Dict[str, Any]:
        """Perform hybrid search with query priming"""
        search_result = {
            "query": query,
            "primed_query": None,
            "results": [],
            "relevance_scores": [],
            "suppression_indicators": [],
            "search_score": 0.0
        }
        
        # Prime query
        primed = self.query_priming.prime_query(query, context)
        search_result["primed_query"] = primed
        
        # Retrieve context
        retrieval = self.context_retrieval.retrieve_context(
            primed["expanded_query"],
            k=k,
            use_semantic=True,
            use_keyword=True,
            use_temporal=True
        )
        
        search_result["results"] = retrieval["combined_results"]
        search_result["relevance_scores"] = [
            r.get("combined_score", 0.0) for r in retrieval["combined_results"]
        ]
        
        # Check for suppression indicators
        if suppression_aware:
            search_result["suppression_indicators"] = self._detect_suppression_indicators(
                retrieval["combined_results"]
            )
        
        # Calculate overall search score
        search_result["search_score"] = (
            primed["priming_score"] * 0.3 +
            retrieval["retrieval_score"] * 0.7
        )
        
        return search_result
    
    def _detect_suppression_indicators(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect suppression indicators in results"""
        indicators = []
        
        suppression_keywords = [
            "classified", "confidential", "top secret", "restricted",
            "suppressed", "hidden", "concealed", "classified"
        ]
        
        for result in results:
            document = result.get("document", "").lower()
            metadata = result.get("metadata", {})
            
            # Check document content
            for keyword in suppression_keywords:
                if keyword in document:
                    indicators.append({
                        "type": "suppression_keyword",
                        "keyword": keyword,
                        "result_index": results.index(result),
                        "severity": "medium"
                    })
            
            # Check metadata
            for key in ["classification", "security_level", "access_level"]:
                if key in metadata:
                    level = str(metadata[key]).lower()
                    if any(kw in level for kw in suppression_keywords):
                        indicators.append({
                            "type": "suppression_metadata",
                            "key": key,
                            "value": metadata[key],
                            "result_index": results.index(result),
                            "severity": "high"
                        })
        
        return indicators
    
    def rank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        suppression_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """Rank results with suppression awareness"""
        ranked = []
        
        for result in results:
            score = result.get("combined_score", 0.0)
            
            # Boost suppression-related results
            if suppression_aware:
                document = result.get("document", "").lower()
                if any(kw in document for kw in ["suppressed", "hidden", "classified"]):
                    score *= 1.2  # Boost suppression-related results
            
            ranked.append({
                **result,
                "final_score": score
            })
        
        # Sort by final score
        ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        return ranked

