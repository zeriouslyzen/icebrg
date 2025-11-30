"""
Context Retrieval
Multi-stage retrieval with keyword, semantic, and temporal search
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from ..vectorstore import VectorStore


class ContextRetrieval:
    """Multi-stage context retrieval"""
    
    def __init__(self, vectorstore: Optional[VectorStore] = None):
        self.vectorstore = vectorstore
        self.retrieval_history: List[Dict[str, Any]] = []
    
    def retrieve_context(
        self,
        query: str,
        k: int = 10,
        use_semantic: bool = True,
        use_keyword: bool = True,
        use_temporal: bool = True
    ) -> Dict[str, Any]:
        """Retrieve context using multi-stage retrieval"""
        context = {
            "query": query,
            "semantic_results": [],
            "keyword_results": [],
            "temporal_results": [],
            "combined_results": [],
            "retrieval_score": 0.0
        }
        
        # Stage 1: Semantic search
        if use_semantic and self.vectorstore:
            try:
                semantic_hits = self.vectorstore.semantic_search(query, k=k)
                context["semantic_results"] = [
                    {
                        "document": hit.document,
                        "metadata": hit.metadata,
                        "score": hit.score if hasattr(hit, "score") else 0.0
                    }
                    for hit in semantic_hits
                ]
            except Exception as e:
                pass
        
        # Stage 2: Keyword search
        if use_keyword and self.vectorstore:
            try:
                # Simple keyword matching
                keywords = query.lower().split()
                keyword_results = []
                
                # This would need to be implemented based on vectorstore capabilities
                # For now, we'll use semantic results as keyword results
                context["keyword_results"] = context["semantic_results"][:k//2]
            except Exception as e:
                pass
        
        # Stage 3: Temporal search
        if use_temporal:
            try:
                # Filter by temporal relevance
                temporal_results = self._filter_temporal(context["semantic_results"])
                context["temporal_results"] = temporal_results
            except Exception as e:
                pass
        
        # Combine results
        context["combined_results"] = self._combine_results(
            context["semantic_results"],
            context["keyword_results"],
            context["temporal_results"],
            k
        )
        
        # Calculate retrieval score
        context["retrieval_score"] = self._calculate_retrieval_score(context)
        
        # Record retrieval
        self.retrieval_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(context["combined_results"]),
            "score": context["retrieval_score"]
        })
        
        return context
    
    def _filter_temporal(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by temporal relevance"""
        temporal = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            # Check for timestamp
            for key in ["timestamp", "date", "created_at", "updated_at"]:
                if key in metadata:
                    temporal.append(result)
                    break
        
        # Sort by temporal relevance (newer first)
        temporal.sort(
            key=lambda x: self._extract_timestamp(x),
            reverse=True
        )
        
        return temporal
    
    def _extract_timestamp(self, result: Dict[str, Any]) -> float:
        """Extract timestamp from result"""
        metadata = result.get("metadata", {})
        
        for key in ["timestamp", "date", "created_at", "updated_at"]:
            if key in metadata:
                try:
                    value = metadata[key]
                    if isinstance(value, str):
                        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        return dt.timestamp()
                    elif isinstance(value, datetime):
                        return value.timestamp()
                except Exception:
                    pass
        
        return 0.0
    
    def _combine_results(
        self,
        semantic: List[Dict[str, Any]],
        keyword: List[Dict[str, Any]],
        temporal: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """Combine results from different stages"""
        combined = {}
        
        # Add semantic results with weight 0.5
        for result in semantic:
            doc_id = result.get("document", "")[:100]  # Use first 100 chars as ID
            if doc_id not in combined:
                combined[doc_id] = {
                    **result,
                    "combined_score": result.get("score", 0.0) * 0.5
                }
        
        # Add keyword results with weight 0.3
        for result in keyword:
            doc_id = result.get("document", "")[:100]
            if doc_id in combined:
                combined[doc_id]["combined_score"] += result.get("score", 0.0) * 0.3
            else:
                combined[doc_id] = {
                    **result,
                    "combined_score": result.get("score", 0.0) * 0.3
                }
        
        # Add temporal results with weight 0.2
        for result in temporal:
            doc_id = result.get("document", "")[:100]
            if doc_id in combined:
                combined[doc_id]["combined_score"] += 0.2
            else:
                combined[doc_id] = {
                    **result,
                    "combined_score": 0.2
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x.get("combined_score", 0.0),
            reverse=True
        )
        
        return sorted_results[:k]
    
    def _calculate_retrieval_score(self, context: Dict[str, Any]) -> float:
        """Calculate overall retrieval score"""
        score = 0.0
        
        # Score from semantic results
        if context["semantic_results"]:
            avg_semantic = sum(r.get("score", 0.0) for r in context["semantic_results"]) / len(context["semantic_results"])
            score += avg_semantic * 0.5
        
        # Score from combined results
        if context["combined_results"]:
            avg_combined = sum(r.get("combined_score", 0.0) for r in context["combined_results"]) / len(context["combined_results"])
            score += avg_combined * 0.5
        
        return min(1.0, score)
    
    def get_retrieval_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get retrieval history"""
        return self.retrieval_history[-limit:] if self.retrieval_history else []

