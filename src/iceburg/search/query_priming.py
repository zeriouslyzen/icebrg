"""
Query Priming
Context-aware query priming and expansion
"""

from typing import Any, Dict, Optional, List
import re


class QueryPriming:
    """Context-aware query priming"""
    
    def __init__(self):
        self.expansion_patterns = {
            "research": ["study", "investigation", "analysis", "research"],
            "suppression": ["hidden", "concealed", "classified", "suppressed"],
            "truth": ["fact", "reality", "evidence", "truth"],
            "discovery": ["finding", "breakthrough", "innovation", "discovery"]
        }
    
    def prime_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prime query with context-aware expansion"""
        primed = {
            "original_query": query,
            "expanded_query": query,
            "keywords": [],
            "context_terms": [],
            "related_queries": [],
            "priming_score": 0.0
        }
        
        # Extract keywords
        primed["keywords"] = self._extract_keywords(query)
        
        # Expand query
        primed["expanded_query"] = self._expand_query(query)
        
        # Add context terms
        if context:
            primed["context_terms"] = self._extract_context_terms(context)
            primed["expanded_query"] = f"{primed['expanded_query']} {' '.join(primed['context_terms'])}"
        
        # Generate related queries
        primed["related_queries"] = self._generate_related_queries(query)
        
        # Calculate priming score
        primed["priming_score"] = self._calculate_priming_score(primed)
        
        return primed
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "must", "can"}
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        expanded = query
        
        # Check for expansion patterns
        for pattern, terms in self.expansion_patterns.items():
            if pattern in query.lower():
                # Add related terms
                related = [t for t in terms if t not in query.lower()]
                if related:
                    expanded = f"{expanded} {' '.join(related[:2])}"
        
        return expanded
    
    def _extract_context_terms(self, context: Dict[str, Any]) -> List[str]:
        """Extract context terms from context"""
        terms = []
        
        # Extract from different context fields
        for key in ["domain", "topic", "category", "subject"]:
            if key in context:
                value = context[key]
                if isinstance(value, str):
                    terms.append(value)
                elif isinstance(value, list):
                    terms.extend(value)
        
        return terms
    
    def _generate_related_queries(self, query: str) -> List[str]:
        """Generate related queries"""
        related = []
        
        # Generate variations
        variations = [
            f"What is {query}?",
            f"Explain {query}",
            f"Analyze {query}",
            f"Research {query}",
            f"Investigate {query}"
        ]
        
        # Add variations that make sense
        query_lower = query.lower()
        if "how" in query_lower:
            related.append(query.replace("how", "what"))
        if "what" in query_lower:
            related.append(query.replace("what", "how"))
        if "why" in query_lower:
            related.append(query.replace("why", "what"))
        
        return related[:5]  # Limit to 5 related queries
    
    def _calculate_priming_score(self, primed: Dict[str, Any]) -> float:
        """Calculate priming score"""
        score = 0.0
        
        # Base score from keywords
        score += len(primed["keywords"]) * 0.1
        
        # Score from expansion
        if primed["expanded_query"] != primed["original_query"]:
            score += 0.2
        
        # Score from context
        score += len(primed["context_terms"]) * 0.15
        
        # Score from related queries
        score += len(primed["related_queries"]) * 0.1
        
        return min(1.0, score)
    
    def refine_query(self, query: str, feedback: Optional[Dict[str, Any]] = None) -> str:
        """Refine query based on feedback"""
        refined = query
        
        if feedback:
            # Add relevant terms from feedback
            if "relevant_terms" in feedback:
                terms = feedback["relevant_terms"]
                if terms:
                    refined = f"{refined} {' '.join(terms[:3])}"
            
            # Remove irrelevant terms
            if "irrelevant_terms" in feedback:
                for term in feedback["irrelevant_terms"]:
                    refined = refined.replace(term, "")
        
        return refined.strip()

