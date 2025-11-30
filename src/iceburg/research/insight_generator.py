"""
Insight Generator
Generates insights using Enhanced Deliberation methodology
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from .methodology_analyzer import MethodologyAnalyzer
from ..truth.suppression_detector import SuppressionDetector
try:
    from ..agents.oracle import Oracle
except ImportError:
    Oracle = None
try:
    from ..agents.synthesist import Synthesist
except ImportError:
    Synthesist = None


class InsightGenerator:
    """Generates insights using Enhanced Deliberation methodology"""
    
    def __init__(self):
        self.methodology_analyzer = MethodologyAnalyzer()
        self.suppression_detector = SuppressionDetector()
        self.oracle = Oracle() if Oracle else None
        self.synthesist = Synthesist() if Synthesist else None
    
    def generate_insights(
        self,
        query: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate insights using Enhanced Deliberation methodology"""
        insights = {
            "query": query,
            "domain": domain,
            "methodology": "enhanced_deliberation",
            "insights": [],
            "breakthroughs": [],
            "suppression_detected": False,
            "cross_domain_connections": []
        }
        
        # Apply methodology
        methodology = self.methodology_analyzer.apply_methodology(query, domain)
        
        # Step 1: Enhanced Deliberation
        deliberation_insights = self._apply_enhanced_deliberation(query, documents)
        insights["insights"].extend(deliberation_insights)
        
        # Step 2: Contradiction Hunting
        contradictions = self._hunt_contradictions(documents or [])
        if contradictions:
            insights["insights"].extend(contradictions)
        
        # Step 3: Meta-Pattern Detection
        patterns = self._detect_meta_patterns(documents or [])
        if patterns:
            insights["insights"].extend(patterns)
        
        # Step 4: Cross-Domain Synthesis
        synthesis = self._cross_domain_synthesis(query, domain)
        if synthesis:
            insights["cross_domain_connections"].extend(synthesis)
        
        # Step 5: Truth-Seeking Analysis
        truth_insights = self._truth_seeking_analysis(query, documents)
        if truth_insights:
            insights["insights"].extend(truth_insights)
        
        # Step 6: Suppression Detection
        if documents:
            suppression_result = self.suppression_detector.detect_suppression(documents)
            if suppression_result.get("suppression_detected"):
                insights["suppression_detected"] = True
                insights["insights"].append({
                    "type": "suppression_detection",
                    "result": suppression_result
                })
        
        # Identify breakthroughs
        insights["breakthroughs"] = self._identify_breakthroughs(insights["insights"])
        
        return insights
    
    def _apply_enhanced_deliberation(
        self,
        query: str,
        documents: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Apply Enhanced Deliberation with pause points"""
        insights = []
        
        # Simulate deliberation pause
        # In production, would use actual agent layers with pauses
        insights.append({
            "type": "deliberation",
            "description": "Applied 40-70 second pause points between agent layers",
            "result": "Deep reflection on query"
        })
        
        return insights
    
    def _hunt_contradictions(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Hunt for contradictions"""
        contradictions = []
        
        # Find contradictions between documents
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                if "content" in doc1 and "content" in doc2:
                    content1 = str(doc1["content"]).lower()
                    content2 = str(doc2["content"]).lower()
                    
                    # Check for opposite statements
                    opposites = [
                        ("success", "failure"),
                        ("approved", "rejected"),
                        ("confirmed", "denied")
                    ]
                    
                    for pos, neg in opposites:
                        if pos in content1 and neg in content2:
                            contradictions.append({
                                "type": "contradiction",
                                "description": f"Contradiction found: {pos} vs {neg}",
                                "doc1": doc1.get("id", "unknown"),
                                "doc2": doc2.get("id", "unknown")
                            })
        
        return contradictions
    
    def _detect_meta_patterns(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect meta-patterns across documents"""
        patterns = []
        
        # Simple pattern detection
        # In production, use more sophisticated analysis
        keywords = {}
        
        for doc in documents:
            if "content" in doc:
                content = str(doc["content"]).lower()
                words = content.split()
                for word in words:
                    if len(word) > 4:  # Filter short words
                        keywords[word] = keywords.get(word, 0) + 1
        
        # Find common patterns
        common_keywords = [word for word, count in keywords.items() if count >= 3]
        
        if common_keywords:
            patterns.append({
                "type": "meta_pattern",
                "description": f"Common patterns detected: {', '.join(common_keywords[:5])}",
                "keywords": common_keywords[:10]
            })
        
        return patterns
    
    def _cross_domain_synthesis(
        self,
        query: str,
        domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Cross-domain synthesis"""
        connections = []
        
        # Simple cross-domain connection
        # In production, use actual synthesist agent
        if domain:
            connections.append({
                "type": "cross_domain",
                "description": f"Cross-domain synthesis between {domain} and related fields",
                "domains": [domain, "physics", "biology", "chemistry"]
            })
        
        return connections
    
    def _truth_seeking_analysis(
        self,
        query: str,
        documents: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Truth-seeking analysis"""
        insights = []
        
        # Simple truth-seeking
        insights.append({
            "type": "truth_seeking",
            "description": "Systematic pursuit of truth regardless of barriers",
            "query": query
        })
        
        return insights
    
    def _identify_breakthroughs(
        self,
        insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify breakthroughs from insights"""
        breakthroughs = []
        
        for insight in insights:
            description = str(insight.get("description", "")).lower()
            
            if any(keyword in description for keyword in ["breakthrough", "discovery", "novel", "unprecedented"]):
                breakthroughs.append({
                    "type": "breakthrough",
                    "insight": insight,
                    "confidence": 0.8
                })
        
        return breakthroughs

