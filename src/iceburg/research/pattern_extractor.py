"""
Pattern Extractor
Extracts patterns from past research
"""

from typing import Any, Dict, Optional, List
import re
from collections import Counter


class PatternExtractor:
    """Extracts patterns from research"""
    
    def __init__(self):
        self.patterns: Dict[str, List[Dict[str, Any]]] = {}
    
    def extract_patterns(
        self,
        research_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract patterns from research outputs"""
        patterns = {
            "research_patterns": [],
            "methodology_patterns": [],
            "breakthrough_patterns": [],
            "suppression_patterns": []
        }
        
        for research in research_outputs:
            content = str(research.get("content", ""))
            
            # Extract research patterns
            research_patterns = self._extract_research_patterns(content)
            patterns["research_patterns"].extend(research_patterns)
            
            # Extract methodology patterns
            methodology_patterns = self._extract_methodology_patterns(content)
            patterns["methodology_patterns"].extend(methodology_patterns)
            
            # Extract breakthrough patterns
            breakthrough_patterns = self._extract_breakthrough_patterns(content)
            patterns["breakthrough_patterns"].extend(breakthrough_patterns)
            
            # Extract suppression patterns
            suppression_patterns = self._extract_suppression_patterns(content)
            patterns["suppression_patterns"].extend(suppression_patterns)
        
        return patterns
    
    def _extract_research_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract research patterns"""
        patterns = []
        
        # Look for research patterns
        research_keywords = ["hypothesis", "experiment", "result", "conclusion"]
        
        for keyword in research_keywords:
            if keyword in content.lower():
                patterns.append({
                    "type": "research_pattern",
                    "keyword": keyword,
                    "description": f"Research pattern: {keyword}"
                })
        
        return patterns
    
    def _extract_methodology_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract methodology patterns"""
        patterns = []
        
        # Look for methodology patterns
        methodology_keywords = [
            "deliberation",
            "contradiction",
            "meta-pattern",
            "synthesis",
            "truth-seeking"
        ]
        
        for keyword in methodology_keywords:
            if keyword in content.lower():
                patterns.append({
                    "type": "methodology_pattern",
                    "keyword": keyword,
                    "description": f"Methodology pattern: {keyword}"
                })
        
        return patterns
    
    def _extract_breakthrough_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract breakthrough patterns"""
        patterns = []
        
        # Look for breakthrough indicators
        breakthrough_keywords = [
            "breakthrough",
            "discovery",
            "novel",
            "unprecedented",
            "revolutionary"
        ]
        
        for keyword in breakthrough_keywords:
            if keyword in content.lower():
                patterns.append({
                    "type": "breakthrough_pattern",
                    "keyword": keyword,
                    "description": f"Breakthrough pattern: {keyword}"
                })
        
        return patterns
    
    def _extract_suppression_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract suppression patterns"""
        patterns = []
        
        # Look for suppression indicators
        suppression_keywords = [
            "suppression",
            "suppressed",
            "hidden",
            "classified",
            "concealed"
        ]
        
        for keyword in suppression_keywords:
            if keyword in content.lower():
                patterns.append({
                    "type": "suppression_pattern",
                    "keyword": keyword,
                    "description": f"Suppression pattern: {keyword}"
                })
        
        return patterns
    
    def extract_physical_principles(
        self,
        research_outputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract physical principles from research"""
        principles = []
        
        # Look for physical principles
        principle_keywords = [
            "principle",
            "law",
            "theory",
            "hypothesis",
            "mechanism"
        ]
        
        for research in research_outputs:
            content = str(research.get("content", ""))
            
            for keyword in principle_keywords:
                if keyword in content.lower():
                    # Extract sentences containing principle
                    sentences = content.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            principles.append({
                                "principle": sentence.strip(),
                                "source": research.get("file", "unknown"),
                                "keyword": keyword
                            })
        
        return principles
    
    def extract_device_concepts(
        self,
        research_outputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract device concepts from research"""
        concepts = []
        
        # Look for device-related concepts
        device_keywords = [
            "device",
            "schematic",
            "circuit",
            "blueprint",
            "design",
            "prototype"
        ]
        
        for research in research_outputs:
            content = str(research.get("content", ""))
            
            for keyword in device_keywords:
                if keyword in content.lower():
                    # Extract sentences containing device concept
                    sentences = content.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            concepts.append({
                                "concept": sentence.strip(),
                                "source": research.get("file", "unknown"),
                                "keyword": keyword
                            })
        
        return concepts

