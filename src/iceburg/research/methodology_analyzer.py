"""
Methodology Analyzer
Analyzes Enhanced Deliberation methodology
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime


class MethodologyAnalyzer:
    """Analyzes research methodology"""
    
    def __init__(self):
        self.methodology_path = Path("data/projects/pancreatic-cancer-cure-enhanced-deliberation/ENHANCED_DELIBERATION_METHODOLOGY.md")
        self.methodology_components: Dict[str, Any] = {}
        self._load_methodology()
    
    def _load_methodology(self):
        """Load Enhanced Deliberation methodology"""
        if self.methodology_path.exists():
            try:
                with open(self.methodology_path, 'r') as f:
                    content = f.read()
                    self.methodology_components = self._parse_methodology(content)
            except Exception:
                self.methodology_components = self._get_default_methodology()
        else:
            self.methodology_components = self._get_default_methodology()
    
    def _parse_methodology(self, content: str) -> Dict[str, Any]:
        """Parse methodology from content"""
        components = {
            "enhanced_deliberation": {
                "pause_points": "40-70 seconds between agent layers",
                "description": "Deep reflection pauses between agent layers"
            },
            "contradiction_hunting": {
                "description": "Actively seek contradictions in narratives",
                "method": "Systematic contradiction detection"
            },
            "meta_pattern_detection": {
                "description": "Recognize patterns across multiple domains",
                "method": "Cross-domain pattern recognition"
            },
            "cross_domain_synthesis": {
                "description": "Connect unrelated fields through emergent patterns",
                "method": "Synthesis across domains"
            },
            "truth_seeking_analysis": {
                "description": "Systematic pursuit of truth regardless of barriers",
                "method": "Truth-seeking methodology"
            },
            "suppression_detection": {
                "description": "Identify systematic knowledge suppression",
                "method": "7-step suppression detection"
            }
        }
        
        return components
    
    def _get_default_methodology(self) -> Dict[str, Any]:
        """Get default methodology components"""
        return {
            "enhanced_deliberation": {
                "pause_points": "40-70 seconds between agent layers",
                "description": "Deep reflection pauses between agent layers"
            },
            "contradiction_hunting": {
                "description": "Actively seek contradictions in narratives",
                "method": "Systematic contradiction detection"
            },
            "meta_pattern_detection": {
                "description": "Recognize patterns across multiple domains",
                "method": "Cross-domain pattern recognition"
            },
            "cross_domain_synthesis": {
                "description": "Connect unrelated fields through emergent patterns",
                "method": "Synthesis across domains"
            },
            "truth_seeking_analysis": {
                "description": "Systematic pursuit of truth regardless of barriers",
                "method": "Truth-seeking methodology"
            },
            "suppression_detection": {
                "description": "Identify systematic knowledge suppression",
                "method": "7-step suppression detection"
            }
        }
    
    def analyze_methodology(
        self,
        research_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze how methodology was applied"""
        analysis = {
            "methodology_components_used": [],
            "effectiveness": {},
            "insights": []
        }
        
        # Check which components were used
        for component, details in self.methodology_components.items():
            if self._was_component_used(component, research_output):
                analysis["methodology_components_used"].append(component)
        
        # Analyze effectiveness
        analysis["effectiveness"] = self._analyze_effectiveness(research_output)
        
        # Extract insights
        analysis["insights"] = self._extract_insights(research_output)
        
        return analysis
    
    def _was_component_used(
        self,
        component: str,
        research_output: Dict[str, Any]
    ) -> bool:
        """Check if methodology component was used"""
        content = str(research_output).lower()
        
        component_keywords = {
            "enhanced_deliberation": ["deliberation", "pause", "reflection"],
            "contradiction_hunting": ["contradiction", "contradictory", "conflict"],
            "meta_pattern_detection": ["pattern", "meta", "cross-domain"],
            "cross_domain_synthesis": ["synthesis", "cross-domain", "emergent"],
            "truth_seeking_analysis": ["truth", "seeking", "analysis"],
            "suppression_detection": ["suppression", "suppressed", "hidden"]
        }
        
        keywords = component_keywords.get(component, [])
        return any(keyword in content for keyword in keywords)
    
    def _analyze_effectiveness(
        self,
        research_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze methodology effectiveness"""
        effectiveness = {
            "breakthrough_detected": False,
            "suppression_detected": False,
            "novel_insights": 0,
            "cross_domain_connections": 0
        }
        
        content = str(research_output).lower()
        
        if "breakthrough" in content or "discovery" in content:
            effectiveness["breakthrough_detected"] = True
        
        if "suppression" in content or "suppressed" in content:
            effectiveness["suppression_detected"] = True
        
        # Count novel insights
        if "novel" in content or "new" in content:
            effectiveness["novel_insights"] = content.count("novel") + content.count("new")
        
        # Count cross-domain connections
        if "cross-domain" in content or "synthesis" in content:
            effectiveness["cross_domain_connections"] = content.count("cross-domain") + content.count("synthesis")
        
        return effectiveness
    
    def _extract_insights(
        self,
        research_output: Dict[str, Any]
    ) -> List[str]:
        """Extract insights from research output"""
        insights = []
        
        # Simple insight extraction
        # In production, use more sophisticated NLP
        content = str(research_output)
        
        # Look for key phrases
        key_phrases = [
            "discovered",
            "found that",
            "revealed",
            "identified",
            "uncovered"
        ]
        
        for phrase in key_phrases:
            if phrase in content.lower():
                # Extract sentence containing phrase
                sentences = content.split('.')
                for sentence in sentences:
                    if phrase in sentence.lower():
                        insights.append(sentence.strip())
        
        return insights[:10]  # Limit to 10 insights
    
    def apply_methodology(
        self,
        query: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Apply Enhanced Deliberation methodology to query"""
        methodology_application = {
            "query": query,
            "domain": domain,
            "methodology": "enhanced_deliberation",
            "steps": []
        }
        
        # Step 1: Enhanced Deliberation
        methodology_application["steps"].append({
            "step": 1,
            "name": "Enhanced Deliberation",
            "action": "Apply 40-70 second pause points between agent layers",
            "duration": 60
        })
        
        # Step 2: Contradiction Hunting
        methodology_application["steps"].append({
            "step": 2,
            "name": "Contradiction Hunting",
            "action": "Actively seek contradictions in narratives",
            "duration": 30
        })
        
        # Step 3: Meta-Pattern Detection
        methodology_application["steps"].append({
            "step": 3,
            "name": "Meta-Pattern Detection",
            "action": "Recognize patterns across multiple domains",
            "duration": 45
        })
        
        # Step 4: Cross-Domain Synthesis
        methodology_application["steps"].append({
            "step": 4,
            "name": "Cross-Domain Synthesis",
            "action": "Connect unrelated fields through emergent patterns",
            "duration": 60
        })
        
        # Step 5: Truth-Seeking Analysis
        methodology_application["steps"].append({
            "step": 5,
            "name": "Truth-Seeking Analysis",
            "action": "Systematic pursuit of truth regardless of barriers",
            "duration": 45
        })
        
        # Step 6: Suppression Detection
        methodology_application["steps"].append({
            "step": 6,
            "name": "Suppression Detection",
            "action": "Identify systematic knowledge suppression",
            "duration": 30
        })
        
        return methodology_application
    
    def get_methodology_components(self) -> Dict[str, Any]:
        """Get methodology components"""
        return self.methodology_components

