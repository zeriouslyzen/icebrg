"""
Historical Analyzer
Analyzes past research to understand how ICEBURG generates insights
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime


class HistoricalAnalyzer:
    """Analyzes past research outputs"""
    
    def __init__(self):
        self.research_outputs_dir = Path("data/research_outputs")
        self.projects_dir = Path("data/projects")
        self.blockchain_dir = Path("data/blockchain_verification")
        self.research_history: List[Dict[str, Any]] = []
        self._load_research_history()
    
    def _load_research_history(self):
        """Load research history from storage"""
        # Load from research outputs
        if self.research_outputs_dir.exists():
            for research_file in self.research_outputs_dir.glob("*.md"):
                try:
                    with open(research_file, 'r') as f:
                        content = f.read()
                        self.research_history.append({
                            "file": str(research_file),
                            "content": content,
                            "type": "research_output"
                        })
                except Exception:
                    pass
        
        # Load from projects
        if self.projects_dir.exists():
            for project_dir in self.projects_dir.iterdir():
                if project_dir.is_dir():
                    for project_file in project_dir.glob("*.md"):
                        try:
                            with open(project_file, 'r') as f:
                                content = f.read()
                                self.research_history.append({
                                    "file": str(project_file),
                                    "content": content,
                                    "type": "project",
                                    "project": project_dir.name
                                })
                        except Exception:
                            pass
        
        # Load from blockchain
        if self.blockchain_dir.exists():
            blockchain_file = self.blockchain_dir / "blockchain_chain.json"
            if blockchain_file.exists():
                try:
                    with open(blockchain_file, 'r') as f:
                        blockchain_data = json.load(f)
                        self.research_history.append({
                            "file": str(blockchain_file),
                            "content": str(blockchain_data),
                            "type": "blockchain"
                        })
                except Exception:
                    pass
    
    def analyze_past_research(
        self,
        research_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze past research"""
        analysis = {
            "total_research": len(self.research_history),
            "research_by_type": {},
            "patterns_identified": [],
            "methodology_usage": {},
            "breakthroughs": [],
            "suppression_detected": []
        }
        
        # Group by type
        for research in self.research_history:
            research_type_item = research.get("type", "unknown")
            analysis["research_by_type"][research_type_item] = \
                analysis["research_by_type"].get(research_type_item, 0) + 1
        
        # Analyze patterns
        analysis["patterns_identified"] = self._identify_patterns()
        
        # Analyze methodology usage
        analysis["methodology_usage"] = self._analyze_methodology_usage()
        
        # Find breakthroughs
        analysis["breakthroughs"] = self._find_breakthroughs()
        
        # Find suppression detections
        analysis["suppression_detected"] = self._find_suppression_detections()
        
        return analysis
    
    def _identify_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in past research"""
        patterns = []
        
        # Simple pattern identification
        # In production, use more sophisticated analysis
        keywords = {}
        
        for research in self.research_history:
            content = str(research.get("content", "")).lower()
            words = content.split()
            for word in words:
                if len(word) > 5:  # Filter short words
                    keywords[word] = keywords.get(word, 0) + 1
        
        # Find common patterns
        common_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for word, count in common_keywords:
            patterns.append({
                "pattern": word,
                "frequency": count,
                "description": f"Common pattern: {word} appears {count} times"
            })
        
        return patterns
    
    def _analyze_methodology_usage(self) -> Dict[str, Any]:
        """Analyze methodology usage in past research"""
        methodology_usage = {
            "enhanced_deliberation": 0,
            "contradiction_hunting": 0,
            "meta_pattern_detection": 0,
            "cross_domain_synthesis": 0,
            "truth_seeking": 0,
            "suppression_detection": 0
        }
        
        for research in self.research_history:
            content = str(research.get("content", "")).lower()
            
            if "deliberation" in content:
                methodology_usage["enhanced_deliberation"] += 1
            
            if "contradiction" in content:
                methodology_usage["contradiction_hunting"] += 1
            
            if "pattern" in content or "meta" in content:
                methodology_usage["meta_pattern_detection"] += 1
            
            if "synthesis" in content or "cross-domain" in content:
                methodology_usage["cross_domain_synthesis"] += 1
            
            if "truth" in content or "seeking" in content:
                methodology_usage["truth_seeking"] += 1
            
            if "suppression" in content:
                methodology_usage["suppression_detection"] += 1
        
        return methodology_usage
    
    def _find_breakthroughs(self) -> List[Dict[str, Any]]:
        """Find breakthroughs in past research"""
        breakthroughs = []
        
        for research in self.research_history:
            content = str(research.get("content", ""))
            content_lower = content.lower()
            
            if any(keyword in content_lower for keyword in ["breakthrough", "discovery", "novel", "unprecedented"]):
                breakthroughs.append({
                    "file": research.get("file"),
                    "type": research.get("type"),
                    "description": "Breakthrough detected in research"
                })
        
        return breakthroughs
    
    def _find_suppression_detections(self) -> List[Dict[str, Any]]:
        """Find suppression detections in past research"""
        suppressions = []
        
        for research in self.research_history:
            content = str(research.get("content", "")).lower()
            
            if "suppression" in content or "suppressed" in content:
                suppressions.append({
                    "file": research.get("file"),
                    "type": research.get("type"),
                    "description": "Suppression detection found in research"
                })
        
        return suppressions
    
    def analyze_pancreatic_cancer_research(self) -> Dict[str, Any]:
        """Analyze pancreatic cancer research specifically"""
        pancreatic_research = [
            r for r in self.research_history
            if "pancreatic" in str(r.get("content", "")).lower() or
               "pancreatic" in str(r.get("file", "")).lower()
        ]
        
        analysis = {
            "research_count": len(pancreatic_research),
            "methodology_used": [],
            "breakthroughs": [],
            "suppression_detected": False
        }
        
        for research in pancreatic_research:
            content = str(research.get("content", "")).lower()
            
            # Check methodology
            if "deliberation" in content:
                analysis["methodology_used"].append("enhanced_deliberation")
            
            if "contradiction" in content:
                analysis["methodology_used"].append("contradiction_hunting")
            
            # Check for breakthroughs
            if "breakthrough" in content or "discovery" in content:
                analysis["breakthroughs"].append({
                    "file": research.get("file"),
                    "description": "Breakthrough in pancreatic cancer research"
                })
            
            # Check for suppression
            if "suppression" in content:
                analysis["suppression_detected"] = True
        
        return analysis
    
    def analyze_quantum_consciousness_research(self) -> Dict[str, Any]:
        """Analyze quantum consciousness research"""
        quantum_research = [
            r for r in self.research_history
            if "quantum" in str(r.get("content", "")).lower() and
               "consciousness" in str(r.get("content", "")).lower()
        ]
        
        analysis = {
            "research_count": len(quantum_research),
            "cross_domain_synthesis": False,
            "emergent_patterns": []
        }
        
        for research in quantum_research:
            content = str(research.get("content", "")).lower()
            
            if "synthesis" in content or "cross-domain" in content:
                analysis["cross_domain_synthesis"] = True
            
            if "emergent" in content or "pattern" in content:
                analysis["emergent_patterns"].append({
                    "file": research.get("file"),
                    "description": "Emergent pattern detected"
                })
        
        return analysis
    
    def get_research_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get research history"""
        return self.research_history[-limit:] if self.research_history else []

