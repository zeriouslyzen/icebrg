from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import json
import uuid
from datetime import datetime


class EmergenceDetector:
    def __init__(self):
        self.emergence_patterns = [
            "cross_domain_synthesis",
            "assumption_challenge", 
            "novel_prediction",
            "framework_departure",
            "evidence_conflict",
            "emergent_property",
            "paradigm_shift"
        ]
    
    def process(self, oracle_output: str, claims: List[dict], evidence_level: str) -> Optional[Dict[str, Any]]:
        """Process existing Oracle output without modifying it"""
        
        # Parse the existing Oracle JSON
        try:
            data = json.loads(oracle_output)
        except:
            return None
        
        # Extract emergence signals from existing data
        emergence_score = self.calculate_emergence_score(data, claims, evidence_level)
        
        if emergence_score > 0.6:  # Threshold for "interesting" emergence
            return self.compress_intelligence(data, claims, evidence_level, emergence_score)
        
        return None
    
    def calculate_emergence_score(self, data: Dict[str, Any], claims: List[dict], evidence_level: str) -> float:
        score = 0.0
        
        # Cross-domain synthesis (existing data)
        domains = data.get("domains", [])
        if len(domains) > 1:
            score += 0.3
        
        # Novel predictions (existing data)
        predictions = data.get("predictions", [])
        if predictions:
            score += 0.2
        
        # Evidence gaps (existing data)
        if evidence_level == "C":  # Hypothesis level
            score += 0.2
        
        # Assumption challenges (from claims)
        if any("assumption" in str(c).lower() for c in claims):
            score += 0.3
        
        # Emergent properties
        principle_name = data.get("principle_name", "").lower()
        if "emergent" in principle_name or "emergence" in principle_name:
            score += 0.3
        
        # Framework departures
        if "alternative" in str(data).lower() or "paradigm" in str(data).lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def compress_intelligence(self, data: Dict[str, Any], claims: List[dict], evidence_level: str, emergence_score: float) -> Dict[str, Any]:
        """Compress complex analysis into actionable intelligence"""
        
        # Extract core emergence
        core_principle = data.get("one_sentence_summary", "")
        principle_name = data.get("principle_name", "")
        

        
        # Extract key predictions
        predictions = data.get("predictions", [])
        key_predictions = []
        for p in predictions[:2]:  # Take first 2 predictions
            if isinstance(p, dict):
                # Handle different prediction formats
                prediction_text = p.get("prediction_text") or p.get("prediction") or str(p)
                key_predictions.append(prediction_text)
            else:
                key_predictions.append(str(p))
        
        # Extract evidence gaps - handle different framing formats
        evidence_gaps = evidence_level
        if "framing" in data:
            if isinstance(data["framing"], dict):
                evidence_gaps = data["framing"].get("primary_evidence", evidence_level)
            else:
                # Handle string format like "Primary Evidence: [C] => 'Hypothesis'"
                framing_str = str(data["framing"])
                if "[" in framing_str and "]" in framing_str:
                    # Extract evidence level from [X] format
                    start = framing_str.find("[") + 1
                    end = framing_str.find("]")
                    if start > 0 and end > start:
                        evidence_gaps = framing_str[start:end]
                else:
                    evidence_gaps = framing_str
        
        # Classify emergence type
        emergence_type = self.classify_emergence(data, claims, evidence_level)
        
        # Assess actionability
        actionability = self.assess_actionability(data, emergence_score)
        
        # Compress into intelligence format
        compressed_intel = {
            "insight_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "emergence_type": emergence_type,
            "emergence_score": emergence_score,
            "core_principle": core_principle,
            "principle_name": principle_name,
            "key_predictions": key_predictions,
            "evidence_gaps": evidence_gaps,
            "domains": data.get("domains", []),
            "confidence_score": self.calculate_confidence(data, emergence_score),
            "actionability": actionability,
            "patterns": self.extract_patterns(data, claims)
        }
        
        return compressed_intel
    
    def classify_emergence(self, data: Dict[str, Any], claims: List[dict], evidence_level: str) -> str:
        """Classify the type of emergence detected"""
        
        domains = data.get("domains", [])
        principle_name = str(data.get("principle_name", "")).lower()
        
        if len(domains) > 1:
            return "cross_domain_synthesis"
        elif "emergent" in principle_name or "emergence" in principle_name:
            return "emergent_property"
        elif evidence_level == "C":
            return "novel_hypothesis"
        elif any("assumption" in str(c).lower() for c in claims):
            return "assumption_challenge"
        else:
            return "pattern_discovery"
    
    def assess_actionability(self, data: Dict[str, Any], emergence_score: float) -> str:
        """Assess how actionable this emergence insight is"""
        
        if emergence_score > 0.8:
            return "high"
        elif emergence_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def calculate_confidence(self, data: Dict[str, Any], emergence_score: float) -> float:
        """Calculate confidence in the emergence detection"""
        
        # Base confidence on emergence score
        base_confidence = emergence_score
        
        # Adjust based on evidence quality
        if "study_design" in data:
            base_confidence += 0.1
        
        if "predictions" in data and data["predictions"]:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def extract_patterns(self, data: Dict[str, Any], claims: List[dict]) -> List[str]:
        """Extract key patterns from the emergence"""
        
        patterns = []
        
        # Domain patterns
        domains = data.get("domains", [])
        if len(domains) > 1:
            patterns.append("cross_domain")
        
        # Evidence patterns
        if any("assumption" in str(c).lower() for c in claims):
            patterns.append("assumption_challenge")
        
        # Property patterns
        principle_name = str(data.get("principle_name", "")).lower()
        if "emergent" in principle_name:
            patterns.append("emergent_property")
        
        # Framework patterns
        if "alternative" in str(data).lower():
            patterns.append("framework_departure")
        
        return patterns
