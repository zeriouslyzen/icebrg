"""Breakthrough Storage System - Stores and manages breakthrough discoveries."""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BreakthroughDiscovery:
    """Represents a breakthrough discovery."""
    id: str
    title: str
    description: str
    discovery_type: str
    confidence: float
    novelty_score: float
    impact_score: float
    domains: List[str]
    quantum_signature: Optional[str] = None
    temporal_signature: Optional[str] = None
    evidence: List[str] = None
    predictions: List[str] = None
    implications: List[str] = None
    timestamp: float = None
    source_agent: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.predictions is None:
            self.predictions = []
        if self.implications is None:
            self.implications = []
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NovelInsight:
    """Represents a novel insight."""
    id: str
    insight_type: str
    content: str
    novelty_score: float
    uncertainty_score: float
    domains: List[str]
    related_breakthroughs: List[str] = None
    timestamp: float = None
    source_agent: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.related_breakthroughs is None:
            self.related_breakthroughs = []
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


class BreakthroughStorage:
    """Manages storage and retrieval of breakthrough discoveries and novel insights."""
    
    def __init__(self, storage_dir: str = "data/universal_knowledge_base"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.breakthroughs_file = self.storage_dir / "breakthrough_discoveries.json"
        self.insights_file = self.storage_dir / "novel_insights.json"
        self.index_file = self.storage_dir / "breakthrough_index.json"
        
        # Load existing data
        self.breakthroughs = self._load_breakthroughs()
        self.insights = self._load_insights()
        self.index = self._load_index()
    
    def _load_breakthroughs(self) -> List[BreakthroughDiscovery]:
        """Load existing breakthrough discoveries."""
        if not self.breakthroughs_file.exists():
            return []
        
        try:
            with open(self.breakthroughs_file, 'r') as f:
                data = json.load(f)
            
            breakthroughs = []
            for item in data:
                if isinstance(item, dict):
                    breakthroughs.append(BreakthroughDiscovery(**item))
            
            logger.info(f"Loaded {len(breakthroughs)} breakthrough discoveries")
            return breakthroughs
        except Exception as e:
            logger.error(f"Error loading breakthroughs: {e}")
            return []
    
    def _load_insights(self) -> List[NovelInsight]:
        """Load existing novel insights."""
        if not self.insights_file.exists():
            return []
        
        try:
            with open(self.insights_file, 'r') as f:
                data = json.load(f)
            
            insights = []
            for item in data:
                if isinstance(item, dict):
                    insights.append(NovelInsight(**item))
            
            logger.info(f"Loaded {len(insights)} novel insights")
            return insights
        except Exception as e:
            logger.error(f"Error loading insights: {e}")
            return []
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the breakthrough index."""
        if not self.index_file.exists():
            return {"breakthrough_count": 0, "insight_count": 0, "last_updated": 0}
        
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return {"breakthrough_count": 0, "insight_count": 0, "last_updated": 0}
    
    def _save_breakthroughs(self):
        """Save breakthrough discoveries to file."""
        try:
            data = [asdict(breakthrough) for breakthrough in self.breakthroughs]
            with open(self.breakthroughs_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.breakthroughs)} breakthrough discoveries")
        except Exception as e:
            logger.error(f"Error saving breakthroughs: {e}")
    
    def _save_insights(self):
        """Save novel insights to file."""
        try:
            data = [asdict(insight) for insight in self.insights]
            with open(self.insights_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.insights)} novel insights")
        except Exception as e:
            logger.error(f"Error saving insights: {e}")
    
    def _save_index(self):
        """Save the breakthrough index."""
        try:
            self.index.update({
                "breakthrough_count": len(self.breakthroughs),
                "insight_count": len(self.insights),
                "last_updated": time.time()
            })
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def store_breakthrough_from_emergence(
        self, 
        emergence_data: Dict[str, Any], 
        agent_output: str,
        source_agent: str = "unknown"
    ) -> Optional[BreakthroughDiscovery]:
        """Store a breakthrough discovery from emergence detection data."""
        try:
            # Extract breakthrough information from emergence data
            title = self._extract_breakthrough_title(agent_output, emergence_data)
            description = self._extract_breakthrough_description(agent_output, emergence_data)
            discovery_type = self._determine_discovery_type(emergence_data)
            confidence = emergence_data.get("confidence", 0.0)
            novelty_score = self._calculate_novelty_score(agent_output, emergence_data)
            impact_score = self._calculate_impact_score(agent_output, emergence_data)
            domains = self._extract_domains(agent_output, emergence_data)
            
            # Extract signatures if available
            quantum_signature = emergence_data.get("quantum_signature")
            temporal_signature = emergence_data.get("temporal_signature")
            
            # Generate unique ID
            breakthrough_id = self._generate_breakthrough_id(title, description, confidence)
            
            # Create breakthrough discovery
            breakthrough = BreakthroughDiscovery(
                id=breakthrough_id,
                title=title,
                description=description,
                discovery_type=discovery_type,
                confidence=confidence,
                novelty_score=novelty_score,
                impact_score=impact_score,
                domains=domains,
                quantum_signature=quantum_signature,
                temporal_signature=temporal_signature,
                evidence=self._extract_evidence(agent_output),
                predictions=self._extract_predictions(agent_output),
                implications=self._extract_implications(agent_output),
                source_agent=source_agent,
                metadata=emergence_data.get("metadata", {})
            )
            
            # Check for duplicates
            if not self._is_duplicate_breakthrough(breakthrough):
                self.breakthroughs.append(breakthrough)
                self._save_breakthroughs()
                self._save_index()
                
                logger.info(f"Stored breakthrough discovery: {title}")
                return breakthrough
            else:
                logger.info(f"Duplicate breakthrough detected, not storing: {title}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing breakthrough: {e}")
            return None
    
    def store_insight_from_uncertainty(
        self, 
        uncertainty_data: Dict[str, Any], 
        content: str,
        source_agent: str = "unknown"
    ) -> Optional[NovelInsight]:
        """Store a novel insight from uncertainty detection."""
        try:
            # Extract insight information
            insight_type = uncertainty_data.get("insight_type", "uncertainty_driven")
            novelty_score = uncertainty_data.get("novelty_score", 0.0)
            uncertainty_score = uncertainty_data.get("uncertainty_score", 0.0)
            domains = uncertainty_data.get("domains", [])
            
            # Generate unique ID
            insight_id = self._generate_insight_id(content, novelty_score)
            
            # Create novel insight
            insight = NovelInsight(
                id=insight_id,
                insight_type=insight_type,
                content=content,
                novelty_score=novelty_score,
                uncertainty_score=uncertainty_score,
                domains=domains,
                source_agent=source_agent,
                metadata=uncertainty_data.get("metadata", {})
            )
            
            # Check for duplicates
            if not self._is_duplicate_insight(insight):
                self.insights.append(insight)
                self._save_insights()
                self._save_index()
                
                logger.info(f"Stored novel insight: {insight_type}")
                return insight
            else:
                logger.info(f"Duplicate insight detected, not storing: {insight_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing insight: {e}")
            return None
    
    def _extract_breakthrough_title(self, text: str, emergence_data: Dict[str, Any]) -> str:
        """Extract a title for the breakthrough discovery."""
        # Try to find a principle name or key concept
        if "principle_name" in emergence_data:
            return emergence_data["principle_name"]
        
        # Look for patterns in the text
        text_lower = text.lower()
        
        # Look for "principle of" or similar patterns
        if "principle of" in text_lower:
            start = text_lower.find("principle of")
            end = text_lower.find(".", start)
            if end == -1:
                end = start + 100
            return text[start:end].strip()
        
        # Look for "the" + noun patterns
        if "the " in text_lower:
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() == "the" and i + 1 < len(words):
                    # Take next few words as title
                    title_words = words[i:i+4]
                    return " ".join(title_words).strip()
        
        # Fallback: first sentence or first 50 characters
        first_sentence = text.split('.')[0]
        if len(first_sentence) > 50:
            return first_sentence[:50] + "..."
        return first_sentence
    
    def _extract_breakthrough_description(self, text: str, emergence_data: Dict[str, Any]) -> str:
        """Extract a description for the breakthrough discovery."""
        # Use the full text as description, truncated if too long
        if len(text) > 1000:
            return text[:1000] + "..."
        return text
    
    def _determine_discovery_type(self, emergence_data: Dict[str, Any]) -> str:
        """Determine the type of discovery based on emergence data."""
        patterns = emergence_data.get("metadata", {}).get("patterns", [])
        
        if any("quantum_terminology" in p for p in patterns):
            return "quantum_breakthrough"
        elif any("cross_domain_synthesis" in p for p in patterns):
            return "cross_domain_synthesis"
        elif any("novel_predictions" in p for p in patterns):
            return "predictive_breakthrough"
        elif any("mathematical_rigor" in p for p in patterns):
            return "mathematical_breakthrough"
        else:
            return "general_breakthrough"
    
    def _calculate_novelty_score(self, text: str, emergence_data: Dict[str, Any]) -> float:
        """Calculate novelty score for the breakthrough."""
        # Use emergence confidence as base
        base_score = emergence_data.get("confidence", 0.0)
        
        # Boost for novel terms
        novelty_terms = ["unprecedented", "novel", "breakthrough", "revolutionary", "first_time"]
        text_lower = text.lower()
        novelty_count = sum(1 for term in novelty_terms if term in text_lower)
        
        novelty_boost = min(0.3, novelty_count * 0.1)
        return min(1.0, base_score + novelty_boost)
    
    def _calculate_impact_score(self, text: str, emergence_data: Dict[str, Any]) -> float:
        """Calculate impact score for the breakthrough."""
        # Use confidence as base
        base_score = emergence_data.get("confidence", 0.0)
        
        # Boost for impact terms
        impact_terms = ["revolutionary", "transformative", "paradigm_shift", "breakthrough", "game_changing"]
        text_lower = text.lower()
        impact_count = sum(1 for term in impact_terms if term in text_lower)
        
        impact_boost = min(0.4, impact_count * 0.15)
        return min(1.0, base_score + impact_boost)
    
    def _extract_domains(self, text: str, emergence_data: Dict[str, Any]) -> List[str]:
        """Extract relevant domains from the text."""
        domains = []
        text_lower = text.lower()
        
        domain_keywords = {
            "physics": ["quantum", "particle", "field", "energy", "wave", "physics"],
            "biology": ["cell", "organism", "evolution", "genetic", "protein", "biology"],
            "consciousness": ["mind", "awareness", "experience", "perception", "consciousness"],
            "information": ["data", "signal", "processing", "computation", "information"],
            "mathematics": ["equation", "function", "topology", "geometry", "mathematics"],
            "chemistry": ["molecule", "reaction", "bond", "chemical", "chemistry"],
            "psychology": ["behavior", "cognitive", "mental", "psychological", "psychology"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["general"]
    
    def _extract_evidence(self, text: str) -> List[str]:
        """Extract evidence from the text."""
        evidence = []
        sentences = text.split('.')
        
        evidence_keywords = ["evidence", "data", "study", "research", "experiment", "observation"]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in evidence_keywords):
                evidence.append(sentence.strip())
        
        return evidence[:5]  # Limit to 5 pieces of evidence
    
    def _extract_predictions(self, text: str) -> List[str]:
        """Extract predictions from the text."""
        predictions = []
        sentences = text.split('.')
        
        prediction_keywords = ["predict", "hypothesis", "will", "should", "expected", "forecast"]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in prediction_keywords):
                predictions.append(sentence.strip())
        
        return predictions[:3]  # Limit to 3 predictions
    
    def _extract_implications(self, text: str) -> List[str]:
        """Extract implications from the text."""
        implications = []
        sentences = text.split('.')
        
        implication_keywords = ["implication", "consequence", "impact", "significance", "meaning"]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in implication_keywords):
                implications.append(sentence.strip())
        
        return implications[:3]  # Limit to 3 implications
    
    def _generate_breakthrough_id(self, title: str, description: str, confidence: float) -> str:
        """Generate a unique ID for the breakthrough."""
        content = f"{title}:{description}:{confidence}"
        hash_obj = hashlib.md5(content.encode())
        return f"breakthrough_{hash_obj.hexdigest()[:12]}"
    
    def _generate_insight_id(self, content: str, novelty_score: float) -> str:
        """Generate a unique ID for the insight."""
        content_hash = f"{content}:{novelty_score}"
        hash_obj = hashlib.md5(content_hash.encode())
        return f"insight_{hash_obj.hexdigest()[:12]}"
    
    def _is_duplicate_breakthrough(self, breakthrough: BreakthroughDiscovery) -> bool:
        """Check if the breakthrough is a duplicate."""
        for existing in self.breakthroughs:
            # Check title similarity
            if self._text_similarity(breakthrough.title, existing.title) > 0.8:
                return True
            
            # Check description similarity
            if self._text_similarity(breakthrough.description, existing.description) > 0.7:
                return True
        
        return False
    
    def _is_duplicate_insight(self, insight: NovelInsight) -> bool:
        """Check if the insight is a duplicate."""
        for existing in self.insights:
            if self._text_similarity(insight.content, existing.content) > 0.8:
                return True
        
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_breakthroughs_by_domain(self, domain: str) -> List[BreakthroughDiscovery]:
        """Get breakthroughs by domain."""
        return [b for b in self.breakthroughs if domain in b.domains]
    
    def get_breakthroughs_by_confidence(self, min_confidence: float) -> List[BreakthroughDiscovery]:
        """Get breakthroughs above minimum confidence."""
        return [b for b in self.breakthroughs if b.confidence >= min_confidence]
    
    def get_recent_breakthroughs(self, hours: int = 24) -> List[BreakthroughDiscovery]:
        """Get breakthroughs from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [b for b in self.breakthroughs if b.timestamp >= cutoff_time]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_breakthroughs": len(self.breakthroughs),
            "total_insights": len(self.insights),
            "domains": list(set(domain for b in self.breakthroughs for domain in b.domains)),
            "avg_confidence": sum(b.confidence for b in self.breakthroughs) / max(1, len(self.breakthroughs)),
            "avg_novelty": sum(b.novelty_score for b in self.breakthroughs) / max(1, len(self.breakthroughs)),
            "last_updated": self.index.get("last_updated", 0)
        }
