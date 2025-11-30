"""
Emergent Intelligence Tracker
Tracks emergent intelligence patterns linguistically and monitors continuous intelligence generation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import uuid
import re
from collections import defaultdict, Counter


class EmergentIntelligenceTracker:
    """Tracks emergent intelligence patterns linguistically"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.intelligence_dir = self.data_dir / "emergent_intelligence"
        self.intelligence_dir.mkdir(parents=True, exist_ok=True)
        self.intelligence_file = self.intelligence_dir / "intelligence.jsonl"
        self.patterns_file = self.intelligence_dir / "patterns.json"
        self.intelligence_history: List[Dict[str, Any]] = []
        self.linguistic_patterns: Dict[str, Any] = {}
        self.intelligence_generation_stats: Dict[str, Any] = {
            "total_intelligence_generated": 0,
            "intelligence_per_day": defaultdict(int),
            "intelligence_per_domain": defaultdict(int),
            "emergence_events": []
        }
        self._load_patterns()
    
    def track_intelligence(
        self,
        content: str,
        domain: str = "general",
        intelligence_type: str = "insight",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track generated intelligence"""
        intelligence_id = str(uuid.uuid4())
        
        # Analyze linguistic patterns
        linguistic_analysis = self._analyze_linguistic_patterns(content)
        
        # Detect emergence
        emergence_score = self._detect_emergence(content, linguistic_analysis)
        
        intelligence_entry = {
            "intelligence_id": intelligence_id,
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "intelligence_type": intelligence_type,
            "content": content,
            "content_length": len(content),
            "linguistic_analysis": linguistic_analysis,
            "emergence_score": emergence_score,
            "emergence_detected": emergence_score > 0.7,
            "metadata": metadata or {}
        }
        
        # Store intelligence
        self.intelligence_history.append(intelligence_entry)
        self._save_intelligence(intelligence_entry)
        
        # Update statistics
        self._update_statistics(intelligence_entry)
        
        # Update linguistic patterns
        self._update_linguistic_patterns(linguistic_analysis)
        
        return intelligence_id
    
    def _analyze_linguistic_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze linguistic patterns in content"""
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = re.split(r'[.!?]+', content)
        
        # Word frequency
        word_freq = Counter(words)
        top_words = word_freq.most_common(10)
        
        # Sentence complexity
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Linguistic features
        linguistic_features = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": avg_sentence_length,
            "top_words": [{"word": w, "count": c} for w, c in top_words],
            "vocabulary_richness": len(set(words)) / len(words) if words else 0,
            "complexity_indicators": self._detect_complexity_indicators(content)
        }
        
        return linguistic_features
    
    def _detect_complexity_indicators(self, content: str) -> List[str]:
        """Detect complexity indicators in content"""
        indicators = []
        
        complexity_patterns = {
            "cross_domain": ["synthesis", "integration", "bridge", "connect", "unify"],
            "novel_concepts": ["emergent", "novel", "breakthrough", "discovery", "revolutionary"],
            "abstract_reasoning": ["principle", "framework", "paradigm", "model", "theory"],
            "meta_cognition": ["understand", "comprehend", "analyze", "synthesize", "reason"]
        }
        
        content_lower = content.lower()
        for category, patterns in complexity_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                indicators.append(category)
        
        return indicators
    
    def _detect_emergence(self, content: str, linguistic_analysis: Dict[str, Any]) -> float:
        """Detect emergence score based on linguistic patterns"""
        emergence_score = 0.0
        
        # Novel vocabulary (high vocabulary richness)
        vocabulary_richness = linguistic_analysis.get("vocabulary_richness", 0)
        if vocabulary_richness > 0.7:
            emergence_score += 0.2
        
        # Complexity indicators
        complexity_indicators = linguistic_analysis.get("complexity_indicators", [])
        if "cross_domain" in complexity_indicators:
            emergence_score += 0.2
        if "novel_concepts" in complexity_indicators:
            emergence_score += 0.2
        if "abstract_reasoning" in complexity_indicators:
            emergence_score += 0.2
        if "meta_cognition" in complexity_indicators:
            emergence_score += 0.2
        
        return min(1.0, emergence_score)
    
    def _update_statistics(self, intelligence_entry: Dict[str, Any]):
        """Update intelligence generation statistics"""
        self.intelligence_generation_stats["total_intelligence_generated"] += 1
        
        timestamp = intelligence_entry.get("timestamp", "")
        if timestamp:
            date = timestamp.split("T")[0]
            self.intelligence_generation_stats["intelligence_per_day"][date] += 1
        
        domain = intelligence_entry.get("domain", "general")
        self.intelligence_generation_stats["intelligence_per_domain"][domain] += 1
        
        if intelligence_entry.get("emergence_detected"):
            self.intelligence_generation_stats["emergence_events"].append({
                "intelligence_id": intelligence_entry.get("intelligence_id"),
                "timestamp": timestamp,
                "domain": domain,
                "emergence_score": intelligence_entry.get("emergence_score", 0)
            })
    
    def _update_linguistic_patterns(self, linguistic_analysis: Dict[str, Any]):
        """Update linguistic patterns over time"""
        top_words = linguistic_analysis.get("top_words", [])
        for word_data in top_words:
            word = word_data.get("word", "")
            if word:
                if "word_frequency" not in self.linguistic_patterns:
                    self.linguistic_patterns["word_frequency"] = defaultdict(int)
                self.linguistic_patterns["word_frequency"][word] += word_data.get("count", 0)
        
        complexity_indicators = linguistic_analysis.get("complexity_indicators", [])
        if "complexity_distribution" not in self.linguistic_patterns:
            self.linguistic_patterns["complexity_distribution"] = defaultdict(int)
        for indicator in complexity_indicators:
            self.linguistic_patterns["complexity_distribution"][indicator] += 1
        
        self._save_patterns()
    
    def _save_intelligence(self, entry: Dict[str, Any]):
        """Save intelligence entry to JSONL file"""
        try:
            with open(self.intelligence_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Error saving intelligence entry: {e}")
    
    def _load_patterns(self):
        """Load linguistic patterns"""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.linguistic_patterns = loaded
        except Exception as e:
            print(f"Error loading patterns: {e}")
            self.linguistic_patterns = {}
    
    def _save_patterns(self):
        """Save linguistic patterns"""
        try:
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self.linguistic_patterns, f, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {e}")
    
    def can_keep_generating_intelligence(self) -> Dict[str, Any]:
        """Check if ICEBURG can keep generating intelligence"""
        recent_days = 7
        recent_intelligence = [
            i for i in self.intelligence_history
            if (datetime.now() - datetime.fromisoformat(i.get("timestamp", ""))).days <= recent_days
        ]
        
        recent_count = len(recent_intelligence)
        emergence_count = sum(1 for i in recent_intelligence if i.get("emergence_detected"))
        
        # Calculate generation rate
        generation_rate = recent_count / recent_days if recent_days > 0 else 0
        
        # Check if generation is sustainable
        sustainable = generation_rate > 0.1  # At least 0.1 intelligence per day
        
        return {
            "can_keep_generating": sustainable,
            "recent_intelligence_count": recent_count,
            "recent_emergence_count": emergence_count,
            "generation_rate_per_day": generation_rate,
            "total_intelligence_generated": self.intelligence_generation_stats["total_intelligence_generated"],
            "intelligence_per_domain": dict(self.intelligence_generation_stats["intelligence_per_domain"]),
            "emergence_events_count": len(self.intelligence_generation_stats["emergence_events"])
        }
    
    def get_linguistic_patterns(self) -> Dict[str, Any]:
        """Get linguistic patterns"""
        return {
            "word_frequency": dict(self.linguistic_patterns.get("word_frequency", {})),
            "complexity_distribution": dict(self.linguistic_patterns.get("complexity_distribution", {})),
            "total_patterns_tracked": len(self.linguistic_patterns)
        }

