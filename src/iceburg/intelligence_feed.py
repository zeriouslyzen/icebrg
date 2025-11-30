from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional
import json
from pathlib import Path
import os
from datetime import datetime


class IntelligenceFeed:
    def __init__(self, data_dir: Optional[Path] = None):
        self.intel_queue: List[Dict[str, Any]] = []
        self.subscribers: List[Callable] = []
        self.data_dir = data_dir or Path("./data")
        self.intel_dir = self.data_dir / "intelligence"
        self.intel_dir.mkdir(parents=True, exist_ok=True)
        
        # Intelligence storage
        self.intel_file = self.intel_dir / "emergence_intel.jsonl"
        self.patterns_file = self.intel_dir / "patterns.json"
        self.trends_file = self.intel_dir / "trends.json"
        
        # Load existing intelligence
        self.load_existing_intelligence()
    
    def send(self, emergence_intel: Dict[str, Any]) -> None:
        """Send emergence intelligence to subscribers and store it"""
        if emergence_intel:
            # Add to queue
            self.intel_queue.append(emergence_intel)
            
            # Store to file
            self.store_intelligence(emergence_intel)
            
            # Update patterns and trends
            self.update_patterns_and_trends(emergence_intel)
            
            # Notify subscribers
            self.notify_subscribers(emergence_intel)
    
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to emergence intelligence updates"""
        self.subscribers.append(callback)
    
    def notify_subscribers(self, intel: Dict[str, Any]) -> None:
        """Notify all subscribers of new intelligence"""
        for subscriber in self.subscribers:
            try:
                subscriber(intel)
            except Exception as e:
    
    def store_intelligence(self, intel: Dict[str, Any]) -> None:
        """Store intelligence to JSONL file"""
        try:
            with open(self.intel_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(intel, ensure_ascii=False) + "\n")
        except Exception as e:
    
    def load_existing_intelligence(self) -> None:
        """Load existing intelligence from storage"""
        try:
            if self.intel_file.exists():
                with open(self.intel_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            intel = json.loads(line.strip())
                            self.intel_queue.append(intel)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
    
    def update_patterns_and_trends(self, new_intel: Dict[str, Any]) -> None:
        """Update pattern and trend analysis with new intelligence"""
        try:
            # Load existing patterns
            patterns = self.load_patterns()
            
            # Update patterns with new intelligence
            emergence_type = new_intel.get("emergence_type", "unknown")
            domains = new_intel.get("domains", [])
            patterns_found = new_intel.get("patterns", [])
            
            # Update emergence type counts
            if emergence_type not in patterns:
                patterns[emergence_type] = 0
            patterns[emergence_type] += 1
            
            # Update domain patterns
            if "domains" not in patterns:
                patterns["domains"] = {}
            for domain in domains:
                if domain not in patterns["domains"]:
                    patterns["domains"][domain] = 0
                patterns["domains"][domain] += 1
            
            # Update pattern counts
            if "pattern_types" not in patterns:
                patterns["pattern_types"] = {}
            for pattern in patterns_found:
                if pattern not in patterns["pattern_types"]:
                    patterns["pattern_types"][pattern] = 0
                patterns["pattern_types"][pattern] += 1
            
            # Save updated patterns
            self.save_patterns(patterns)
            
            # Update trends
            self.update_trends(new_intel)
            
        except Exception as e:
    
    def load_patterns(self) -> Dict[str, Any]:
        """Load existing patterns from storage"""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
        return {}
    
    def save_patterns(self, patterns: Dict[str, Any]) -> None:
        """Save patterns to storage"""
        try:
            with open(self.patterns_file, "w", encoding="utf-8") as f:
                json.dump(patterns, f, indent=2, ensure_ascii=False)
        except Exception as e:
    
    def update_trends(self, new_intel: Dict[str, Any]) -> None:
        """Update trend analysis with new intelligence"""
        try:
            # Load existing trends
            trends = self.load_trends()
            
            # Add new intelligence to trends
            timestamp = new_intel.get("timestamp", datetime.utcnow().isoformat())
            emergence_score = new_intel.get("emergence_score", 0.0)
            actionability = new_intel.get("actionability", "low")
            
            # Update score trends
            if "score_trends" not in trends:
                trends["score_trends"] = []
            trends["score_trends"].append({
                "timestamp": timestamp,
                "score": emergence_score
            })
            
            # Keep only last 100 scores for trend analysis
            if len(trends["score_trends"]) > 100:
                trends["score_trends"] = trends["score_trends"][-100:]
            
            # Update actionability trends
            if "actionability_trends" not in trends:
                trends["actionability_trends"] = {"high": 0, "medium": 0, "low": 0}
            trends["actionability_trends"][actionability] += 1
            
            # Save updated trends
            self.save_trends(trends)
            
        except Exception as e:
    
    def load_trends(self) -> Dict[str, Any]:
        """Load existing trends from storage"""
        try:
            if self.trends_file.exists():
                with open(self.trends_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
        return {}
    
    def save_trends(self, trends: Dict[str, Any]) -> None:
        """Save trends to storage"""
        try:
            with open(self.trends_file, "w", encoding="utf-8") as f:
                json.dump(trends, f, indent=2, ensure_ascii=False)
        except Exception as e:
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get a summary of collected intelligence"""
        try:
            patterns = self.load_patterns()
            trends = self.load_trends()
            
            return {
                "total_intelligence": len(self.intel_queue),
                "patterns": patterns,
                "trends": trends,
                "recent_intelligence": self.intel_queue[-10:] if self.intel_queue else []
            }
        except Exception as e:
            return {}
    
    def clear_intelligence(self) -> None:
        """Clear all stored intelligence (for testing/reset)"""
        self.intel_queue.clear()
        try:
            if self.intel_file.exists():
                self.intel_file.unlink()
            if self.patterns_file.exists():
                self.patterns_file.unlink()
            if self.trends_file.exists():
                self.trends_file.unlink()
        except Exception as e:
