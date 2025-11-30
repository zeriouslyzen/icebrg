"""
Metadata Analyzer
Analyzes document metadata for suppression indicators
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import re


class MetadataAnalyzer:
    """Analyzes metadata for suppression indicators"""
    
    def __init__(self):
        self.suppression_patterns = [
            "CLASSIFIED",
            "CONFIDENTIAL",
            "TOP SECRET",
            "RESTRICTED",
            "FOR OFFICIAL USE ONLY"
        ]
        self.timeline_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # Date format
            r"\d{1,2}/\d{1,2}/\d{4}",  # Date format
        ]
    
    def analyze_timestamps(
        self,
        timestamps: List[str],
        reference_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze timestamps for delays and gaps"""
        analysis = {
            "timestamps": timestamps,
            "delays": [],
            "gaps": [],
            "anomalies": []
        }
        
        if not timestamps:
            return analysis
        
        # Parse timestamps
        parsed_times = []
        for ts in timestamps:
            try:
                if isinstance(ts, str):
                    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    parsed = ts
                parsed_times.append(parsed)
            except Exception:
                continue
        
        if not parsed_times:
            return analysis
        
        # Sort by date
        parsed_times.sort()
        
        # Find delays
        if reference_date:
            for ts in parsed_times:
                delay = (ts - reference_date).days
                if delay > 365:  # More than 1 year
                    analysis["delays"].append({
                        "timestamp": ts.isoformat(),
                        "delay_days": delay,
                        "severity": "high" if delay > 365 * 5 else "medium"
                    })
        
        # Find gaps
        for i in range(len(parsed_times) - 1):
            gap = (parsed_times[i+1] - parsed_times[i]).days
            if gap > 365:  # More than 1 year gap
                analysis["gaps"].append({
                    "start": parsed_times[i].isoformat(),
                    "end": parsed_times[i+1].isoformat(),
                    "gap_days": gap,
                    "severity": "high" if gap > 365 * 10 else "medium"
                })
        
        return analysis
    
    def analyze_classification_levels(
        self,
        classification_levels: List[str]
    ) -> Dict[str, Any]:
        """Analyze classification levels"""
        analysis = {
            "levels": classification_levels,
            "high_classification_count": 0,
            "suppression_indicators": []
        }
        
        high_levels = ["TOP SECRET", "CONFIDENTIAL", "SECRET"]
        
        for level in classification_levels:
            if level.upper() in high_levels:
                analysis["high_classification_count"] += 1
                analysis["suppression_indicators"].append({
                    "type": "high_classification",
                    "level": level,
                    "severity": "high" if level == "TOP SECRET" else "medium"
                })
        
        return analysis
    
    def extract_metadata(
        self,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata from document"""
        metadata = {
            "timestamps": [],
            "classification_levels": [],
            "authors": [],
            "organizations": [],
            "keywords": []
        }
        
        # Extract timestamps
        for key in ["date", "timestamp", "created_at", "updated_at", "published_at"]:
            if key in document:
                metadata["timestamps"].append(str(document[key]))
        
        # Extract classification levels
        for key in ["classification", "classification_level", "security_level"]:
            if key in document:
                metadata["classification_levels"].append(str(document[key]))
        
        # Extract authors
        for key in ["author", "authors", "creator", "created_by"]:
            if key in document:
                if isinstance(document[key], list):
                    metadata["authors"].extend(document[key])
                else:
                    metadata["authors"].append(str(document[key]))
        
        # Extract organizations
        for key in ["organization", "org", "department", "agency"]:
            if key in document:
                if isinstance(document[key], list):
                    metadata["organizations"].extend(document[key])
                else:
                    metadata["organizations"].append(str(document[key]))
        
        # Extract keywords
        if "keywords" in document:
            if isinstance(document["keywords"], list):
                metadata["keywords"].extend(document["keywords"])
            else:
                metadata["keywords"].append(str(document["keywords"]))
        
        return metadata
    
    def detect_suppression_patterns(
        self,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect suppression patterns in metadata"""
        patterns = []
        
        # Check classification levels
        for level in metadata.get("classification_levels", []):
            if any(pattern in level.upper() for pattern in self.suppression_patterns):
                patterns.append({
                    "type": "classification_suppression",
                    "level": level,
                    "severity": "high"
                })
        
        # Check for timeline gaps
        timestamps = metadata.get("timestamps", [])
        if len(timestamps) >= 2:
            analysis = self.analyze_timestamps(timestamps)
            if analysis["gaps"]:
                patterns.extend([
                    {
                        "type": "timeline_gap",
                        "gap": gap,
                        "severity": gap["severity"]
                    }
                    for gap in analysis["gaps"]
                ])
        
        return patterns

