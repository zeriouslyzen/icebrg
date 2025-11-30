"""
Timeline Correlator
Correlates events across timelines to find suppression patterns
"""

from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict


class TimelineCorrelator:
    """Correlates timelines to find suppression patterns"""
    
    def __init__(self):
        self.timelines: Dict[str, List[Dict[str, Any]]] = {}
        self.reference_timeline: Optional[List[Dict[str, Any]]] = None
    
    def add_timeline(
        self,
        name: str,
        events: List[Dict[str, Any]]
    ) -> bool:
        """Add a timeline for correlation"""
        self.timelines[name] = events
        return True
    
    def set_reference_timeline(
        self,
        events: List[Dict[str, Any]]
    ) -> bool:
        """Set reference timeline"""
        self.reference_timeline = events
        return True
    
    def correlate_events(
        self,
        timeline1_name: str,
        timeline2_name: str
    ) -> Dict[str, Any]:
        """Correlate events between two timelines"""
        if timeline1_name not in self.timelines or timeline2_name not in self.timelines:
            return {"error": "Timeline not found"}
        
        timeline1 = self.timelines[timeline1_name]
        timeline2 = self.timelines[timeline2_name]
        
        correlation = {
            "timeline1": timeline1_name,
            "timeline2": timeline2_name,
            "matching_events": [],
            "unique_events_t1": [],
            "unique_events_t2": [],
            "correlation_score": 0.0
        }
        
        # Find matching events
        for event1 in timeline1:
            event1_date = self._extract_date(event1)
            if not event1_date:
                continue
            
            for event2 in timeline2:
                event2_date = self._extract_date(event2)
                if not event2_date:
                    continue
                
                # Check if events are within 30 days of each other
                date_diff = abs((event1_date - event2_date).days)
                if date_diff <= 30:
                    correlation["matching_events"].append({
                        "event1": event1,
                        "event2": event2,
                        "date_diff_days": date_diff
                    })
        
        # Find unique events
        event1_dates = {self._extract_date(e) for e in timeline1 if self._extract_date(e)}
        event2_dates = {self._extract_date(e) for e in timeline2 if self._extract_date(e)}
        
        unique_t1_dates = event1_dates - event2_dates
        unique_t2_dates = event2_dates - event1_dates
        
        correlation["unique_events_t1"] = [
            e for e in timeline1
            if self._extract_date(e) in unique_t1_dates
        ]
        correlation["unique_events_t2"] = [
            e for e in timeline2
            if self._extract_date(e) in unique_t2_dates
        ]
        
        # Calculate correlation score
        total_events = len(timeline1) + len(timeline2)
        if total_events > 0:
            correlation["correlation_score"] = len(correlation["matching_events"]) / total_events
        
        return correlation
    
    def find_timeline_gaps(
        self,
        timeline_name: str,
        reference_timeline: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Find gaps in timeline compared to reference"""
        if timeline_name not in self.timelines:
            return {"error": "Timeline not found"}
        
        timeline = self.timelines[timeline_name]
        reference = reference_timeline or self.reference_timeline
        
        if not reference:
            return {"error": "No reference timeline"}
        
        gaps = {
            "timeline": timeline_name,
            "missing_events": [],
            "gaps": []
        }
        
        # Find missing events
        timeline_dates = {self._extract_date(e) for e in timeline if self._extract_date(e)}
        ref_dates = {self._extract_date(e) for e in reference if self._extract_date(e)}
        
        missing_dates = ref_dates - timeline_dates
        
        for ref_event in reference:
            ref_date = self._extract_date(ref_event)
            if ref_date and ref_date in missing_dates:
                gaps["missing_events"].append({
                    "date": ref_date.isoformat(),
                    "event": ref_event,
                    "severity": "high"
                })
        
        # Find time gaps
        timeline_sorted = sorted(
            [e for e in timeline if self._extract_date(e)],
            key=self._extract_date
        )
        
        for i in range(len(timeline_sorted) - 1):
            date1 = self._extract_date(timeline_sorted[i])
            date2 = self._extract_date(timeline_sorted[i+1])
            
            if date1 and date2:
                gap_days = (date2 - date1).days
                if gap_days > 365:  # More than 1 year
                    gaps["gaps"].append({
                        "start": date1.isoformat(),
                        "end": date2.isoformat(),
                        "gap_days": gap_days,
                        "severity": "high" if gap_days > 365 * 10 else "medium"
                    })
        
        return gaps
    
    def _extract_date(self, event: Dict[str, Any]) -> Optional[datetime]:
        """Extract date from event"""
        for key in ["date", "timestamp", "created_at", "updated_at", "published_at"]:
            if key in event:
                try:
                    value = event[key]
                    if isinstance(value, str):
                        return datetime.fromisoformat(value.replace("Z", "+00:00"))
                    elif isinstance(value, datetime):
                        return value
                except Exception:
                    continue
        return None
    
    def correlate_all_timelines(self) -> Dict[str, Any]:
        """Correlate all timelines"""
        correlations = {}
        
        timeline_names = list(self.timelines.keys())
        for i, name1 in enumerate(timeline_names):
            for name2 in timeline_names[i+1:]:
                key = f"{name1}_vs_{name2}"
                correlations[key] = self.correlate_events(name1, name2)
        
        return correlations

