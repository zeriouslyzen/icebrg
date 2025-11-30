"""
Source Citation Tracker
Tracks all sources, citations, and references for ICEBURG responses
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import uuid


class SourceCitationTracker:
    """Tracks sources and citations for all ICEBURG responses"""
    
    def __init__(self, config=None, data_dir: Optional[Path] = None):
        # Support both config object and direct data_dir
        if config and hasattr(config, 'data_dir'):
            self.data_dir = Path(config.data_dir)
        else:
            self.data_dir = data_dir or Path("data")
        self.citations_dir = self.data_dir / "citations"
        self.citations_dir.mkdir(parents=True, exist_ok=True)
        self.citations_file = self.citations_dir / "citations.jsonl"
        self.sources_file = self.citations_dir / "sources.jsonl"
        self.citations: List[Dict[str, Any]] = []
        self.sources: Dict[str, Dict[str, Any]] = {}
    
    def track_citation(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track a citation for a query-response pair"""
        citation_id = str(uuid.uuid4())
        
        citation = {
            "citation_id": citation_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "sources": sources,
            "source_count": len(sources),
            "metadata": metadata or {}
        }
        
        # Store citation
        self.citations.append(citation)
        self._save_citation(citation)
        
        # Track individual sources
        for source in sources:
            self._track_source(source)
        
        return citation_id
    
    def _track_source(self, source: Dict[str, Any]):
        """Track an individual source"""
        source_id = source.get("url") or source.get("id") or source.get("title", "")
        if not source_id:
            return
        
        if source_id not in self.sources:
            # Calculate reliability score
            reliability_score = self._calculate_reliability_score(source)
            
            self.sources[source_id] = {
                "source_id": source_id,
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "summary": source.get("summary", ""),
                "source_type": source.get("source_type", "unknown"),
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "citation_count": 0,
                "copyright_status": source.get("copyright_status", "unknown"),
                "reliability_score": reliability_score,
                "reliability_grade": self._get_reliability_grade(reliability_score),
                "metadata": source.get("metadata", {})
            }
        else:
            # Update reliability score (may change over time)
            reliability_score = self._calculate_reliability_score(source)
            self.sources[source_id]["reliability_score"] = reliability_score
            self.sources[source_id]["reliability_grade"] = self._get_reliability_grade(reliability_score)
        
        # Update source tracking
        self.sources[source_id]["last_seen"] = datetime.now().isoformat()
        self.sources[source_id]["citation_count"] += 1
        
        # Save source
        self._save_source(self.sources[source_id])
    
    def _calculate_reliability_score(self, source: Dict[str, Any]) -> float:
        """
        Calculate reliability score for a source (0.0 to 1.0).
        
        Factors:
        - Source type (mainstream > alternative > unknown)
        - URL domain (.edu, .gov > .org > .com > others)
        - Peer-reviewed status
        - Citation count (more citations = more reliable)
        - Evidence level (A/B/C/S/X)
        """
        score = 0.5  # Base score
        
        # Source type scoring
        source_type = source.get("source_type", "unknown").lower()
        type_scores = {
            "mainstream": 0.3,
            "peer-reviewed": 0.3,
            "academic": 0.3,
            "alternative": 0.1,
            "suppressed": 0.05,
            "unknown": 0.0
        }
        score += type_scores.get(source_type, 0.0)
        
        # URL domain scoring
        url = source.get("url", "")
        if url:
            if ".edu" in url or ".gov" in url:
                score += 0.15
            elif ".org" in url:
                score += 0.1
            elif ".com" in url:
                score += 0.05
        
        # Evidence level scoring
        evidence_level = source.get("evidence_level", "").upper()
        evidence_scores = {
            "A": 0.1,  # Well-established
            "B": 0.05,  # Plausible
            "C": 0.0,   # Speculative
            "S": 0.02,  # Suppressed but valid
            "X": 0.0    # Actively censored
        }
        score += evidence_scores.get(evidence_level, 0.0)
        
        # Peer-reviewed status
        if source.get("peer_reviewed", False):
            score += 0.1
        
        # Citation count boost (capped at 0.1)
        citation_count = source.get("citation_count", 0)
        if citation_count > 0:
            score += min(0.1, citation_count * 0.01)
        
        return min(1.0, max(0.0, score))
    
    def _get_reliability_grade(self, score: float) -> str:
        """Get reliability grade from score"""
        if score >= 0.8:
            return "A"  # Highly reliable
        elif score >= 0.6:
            return "B"  # Reliable
        elif score >= 0.4:
            return "C"  # Moderate
        elif score >= 0.2:
            return "D"  # Low reliability
        else:
            return "F"  # Very low reliability
    
    def _save_citation(self, citation: Dict[str, Any]):
        """Save citation to JSONL file"""
        try:
            with open(self.citations_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(citation) + '\n')
        except Exception as e:
            print(f"Error saving citation: {e}")
    
    def _save_source(self, source: Dict[str, Any]):
        """Save source to JSONL file"""
        try:
            with open(self.sources_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(source) + '\n')
        except Exception as e:
            print(f"Error saving source: {e}")
    
    def get_citations_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Get all citations for a query"""
        return [c for c in self.citations if query.lower() in c.get("query", "").lower()]
    
    def get_citation_by_id(self, citation_id: str) -> Optional[Dict[str, Any]]:
        """Get citation by ID"""
        return next((c for c in self.citations if c.get("citation_id") == citation_id), None)
    
    def get_source_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked sources"""
        total_sources = len(self.sources)
        total_citations = len(self.citations)
        
        source_types = {}
        copyright_statuses = {}
        
        for source in self.sources.values():
            source_type = source.get("source_type", "unknown")
            source_types[source_type] = source_types.get(source_type, 0) + 1
            
            copyright_status = source.get("copyright_status", "unknown")
            copyright_statuses[copyright_status] = copyright_statuses.get(copyright_status, 0) + 1
        
        return {
            "total_sources": total_sources,
            "total_citations": total_citations,
            "source_types": source_types,
            "copyright_statuses": copyright_statuses,
            "average_citations_per_source": total_citations / total_sources if total_sources > 0 else 0
        }
    
    def extract_sources(self, content: str) -> List[str]:
        """Extract sources from response content"""
        sources = []
        
        # Look for markdown links [text](url)
        import re
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        matches = re.findall(link_pattern, content)
        for title, url in matches:
            if url.startswith('http'):
                sources.append(url)
        
        # Look for URLs
        url_pattern = r'https?://[^\s\)]+'
        url_matches = re.findall(url_pattern, content)
        sources.extend(url_matches)
        
        # Look for citation patterns like [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        citation_matches = re.findall(citation_pattern, content)
        
        # Remove duplicates and return
        return list(set(sources))
    
    def format_citations_for_response(self, citation_id: str, include_evidence_grades: bool = True) -> str:
        """Format citations for inclusion in response with evidence grades and reliability scores"""
        citation = next((c for c in self.citations if c.get("citation_id") == citation_id), None)
        if not citation:
            return ""
        
        sources = citation.get("sources", [])
        if not sources:
            return ""
        
        formatted = "\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            url = source.get("url", "")
            title = source.get("title", "")
            source_type = source.get("source_type", "unknown")
            
            # Get reliability score and grade
            source_id = source.get("url") or source.get("id") or source.get("title", "")
            reliability_info = ""
            if source_id in self.sources:
                reliability_score = self.sources[source_id].get("reliability_score", 0.0)
                reliability_grade = self.sources[source_id].get("reliability_grade", "C")
                reliability_info = f" [Reliability: {reliability_grade} ({reliability_score:.2f})]"
            
            # Get evidence level
            evidence_level = source.get("evidence_level", "")
            evidence_info = ""
            if evidence_level and include_evidence_grades:
                evidence_labels = {
                    "A": "Well-Established",
                    "B": "Plausible",
                    "C": "Speculative",
                    "S": "Suppressed but Valid",
                    "X": "Actively Censored"
                }
                evidence_label = evidence_labels.get(evidence_level.upper(), evidence_level)
                evidence_info = f" [Evidence: {evidence_level} - {evidence_label}]"
            
            if url:
                formatted += f"{i}. [{title or url}]({url}) ({source_type}){reliability_info}{evidence_info}\n"
            else:
                formatted += f"{i}. {title or 'Unknown source'} ({source_type}){reliability_info}{evidence_info}\n"
        
        return formatted

