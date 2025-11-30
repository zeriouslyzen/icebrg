"""
Information Archaeology
Recovers suppressed knowledge through metadata analysis and timeline correlation
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path


class InformationArchaeology:
    """Recovers suppressed information through archaeological methods"""
    
    def __init__(self):
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.timeline_data: List[Dict[str, Any]] = []
        self.recovered_knowledge: List[Dict[str, Any]] = []
    
    def analyze_metadata(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze document metadata for suppression indicators"""
        analysis = {
            "document_path": document_path,
            "timestamp": datetime.now().isoformat(),
            "suppression_indicators": [],
            "classification_levels": [],
            "timeline_gaps": [],
            "contradictions": []
        }
        
        if metadata:
            # Check for classification delays
            if "classification_date" in metadata and "publication_date" in metadata:
                classification_date = datetime.fromisoformat(metadata["classification_date"])
                publication_date = datetime.fromisoformat(metadata["publication_date"])
                delay = (publication_date - classification_date).days
                
                if delay > 365 * 20:  # 20+ years
                    analysis["suppression_indicators"].append({
                        "type": "classification_delay",
                        "delay_days": delay,
                        "severity": "high"
                    })
            
            # Check classification levels
            if "classification_level" in metadata:
                level = metadata["classification_level"]
                analysis["classification_levels"].append(level)
                
                if level in ["TOP SECRET", "CONFIDENTIAL"]:
                    analysis["suppression_indicators"].append({
                        "type": "high_classification",
                        "level": level,
                        "severity": "medium"
                    })
        
        return analysis
    
    def correlate_timeline(
        self,
        events: List[Dict[str, Any]],
        reference_timeline: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Correlate events with reference timeline to find gaps"""
        correlation = {
            "events": events,
            "gaps": [],
            "anomalies": [],
            "correlation_score": 0.0
        }
        
        if reference_timeline:
            # Find gaps in timeline
            event_dates = sorted([e.get("date") for e in events if "date" in e])
            ref_dates = sorted([e.get("date") for e in reference_timeline if "date" in e])
            
            # Find missing events
            for ref_event in reference_timeline:
                ref_date = ref_event.get("date")
                matching_events = [e for e in events if e.get("date") == ref_date]
                
                if not matching_events:
                    correlation["gaps"].append({
                        "date": ref_date,
                        "expected_event": ref_event,
                        "severity": "high"
                    })
        
        return correlation
    
    def reconstruct_from_contradictions(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Reconstruct suppressed information from contradictions"""
        reconstruction = {
            "contradictions": [],
            "reconstructed_info": [],
            "confidence": 0.0
        }
        
        # Find contradictions between documents
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                contradictions = self._find_contradictions(doc1, doc2)
                if contradictions:
                    reconstruction["contradictions"].extend(contradictions)
        
        # Reconstruct information from contradictions
        if reconstruction["contradictions"]:
            reconstruction["reconstructed_info"] = self._reconstruct_info(
                reconstruction["contradictions"]
            )
            reconstruction["confidence"] = min(
                0.9,
                len(reconstruction["reconstructed_info"]) * 0.1
            )
        
        return reconstruction
    
    def _find_contradictions(
        self,
        doc1: Dict[str, Any],
        doc2: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find contradictions between two documents"""
        contradictions = []
        
        # Simple contradiction detection
        # In production, use more sophisticated NLP
        if "content" in doc1 and "content" in doc2:
            content1 = str(doc1["content"]).lower()
            content2 = str(doc2["content"]).lower()
            
            # Check for opposite statements
            opposites = [
                ("success", "failure"),
                ("approved", "rejected"),
                ("confirmed", "denied"),
                ("exists", "does not exist")
            ]
            
            for pos, neg in opposites:
                if pos in content1 and neg in content2:
                    contradictions.append({
                        "type": "opposite_statements",
                        "doc1": doc1.get("id", "unknown"),
                        "doc2": doc2.get("id", "unknown"),
                        "statement1": pos,
                        "statement2": neg
                    })
        
        return contradictions
    
    def _reconstruct_info(
        self,
        contradictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reconstruct information from contradictions"""
        reconstructed = []
        
        for contradiction in contradictions:
            if contradiction["type"] == "opposite_statements":
                # Reconstruct: if one says success and other says failure,
                # the truth might be partial success or conditional
                reconstructed.append({
                    "type": "reconstructed_fact",
                    "source": "contradiction_analysis",
                    "content": f"Reconstructed from contradiction: {contradiction['statement1']} vs {contradiction['statement2']}",
                    "confidence": 0.6
                })
        
        return reconstructed
    
    def recover_suppressed_knowledge(
        self,
        document_paths: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Main method to recover suppressed knowledge"""
        recovery = {
            "documents_analyzed": len(document_paths),
            "metadata_analysis": [],
            "timeline_correlation": None,
            "contradiction_analysis": None,
            "recovered_knowledge": [],
            "confidence": 0.0
        }
        
        # Analyze metadata
        documents = []
        for i, doc_path in enumerate(document_paths):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            analysis = self.analyze_metadata(doc_path, metadata)
            recovery["metadata_analysis"].append(analysis)
            
            documents.append({
                "id": doc_path,
                "metadata": metadata,
                "analysis": analysis
            })
        
        # Correlate timeline
        events = [d["analysis"] for d in documents]
        recovery["timeline_correlation"] = self.correlate_timeline(events)
        
        # Reconstruct from contradictions
        recovery["contradiction_analysis"] = self.reconstruct_from_contradictions(documents)
        
        # Combine recovered knowledge
        recovery["recovered_knowledge"] = recovery["contradiction_analysis"]["reconstructed_info"]
        
        # Calculate overall confidence
        suppression_indicators = sum(
            len(a["suppression_indicators"]) for a in recovery["metadata_analysis"]
        )
        recovery["confidence"] = min(
            0.9,
            0.3 + (suppression_indicators * 0.1) + (len(recovery["recovered_knowledge"]) * 0.1)
        )
        
        return recovery

