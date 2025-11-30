"""
Suppression Detector
7-step systematic process for detecting information suppression
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from .metadata_analyzer import MetadataAnalyzer
from .timeline_correlator import TimelineCorrelator
from .information_archaeology import InformationArchaeology


class SuppressionDetector:
    """Detects information suppression using 7-step process"""
    
    def __init__(self):
        self.metadata_analyzer = MetadataAnalyzer()
        self.timeline_correlator = TimelineCorrelator()
        self.information_archaeology = InformationArchaeology()
    
    def detect_suppression(
        self,
        documents: List[Dict[str, Any]],
        reference_timeline: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """7-step systematic suppression detection"""
        
        result = {
            "step1_classification_delay": None,
            "step2_narrative_rewriting": None,
            "step3_publication_bottleneck": None,
            "step4_funding_misdirection": None,
            "step5_timeline_gap": None,
            "step6_contradiction_amplification": None,
            "step7_recovery_evidence": None,
            "overall_suppression_score": 0.0,
            "suppression_detected": False
        }
        
        # Step 1: Classification Delay Detection
        result["step1_classification_delay"] = self._step1_classification_delay(documents)
        
        # Step 2: Narrative Rewriting Identification
        result["step2_narrative_rewriting"] = self._step2_narrative_rewriting(documents)
        
        # Step 3: Publication Bottleneck Analysis
        result["step3_publication_bottleneck"] = self._step3_publication_bottleneck(documents)
        
        # Step 4: Funding Misdirection Patterns
        result["step4_funding_misdirection"] = self._step4_funding_misdirection(documents)
        
        # Step 5: Timeline Gap Analysis
        result["step5_timeline_gap"] = self._step5_timeline_gap(documents, reference_timeline)
        
        # Step 6: Contradiction Amplification
        result["step6_contradiction_amplification"] = self._step6_contradiction_amplification(documents)
        
        # Step 7: Recovery Evidence Validation
        result["step7_recovery_evidence"] = self._step7_recovery_evidence(documents)
        
        # Calculate overall suppression score
        scores = [
            result["step1_classification_delay"].get("score", 0.0),
            result["step2_narrative_rewriting"].get("score", 0.0),
            result["step3_publication_bottleneck"].get("score", 0.0),
            result["step4_funding_misdirection"].get("score", 0.0),
            result["step5_timeline_gap"].get("score", 0.0),
            result["step6_contradiction_amplification"].get("score", 0.0),
            result["step7_recovery_evidence"].get("score", 0.0)
        ]
        
        result["overall_suppression_score"] = sum(scores) / len(scores) if scores else 0.0
        result["suppression_detected"] = result["overall_suppression_score"] > 0.5
        
        return result
    
    def _step1_classification_delay(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 1: Classification delay detection"""
        delays = []
        
        for doc in documents:
            metadata = self.metadata_analyzer.extract_metadata(doc)
            timestamps = metadata.get("timestamps", [])
            
            if len(timestamps) >= 2:
                analysis = self.metadata_analyzer.analyze_timestamps(timestamps)
                delays.extend(analysis.get("delays", []))
        
        score = min(1.0, len(delays) * 0.2) if delays else 0.0
        
        return {
            "delays": delays,
            "count": len(delays),
            "score": score,
            "detected": len(delays) > 0
        }
    
    def _step2_narrative_rewriting(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 2: Narrative rewriting identification"""
        rewrites = []
        
        # Check for conflicting narratives
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                if "content" in doc1 and "content" in doc2:
                    content1 = str(doc1["content"]).lower()
                    content2 = str(doc2["content"]).lower()
                    
                    # Check for narrative changes
                    if "discovered" in content1 and "invented" in content2:
                        rewrites.append({
                            "type": "discovery_to_invention",
                            "doc1": doc1.get("id", "unknown"),
                            "doc2": doc2.get("id", "unknown")
                        })
        
        score = min(1.0, len(rewrites) * 0.3) if rewrites else 0.0
        
        return {
            "rewrites": rewrites,
            "count": len(rewrites),
            "score": score,
            "detected": len(rewrites) > 0
        }
    
    def _step3_publication_bottleneck(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 3: Publication bottleneck analysis"""
        bottlenecks = []
        
        for doc in documents:
            metadata = self.metadata_analyzer.extract_metadata(doc)
            
            # Check for long delays between creation and publication
            if "created_at" in metadata.get("timestamps", []) and "published_at" in metadata.get("timestamps", []):
                try:
                    created = datetime.fromisoformat(metadata["timestamps"][0])
                    published = datetime.fromisoformat(metadata["timestamps"][-1])
                    delay = (published - created).days
                    
                    if delay > 365 * 5:  # More than 5 years
                        bottlenecks.append({
                            "document": doc.get("id", "unknown"),
                            "delay_days": delay,
                            "severity": "high"
                        })
                except Exception:
                    pass
        
        score = min(1.0, len(bottlenecks) * 0.25) if bottlenecks else 0.0
        
        return {
            "bottlenecks": bottlenecks,
            "count": len(bottlenecks),
            "score": score,
            "detected": len(bottlenecks) > 0
        }
    
    def _step4_funding_misdirection(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 4: Funding misdirection patterns"""
        misdirections = []
        
        # Check for funding patterns that suggest misdirection
        for doc in documents:
            content = str(doc.get("content", "")).lower()
            
            # Look for funding-related contradictions
            if "funded" in content and "not funded" in content:
                misdirections.append({
                    "type": "funding_contradiction",
                    "document": doc.get("id", "unknown")
                })
        
        score = min(1.0, len(misdirections) * 0.2) if misdirections else 0.0
        
        return {
            "misdirections": misdirections,
            "count": len(misdirections),
            "score": score,
            "detected": len(misdirections) > 0
        }
    
    def _step5_timeline_gap(
        self,
        documents: List[Dict[str, Any]],
        reference_timeline: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Step 5: Timeline gap analysis"""
        if not reference_timeline:
            return {"gaps": [], "count": 0, "score": 0.0, "detected": False}
        
        # Create timeline from documents
        timeline_events = []
        for doc in documents:
            metadata = self.metadata_analyzer.extract_metadata(doc)
            dates = metadata.get("timestamps", [])
            if dates:
                timeline_events.append({
                    "date": dates[0],
                    "document": doc
                })
        
        # Find gaps
        gaps = self.timeline_correlator.find_timeline_gaps("documents", reference_timeline)
        
        score = min(1.0, len(gaps.get("missing_events", [])) * 0.15) if gaps.get("missing_events") else 0.0
        
        return {
            "gaps": gaps.get("missing_events", []),
            "count": len(gaps.get("missing_events", [])),
            "score": score,
            "detected": len(gaps.get("missing_events", [])) > 0
        }
    
    def _step6_contradiction_amplification(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 6: Contradiction amplification"""
        reconstruction = self.information_archaeology.reconstruct_from_contradictions(documents)
        
        contradictions = reconstruction.get("contradictions", [])
        score = min(1.0, len(contradictions) * 0.2) if contradictions else 0.0
        
        return {
            "contradictions": contradictions,
            "count": len(contradictions),
            "score": score,
            "detected": len(contradictions) > 0
        }
    
    def _step7_recovery_evidence(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 7: Recovery evidence validation"""
        recovery = self.information_archaeology.recover_suppressed_knowledge(
            [d.get("id", "unknown") for d in documents],
            [self.metadata_analyzer.extract_metadata(d) for d in documents]
        )
        
        recovered = recovery.get("recovered_knowledge", [])
        score = recovery.get("confidence", 0.0)
        
        return {
            "recovered_knowledge": recovered,
            "count": len(recovered),
            "score": score,
            "detected": len(recovered) > 0
        }

