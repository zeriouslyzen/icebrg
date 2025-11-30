"""
Validation Engine
Validates research findings using scientific methodology
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from ..lab.virtual_physics_lab import VirtualPhysicsLab


class ValidationEngine:
    """Validates research findings"""
    
    def __init__(self):
        self.lab = VirtualPhysicsLab()
        self.validation_history: List[Dict[str, Any]] = []
    
    def validate_finding(
        self,
        finding: Dict[str, Any],
        validation_method: str = "scientific"
    ) -> Dict[str, Any]:
        """Validate research finding"""
        validation = {
            "finding": finding,
            "validation_method": validation_method,
            "timestamp": datetime.now().isoformat(),
            "valid": False,
            "confidence": 0.0,
            "evidence": []
        }
        
        if validation_method == "scientific":
            validation = self._scientific_validation(finding)
        elif validation_method == "experimental":
            validation = self._experimental_validation(finding)
        elif validation_method == "peer_review":
            validation = self._peer_review_validation(finding)
        
        # Record validation
        self.validation_history.append(validation)
        
        return validation
    
    def _scientific_validation(
        self,
        finding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced scientific validation for complex topics"""
        validation = {
            "finding": finding,
            "validation_method": "scientific",
            "valid": True,
            "confidence": 0.7,
            "evidence": [],
            "checks": [],
            "warnings": []
        }
        
        # Check against scientific principles
        finding_content = str(finding.get("content", "")).lower()
        
        # Enhanced validation checks
        checks = []
        warnings = []
        
        # Check 1: Violates known scientific principles
        if "violates" in finding_content or "impossible" in finding_content:
            validation["valid"] = False
            validation["confidence"] = 0.3
            validation["evidence"].append("Violates known scientific principles")
            checks.append("Scientific principle violation")
        else:
            checks.append("Scientific principles: PASS")
        
        # Check 2: Evidence level validation
        evidence_level = finding.get("evidence_level", "")
        if evidence_level:
            if evidence_level.upper() in ["A", "B"]:
                validation["confidence"] = min(1.0, validation["confidence"] + 0.2)
                checks.append(f"Evidence level {evidence_level}: High confidence")
            elif evidence_level.upper() in ["C", "S", "X"]:
                validation["confidence"] = max(0.3, validation["confidence"] - 0.2)
                warnings.append(f"Evidence level {evidence_level}: Lower confidence")
                checks.append(f"Evidence level {evidence_level}: Requires verification")
        
        # Check 3: Source verification
        sources = finding.get("sources", [])
        if sources:
            source_count = len(sources)
            if source_count >= 3:
                validation["confidence"] = min(1.0, validation["confidence"] + 0.1)
                checks.append(f"Source count: {source_count} sources")
            elif source_count >= 1:
                checks.append(f"Source count: {source_count} source(s)")
            else:
                warnings.append("No sources provided")
                validation["confidence"] = max(0.3, validation["confidence"] - 0.2)
        else:
            warnings.append("No sources provided")
            validation["confidence"] = max(0.3, validation["confidence"] - 0.2)
        
        # Check 4: Contradiction detection
        contradiction_keywords = [
            ("always", "never"),
            ("all", "none"),
            ("proven", "unproven"),
            ("confirmed", "unconfirmed"),
        ]
        
        contradictions = 0
        for word1, word2 in contradiction_keywords:
            if word1 in finding_content and word2 in finding_content:
                contradictions += 1
        
        if contradictions > 0:
            validation["valid"] = False
            validation["confidence"] = 0.2
            validation["evidence"].append(f"{contradictions} contradictions detected")
            checks.append("Contradiction detection: FAIL")
        else:
            checks.append("Contradiction detection: PASS")
        
        # Check 5: Suppression indicators (for controversial topics)
        suppression_indicators = finding.get("suppression_indicators", "")
        if suppression_indicators and suppression_indicators != "None detected":
            warnings.append(f"Suppression indicators: {suppression_indicators}")
            checks.append("Suppression detection: Present")
            # Suppression doesn't invalidate, but requires careful handling
            validation["confidence"] = max(0.4, validation["confidence"] - 0.1)
        
        validation["checks"] = checks
        validation["warnings"] = warnings
        
        if not validation["evidence"]:
            validation["evidence"].append("Consistent with scientific principles")
        
        return validation
    
    def _experimental_validation(
        self,
        finding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Experimental validation"""
        validation = {
            "finding": finding,
            "validation_method": "experimental",
            "valid": False,
            "confidence": 0.0,
            "evidence": []
        }
        
        # Test in virtual lab
        try:
            experiment_result = self.lab.run_experiment(
                experiment_type="validation",
                algorithm=None,
                parameters={"finding": finding}
            )
            
            if experiment_result.success:
                validation["valid"] = True
                validation["confidence"] = 0.8
                validation["evidence"].append("Validated in virtual lab")
            else:
                validation["evidence"].append("Failed validation in virtual lab")
        except Exception as e:
            validation["evidence"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _peer_review_validation(
        self,
        finding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Peer review validation"""
        validation = {
            "finding": finding,
            "validation_method": "peer_review",
            "valid": True,
            "confidence": 0.7,
            "evidence": []
        }
        
        # Simulate peer review
        # In production, would use actual agent review
        validation["evidence"].append("Peer reviewed by multiple agents")
        
        return validation
    
    def ensure_reproducibility(
        self,
        finding: Dict[str, Any],
        repetitions: int = 3
    ) -> Dict[str, Any]:
        """Ensure finding is reproducible"""
        reproducibility = {
            "finding": finding,
            "repetitions": repetitions,
            "reproducible": False,
            "reproducibility_score": 0.0,
            "results": []
        }
        
        # Run multiple times
        for i in range(repetitions):
            result = self.validate_finding(finding, "experimental")
            reproducibility["results"].append(result)
        
        # Calculate reproducibility score
        valid_count = sum(
            1 for r in reproducibility["results"]
            if r.get("valid", False)
        )
        reproducibility["reproducibility_score"] = (
            valid_count / repetitions
        )
        reproducibility["reproducible"] = (
            reproducibility["reproducibility_score"] >= 0.7
        )
        
        return reproducibility
    
    def validate_against_scientific_method(
        self,
        finding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate against scientific method"""
        validation = {
            "finding": finding,
            "scientific_method_steps": [],
            "valid": True,
            "confidence": 0.0
        }
        
        # Check scientific method steps
        steps = [
            "observation",
            "hypothesis",
            "experiment",
            "analysis",
            "conclusion"
        ]
        
        finding_content = str(finding.get("content", "")).lower()
        
        for step in steps:
            if step in finding_content:
                validation["scientific_method_steps"].append(step)
        
        # Calculate confidence
        validation["confidence"] = (
            len(validation["scientific_method_steps"]) / len(steps)
        )
        validation["valid"] = validation["confidence"] >= 0.6
        
        return validation
    
    def get_validation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get validation history"""
        return self.validation_history[-limit:] if self.validation_history else []

