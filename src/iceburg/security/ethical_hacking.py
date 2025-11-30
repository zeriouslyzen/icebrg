"""
Ethical Hacking
Ethical hacking framework with consent management
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from .penetration_tester import PenetrationTester


class EthicalHacking:
    """Ethical hacking framework"""
    
    def __init__(self):
        self.penetration_tester = PenetrationTester()
        self.consents: Dict[str, Dict[str, Any]] = {}
        self.test_scopes: Dict[str, Dict[str, Any]] = {}
        self.test_results: List[Dict[str, Any]] = []
    
    def request_consent(
        self,
        target: str,
        scope: Dict[str, Any],
        requester: str
    ) -> Dict[str, Any]:
        """Request consent for penetration testing"""
        consent = {
            "target": target,
            "scope": scope,
            "requester": requester,
            "requested_at": datetime.now().isoformat(),
            "status": "pending",
            "consent_id": f"consent_{int(datetime.now().timestamp())}"
        }
        
        self.consents[consent["consent_id"]] = consent
        return consent
    
    def grant_consent(
        self,
        consent_id: str,
        granted_by: str,
        conditions: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Grant consent for penetration testing"""
        if consent_id not in self.consents:
            return False
        
        consent = self.consents[consent_id]
        consent["status"] = "granted"
        consent["granted_by"] = granted_by
        consent["granted_at"] = datetime.now().isoformat()
        consent["conditions"] = conditions or {}
        
        # Set test scope
        self.test_scopes[consent["target"]] = {
            "scope": consent["scope"],
            "conditions": consent["conditions"],
            "consent_id": consent_id
        }
        
        return True
    
    def revoke_consent(self, consent_id: str) -> bool:
        """Revoke consent"""
        if consent_id not in self.consents:
            return False
        
        consent = self.consents[consent_id]
        consent["status"] = "revoked"
        consent["revoked_at"] = datetime.now().isoformat()
        
        # Remove test scope
        if consent["target"] in self.test_scopes:
            del self.test_scopes[consent["target"]]
        
        return True
    
    def perform_ethical_test(
        self,
        target: str,
        test_type: str
    ) -> Dict[str, Any]:
        """Perform ethical penetration test with consent check"""
        # Check consent
        if target not in self.test_scopes:
            return {
                "error": "No consent granted for target",
                "target": target,
                "test_type": test_type
            }
        
        scope = self.test_scopes[target]
        
        # Check if test is within scope
        if not self._is_within_scope(test_type, scope["scope"]):
            return {
                "error": "Test type not within granted scope",
                "target": target,
                "test_type": test_type,
                "scope": scope["scope"]
            }
        
        # Perform test
        if test_type == "network":
            result = self.penetration_tester.network_penetration_test(target)
        elif test_type == "web_application":
            result = self.penetration_tester.web_application_test(target)
        elif test_type == "api":
            result = self.penetration_tester.api_security_test(target)
        else:
            return {
                "error": f"Unknown test type: {test_type}"
            }
        
        # Add consent information
        result["consent_id"] = scope["consent_id"]
        result["scope"] = scope["scope"]
        result["ethical_test"] = True
        
        # Store result
        self.test_results.append(result)
        
        return result
    
    def _is_within_scope(
        self,
        test_type: str,
        scope: Dict[str, Any]
    ) -> bool:
        """Check if test is within scope"""
        allowed_tests = scope.get("allowed_tests", [])
        return test_type in allowed_tests or "all" in allowed_tests
    
    def responsible_disclosure(
        self,
        vulnerability: Dict[str, Any],
        target: str
    ) -> Dict[str, Any]:
        """Handle responsible disclosure of vulnerability"""
        disclosure = {
            "vulnerability": vulnerability,
            "target": target,
            "disclosed_at": datetime.now().isoformat(),
            "status": "pending",
            "disclosure_id": f"disclosure_{int(datetime.now().timestamp())}"
        }
        
        # In production, would send to responsible party
        # For now, just store
        return disclosure
    
    def get_consent_status(self, target: str) -> Optional[Dict[str, Any]]:
        """Get consent status for target"""
        if target in self.test_scopes:
            consent_id = self.test_scopes[target]["consent_id"]
            return self.consents.get(consent_id)
        return None
    
    def get_test_history(self, target: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get test history"""
        if target:
            return [r for r in self.test_results if r.get("target") == target]
        return self.test_results

