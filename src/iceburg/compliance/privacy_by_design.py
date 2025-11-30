"""
Privacy by Design
Implements privacy by design principles
"""

from typing import Any, Dict, Optional, List
import hashlib
import json


class PrivacyByDesign:
    """Privacy by design manager"""
    
    def __init__(self):
        self.encryption_enabled = True
        self.anonymization_enabled = True
        self.pseudonymization_enabled = True
    
    def anonymize_data(self, data: Dict[str, Any], fields_to_anonymize: List[str]) -> Dict[str, Any]:
        """Anonymize specified fields in data"""
        anonymized = data.copy()
        
        for field in fields_to_anonymize:
            if field in anonymized:
                anonymized[field] = self._anonymize_value(anonymized[field])
        
        return anonymized
    
    def _anonymize_value(self, value: Any) -> str:
        """Anonymize a single value"""
        if isinstance(value, str):
            # Hash the value
            return hashlib.sha256(value.encode()).hexdigest()[:16]
        elif isinstance(value, (int, float)):
            # Round to nearest 10
            return str(int(value / 10) * 10)
        else:
            return "***"
    
    def pseudonymize_data(self, data: Dict[str, Any], identifier_field: str) -> Dict[str, Any]:
        """Pseudonymize identifier field"""
        pseudonymized = data.copy()
        
        if identifier_field in pseudonymized:
            original_value = str(pseudonymized[identifier_field])
            pseudonym = hashlib.sha256(original_value.encode()).hexdigest()[:16]
            pseudonymized[identifier_field] = pseudonym
        
        return pseudonymized
    
    def encrypt_sensitive_fields(
        self,
        data: Dict[str, Any],
        sensitive_fields: List[str],
        encryption_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Encrypt sensitive fields"""
        encrypted = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted:
                # Simple encryption (in production, use proper encryption)
                value = str(encrypted[field])
                if encryption_key:
                    encrypted_value = hashlib.sha256((value + encryption_key).encode()).hexdigest()
                else:
                    encrypted_value = hashlib.sha256(value.encode()).hexdigest()
                encrypted[field] = encrypted_value
        
        return encrypted
    
    def check_privacy_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if data complies with privacy by design"""
        issues = []
        
        # Check for direct identifiers
        direct_identifiers = ["email", "phone", "ssn", "passport"]
        for identifier in direct_identifiers:
            if identifier in data:
                issues.append(f"Direct identifier found: {identifier}")
        
        # Check for indirect identifiers
        indirect_identifiers = ["zip_code", "date_of_birth", "gender"]
        found_indirect = [id for id in indirect_identifiers if id in data]
        if len(found_indirect) >= 2:
            issues.append(f"Multiple indirect identifiers found: {found_indirect}")
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues
        }
    
    def apply_privacy_by_design(
        self,
        data: Dict[str, Any],
        anonymize_fields: Optional[List[str]] = None,
        pseudonymize_identifier: Optional[str] = None,
        encrypt_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Apply privacy by design principles"""
        processed = data.copy()
        
        if anonymize_fields:
            processed = self.anonymize_data(processed, anonymize_fields)
        
        if pseudonymize_identifier:
            processed = self.pseudonymize_data(processed, pseudonymize_identifier)
        
        if encrypt_fields:
            processed = self.encrypt_sensitive_fields(processed, encrypt_fields)
        
        return processed

