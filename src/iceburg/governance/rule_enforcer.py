"""
Rule Enforcer
Enforces constitutional rules and data policies
"""

from typing import Any, Dict, Optional, List
from .constitution import Constitution
from .data_policy import DataPolicy


class RuleEnforcer:
    """Enforces rules and policies"""
    
    def __init__(self, constitution: Optional[Constitution] = None):
        self.constitution = constitution or Constitution()
        self.policies: List[DataPolicy] = []
    
    def add_policy(self, policy: DataPolicy) -> bool:
        """Add a data policy"""
        self.policies.append(policy)
        return True
    
    def remove_policy(self, policy_name: str) -> bool:
        """Remove a data policy"""
        self.policies = [p for p in self.policies if p.name != policy_name]
        return True
    
    def enforce_constitution(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce constitutional rules"""
        return self.constitution.check_action(action)
    
    def enforce_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce data policies"""
        all_violations = []
        all_warnings = []
        
        for policy in self.policies:
            result = policy.validate(data)
            if not result["valid"]:
                if policy.enforcement_level == "strict":
                    all_violations.extend(result["violations"])
                elif policy.enforcement_level == "moderate":
                    all_warnings.extend(result["violations"])
                else:
                    all_warnings.extend(result["violations"])
        
        return {
            "valid": len(all_violations) == 0,
            "violations": all_violations,
            "warnings": all_warnings
        }
    
    def enforce_all(self, action: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enforce both constitution and policies"""
        constitution_result = self.enforce_constitution(action)
        policy_result = self.enforce_policies(data or {})
        
        allowed = constitution_result["allowed"] and policy_result["valid"]
        
        return {
            "allowed": allowed,
            "constitution": constitution_result,
            "policies": policy_result
        }

