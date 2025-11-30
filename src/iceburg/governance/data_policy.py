"""
Data Policy
Manages data governance policies
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class DataPolicy:
    """Data governance policy"""
    
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement_level: str = "strict"  # strict, moderate, lenient
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()
    
    def add_rule(self, rule: Dict[str, Any]) -> bool:
        """Add a rule to the policy"""
        self.rules.append(rule)
        self.updated_at = datetime.now().isoformat()
        return True
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule from the policy"""
        self.rules = [r for r in self.rules if r.get("name") != rule_name]
        self.updated_at = datetime.now().isoformat()
        return True
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against policy"""
        violations = []
        for rule in self.rules:
            if not self._check_rule(rule, data):
                violations.append({
                    "rule": rule.get("name", "unknown"),
                    "message": rule.get("message", "Rule violation")
                })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations
        }
    
    def _check_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Check if data satisfies a rule"""
        rule_type = rule.get("type")
        
        if rule_type == "required_field":
            field = rule.get("field")
            return field in data and data[field] is not None
        
        elif rule_type == "field_type":
            field = rule.get("field")
            expected_type = rule.get("type_value")
            if field not in data:
                return False
            return isinstance(data[field], expected_type)
        
        elif rule_type == "field_range":
            field = rule.get("field")
            min_val = rule.get("min")
            max_val = rule.get("max")
            if field not in data:
                return False
            value = data[field]
            if not isinstance(value, (int, float)):
                return False
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True
        
        elif rule_type == "custom":
            check_func = rule.get("check_function")
            if check_func:
                return check_func(data)
            return True
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "rules": self.rules,
            "enforcement_level": self.enforcement_level,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPolicy":
        """Create policy from dictionary"""
        return cls(
            name=data["name"],
            description=data["description"],
            rules=data["rules"],
            enforcement_level=data.get("enforcement_level", "strict"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )

