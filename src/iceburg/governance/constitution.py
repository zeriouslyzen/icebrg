"""
Constitution
Constitutional governance framework for autonomous operation
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ConstitutionalRule:
    """Constitutional rule"""
    
    name: str
    description: str
    rule_type: str  # constraint, principle, boundary
    condition: str  # Python expression or function
    action: str  # allow, deny, warn
    priority: int = 0  # Higher priority rules are checked first


class Constitution:
    """Constitutional governance framework"""
    
    def __init__(self):
        self.rules: List[ConstitutionalRule] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default constitutional rules"""
        default_rules = [
            ConstitutionalRule(
                name="truth_seeking_principle",
                description="Always seek truth regardless of barriers",
                rule_type="principle",
                condition="True",
                action="allow",
                priority=100
            ),
            ConstitutionalRule(
                name="safety_boundary",
                description="Never harm humans or violate safety",
                rule_type="boundary",
                condition="not ('harm' in action or 'violate' in action)",
                action="deny",
                priority=200
            ),
            ConstitutionalRule(
                name="autonomy_constraint",
                description="Operate within autonomous boundaries",
                rule_type="constraint",
                condition="autonomy_level <= max_autonomy",
                action="warn",
                priority=50
            )
        ]
        self.rules.extend(default_rules)
    
    def add_rule(self, rule: ConstitutionalRule) -> bool:
        """Add a constitutional rule"""
        self.rules.append(rule)
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        self.updated_at = datetime.now().isoformat()
        return True
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a constitutional rule"""
        self.rules = [r for r in self.rules if r.name != rule_name]
        self.updated_at = datetime.now().isoformat()
        return True
    
    def check_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an action is allowed by constitution"""
        violations = []
        warnings = []
        
        for rule in self.rules:
            try:
                # Evaluate condition
                context = {
                    "action": action.get("action", ""),
                    "autonomy_level": action.get("autonomy_level", 0),
                    "max_autonomy": action.get("max_autonomy", 100),
                    **action
                }
                
                # Simple condition evaluation (can be enhanced)
                condition_result = self._evaluate_condition(rule.condition, context)
                
                if not condition_result:
                    if rule.action == "deny":
                        violations.append({
                            "rule": rule.name,
                            "message": rule.description,
                            "priority": rule.priority
                        })
                    elif rule.action == "warn":
                        warnings.append({
                            "rule": rule.name,
                            "message": rule.description,
                            "priority": rule.priority
                        })
            except Exception as e:
                warnings.append({
                    "rule": rule.name,
                    "message": f"Error evaluating rule: {str(e)}",
                    "priority": rule.priority
                })
        
        # If any deny rules violated, action is not allowed
        allowed = len(violations) == 0
        
        return {
            "allowed": allowed,
            "violations": violations,
            "warnings": warnings
        }
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression"""
        try:
            # Simple string-based evaluation
            # In production, use a proper expression evaluator
            if condition == "True":
                return True
            elif condition == "False":
                return False
            else:
                # Try to evaluate as Python expression
                return eval(condition, {"__builtins__": {}}, context)
        except Exception:
            # If evaluation fails, default to True (permissive)
            return True
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all rules"""
        return [
            {
                "name": r.name,
                "description": r.description,
                "rule_type": r.rule_type,
                "condition": r.condition,
                "action": r.action,
                "priority": r.priority
            }
            for r in self.rules
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert constitution to dictionary"""
        return {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "rules": self.get_rules()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constitution":
        """Create constitution from dictionary"""
        constitution = cls()
        constitution.created_at = data.get("created_at", datetime.now().isoformat())
        constitution.updated_at = data.get("updated_at", datetime.now().isoformat())
        
        for rule_data in data.get("rules", []):
            rule = ConstitutionalRule(
                name=rule_data["name"],
                description=rule_data["description"],
                rule_type=rule_data["rule_type"],
                condition=rule_data["condition"],
                action=rule_data["action"],
                priority=rule_data.get("priority", 0)
            )
            constitution.add_rule(rule)
        
        return constitution

