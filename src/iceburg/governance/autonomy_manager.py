"""
Autonomy Manager
Manages autonomous operation boundaries and constraints
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from .constitution import Constitution
from .rule_enforcer import RuleEnforcer


class AutonomyManager:
    """Manages autonomous operation boundaries"""
    
    def __init__(self, constitution: Optional[Constitution] = None):
        self.constitution = constitution or Constitution()
        self.rule_enforcer = RuleEnforcer(self.constitution)
        self.autonomy_level: int = 0
        self.max_autonomy: int = 100
        self.autonomy_history: List[Dict[str, Any]] = []
        self.boundaries: Dict[str, Any] = {
            "max_autonomy_level": 100,
            "min_autonomy_level": 0,
            "safety_threshold": 80,
            "warning_threshold": 60
        }
    
    def set_autonomy_level(self, level: int) -> bool:
        """Set current autonomy level"""
        if level < 0 or level > self.max_autonomy:
            return False
        
        self.autonomy_level = level
        self.autonomy_history.append({
            "level": level,
            "timestamp": datetime.now().isoformat()
        })
        return True
    
    def increase_autonomy(self, increment: int = 1) -> bool:
        """Increase autonomy level"""
        new_level = min(self.autonomy_level + increment, self.max_autonomy)
        return self.set_autonomy_level(new_level)
    
    def decrease_autonomy(self, decrement: int = 1) -> bool:
        """Decrease autonomy level"""
        new_level = max(self.autonomy_level - decrement, 0)
        return self.set_autonomy_level(new_level)
    
    def check_autonomy_boundary(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if action is within autonomy boundaries"""
        result = {
            "allowed": True,
            "warnings": [],
            "violations": []
        }
        
        # Check autonomy level
        if self.autonomy_level > self.boundaries["safety_threshold"]:
            result["warnings"].append({
                "type": "high_autonomy",
                "message": f"Autonomy level {self.autonomy_level} exceeds safety threshold {self.boundaries['safety_threshold']}",
                "severity": "high"
            })
        
        # Check constitutional rules
        constitution_result = self.rule_enforcer.enforce_constitution({
            **action,
            "autonomy_level": self.autonomy_level,
            "max_autonomy": self.max_autonomy
        })
        
        if not constitution_result["allowed"]:
            result["allowed"] = False
            result["violations"].extend(constitution_result["violations"])
        
        result["warnings"].extend(constitution_result["warnings"])
        
        return result
    
    def can_perform_action(self, action: Dict[str, Any]) -> bool:
        """Check if action can be performed"""
        boundary_check = self.check_autonomy_boundary(action)
        return boundary_check["allowed"]
    
    def get_autonomy_status(self) -> Dict[str, Any]:
        """Get current autonomy status"""
        return {
            "current_level": self.autonomy_level,
            "max_level": self.max_autonomy,
            "safety_threshold": self.boundaries["safety_threshold"],
            "warning_threshold": self.boundaries["warning_threshold"],
            "within_safety": self.autonomy_level <= self.boundaries["safety_threshold"],
            "within_warning": self.autonomy_level <= self.boundaries["warning_threshold"],
            "history": self.autonomy_history[-10:] if self.autonomy_history else []
        }
    
    def set_boundaries(self, boundaries: Dict[str, Any]) -> bool:
        """Set autonomy boundaries"""
        self.boundaries.update(boundaries)
        return True
    
    def reset_autonomy(self) -> bool:
        """Reset autonomy to minimum level"""
        return self.set_autonomy_level(0)
    
    def enable_full_autonomy(self) -> bool:
        """Enable full autonomy (within safety limits)"""
        safe_level = min(self.boundaries["safety_threshold"], self.max_autonomy)
        return self.set_autonomy_level(safe_level)

