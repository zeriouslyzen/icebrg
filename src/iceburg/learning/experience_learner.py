"""
Experience Learner
Learn from historical fixes without fine-tuning
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ExperienceLearner:
    """Learn from historical fixes."""
    
    def __init__(self, config: Dict[str, Any] = None, cfg=None):
        """
        Initialize experience learner.
        
        Args:
            config: Configuration for experience learner
            cfg: ICEBURG configuration
        """
        self.config = config or {}
        self.cfg = cfg
        
        # Learning state
        self.fix_patterns = {}
        self.success_patterns = {}
        self.failure_patterns = {}
    
    async def learn_from_fixes(
        self,
        historical_fixes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Learn from historical fix outcomes.
        
        Args:
            historical_fixes: List of historical fix outcomes
            
        Returns:
            Learning results
        """
        logger.info(f"Learning from {len(historical_fixes)} historical fixes")
        
        # Extract patterns from historical fixes
        for fix in historical_fixes:
            action = fix.get("action")
            success = fix.get("success", False)
            metrics_before = fix.get("metrics_before", {})
            metrics_after = fix.get("metrics_after", {})
            
            if action:
                # Track fix patterns
                if action not in self.fix_patterns:
                    self.fix_patterns[action] = {"success": 0, "failure": 0}
                
                if success:
                    self.fix_patterns[action]["success"] += 1
                    # Store success pattern
                    self.success_patterns[action] = {
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after
                    }
                else:
                    self.fix_patterns[action]["failure"] += 1
                    # Store failure pattern
                    self.failure_patterns[action] = {
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after
                    }
        
        # Calculate improvement
        improvement = self._calculate_improvement()
        
        logger.info(f"Learned patterns for {len(self.fix_patterns)} actions")
        
        return {
            "success": True,
            "fix_patterns": len(self.fix_patterns),
            "success_patterns": len(self.success_patterns),
            "failure_patterns": len(self.failure_patterns),
            "improvement": improvement
        }
    
    def _calculate_improvement(self) -> float:
        """Calculate improvement from learning."""
        # Calculate success rate
        total_success = sum(p["success"] for p in self.fix_patterns.values())
        total_failure = sum(p["failure"] for p in self.fix_patterns.values())
        total = total_success + total_failure
        
        if total == 0:
            return 0.0
        
        success_rate = total_success / total
        
        # Improvement is based on success rate (higher success rate = more improvement)
        return success_rate * 10.0  # Scale to 0-10% improvement
    
    def get_best_action(self, bottleneck_type: str) -> Optional[str]:
        """Get best action for bottleneck type based on learned patterns."""
        # Find action with highest success rate for this bottleneck type
        best_action = None
        best_success_rate = 0.0
        
        for action, patterns in self.fix_patterns.items():
            total = patterns["success"] + patterns["failure"]
            if total > 0:
                success_rate = patterns["success"] / total
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_action = action
        
        return best_action

