"""
Social Norm System for AGI Civilization

Implements real norm tracking, violation detection, and enforcement.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class NormType(Enum):
    """Types of social norms."""
    COOPERATION = "cooperation"
    SHARING = "sharing"
    HONESTY = "honesty"
    RECIPROCITY = "reciprocity"
    FAIRNESS = "fairness"
    PUNISHMENT = "punishment"


class NormStrength(Enum):
    """Strength levels for norms."""
    WEAK = 0.25
    MODERATE = 0.5
    STRONG = 0.75
    ABSOLUTE = 1.0


@dataclass
class SocialNorm:
    """Represents a social norm in the civilization."""
    norm_id: str
    norm_type: NormType
    description: str
    strength: float  # 0.0 to 1.0
    adherence_rate: float  # How often agents follow it
    violations: int
    enforcements: int
    created_time: float
    last_updated: float
    supporting_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormViolation:
    """Represents a violation of a norm."""
    violation_id: str
    norm_id: str
    violator_id: str
    witness_ids: List[str]
    severity: float  # 0.0 to 1.0
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    punished: bool = False


@dataclass
class NormEnforcement:
    """Represents an enforcement action."""
    enforcement_id: str
    norm_id: str
    enforcer_id: str
    target_id: str
    enforcement_type: str  # "warning", "reputation_penalty", "resource_penalty"
    severity: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SocialNormSystem:
    """
    Complete social norm formation and enforcement system.
    
    Features:
    - Norm emergence from agent behavior patterns
    - Violation detection and tracking
    - Multi-level enforcement (warnings, penalties)
    - Norm evolution over time
    - Support/opposition tracking
    """
    
    def __init__(self, max_norms: int = 50, max_violations: int = 1000):
        """
        Initialize the social norm system.
        
        Args:
            max_norms: Maximum number of norms to track
            max_violations: Maximum violation history to keep
        """
        self.max_norms = max_norms
        self.max_violations = max_violations
        
        # Norm storage
        self.norms: Dict[str, SocialNorm] = {}
        self.norm_counter = 0
        
        # Violation tracking
        self.violations: List[NormViolation] = []
        self.violation_counter = 0
        
        # Enforcement tracking
        self.enforcements: List[NormEnforcement] = []
        self.enforcement_counter = 0
        
        # Agent norm adherence tracking
        self.agent_adherence: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Behavior pattern tracking for norm emergence
        self.behavior_patterns: Dict[str, int] = defaultdict(int)
        self.pattern_threshold = 10  # Behaviors must occur this many times to form a norm
        
        # Statistics
        self.stats = {
            "total_norms": 0,
            "active_norms": 0,
            "total_violations": 0,
            "total_enforcements": 0,
            "avg_adherence_rate": 0.0
        }
    
    def initialize(self, initial_norms: List[Dict[str, Any]] = None):
        """
        Initialize with optional predefined norms.
        
        Args:
            initial_norms: List of initial norm definitions
        """
        # Create default fundamental norms
        default_norms = [
            {
                "norm_type": NormType.COOPERATION,
                "description": "Agents should cooperate when mutual benefit is possible",
                "strength": 0.6
            },
            {
                "norm_type": NormType.SHARING,
                "description": "Agents should share excess resources with those in need",
                "strength": 0.5
            },
            {
                "norm_type": NormType.RECIPROCITY,
                "description": "Agents should reciprocate favors received",
                "strength": 0.7
            }
        ]
        
        norms_to_create = initial_norms or default_norms
        
        for norm_def in norms_to_create:
            self.create_norm(
                norm_type=norm_def.get("norm_type", NormType.COOPERATION),
                description=norm_def.get("description", ""),
                strength=norm_def.get("strength", 0.5)
            )
        
        logger.info(f"Social norm system initialized with {len(self.norms)} norms")
    
    def create_norm(self, 
                    norm_type: NormType,
                    description: str,
                    strength: float = 0.5) -> str:
        """
        Create a new social norm.
        
        Args:
            norm_type: Type of norm
            description: Description of the norm
            strength: Initial strength (0.0 to 1.0)
            
        Returns:
            Norm ID
        """
        if len(self.norms) >= self.max_norms:
            # Remove weakest norm
            self._remove_weakest_norm()
        
        norm = SocialNorm(
            norm_id=f"norm_{self.norm_counter}",
            norm_type=norm_type,
            description=description,
            strength=max(0.0, min(1.0, strength)),
            adherence_rate=1.0,  # Start optimistic
            violations=0,
            enforcements=0,
            created_time=time.time(),
            last_updated=time.time()
        )
        
        self.norms[norm.norm_id] = norm
        self.norm_counter += 1
        self.stats["total_norms"] += 1
        self.stats["active_norms"] = len(self.norms)
        
        logger.info(f"Created norm: {description} (strength: {strength})")
        return norm.norm_id
    
    def _remove_weakest_norm(self):
        """Remove the weakest norm to make room for new ones."""
        if not self.norms:
            return
        
        weakest = min(self.norms.values(), key=lambda n: n.strength * n.adherence_rate)
        del self.norms[weakest.norm_id]
        self.stats["active_norms"] = len(self.norms)
    
    def detect_violation(self, 
                         action: Dict[str, Any],
                         agent_id: str,
                         witnesses: List[str] = None) -> Optional[NormViolation]:
        """
        Check if an action violates any norms.
        
        Args:
            action: The action taken
            agent_id: Agent who took the action
            witnesses: List of agent IDs who witnessed
            
        Returns:
            NormViolation if a violation was detected, None otherwise
        """
        if witnesses is None:
            witnesses = []
        
        action_type = action.get("type", "")
        
        # Check each norm for potential violations
        for norm in self.norms.values():
            violation_detected = False
            severity = 0.0
            
            # Check based on norm type
            if norm.norm_type == NormType.COOPERATION:
                if action_type in ["defect", "compete", "sabotage"]:
                    violation_detected = True
                    severity = 0.5 if action_type == "compete" else 0.8
                    
            elif norm.norm_type == NormType.SHARING:
                if action_type == "hoard" and action.get("has_excess", False):
                    violation_detected = True
                    severity = 0.6
                    
            elif norm.norm_type == NormType.RECIPROCITY:
                if action_type == "ignore_favor" or action.get("reciprocity_debt", 0) > 3:
                    violation_detected = True
                    severity = 0.4
                    
            elif norm.norm_type == NormType.FAIRNESS:
                if action_type == "unfair_distribution":
                    violation_detected = True
                    severity = 0.7
            
            if violation_detected:
                violation = NormViolation(
                    violation_id=f"violation_{self.violation_counter}",
                    norm_id=norm.norm_id,
                    violator_id=agent_id,
                    witness_ids=witnesses,
                    severity=severity * norm.strength,
                    timestamp=time.time(),
                    context={"action": action}
                )
                
                self.violations.append(violation)
                self.violation_counter += 1
                self.stats["total_violations"] += 1
                
                # Update norm stats
                norm.violations += 1
                norm.adherence_rate = max(0.0, norm.adherence_rate - 0.01)
                norm.last_updated = time.time()
                
                # Trim old violations
                if len(self.violations) > self.max_violations:
                    self.violations = self.violations[-self.max_violations:]
                
                logger.debug(f"Violation detected: {agent_id} violated {norm.norm_id}")
                return violation
        
        return None
    
    def enforce_norm(self,
                     violation: NormViolation,
                     enforcer_id: str,
                     enforcement_type: str = "reputation_penalty") -> NormEnforcement:
        """
        Enforce a norm after a violation.
        
        Args:
            violation: The violation to address
            enforcer_id: Agent doing the enforcing
            enforcement_type: Type of enforcement action
            
        Returns:
            NormEnforcement record
        """
        enforcement = NormEnforcement(
            enforcement_id=f"enforcement_{self.enforcement_counter}",
            norm_id=violation.norm_id,
            enforcer_id=enforcer_id,
            target_id=violation.violator_id,
            enforcement_type=enforcement_type,
            severity=violation.severity,
            timestamp=time.time()
        )
        
        self.enforcements.append(enforcement)
        self.enforcement_counter += 1
        self.stats["total_enforcements"] += 1
        
        # Mark violation as punished
        violation.punished = True
        
        # Update norm stats
        if violation.norm_id in self.norms:
            self.norms[violation.norm_id].enforcements += 1
            self.norms[violation.norm_id].last_updated = time.time()
        
        logger.debug(f"Norm enforced: {enforcer_id} punished {violation.violator_id}")
        return enforcement
    
    def record_behavior(self, behavior_type: str, agent_id: str):
        """
        Record a behavior pattern for potential norm emergence.
        
        Args:
            behavior_type: Type of behavior observed
            agent_id: Agent who exhibited the behavior
        """
        self.behavior_patterns[behavior_type] += 1
        
        # Check if a new norm should emerge
        if self.behavior_patterns[behavior_type] >= self.pattern_threshold:
            self._attempt_norm_emergence(behavior_type)
    
    def _attempt_norm_emergence(self, behavior_type: str):
        """Attempt to create a norm from observed behavior patterns."""
        # Map behavior types to potential norms
        behavior_norm_map = {
            "share_resources": (NormType.SHARING, "Share resources with others"),
            "help_others": (NormType.COOPERATION, "Help other agents in need"),
            "return_favors": (NormType.RECIPROCITY, "Return favors received"),
            "fair_split": (NormType.FAIRNESS, "Split resources fairly")
        }
        
        if behavior_type in behavior_norm_map:
            norm_type, description = behavior_norm_map[behavior_type]
            
            # Check if similar norm already exists
            for norm in self.norms.values():
                if norm.norm_type == norm_type:
                    # Strengthen existing norm instead
                    norm.strength = min(1.0, norm.strength + 0.05)
                    norm.last_updated = time.time()
                    return
            
            # Create new emergent norm
            self.create_norm(norm_type, description, strength=0.4)
            logger.info(f"New norm emerged from behavior: {description}")
    
    def update(self, actions: List[Dict[str, Any]]):
        """
        Update norms based on agent actions.
        
        Args:
            actions: List of agent actions to process
        """
        for action in actions:
            agent_id = action.get("agent_id", "unknown")
            action_type = action.get("type", "")
            
            # Record behavior for norm emergence
            self.record_behavior(action_type, agent_id)
            
            # Check for violations
            self.detect_violation(action, agent_id)
        
        # Decay norm strength over time for unused norms
        self._decay_norms()
        
        # Update statistics
        self._update_stats()
    
    def _decay_norms(self):
        """Apply time-based decay to norm strength."""
        current_time = time.time()
        
        for norm in self.norms.values():
            time_since_update = current_time - norm.last_updated
            
            # Decay if not reinforced in last hour
            if time_since_update > 3600:
                decay = 0.001 * (time_since_update / 3600)
                norm.strength = max(0.1, norm.strength - decay)
    
    def _update_stats(self):
        """Update aggregate statistics."""
        if self.norms:
            self.stats["avg_adherence_rate"] = np.mean([n.adherence_rate for n in self.norms.values()])
        self.stats["active_norms"] = len(self.norms)
    
    def get_norms(self) -> Dict[str, Any]:
        """Get all current norms."""
        return {
            norm_id: {
                "type": norm.norm_type.value,
                "description": norm.description,
                "strength": norm.strength,
                "adherence_rate": norm.adherence_rate,
                "violations": norm.violations,
                "enforcements": norm.enforcements
            }
            for norm_id, norm in self.norms.items()
        }
    
    def get_norm_by_id(self, norm_id: str) -> Optional[SocialNorm]:
        """Get a specific norm by ID."""
        return self.norms.get(norm_id)
    
    def get_violations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent violations."""
        return [
            {
                "violation_id": v.violation_id,
                "norm_id": v.norm_id,
                "violator_id": v.violator_id,
                "severity": v.severity,
                "punished": v.punished,
                "timestamp": v.timestamp
            }
            for v in self.violations[-limit:]
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get social norm statistics."""
        return self.stats.copy()
    
    def get_agent_adherence(self, agent_id: str) -> Dict[str, float]:
        """Get an agent's adherence rate per norm."""
        return dict(self.agent_adherence.get(agent_id, {}))
