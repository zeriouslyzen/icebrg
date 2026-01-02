"""
Emergence Detector for AGI Civilization

Detects emergent patterns, novel behaviors, and breakthroughs in the simulation.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import hashlib

logger = logging.getLogger(__name__)


class EmergenceType(Enum):
    """Types of emergent phenomena."""
    BEHAVIORAL = "behavioral"      # New behavior patterns
    STRUCTURAL = "structural"      # New organizational structures
    CULTURAL = "cultural"          # New norms or conventions
    TECHNOLOGICAL = "technological"  # New capabilities or tools
    SOCIAL = "social"              # New social dynamics
    UNKNOWN = "unknown"            # Unclassified emergence


class EmergenceSeverity(Enum):
    """Severity/importance of emergence events."""
    MINOR = 1        # Small variations
    NOTABLE = 2      # Worth tracking
    SIGNIFICANT = 3  # Important change
    MAJOR = 4        # Substantial shift
    BREAKTHROUGH = 5 # Revolutionary change


@dataclass
class EmergenceEvent:
    """Represents a detected emergence event."""
    event_id: str
    event_type: EmergenceType
    severity: EmergenceSeverity
    description: str
    timestamp: float
    triggering_agents: List[str]
    novelty_score: float  # 0.0 to 1.0
    impact_score: float   # 0.0 to 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorPattern:
    """Tracks a pattern of behavior for emergence detection."""
    pattern_id: str
    pattern_hash: str  # For quick duplicate detection
    action_sequence: List[str]
    frequency: int
    first_seen: float
    last_seen: float
    participating_agents: List[str] = field(default_factory=list)


class EmergenceDetector:
    """
    Complete emergence detection system for AGI civilization.
    
    Features:
    - Behavior pattern mining
    - Novelty detection using compression-based metrics
    - Cross-agent coordination detection
    - Trend analysis
    - Emergence event generation and tracking
    """
    
    def __init__(self, 
                 emergence_threshold: float = 0.7,
                 pattern_window: int = 100,
                 min_pattern_length: int = 2,
                 max_pattern_length: int = 10):
        """
        Initialize the emergence detector.
        
        Args:
            emergence_threshold: Threshold for classifying as emergent
            pattern_window: Number of recent actions to analyze
            min_pattern_length: Minimum sequence length for patterns
            max_pattern_length: Maximum sequence length for patterns
        """
        self.emergence_threshold = emergence_threshold
        self.pattern_window = pattern_window
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        
        # Pattern tracking
        self.patterns: Dict[str, BehaviorPattern] = {}
        self.pattern_counter = 0
        
        # Action history per agent
        self.action_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Global action history
        self.global_actions: List[Dict[str, Any]] = []
        
        # Emergence events
        self.events: List[EmergenceEvent] = []
        self.event_counter = 0
        
        # Baseline statistics for comparison
        self.baseline_stats: Dict[str, Any] = {
            "action_distribution": Counter(),
            "coordination_rate": 0.0,
            "avg_interaction_length": 0.0
        }
        
        # Compression dictionary for novelty detection
        self.compression_dict: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            "patterns_detected": 0,
            "emergence_events": 0,
            "avg_novelty_score": 0.0,
            "last_detection_time": 0.0
        }
    
    def initialize(self, initial_baseline: Dict[str, Any] = None):
        """
        Initialize with optional baseline data.
        
        Args:
            initial_baseline: Initial baseline statistics
        """
        if initial_baseline:
            self.baseline_stats.update(initial_baseline)
        
        logger.info("Emergence detector initialized")
    
    def record_action(self, action: Dict[str, Any], agent_id: str):
        """
        Record an action for pattern analysis.
        
        Args:
            action: Action taken
            agent_id: Agent who took the action
        """
        action_record = {
            "type": action.get("type", "unknown"),
            "timestamp": time.time(),
            "agent_id": agent_id,
            "metadata": action
        }
        
        # Add to history
        self.action_history[agent_id].append(action_record)
        self.global_actions.append(action_record)
        
        # Trim history
        if len(self.action_history[agent_id]) > self.pattern_window:
            self.action_history[agent_id] = self.action_history[agent_id][-self.pattern_window:]
        
        if len(self.global_actions) > self.pattern_window * 10:
            self.global_actions = self.global_actions[-(self.pattern_window * 10):]
        
        # Update baseline
        self.baseline_stats["action_distribution"][action_record["type"]] += 1
    
    def check(self, world_state: Any) -> List[EmergenceEvent]:
        """
        Check for emergence events based on current world state.
        
        Args:
            world_state: Current world state object
            
        Returns:
            List of detected emergence events
        """
        detected_events = []
        
        # 1. Check for new behavior patterns
        pattern_events = self._detect_behavior_patterns()
        detected_events.extend(pattern_events)
        
        # 2. Check for coordination emergence
        coordination_events = self._detect_coordination()
        detected_events.extend(coordination_events)
        
        # 3. Check for structural changes
        if world_state:
            structural_events = self._detect_structural_changes(world_state)
            detected_events.extend(structural_events)
        
        # 4. Check for novelty spikes
        novelty_events = self._detect_novelty()
        detected_events.extend(novelty_events)
        
        # Store events
        self.events.extend(detected_events)
        
        # Update stats
        self.stats["emergence_events"] = len(self.events)
        self.stats["last_detection_time"] = time.time()
        
        if detected_events:
            logger.info(f"Detected {len(detected_events)} emergence events")
        
        return detected_events
    
    def _detect_behavior_patterns(self) -> List[EmergenceEvent]:
        """Detect new or unusual behavior patterns."""
        events = []
        
        for agent_id, actions in self.action_history.items():
            if len(actions) < self.min_pattern_length:
                continue
            
            # Extract action sequences
            action_types = [a["type"] for a in actions[-20:]]  # Recent 20 actions
            
            # Look for repeated sequences
            for length in range(self.min_pattern_length, min(len(action_types), self.max_pattern_length) + 1):
                for i in range(len(action_types) - length + 1):
                    sequence = tuple(action_types[i:i + length])
                    pattern_hash = hashlib.md5(str(sequence).encode()).hexdigest()[:8]
                    
                    if pattern_hash in self.patterns:
                        pattern = self.patterns[pattern_hash]
                        pattern.frequency += 1
                        pattern.last_seen = time.time()
                        if agent_id not in pattern.participating_agents:
                            pattern.participating_agents.append(agent_id)
                        
                        # Check for emergence (pattern becomes widespread)
                        if pattern.frequency == 5 and len(pattern.participating_agents) >= 2:
                            events.append(self._create_emergence_event(
                                event_type=EmergenceType.BEHAVIORAL,
                                severity=EmergenceSeverity.NOTABLE,
                                description=f"Behavior pattern emerged: {' -> '.join(sequence)}",
                                triggering_agents=pattern.participating_agents.copy(),
                                novelty_score=0.6,
                                impact_score=0.4,
                                evidence={"pattern": sequence, "frequency": pattern.frequency}
                            ))
                    else:
                        # New pattern
                        pattern = BehaviorPattern(
                            pattern_id=f"pattern_{self.pattern_counter}",
                            pattern_hash=pattern_hash,
                            action_sequence=list(sequence),
                            frequency=1,
                            first_seen=time.time(),
                            last_seen=time.time(),
                            participating_agents=[agent_id]
                        )
                        self.patterns[pattern_hash] = pattern
                        self.pattern_counter += 1
                        self.stats["patterns_detected"] += 1
        
        return events
    
    def _detect_coordination(self) -> List[EmergenceEvent]:
        """Detect emergence of coordinated behavior between agents."""
        events = []
        
        if len(self.global_actions) < 10:
            return events
        
        # Look for simultaneous similar actions (within 1 second)
        recent_actions = self.global_actions[-50:]
        
        time_buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for action in recent_actions:
            bucket = int(action["timestamp"])
            time_buckets[bucket].append(action)
        
        for bucket, actions in time_buckets.items():
            if len(actions) >= 3:  # 3+ agents doing something at same time
                action_types = [a["type"] for a in actions]
                agents = list(set([a["agent_id"] for a in actions]))
                
                # Check for identical actions
                type_counts = Counter(action_types)
                most_common = type_counts.most_common(1)
                
                if most_common and most_common[0][1] >= 3:
                    common_action = most_common[0][0]
                    
                    # This might be emergent coordination
                    events.append(self._create_emergence_event(
                        event_type=EmergenceType.SOCIAL,
                        severity=EmergenceSeverity.SIGNIFICANT,
                        description=f"Coordinated action detected: {len(agents)} agents performed '{common_action}' simultaneously",
                        triggering_agents=agents,
                        novelty_score=0.7,
                        impact_score=0.5,
                        evidence={"action": common_action, "agent_count": len(agents)}
                    ))
        
        return events
    
    def _detect_structural_changes(self, world_state: Any) -> List[EmergenceEvent]:
        """Detect structural changes in the world."""
        events = []
        
        # Check for resource distribution changes
        if hasattr(world_state, 'resources'):
            total_resources = sum(r.amount for r in world_state.resources.values())
            
            if hasattr(self, '_last_total_resources'):
                change = abs(total_resources - self._last_total_resources)
                relative_change = change / max(self._last_total_resources, 1)
                
                if relative_change > 0.5:  # 50% change
                    events.append(self._create_emergence_event(
                        event_type=EmergenceType.STRUCTURAL,
                        severity=EmergenceSeverity.MAJOR,
                        description=f"Major resource shift detected: {relative_change:.1%} change",
                        triggering_agents=[],
                        novelty_score=0.8,
                        impact_score=0.7,
                        evidence={"change_percent": relative_change, "total_resources": total_resources}
                    ))
            
            self._last_total_resources = total_resources
        
        return events
    
    def _detect_novelty(self) -> List[EmergenceEvent]:
        """Detect novel action sequences using compression-based metrics."""
        events = []
        
        if len(self.global_actions) < 20:
            return events
        
        # Get recent action sequence
        recent_types = [a["type"] for a in self.global_actions[-20:]]
        sequence_str = "_".join(recent_types)
        
        # Calculate novelty based on how well the sequence compresses
        # (novel sequences compress poorly because they haven't been seen before)
        
        existing_count = self.compression_dict.get(sequence_str, 0)
        novelty_score = 1.0 / (1.0 + existing_count)
        
        self.compression_dict[sequence_str] = existing_count + 1
        
        # Check for highly novel sequences
        if novelty_score > self.emergence_threshold:
            # Track running average
            if self.stats["avg_novelty_score"] > 0:
                self.stats["avg_novelty_score"] = 0.9 * self.stats["avg_novelty_score"] + 0.1 * novelty_score
            else:
                self.stats["avg_novelty_score"] = novelty_score
            
            if novelty_score > 0.95:  # Very novel
                events.append(self._create_emergence_event(
                    event_type=EmergenceType.UNKNOWN,
                    severity=EmergenceSeverity.NOTABLE,
                    description=f"Highly novel behavior sequence detected (novelty: {novelty_score:.2f})",
                    triggering_agents=[a["agent_id"] for a in self.global_actions[-5:]],
                    novelty_score=novelty_score,
                    impact_score=0.3,
                    evidence={"sequence": recent_types[-5:]}
                ))
        
        return events
    
    def _create_emergence_event(self,
                                 event_type: EmergenceType,
                                 severity: EmergenceSeverity,
                                 description: str,
                                 triggering_agents: List[str],
                                 novelty_score: float,
                                 impact_score: float,
                                 evidence: Dict[str, Any]) -> EmergenceEvent:
        """Create a new emergence event."""
        event = EmergenceEvent(
            event_id=f"emergence_{self.event_counter}",
            event_type=event_type,
            severity=severity,
            description=description,
            timestamp=time.time(),
            triggering_agents=triggering_agents,
            novelty_score=novelty_score,
            impact_score=impact_score,
            evidence=evidence
        )
        
        self.event_counter += 1
        logger.info(f"Emergence event: {description}")
        
        return event
    
    def get_events(self, limit: int = 50, min_severity: int = 1) -> List[Dict[str, Any]]:
        """
        Get recent emergence events.
        
        Args:
            limit: Maximum events to return
            min_severity: Minimum severity level
            
        Returns:
            List of event dictionaries
        """
        filtered = [e for e in self.events if e.severity.value >= min_severity]
        
        return [
            {
                "event_id": e.event_id,
                "type": e.event_type.value,
                "severity": e.severity.name,
                "severity_level": e.severity.value,
                "description": e.description,
                "timestamp": e.timestamp,
                "agents": e.triggering_agents,
                "novelty_score": e.novelty_score,
                "impact_score": e.impact_score,
                "evidence": e.evidence
            }
            for e in filtered[-limit:]
        ]
    
    def get_patterns(self, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """
        Get detected behavior patterns.
        
        Args:
            min_frequency: Minimum frequency to include
            
        Returns:
            List of pattern dictionaries
        """
        filtered = [p for p in self.patterns.values() if p.frequency >= min_frequency]
        
        return sorted(
            [
                {
                    "pattern_id": p.pattern_id,
                    "sequence": p.action_sequence,
                    "frequency": p.frequency,
                    "agent_count": len(p.participating_agents),
                    "first_seen": p.first_seen,
                    "last_seen": p.last_seen
                }
                for p in filtered
            ],
            key=lambda x: x["frequency"],
            reverse=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get emergence detection statistics."""
        return self.stats.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of emergence activity."""
        return {
            "total_events": len(self.events),
            "total_patterns": len(self.patterns),
            "avg_novelty": self.stats["avg_novelty_score"],
            "event_breakdown": Counter(e.event_type.value for e in self.events),
            "severity_breakdown": Counter(e.severity.name for e in self.events),
            "most_active_agents": self._get_most_active_agents(),
            "recent_events": self.get_events(5)
        }
    
    def _get_most_active_agents(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Get agents most involved in emergence events."""
        agent_counts: Counter = Counter()
        for event in self.events:
            for agent in event.triggering_agents:
                agent_counts[agent] += 1
        
        return agent_counts.most_common(top_k)
