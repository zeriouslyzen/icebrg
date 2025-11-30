"""
Persistent Agents with Memory, Goals, and Reputation
Implements agent identities with persistent memory and goal-driven behavior.
"""

import time
import uuid
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in the society."""
    RESEARCHER = "researcher"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"
    LEADER = "leader"
    FOLLOWER = "follower"


class GoalPriority(Enum):
    """Goal priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AgentIdentity:
    """Represents an agent's identity and characteristics."""
    agent_id: str
    role: AgentRole
    personality_traits: Dict[str, float]  # Trait name -> value (0.0 to 1.0)
    capabilities: List[str]
    preferences: Dict[str, Any]
    created_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Goal:
    """Represents an agent goal."""
    goal_id: str
    description: str
    priority: GoalPriority
    deadline: Optional[float]  # Unix timestamp
    progress: float  # 0.0 to 1.0
    dependencies: List[str]  # Other goal IDs
    created_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    """Represents a memory item."""
    memory_id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural"
    importance: float  # 0.0 to 1.0
    timestamp: float
    associated_goals: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class GoalHierarchy:
    """
    Manages agent goals with hierarchical structure and dependencies.
    
    Features:
    - Goal prioritization
    - Dependency management
    - Progress tracking
    - Goal completion
    - Conflict resolution
    """
    
    def __init__(self, max_goals: int = 50):
        """
        Initialize goal hierarchy.
        
        Args:
            max_goals: Maximum number of goals to track
        """
        self.max_goals = max_goals
        self.goals: Dict[str, Goal] = {}
        self.goal_counter = 0
        
        # Goal relationships
        self.goal_dependencies: Dict[str, Set[str]] = {}  # goal_id -> set of dependent goal_ids
        self.goal_conflicts: Dict[str, Set[str]] = {}  # goal_id -> set of conflicting goal_ids
        
        # Goal statistics
        self.goal_stats = {
            "total_goals": 0,
            "completed_goals": 0,
            "failed_goals": 0,
            "active_goals": 0,
            "avg_completion_time": 0.0
        }
    
    def add_goal(self, 
                description: str, 
                priority: GoalPriority = GoalPriority.MEDIUM,
                deadline: Optional[float] = None,
                dependencies: List[str] = None,
                metadata: Dict[str, Any] = None) -> str:
        """
        Add a new goal to the hierarchy.
        
        Args:
            description: Goal description
            priority: Goal priority
            deadline: Optional deadline
            dependencies: List of goal IDs this goal depends on
            metadata: Additional metadata
            
        Returns:
            Goal ID
        """
        if len(self.goals) >= self.max_goals:
            # Remove lowest priority goal
            self._remove_lowest_priority_goal()
        
        if dependencies is None:
            dependencies = []
        
        if metadata is None:
            metadata = {}
        
        goal = Goal(
            goal_id=f"goal_{self.goal_counter}",
            description=description,
            priority=priority,
            deadline=deadline,
            progress=0.0,
            dependencies=dependencies,
            created_time=time.time(),
            metadata=metadata
        )
        
        self.goals[goal.goal_id] = goal
        self.goal_dependencies[goal.goal_id] = set(dependencies)
        self.goal_conflicts[goal.goal_id] = set()
        self.goal_counter += 1
        self.goal_stats["total_goals"] += 1
        self.goal_stats["active_goals"] += 1
        
        logger.info(f"Added goal: {description} (priority: {priority.value})")
        return goal.goal_id
    
    def _remove_lowest_priority_goal(self):
        """Remove the lowest priority goal."""
        if not self.goals:
            return
        
        # Find lowest priority goal
        priority_order = {
            GoalPriority.CRITICAL: 4,
            GoalPriority.HIGH: 3,
            GoalPriority.MEDIUM: 2,
            GoalPriority.LOW: 1
        }
        
        lowest_priority_goal = min(
            self.goals.values(),
            key=lambda g: priority_order[g.priority]
        )
        
        self.remove_goal(lowest_priority_goal.goal_id)
    
    def remove_goal(self, goal_id: str) -> bool:
        """
        Remove a goal from the hierarchy.
        
        Args:
            goal_id: ID of goal to remove
            
        Returns:
            True if goal was removed
        """
        if goal_id not in self.goals:
            return False
        
        # Remove from dependencies
        if goal_id in self.goal_dependencies:
            del self.goal_dependencies[goal_id]
        
        if goal_id in self.goal_conflicts:
            del self.goal_conflicts[goal_id]
        
        # Remove from other goals' dependencies
        for other_goal_id, deps in self.goal_dependencies.items():
            deps.discard(goal_id)
        
        del self.goals[goal_id]
        self.goal_stats["active_goals"] -= 1
        
        logger.info(f"Removed goal: {goal_id}")
        return True
    
    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        """
        Update goal progress.
        
        Args:
            goal_id: ID of goal to update
            progress: New progress value (0.0 to 1.0)
            
        Returns:
            True if goal was updated
        """
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        goal.progress = max(0.0, min(1.0, progress))
        
        # Check if goal is completed
        if goal.progress >= 1.0:
            self._complete_goal(goal_id)
        
        logger.debug(f"Updated goal {goal_id} progress to {progress:.2f}")
        return True
    
    def _complete_goal(self, goal_id: str):
        """Mark a goal as completed."""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        completion_time = time.time() - goal.created_time
        
        # Update statistics
        self.goal_stats["completed_goals"] += 1
        self.goal_stats["active_goals"] -= 1
        
        # Update average completion time
        total_completed = self.goal_stats["completed_goals"]
        current_avg = self.goal_stats["avg_completion_time"]
        new_avg = ((current_avg * (total_completed - 1)) + completion_time) / total_completed
        self.goal_stats["avg_completion_time"] = new_avg
        
        logger.info(f"Completed goal: {goal.description}")
    
    def get_ready_goals(self) -> List[Goal]:
        """Get goals that are ready to be worked on (dependencies satisfied)."""
        ready_goals = []
        
        for goal_id, goal in self.goals.items():
            if goal.progress >= 1.0:  # Already completed
                continue
            
            # Check if dependencies are satisfied
            dependencies = self.goal_dependencies.get(goal_id, set())
            if all(dep_id in self.goals and self.goals[dep_id].progress >= 1.0 
                   for dep_id in dependencies):
                ready_goals.append(goal)
        
        # Sort by priority
        priority_order = {
            GoalPriority.CRITICAL: 4,
            GoalPriority.HIGH: 3,
            GoalPriority.MEDIUM: 2,
            GoalPriority.LOW: 1
        }
        
        ready_goals.sort(key=lambda g: priority_order[g.priority], reverse=True)
        return ready_goals
    
    def get_goal_stats(self) -> Dict[str, Any]:
        """Get goal hierarchy statistics."""
        return self.goal_stats.copy()


class AgentMemory:
    """
    Manages agent memory with different types and importance.
    
    Features:
    - Episodic memory (experiences)
    - Semantic memory (knowledge)
    - Procedural memory (skills)
    - Memory consolidation
    - Forgetting mechanisms
    - Database persistence
    """
    
    def __init__(self, max_memories: int = 1000, enable_persistence: bool = True):
        """
        Initialize agent memory.
        
        Args:
            max_memories: Maximum number of memories to store
            enable_persistence: Whether to persist memories to database
        """
        self.max_memories = max_memories
        self.memories: Dict[str, Memory] = {}
        self.memory_counter = 0
        
        # Memory consolidation
        self.consolidation_threshold = 0.7
        self.forgetting_rate = 0.01
        
        # Memory statistics
        self.memory_stats = {
            "total_memories": 0,
            "episodic_memories": 0,
            "semantic_memories": 0,
            "procedural_memories": 0,
            "forgotten_memories": 0
        }
        
        # Persistence layer
        self.enable_persistence = enable_persistence
        self.unified_db = None
        if enable_persistence:
            try:
                from ..database.unified_database import UnifiedDatabase
                from ..config import IceburgConfig
                cfg = IceburgConfig()
                self.unified_db = UnifiedDatabase(cfg)
                # Load existing memories from database
                self._load_from_database()
            except Exception as e:
                logger.warning(f"Could not initialize database persistence: {e}")
                self.unified_db = None
    
    def add_memory(self, 
                   content: str, 
                   memory_type: str = "episodic",
                   importance: float = 0.5,
                   associated_goals: List[str] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Add a memory to the agent's memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance of memory (0.0 to 1.0)
            associated_goals: List of associated goal IDs
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        if len(self.memories) >= self.max_memories:
            # Remove least important memory
            self._remove_least_important_memory()
        
        if associated_goals is None:
            associated_goals = []
        
        if metadata is None:
            metadata = {}
        
        memory = Memory(
            memory_id=f"memory_{self.memory_counter}",
            content=content,
            memory_type=memory_type,
            importance=importance,
            timestamp=time.time(),
            associated_goals=associated_goals,
            metadata=metadata
        )
        
        self.memories[memory.memory_id] = memory
        self.memory_counter += 1
        self.memory_stats["total_memories"] += 1
        self.memory_stats[f"{memory_type}_memories"] += 1
        
        # Persist to database
        if self.enable_persistence and self.unified_db:
            try:
                self._persist_memory(memory)
            except Exception as e:
                logger.warning(f"Could not persist memory to database: {e}")
        
        logger.debug(f"Added {memory_type} memory: {content[:50]}...")
        return memory.memory_id
    
    def _remove_least_important_memory(self):
        """Remove the least important memory."""
        if not self.memories:
            return
        
        least_important = min(
            self.memories.values(),
            key=lambda m: m.importance
        )
        
        self.remove_memory(least_important.memory_id)
    
    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the agent's memory.
        
        Args:
            memory_id: ID of memory to remove
            
        Returns:
            True if memory was removed
        """
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        del self.memories[memory_id]
        self.memory_stats["forgotten_memories"] += 1
        
        logger.debug(f"Removed memory: {memory.content[:50]}...")
        return True
    
    def get_memories_by_type(self, memory_type: str) -> List[Memory]:
        """Get memories of a specific type."""
        return [
            memory for memory in self.memories.values()
            if memory.memory_type == memory_type
        ]
    
    def get_memories_by_importance(self, min_importance: float = 0.0) -> List[Memory]:
        """Get memories above a minimum importance threshold."""
        return [
            memory for memory in self.memories.values()
            if memory.importance >= min_importance
        ]
    
    def get_memories_by_goals(self, goal_ids: List[str]) -> List[Memory]:
        """Get memories associated with specific goals."""
        return [
            memory for memory in self.memories.values()
            if any(goal_id in memory.associated_goals for goal_id in goal_ids)
        ]
    
    def consolidate_memories(self):
        """Consolidate memories based on importance and recency."""
        current_time = time.time()
        
        for memory in self.memories.values():
            # Calculate consolidation score
            age = current_time - memory.timestamp
            consolidation_score = memory.importance * (1.0 / (1.0 + age / 3600.0))  # Decay over hours
            
            # If consolidation score is high, strengthen the memory
            if consolidation_score > self.consolidation_threshold:
                memory.importance = min(1.0, memory.importance + 0.1)
            
            # If consolidation score is low, weaken the memory
            elif consolidation_score < self.forgetting_rate:
                memory.importance = max(0.0, memory.importance - 0.1)
                
                # Remove if importance becomes too low
                if memory.importance <= 0.0:
                    self.remove_memory(memory.memory_id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory_stats.copy()
    
    # ========== Persistence Methods ==========
    
    def _load_from_database(self):
        """Load existing memories from database"""
        if not self.unified_db:
            return
        
        try:
            import asyncio
            
            query = '''
                SELECT * FROM memory_entries
                WHERE memory_type IN ('episodic', 'semantic', 'procedural')
                ORDER BY created_at DESC
                LIMIT ?
            '''
            
            result = asyncio.run(self.unified_db.execute_query(query, (self.max_memories,)))
            
            for row in result.data:
                try:
                    memory = Memory(
                        memory_id=row.get('memory_id', ''),
                        content=row.get('content', ''),
                        memory_type=row.get('memory_type', 'episodic'),
                        importance=row.get('importance', 0.5),
                        timestamp=row.get('created_at', time.time()),
                        associated_goals=json.loads(row.get('associations', '[]')) if isinstance(row.get('associations'), str) else [],
                        metadata=json.loads(row.get('metadata', '{}')) if isinstance(row.get('metadata'), str) else {}
                    )
                    
                    if memory.memory_id not in self.memories:
                        self.memories[memory.memory_id] = memory
                        self.memory_counter = max(self.memory_counter, int(memory.memory_id.split('_')[-1]) if '_' in memory.memory_id else 0) + 1
                        self.memory_stats["total_memories"] += 1
                        self.memory_stats[f"{memory.memory_type}_memories"] += 1
                except Exception as e:
                    logger.warning(f"Could not load memory {row.get('memory_id', 'unknown')}: {e}")
            
            logger.info(f"Loaded {len(result.data)} memories from database")
            
        except Exception as e:
            logger.warning(f"Could not load memories from database: {e}")
    
    def _persist_memory(self, memory: Memory):
        """Persist memory to database"""
        if not self.unified_db:
            return
        
        try:
            import asyncio
            import json
            
            # Extract domain from metadata or use default
            domain = memory.metadata.get('domain', 'general') if isinstance(memory.metadata, dict) else 'general'
            
            query = '''
                INSERT OR REPLACE INTO memory_entries (
                    memory_id, memory_type, content, domain, importance,
                    associations, created_at, last_accessed, memory_strength,
                    cross_references, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                memory.memory_id,
                memory.memory_type,
                memory.content,
                domain,
                memory.importance,
                json.dumps(memory.associated_goals),
                memory.timestamp,
                time.time(),  # last_accessed
                memory.importance,  # memory_strength
                json.dumps([]),  # cross_references
                json.dumps(memory.metadata)
            )
            
            result = asyncio.run(self.unified_db.execute_query(query, params, fetch=False))
            if result.success:
                logger.debug(f"Persisted {memory.memory_type} memory: {memory.memory_id}")
            
        except Exception as e:
            logger.warning(f"Could not persist memory: {e}")
    
    def sync_to_database(self):
        """Sync all in-memory memories to database"""
        if not self.unified_db:
            return
        
        try:
            for memory in self.memories.values():
                self._persist_memory(memory)
            logger.info(f"Synced {len(self.memories)} memories to database")
        except Exception as e:
            logger.warning(f"Could not sync memories to database: {e}")


class PersistentAgent:
    """
    Persistent agent with memory, goals, and reputation.
    
    Features:
    - Persistent identity across sessions
    - Goal-driven behavior
    - Memory-based decision making
    - Social reputation tracking
    - Learning from experience
    """
    
    def __init__(self, 
                 agent_id: str = None,
                 role: AgentRole = AgentRole.GENERALIST,
                 personality_traits: Dict[str, float] = None,
                 capabilities: List[str] = None,
                 preferences: Dict[str, Any] = None):
        """
        Initialize a persistent agent.
        
        Args:
            agent_id: Unique agent identifier
            role: Agent role in society
            personality_traits: Personality trait values
            capabilities: List of agent capabilities
            preferences: Agent preferences
        """
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self.role = role
        
        # Personality traits (default values)
        if personality_traits is None:
            personality_traits = {
                "cooperation": 0.5,
                "aggression": 0.3,
                "curiosity": 0.7,
                "persistence": 0.6,
                "creativity": 0.5,
                "analytical": 0.7
            }
        self.personality_traits = personality_traits
        
        # Capabilities and preferences
        self.capabilities = capabilities or []
        self.preferences = preferences or {}
        
        # Core components
        self.identity = AgentIdentity(
            agent_id=self.agent_id,
            role=role,
            personality_traits=personality_traits,
            capabilities=capabilities or [],
            preferences=preferences or {},
            created_time=time.time()
        )
        
        self.goals = GoalHierarchy()
        self.memory = AgentMemory()
        
        # Social attributes
        self.reputation = 0.5  # 0.0 to 1.0
        self.social_connections: Set[str] = set()
        self.cooperation_history: List[bool] = []
        
        # Current state
        self.location: Tuple[float, float] = (0.0, 0.0)
        self.resources: Dict[str, float] = {}
        self.energy = 1.0  # 0.0 to 1.0
        self.mood = 0.5  # -1.0 to 1.0
        
        # Performance tracking
        self.performance_history: List[float] = []
        self.decision_history: List[Dict[str, Any]] = []
        
        logger.info(f"Created persistent agent {self.agent_id} with role {role.value}")
    
    def decide_action(self, 
                     perception: Any, 
                     social_norms: Any = None, 
                     resource_economy: Any = None) -> Optional[Dict[str, Any]]:
        """
        Decide on an action based on perception, goals, and memory.
        
        Args:
            perception: Agent perception from world
            social_norms: Social norm system
            resource_economy: Resource economy system
            
        Returns:
            Action dictionary or None
        """
        # Get current goals
        ready_goals = self.goals.get_ready_goals()
        if not ready_goals:
            return None
        
        # Select highest priority goal
        current_goal = ready_goals[0]
        
        # Generate action based on goal and perception
        action = self._generate_action_for_goal(current_goal, perception)
        
        # Record decision
        self.decision_history.append({
            "timestamp": time.time(),
            "goal_id": current_goal.goal_id,
            "action": action,
            "perception": str(perception)[:200]  # Truncate for storage
        })
        
        return action
    
    def _generate_action_for_goal(self, goal: Goal, perception: Any) -> Dict[str, Any]:
        """Generate an action to work toward a goal."""
        # Simple action generation based on goal type and personality
        action_type = self._determine_action_type(goal, perception)
        
        action = {
            "type": action_type,
            "goal_id": goal.goal_id,
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "metadata": {
                "personality_traits": self.personality_traits,
                "reputation": self.reputation,
                "energy": self.energy
            }
        }
        
        # Add action-specific parameters
        if action_type == "consume_resource":
            action["resource_name"] = "energy"
            action["amount"] = 0.1
        elif action_type == "create_resource":
            action["name"] = f"resource_{goal.goal_id}"
            action["amount"] = 1.0
            action["location"] = self.location
        elif action_type == "move":
            # Move toward goal location
            action["new_location"] = (
                self.location[0] + np.random.uniform(-1, 1),
                self.location[1] + np.random.uniform(-1, 1)
            )
        
        return action
    
    def _determine_action_type(self, goal: Goal, perception: Any) -> str:
        """Determine action type based on goal and personality."""
        # Simple heuristics based on personality traits
        if self.personality_traits["cooperation"] > 0.7:
            return "cooperate"
        elif self.personality_traits["aggression"] > 0.7:
            return "compete"
        elif self.personality_traits["curiosity"] > 0.7:
            return "explore"
        else:
            return "consume_resource"
    
    def update_from_action_result(self, action: Dict[str, Any], result: Dict[str, Any]):
        """Update agent state based on action result."""
        # Update goal progress
        goal_id = action.get("goal_id")
        if goal_id and goal_id in self.goals.goals:
            success = result.get("success", False)
            if success:
                progress_increase = 0.1
                self.goals.update_goal_progress(goal_id, 
                    self.goals.goals[goal_id].progress + progress_increase)
        
        # Update memory
        self.memory.add_memory(
            content=f"Action {action['type']} resulted in {result.get('outcome', 'unknown')}",
            memory_type="episodic",
            importance=0.5,
            associated_goals=[goal_id] if goal_id else []
        )
        
        # Update performance
        performance = result.get("performance", 0.0)
        self.performance_history.append(performance)
        
        # Update reputation
        if "reputation_change" in result:
            self.reputation = max(0.0, min(1.0, 
                self.reputation + result["reputation_change"]))
        
        # Update energy and mood
        self.energy = max(0.0, min(1.0, self.energy - 0.1))
        if result.get("success", False):
            self.mood = min(1.0, self.mood + 0.1)
        else:
            self.mood = max(-1.0, self.mood - 0.1)
    
    def get_location(self) -> Tuple[float, float]:
        """Get agent's current location."""
        return self.location
    
    def set_location(self, location: Tuple[float, float]):
        """Set agent's location."""
        self.location = location
    
    def get_perception_radius(self) -> float:
        """Get agent's perception radius."""
        # Base radius modified by personality
        base_radius = 10.0
        curiosity_modifier = self.personality_traits.get("curiosity", 0.5)
        return base_radius * (0.5 + curiosity_modifier)
    
    def get_resources(self) -> Dict[str, float]:
        """Get agent's current resources."""
        return self.resources.copy()
    
    def needs_resources(self) -> bool:
        """Check if agent needs resources."""
        return len(self.resources) == 0 or sum(self.resources.values()) < 1.0
    
    def vote(self, options: List[str], context: Dict[str, Any]) -> Optional[str]:
        """Vote on a decision."""
        # Simple voting based on personality and preferences
        if not options:
            return None
        
        # Use personality traits to influence voting
        if self.personality_traits["cooperation"] > 0.7:
            # Prefer cooperative options
            cooperative_options = [opt for opt in options if "cooperate" in opt.lower()]
            if cooperative_options:
                return np.random.choice(cooperative_options)
        
        # Default to random choice
        return np.random.choice(options)
    
    def get_agent_profile(self) -> Dict[str, Any]:
        """Get comprehensive agent profile."""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "personality_traits": self.personality_traits,
            "reputation": self.reputation,
            "location": self.location,
            "energy": self.energy,
            "mood": self.mood,
            "goals": {
                "total": len(self.goals.goals),
                "active": self.goals.goal_stats["active_goals"],
                "completed": self.goals.goal_stats["completed_goals"]
            },
            "memory": {
                "total": len(self.memory.memories),
                "stats": self.memory.get_memory_stats()
            },
            "performance": {
                "avg_performance": np.mean(self.performance_history) if self.performance_history else 0.0,
                "total_decisions": len(self.decision_history)
            }
        }
