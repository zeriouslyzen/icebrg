"""
AGI Civilization World Model - Persistent simulation state
Implements foundation world models like Genie 3 for agent societies.
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SimulationState(Enum):
    """World simulation states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"



@dataclass
class Resource:
    """Represents a resource in the world."""
    name: str
    amount: float
    max_amount: float
    regeneration_rate: float = 0.0
    decay_rate: float = 0.0
    location: Tuple[float, float] = (0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldEvent:
    """Represents an event in the world."""
    event_id: str
    event_type: str
    timestamp: float
    location: Tuple[float, float]
    magnitude: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerception:
    """Represents what an agent perceives from the world."""
    agent_id: str
    timestamp: float
    visible_resources: List[Resource]
    nearby_agents: List[str]
    world_events: List[WorldEvent]
    environmental_factors: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorldState:
    """
    Persistent world state for AGI civilization simulation.
    
    Features:
    - Resource management and regeneration
    - Environmental events and dynamics
    - Spatial relationships
    - Temporal persistence
    - Event logging and replay
    """
    
    def __init__(self, 
                 world_size: Tuple[float, float] = (100.0, 100.0),
                 max_resources: int = 1000,
                 max_events: int = 10000):
        """
        Initialize the world state.
        
        Args:
            world_size: Size of the world (width, height)
            max_resources: Maximum number of resources
            max_events: Maximum number of events to track
        """
        self.world_size = world_size
        self.max_resources = max_resources
        self.max_events = max_events
        
        # World state
        self.state = SimulationState.INITIALIZING
        self.current_time = time.time()
        self.simulation_step = 0
        
        # Resources
        self.resources: Dict[str, Resource] = {}
        self.resource_locations: Dict[str, Tuple[float, float]] = {}
        
        # Events
        self.events: List[WorldEvent] = []
        self.event_counter = 0
        
        # Environmental factors
        self.environmental_factors = {
            "temperature": 20.0,
            "humidity": 0.5,
            "pressure": 1.0,
            "visibility": 1.0,
            "stability": 1.0
        }
        
        # Spatial grid for efficient queries
        self.grid_size = 10.0  # Grid cell size
        self.grid_width = int(world_size[0] / self.grid_size)
        self.grid_height = int(world_size[1] / self.grid_size)
        self.spatial_grid: Dict[Tuple[int, int], List[str]] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_steps": 0,
            "average_step_time": 0.0,
            "resource_updates": 0,
            "event_generations": 0
        }
    
    def initialize_world(self, initial_resources: List[Dict[str, Any]] = None):
        """Initialize the world with initial resources and state."""
        logger.info("Initializing AGI civilization world...")
        
        self.state = SimulationState.INITIALIZING
        
        # Add initial resources
        if initial_resources:
            for resource_data in initial_resources:
                self.add_resource(**resource_data)
        
        # Initialize spatial grid
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                self.spatial_grid[(x, y)] = []
        
        # Set initial environmental factors
        self._update_environmental_factors()
        
        self.state = SimulationState.RUNNING
        logger.info(f"World initialized with {len(self.resources)} resources")
    
    def add_resource(self, 
                    name: str, 
                    amount: float, 
                    max_amount: float,
                    regeneration_rate: float = 0.0,
                    decay_rate: float = 0.0,
                    location: Tuple[float, float] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """Add a resource to the world."""
        if len(self.resources) >= self.max_resources:
            logger.warning("Maximum resources reached, removing oldest resource")
            # Remove oldest resource
            oldest = min(self.resources.keys(), key=lambda k: self.resources[k].metadata.get("created_time", 0))
            self.remove_resource(oldest)
        
        if location is None:
            location = (
                np.random.uniform(0, self.world_size[0]),
                np.random.uniform(0, self.world_size[1])
            )
        
        if metadata is None:
            metadata = {}
        
        metadata["created_time"] = self.current_time
        
        resource = Resource(
            name=name,
            amount=amount,
            max_amount=max_amount,
            regeneration_rate=regeneration_rate,
            decay_rate=decay_rate,
            location=location,
            metadata=metadata
        )
        
        self.resources[name] = resource
        self.resource_locations[name] = location
        
        # Update spatial grid
        self._update_spatial_grid(name, location)
        
        logger.info(f"Added resource '{name}' at {location} with amount {amount}")
        return name
    
    def remove_resource(self, name: str):
        """Remove a resource from the world."""
        if name in self.resources:
            del self.resources[name]
            if name in self.resource_locations:
                del self.resource_locations[name]
            logger.info(f"Removed resource '{name}'")
    
    def get_perceptions(self, agent_id: str, location: Tuple[float, float], 
                       perception_radius: float = 10.0) -> AgentPerception:
        """
        Get what an agent perceives from the world.
        
        Args:
            agent_id: ID of the perceiving agent
            location: Agent's location
            perception_radius: Radius of perception
            
        Returns:
            AgentPerception object
        """
        # Find visible resources
        visible_resources = []
        for resource_name, resource in self.resources.items():
            distance = np.sqrt(
                (resource.location[0] - location[0])**2 + 
                (resource.location[1] - location[1])**2
            )
            if distance <= perception_radius:
                visible_resources.append(resource)
        
        # Find nearby agents using spatial indexing
        nearby_agents = []
        try:
            # Use agent registry to find nearby agents
            if hasattr(self, 'agent_registry'):
                all_agents = self.agent_registry.get_all_agents()
                for agent in all_agents:
                    if hasattr(agent, 'position'):
                        distance = np.linalg.norm(np.array(agent.position) - np.array(position))
                        if distance <= radius:
                            nearby_agents.append(agent)
        except Exception:
            # Fallback: return empty list if agent tracking not available
            nearby_agents = []
        
        # Get recent world events
        recent_events = [
            event for event in self.events 
            if self.current_time - event.timestamp < 60.0  # Last minute
        ]
        
        return AgentPerception(
            agent_id=agent_id,
            timestamp=self.current_time,
            visible_resources=visible_resources,
            nearby_agents=nearby_agents,
            world_events=recent_events,
            environmental_factors=self.environmental_factors.copy(),
            metadata={"perception_radius": perception_radius}
        )
    
    def apply_actions(self, actions: List[Dict[str, Any]]):
        """
        Apply agent actions to the world.
        
        Args:
            actions: List of agent actions
        """
        for action in actions:
            try:
                self._process_action(action)
            except Exception as e:
                logger.error(f"Failed to process action {action}: {e}")
    
    def _process_action(self, action: Dict[str, Any]):
        """Process a single agent action."""
        action_type = action.get("type", "unknown")
        
        if action_type == "consume_resource":
            self._consume_resource(action)
        elif action_type == "create_resource":
            self._create_resource(action)
        elif action_type == "move_resource":
            self._move_resource(action)
        elif action_type == "modify_environment":
            self._modify_environment(action)
        else:
            logger.warning(f"Unknown action type: {action_type}")
    
    def _consume_resource(self, action: Dict[str, Any]):
        """Process resource consumption action."""
        resource_name = action.get("resource_name")
        amount = action.get("amount", 0.0)
        
        if resource_name in self.resources:
            resource = self.resources[resource_name]
            resource.amount = max(0.0, resource.amount - amount)
            
            # Log event
            self._add_event(
                event_type="resource_consumed",
                location=resource.location,
                magnitude=amount,
                description=f"Resource '{resource_name}' consumed by {amount}"
            )
    
    def _create_resource(self, action: Dict[str, Any]):
        """Process resource creation action."""
        name = action.get("name")
        amount = action.get("amount", 1.0)
        location = action.get("location")
        
        if name and location:
            self.add_resource(
                name=name,
                amount=amount,
                max_amount=amount * 2,
                location=location
            )
    
    def _move_resource(self, action: Dict[str, Any]):
        """Process resource movement action."""
        resource_name = action.get("resource_name")
        new_location = action.get("new_location")
        
        if resource_name in self.resources and new_location:
            old_location = self.resources[resource_name].location
            self.resources[resource_name].location = new_location
            self.resource_locations[resource_name] = new_location
            
            # Update spatial grid
            self._update_spatial_grid(resource_name, new_location)
    
    def _modify_environment(self, action: Dict[str, Any]):
        """Process environment modification action."""
        factor = action.get("factor")
        change = action.get("change", 0.0)
        
        if factor in self.environmental_factors:
            self.environmental_factors[factor] += change
            self.environmental_factors[factor] = max(0.0, min(1.0, self.environmental_factors[factor]))
    
    def update_world(self, delta_time: float = 1.0):
        """
        Update the world state for one simulation step.
        
        Args:
            delta_time: Time step in simulation units
        """
        start_time = time.time()
        
        # Update current time
        self.current_time += delta_time
        self.simulation_step += 1
        
        # Update resources
        self._update_resources(delta_time)
        
        # Update environmental factors
        self._update_environmental_factors()
        
        # Clean up old events
        self._cleanup_old_events()
        
        # Update performance stats
        step_time = time.time() - start_time
        self._update_performance_stats(step_time)
        
        logger.debug(f"World updated - step {self.simulation_step}, time {self.current_time:.2f}")
    
    def _update_resources(self, delta_time: float):
        """Update all resources in the world."""
        # Collect resources to remove (can't modify dict during iteration)
        resources_to_remove = []
        
        for resource in self.resources.values():
            # Regeneration
            if resource.regeneration_rate > 0:
                resource.amount = min(
                    resource.max_amount,
                    resource.amount + resource.regeneration_rate * delta_time
                )
            
            # Decay
            if resource.decay_rate > 0:
                resource.amount = max(0.0, resource.amount - resource.decay_rate * delta_time)
            
            # Mark depleted resources for removal
            if resource.amount <= 0:
                resources_to_remove.append(resource.name)
        
        # Remove depleted resources after iteration
        for resource_name in resources_to_remove:
            self.remove_resource(resource_name)
    
    def _update_environmental_factors(self):
        """Update environmental factors based on world state."""
        # Simple environmental dynamics
        resource_density = len(self.resources) / self.max_resources
        
        # Temperature affects resource regeneration
        self.environmental_factors["temperature"] = 20.0 + resource_density * 10.0
        
        # Stability decreases with resource depletion
        total_resources = sum(r.amount for r in self.resources.values())
        max_total = sum(r.max_amount for r in self.resources.values())
        self.environmental_factors["stability"] = total_resources / max_total if max_total > 0 else 1.0
    
    def _cleanup_old_events(self):
        """Remove old events to prevent memory buildup."""
        if len(self.events) > self.max_events:
            # Keep only recent events
            cutoff_time = self.current_time - 3600.0  # Keep last hour
            self.events = [e for e in self.events if e.timestamp > cutoff_time]
    
    def _update_spatial_grid(self, resource_name: str, location: Tuple[float, float]):
        """Update spatial grid for efficient spatial queries."""
        grid_x = int(location[0] / self.grid_size)
        grid_y = int(location[1] / self.grid_size)
        
        # Remove from old grid cells
        for cell in self.spatial_grid.values():
            if resource_name in cell:
                cell.remove(resource_name)
        
        # Add to new grid cell
        if (grid_x, grid_y) in self.spatial_grid:
            self.spatial_grid[(grid_x, grid_y)].append(resource_name)
    
    def _add_event(self, event_type: str, location: Tuple[float, float], 
                   magnitude: float, description: str, metadata: Dict[str, Any] = None):
        """Add an event to the world."""
        if metadata is None:
            metadata = {}
        
        event = WorldEvent(
            event_id=f"event_{self.event_counter}",
            event_type=event_type,
            timestamp=self.current_time,
            location=location,
            magnitude=magnitude,
            description=description,
            metadata=metadata
        )
        
        self.events.append(event)
        self.event_counter += 1
        
        logger.debug(f"Added event: {event_type} at {location}")
    
    def _update_performance_stats(self, step_time: float):
        """Update performance statistics."""
        self.performance_stats["total_steps"] += 1
        
        # Update average step time
        total_steps = self.performance_stats["total_steps"]
        current_avg = self.performance_stats["average_step_time"]
        new_avg = ((current_avg * (total_steps - 1)) + step_time) / total_steps
        self.performance_stats["average_step_time"] = new_avg
    
    def get_world_state(self) -> Dict[str, Any]:
        """Get current world state summary."""
        return {
            "state": self.state.value,
            "current_time": self.current_time,
            "simulation_step": self.simulation_step,
            "num_resources": len(self.resources),
            "num_events": len(self.events),
            "environmental_factors": self.environmental_factors.copy(),
            "performance_stats": self.performance_stats.copy()
        }
    
    def save_world_state(self, filepath: str):
        """Save world state to file."""
        world_data = {
            "world_size": self.world_size,
            "current_time": self.current_time,
            "simulation_step": self.simulation_step,
            "resources": {
                name: {
                    "amount": resource.amount,
                    "max_amount": resource.max_amount,
                    "regeneration_rate": resource.regeneration_rate,
                    "decay_rate": resource.decay_rate,
                    "location": resource.location,
                    "metadata": resource.metadata
                }
                for name, resource in self.resources.items()
            },
            "events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "location": event.location,
                    "magnitude": event.magnitude,
                    "description": event.description,
                    "metadata": event.metadata
                }
                for event in self.events
            ],
            "environmental_factors": self.environmental_factors,
            "performance_stats": self.performance_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(world_data, f, indent=2)
        
        logger.info(f"World state saved to {filepath}")
    
    def load_world_state(self, filepath: str):
        """Load world state from file."""
        with open(filepath, 'r') as f:
            world_data = json.load(f)
        
        # Restore world state
        self.world_size = tuple(world_data["world_size"])
        self.current_time = world_data["current_time"]
        self.simulation_step = world_data["simulation_step"]
        
        # Restore resources
        self.resources = {}
        for name, data in world_data["resources"].items():
            resource = Resource(
                name=name,
                amount=data["amount"],
                max_amount=data["max_amount"],
                regeneration_rate=data["regeneration_rate"],
                decay_rate=data["decay_rate"],
                location=tuple(data["location"]),
                metadata=data["metadata"]
            )
            self.resources[name] = resource
        
        # Restore events
        self.events = []
        for event_data in world_data["events"]:
            event = WorldEvent(
                event_id=event_data["event_id"],
                event_type=event_data["event_type"],
                timestamp=event_data["timestamp"],
                location=tuple(event_data["location"]),
                magnitude=event_data["magnitude"],
                description=event_data["description"],
                metadata=event_data["metadata"]
            )
            self.events.append(event)
        
        # Restore environmental factors
        self.environmental_factors = world_data["environmental_factors"]
        self.performance_stats = world_data["performance_stats"]
        
        logger.info(f"World state loaded from {filepath}")


class AGICivilization:
    """
    Main AGI Civilization class that orchestrates world simulation.
    
    Features:
    - Persistent world state
    - Multi-agent coordination
    - Resource economy
    - Social norm formation
    - Emergence detection
    """
    
    def __init__(self, 
                 world_size: Tuple[float, float] = (100.0, 100.0),
                 max_agents: int = 100):
        """
        Initialize the AGI civilization.
        
        Args:
            world_size: Size of the world
            max_agents: Maximum number of agents
        """
        self.world_state = WorldState(world_size)
        self.agents = {}
        self.max_agents = max_agents
        
        # Social systems
        self.social_norms = SocialNormSystem()
        self.resource_economy = ResourceEconomy()
        
        # Emergence detection
        self.emergence_detector = EmergenceDetector()
        
        # Simulation state
        self.is_running = False
        self.simulation_thread = None
    
    def initialize_civilization(self, initial_resources: List[Dict[str, Any]] = None):
        """Initialize the civilization with world and agents."""
        logger.info("Initializing AGI civilization...")
        
        # Initialize world
        self.world_state.initialize_world(initial_resources)
        
        # Initialize social systems
        self.social_norms.initialize()
        self.resource_economy.initialize()
        
        logger.info("AGI civilization initialized")
    
    def simulate(self, spec: Dict[str, Any], steps: int = 1000) -> Dict[str, Any]:
        """
        Simulate the AGI civilization for specified steps.
        
        Args:
            spec: Civilization specification
            steps: Number of simulation steps
            
        Returns:
            Simulation results
        """
        logger.info(f"Starting AGI civilization simulation for {steps} steps")
        
        results = {
            "simulation_steps": steps,
            "world_evolution": [],
            "agent_behaviors": [],
            "emergence_events": [],
            "social_norms": [],
            "resource_dynamics": []
        }
        
        for step in range(steps):
            # Update world
            self.world_state.update_world()
            
            # Get agent perceptions
            perceptions = self._get_all_perceptions()
            
            # Agents decide actions
            actions = self._decide_actions(perceptions)
            
            # Execute actions
            self.world_state.apply_actions(actions)
            
            # Update social systems
            self._update_social_systems(actions)
            
            # Detect emergence
            emergence_events = self.emergence_detector.check(self.world_state)
            if emergence_events:
                results["emergence_events"].extend(emergence_events)
            
            # Record state
            if step % 100 == 0:  # Record every 100 steps
                results["world_evolution"].append(self.world_state.get_world_state())
                results["social_norms"].append(self.social_norms.get_norms())
                results["resource_dynamics"].append(self._get_resource_summary())
        
        logger.info("AGI civilization simulation completed")
        return results
    
    def _get_all_perceptions(self) -> List[AgentPerception]:
        """Get perceptions for all agents."""
        perceptions = []
        for agent_id, agent in self.agents.items():
            perception = self.world_state.get_perceptions(
                agent_id, 
                agent.get_location(),
                agent.get_perception_radius()
            )
            perceptions.append(perception)
        return perceptions
    
    def _decide_actions(self, perceptions: List[AgentPerception]) -> List[Dict[str, Any]]:
        """Decide actions for all agents based on perceptions."""
        actions = []
        for perception in perceptions:
            agent = self.agents.get(perception.agent_id)
            if agent:
                action = agent.decide_action(perception, self.social_norms, self.resource_economy)
                if action:
                    actions.append(action)
        return actions
    
    def _update_social_systems(self, actions: List[Dict[str, Any]]):
        """Update social systems based on actions."""
        self.social_norms.update(actions)
        self.resource_economy.update(actions)
    
    def _get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource state."""
        return {
            "total_resources": len(self.world_state.resources),
            "total_amount": sum(r.amount for r in self.world_state.resources.values()),
            "resource_types": list(set(r.name for r in self.world_state.resources.values()))
        }


# Import real implementations from separate modules
# These replace the placeholder classes
from iceburg.civilization.social_norms import SocialNormSystem
from iceburg.civilization.resource_economy import ResourceEconomy
from iceburg.civilization.emergence_detector import EmergenceDetector


# Re-export for backwards compatibility
__all__ = [
    'SimulationState',
    'Resource',
    'WorldEvent',
    'AgentPerception',
    'WorldState',
    'AGICivilization',
    'SocialNormSystem',
    'ResourceEconomy',
    'EmergenceDetector'
]

