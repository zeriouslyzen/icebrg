"""
Civilization API Routes

REST API endpoints for controlling and monitoring the AGI civilization simulation.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import asyncio

from ..civilization.world_model import AGICivilization, WorldState
from ..civilization.persistent_agents import PersistentAgent, AgentRole, GoalPriority
from ..civilization.agent_society import MultiAgentSociety

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/civilization", tags=["civilization"])

# Global civilization instance
_civilization: Optional[AGICivilization] = None
_simulation_task: Optional[asyncio.Task] = None
_is_running: bool = False


# Pydantic models for request/response
class InitializeRequest(BaseModel):
    world_size: tuple = (100.0, 100.0)
    max_agents: int = 20
    initial_resources: List[Dict[str, Any]] = []


class AgentCreateRequest(BaseModel):
    agent_id: str
    role: str = "researcher"
    personality_traits: Optional[Dict[str, float]] = None


class SimulateRequest(BaseModel):
    steps: int = 10
    speed: float = 1.0


def get_civilization() -> AGICivilization:
    """Get or create the civilization instance."""
    global _civilization
    if _civilization is None:
        _civilization = AGICivilization()
    return _civilization


@router.post("/start")
async def start_civilization(request: InitializeRequest):
    """
    Initialize and start the AGI civilization simulation.
    
    Creates a new world with specified parameters and initializes
    default resources and systems.
    """
    global _civilization, _is_running
    
    try:
        # Create new civilization
        _civilization = AGICivilization(
            world_size=tuple(request.world_size),
            max_agents=request.max_agents
        )
        
        # Set up initial resources
        initial_resources = request.initial_resources or [
            {"name": "energy", "amount": 100.0, "max_amount": 500.0, "regeneration_rate": 2.0},
            {"name": "knowledge", "amount": 50.0, "max_amount": 300.0, "regeneration_rate": 1.0},
            {"name": "materials", "amount": 75.0, "max_amount": 400.0, "regeneration_rate": 1.5},
            {"name": "compute", "amount": 25.0, "max_amount": 200.0, "regeneration_rate": 0.5}
        ]
        
        _civilization.initialize_civilization(initial_resources)
        
        # Add some initial agents
        roles = [AgentRole.RESEARCHER, AgentRole.COORDINATOR, AgentRole.SPECIALIST, 
                 AgentRole.GENERALIST, AgentRole.LEADER]
        for i in range(min(5, request.max_agents)):
            agent = PersistentAgent(
                agent_id=f"agent_{i}",
                role=roles[i % len(roles)]
            )
            agent.goals.add_goal(
                description=f"Explore and contribute to civilization",
                priority=GoalPriority.MEDIUM
            )
            _civilization.agents[agent.agent_id] = agent
        
        _is_running = True
        
        logger.info(f"Civilization started with {len(_civilization.agents)} agents")
        
        return {
            "status": "started",
            "world_size": request.world_size,
            "max_agents": request.max_agents,
            "initial_agents": len(_civilization.agents),
            "initial_resources": len(_civilization.world_state.resources)
        }
        
    except Exception as e:
        logger.error(f"Failed to start civilization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_civilization():
    """Stop the currently running civilization simulation."""
    global _is_running, _simulation_task
    
    if _simulation_task and not _simulation_task.done():
        _simulation_task.cancel()
        _simulation_task = None
    
    _is_running = False
    
    return {"status": "stopped"}


@router.get("/status")
async def get_status():
    """
    Get the current status of the civilization simulation.
    
    Returns simulation state, agent count, resource summary, and performance metrics.
    """
    civ = get_civilization()
    
    if not civ.world_state:
        return {
            "status": "not_initialized",
            "is_running": False
        }
    
    world_state = civ.world_state.get_world_state()
    
    return {
        "status": "running" if _is_running else "paused",
        "is_running": _is_running,
        "simulation_step": world_state.get("simulation_step", 0),
        "agent_count": len(civ.agents),
        "resource_count": world_state.get("num_resources", 0),
        "environmental_factors": world_state.get("environmental_factors", {}),
        "performance": world_state.get("performance_stats", {})
    }


@router.get("/world")
async def get_world_state():
    """
    Get the complete world state including resources, events, and environmental factors.
    """
    civ = get_civilization()
    
    if not civ.world_state:
        raise HTTPException(status_code=400, detail="Civilization not initialized")
    
    world = civ.world_state
    
    resources = [
        {
            "name": r.name,
            "amount": r.amount,
            "max_amount": r.max_amount,
            "regeneration_rate": r.regeneration_rate,
            "decay_rate": r.decay_rate,
            "location": r.location
        }
        for r in world.resources.values()
    ]
    
    events = [
        {
            "event_id": e.event_id,
            "event_type": e.event_type,
            "timestamp": e.timestamp,
            "location": e.location,
            "magnitude": e.magnitude,
            "description": e.description
        }
        for e in world.events[-50:]  # Last 50 events
    ]
    
    return {
        "simulation_step": world.simulation_step,
        "world_size": world.world_size,
        "resources": resources,
        "events": events,
        "environmental_factors": world.environmental_factors.copy(),
        "grid_cells": len(world.spatial_grid)
    }


@router.get("/agents")
async def get_agents(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Get all agents with their current state.
    
    Returns agent profiles including role, location, goals, and memory stats.
    """
    civ = get_civilization()
    
    agents_list = list(civ.agents.values())[offset:offset + limit]
    
    return {
        "total": len(civ.agents),
        "offset": offset,
        "limit": limit,
        "agents": [
            {
                "agent_id": agent.agent_id,
                "role": agent.role.value,
                "location": agent.get_location(),
                "reputation": agent.reputation,
                "energy": agent.energy,
                "mood": agent.mood,
                "personality_traits": agent.personality_traits,
                "goals": {
                    "total": len(agent.goals.goals),
                    "active": agent.goals.goal_stats.get("active_goals", 0),
                    "completed": agent.goals.goal_stats.get("completed_goals", 0)
                },
                "memory_count": len(agent.memory.memories),
                "resources": agent.get_resources()
            }
            for agent in agents_list
        ]
    }


@router.get("/agents/{agent_id}")
async def get_agent_details(agent_id: str):
    """
    Get detailed information about a specific agent.
    
    Includes full profile, goals, recent memories, and performance history.
    """
    civ = get_civilization()
    
    if agent_id not in civ.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = civ.agents[agent_id]
    profile = agent.get_agent_profile()
    
    # Add more details
    profile["goals_list"] = [
        {
            "goal_id": g.goal_id,
            "description": g.description,
            "priority": g.priority.value,
            "progress": g.progress,
            "created_time": g.created_time
        }
        for g in list(agent.goals.goals.values())[:10]
    ]
    
    profile["recent_memories"] = [
        {
            "memory_id": m.memory_id,
            "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
            "memory_type": m.memory_type,
            "importance": m.importance,
            "timestamp": m.timestamp
        }
        for m in list(agent.memory.memories.values())[-10:]
    ]
    
    profile["decision_history"] = list(agent.decision_history)[-10:]
    profile["performance_history"] = list(agent.performance_history)[-20:]
    
    return profile


@router.post("/agents")
async def create_agent(request: AgentCreateRequest):
    """
    Add a new agent to the civilization.
    """
    civ = get_civilization()
    
    if request.agent_id in civ.agents:
        raise HTTPException(status_code=400, detail=f"Agent {request.agent_id} already exists")
    
    if len(civ.agents) >= civ.max_agents:
        raise HTTPException(status_code=400, detail="Maximum agent limit reached")
    
    # Map role string to enum
    role_map = {
        "researcher": AgentRole.RESEARCHER,
        "coordinator": AgentRole.COORDINATOR,
        "specialist": AgentRole.SPECIALIST,
        "generalist": AgentRole.GENERALIST,
        "leader": AgentRole.LEADER,
        "follower": AgentRole.FOLLOWER
    }
    role = role_map.get(request.role.lower(), AgentRole.GENERALIST)
    
    agent = PersistentAgent(
        agent_id=request.agent_id,
        role=role,
        personality_traits=request.personality_traits
    )
    
    agent.goals.add_goal(
        description="Integrate into civilization",
        priority=GoalPriority.HIGH
    )
    
    civ.agents[request.agent_id] = agent
    
    return {
        "status": "created",
        "agent_id": request.agent_id,
        "role": role.value
    }


@router.delete("/agents/{agent_id}")
async def remove_agent(agent_id: str):
    """Remove an agent from the civilization."""
    civ = get_civilization()
    
    if agent_id not in civ.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    del civ.agents[agent_id]
    
    return {"status": "removed", "agent_id": agent_id}


@router.post("/step")
async def simulation_step(request: SimulateRequest):
    """
    Advance the simulation by a specified number of steps.
    
    Returns summary of what happened during the simulation.
    """
    civ = get_civilization()
    
    if not civ.world_state:
        raise HTTPException(status_code=400, detail="Civilization not initialized")
    
    try:
        results = civ.simulate(spec={}, steps=request.steps)
        
        return {
            "status": "completed",
            "steps_executed": request.steps,
            "current_step": civ.world_state.simulation_step,
            "emergence_events_count": len(results.get("emergence_events", [])),
            "world_evolution_snapshots": len(results.get("world_evolution", [])),
            "agent_count": len(civ.agents)
        }
        
    except Exception as e:
        logger.error(f"Simulation step failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_emergence_events(
    limit: int = Query(50, ge=1, le=200),
    min_severity: int = Query(1, ge=1, le=5)
):
    """
    Get emergence events detected in the simulation.
    """
    civ = get_civilization()
    
    events = civ.emergence_detector.get_events(limit=limit, min_severity=min_severity)
    
    return {
        "total": len(civ.emergence_detector.events),
        "filtered_count": len(events),
        "events": events
    }


@router.get("/norms")
async def get_social_norms():
    """
    Get all active social norms in the civilization.
    """
    civ = get_civilization()
    
    norms = civ.social_norms.get_norms()
    stats = civ.social_norms.get_stats()
    violations = civ.social_norms.get_violations(limit=20)
    
    return {
        "norms": norms,
        "stats": stats,
        "recent_violations": violations
    }


@router.get("/economy")
async def get_economy_stats():
    """
    Get economic statistics including market prices and wealth distribution.
    """
    civ = get_civilization()
    
    prices = civ.resource_economy.get_market_prices()
    stats = civ.resource_economy.get_stats()
    wealth_dist = civ.resource_economy.get_wealth_distribution()[:20]  # Top 20
    trades = civ.resource_economy.get_trade_history(limit=20)
    
    return {
        "market_prices": prices,
        "stats": stats,
        "wealth_distribution": wealth_dist,
        "recent_trades": trades
    }


@router.get("/patterns")
async def get_behavior_patterns(min_frequency: int = Query(2, ge=1)):
    """
    Get detected behavior patterns from the emergence detector.
    """
    civ = get_civilization()
    
    patterns = civ.emergence_detector.get_patterns(min_frequency=min_frequency)
    summary = civ.emergence_detector.get_summary()
    
    return {
        "patterns": patterns,
        "summary": summary
    }


@router.get("/society")
async def get_society_stats():
    """
    Get society-level statistics from the social learning system.
    """
    civ = get_civilization()
    
    if hasattr(civ, 'agent_society') and civ.agent_society:
        stats = civ.agent_society.get_society_stats()
    else:
        # Fallback if society not initialized
        stats = {
            "society_stats": {"total_agents": len(civ.agents)},
            "social_learning": {},
            "agent_profiles": []
        }
    
    return stats
