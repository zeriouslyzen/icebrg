"""
Pre-Warmed Agent Pool
Pre-loads all agents into memory at startup for instant responses.
Based on always-on AI architecture patterns from frontier AI research.
"""

from __future__ import annotations

import asyncio
import logging
import time
import os
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from threading import Lock
import weakref

logger = logging.getLogger(__name__)


class PreWarmedAgentPool:
    """
    Pre-warmed agent pool that maintains agents ready for instant responses.
    
    Architecture:
    - Initialize all agents at startup
    - Load models into memory
    - Maintain agent pool ready for instant response
    - Connection pooling for agent reuse
    - Health monitoring and auto-recovery
    """
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.agents: Dict[str, Any] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.warmup_complete = False
        self.warmup_lock = Lock()
        self.health_check_interval = 30  # seconds
        
        # Performance tracking
        self.stats = {
            "agents_warmed": 0,
            "warmup_time_seconds": 0,
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "health_checks": 0,
            "recoveries": 0
        }
        
        logger.info("PreWarmedAgentPool initialized")
    
    async def warmup(self):
        """Pre-warm all agents - load models into memory"""
        if self.warmup_complete:
            logger.info("Agents already warmed up")
            return
        
        with self.warmup_lock:
            if self.warmup_complete:
                return
            
            logger.info("Starting agent warmup...")
            start_time = time.time()
            
            try:
                # Initialize core agents
                await self._warmup_core_agents()
                
                # Initialize specialized agents
                await self._warmup_specialized_agents()
                
                # Initialize optional agents
                await self._warmup_optional_agents()
                
                # Mark warmup as complete
                self.warmup_complete = True
                warmup_time = time.time() - start_time
                self.stats["warmup_time_seconds"] = warmup_time
                self.stats["agents_warmed"] = len(self.agents)
                
                logger.info(f"Agent warmup complete: {len(self.agents)} agents warmed in {warmup_time:.2f}s")
                
                # Start health monitoring
                asyncio.create_task(self._health_monitoring_loop())
                
            except Exception as e:
                logger.error(f"Error during agent warmup: {e}", exc_info=True)
                raise
    
    async def _warmup_core_agents(self):
        """Warm up core agents (Surveyor, Dissident, Synthesist, Oracle)"""
        # Only warm up Surveyor by default - other agents loaded on-demand
        # Set ICEBURG_WARM_ALL_AGENTS=1 to warm up all agents
        warm_all = os.getenv("ICEBURG_WARM_ALL_AGENTS", "0") == "1"
        if warm_all:
            core_agents = ["surveyor", "dissident", "synthesist", "oracle"]
        else:
            core_agents = ["surveyor"]  # Only warm up Surveyor for chat mode
        
        for agent_id in core_agents:
            try:
                await self._warmup_agent(agent_id)
            except Exception as e:
                logger.error(f"Error warming up {agent_id}: {e}", exc_info=True)
                # Continue with other agents even if one fails
    
    async def _warmup_specialized_agents(self):
        """Warm up specialized agents (Archaeologist, Supervisor, Scrutineer)"""
        # Skip specialized agents unless explicitly enabled (saves startup time)
        warm_all = os.getenv("ICEBURG_WARM_ALL_AGENTS", "0") == "1"
        if not warm_all:
            logger.info("Skipping specialized agent warmup (only Surveyor warmed by default)")
            return
        
        specialized_agents = ["archaeologist", "supervisor", "scrutineer"]
        
        for agent_id in specialized_agents:
            try:
                await self._warmup_agent(agent_id)
            except Exception as e:
                logger.error(f"Error warming up {agent_id}: {e}", exc_info=True)
    
    async def _warmup_optional_agents(self):
        """Warm up optional agents (Architect, Weaver, Scribe)"""
        # Skip optional agents unless explicitly enabled (saves startup time)
        warm_all = os.getenv("ICEBURG_WARM_ALL_AGENTS", "0") == "1"
        if not warm_all:
            logger.info("Skipping optional agent warmup (only Surveyor warmed by default)")
            return
        
        optional_agents = ["architect", "weaver", "scribe"]
        
        for agent_id in optional_agents:
            try:
                await self._warmup_agent(agent_id)
            except Exception as e:
                logger.warning(f"Optional agent {agent_id} warmup failed: {e}")
                # Optional agents can fail without breaking warmup
    
    async def _warmup_agent(self, agent_id: str):
        """Warm up a single agent"""
        try:
            logger.info(f"Warming up agent: {agent_id}")
            
            # Create agent instance (lazy import to avoid circular dependencies)
            agent_instance = await self._create_agent_instance(agent_id)
            
            if agent_instance:
                # Store agent instance
                self.agents[agent_id] = agent_instance
                
                # Initialize agent state
                self.agent_states[agent_id] = {
                    "status": "ready",
                    "warmed_at": datetime.now().isoformat(),
                    "last_used": None,
                    "request_count": 0,
                    "error_count": 0,
                    "health": "healthy"
                }
                
                logger.info(f"Agent {agent_id} warmed up successfully")
            else:
                logger.warning(f"Could not create agent instance for {agent_id}")
                
        except Exception as e:
            logger.error(f"Error warming up agent {agent_id}: {e}", exc_info=True)
            # Mark agent as failed but continue
            self.agent_states[agent_id] = {
                "status": "failed",
                "warmed_at": None,
                "error": str(e),
                "health": "unhealthy"
            }
    
    async def _create_agent_instance(self, agent_id: str) -> Optional[Any]:
        """Create agent instance (placeholder - would integrate with actual agents)"""
        try:
            # This is a placeholder - actual implementation would create real agent instances
            # For now, just return a dict representing the agent
            return {
                "agent_id": agent_id,
                "status": "ready",
                "warmed_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating agent instance for {agent_id}: {e}", exc_info=True)
            return None
    
    async def _health_monitoring_loop(self):
        """Continuously monitor agent health and recover if needed"""
        logger.info("Health monitoring loop started")
        
        while self.warmup_complete:
            try:
                await self._check_agent_health()
                self.stats["health_checks"] += 1
                
                # Wait before next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_agent_health(self):
        """Check health of all agents and recover if needed"""
        for agent_id, state in self.agent_states.items():
            try:
                # Check if agent is healthy
                if state.get("health") == "unhealthy":
                    # Try to recover
                    await self._recover_agent(agent_id)
                    self.stats["recoveries"] += 1
                
            except Exception as e:
                logger.error(f"Error checking health for {agent_id}: {e}", exc_info=True)
    
    async def _recover_agent(self, agent_id: str):
        """Recover a failed agent"""
        try:
            logger.info(f"Attempting to recover agent: {agent_id}")
            
            # Remove failed agent
            if agent_id in self.agents:
                del self.agents[agent_id]
            
            # Re-warmup agent
            await self._warmup_agent(agent_id)
            
            logger.info(f"Agent {agent_id} recovered successfully")
            
        except Exception as e:
            logger.error(f"Error recovering agent {agent_id}: {e}", exc_info=True)
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Get pre-warmed agent (instant)"""
        if not self.warmup_complete:
            logger.warning("Agents not warmed up yet")
            return None
        
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found in pool")
            return None
        
        # Update agent state
        if agent_id in self.agent_states:
            self.agent_states[agent_id]["last_used"] = datetime.now().isoformat()
            self.agent_states[agent_id]["request_count"] = self.agent_states[agent_id].get("request_count", 0) + 1
        
        self.stats["total_requests"] += 1
        self.stats["cache_hits"] += 1
        
        return self.agents.get(agent_id)
    
    def is_agent_ready(self, agent_id: str) -> bool:
        """Check if agent is ready"""
        if not self.warmup_complete:
            return False
        
        if agent_id not in self.agents:
            return False
        
        state = self.agent_states.get(agent_id, {})
        return state.get("status") == "ready" and state.get("health") == "healthy"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            **self.stats,
            "warmup_complete": self.warmup_complete,
            "agents_count": len(self.agents),
            "ready_agents": sum(1 for state in self.agent_states.values() if state.get("status") == "ready"),
            "agent_states": self.agent_states
        }
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent state"""
        return self.agent_states.get(agent_id)

