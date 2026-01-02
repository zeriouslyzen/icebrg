"""
Swarm Orchestrator
Manages parallel execution of agentic swarms (Grok-style architecture)
"""

import asyncio
import logging
from typing import Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentTask:
    name: str
    func: Callable[..., Awaitable[Any]]
    args: tuple
    kwargs: Dict[str, Any]

class SwarmOrchestrator:
    """
    Orchestrates parallel agent execution (Swarm Pattern).
    Moves away from linear chains to concurrent "debate" streams.
    """
    
    def __init__(self):
        self.active_swarms: Dict[str, List[AgentTask]] = {}
        
    async def run_swarm(self, swarm_name: str, tasks: List[AgentTask]) -> Dict[str, Any]:
        """
        Run a swarm of agents in parallel.
        
        Args:
            swarm_name: Identifier for this swarm operation
            tasks: List of AgentTask objects to execute simultaneously
            
        Returns:
            Dict mapping agent names to their results
        """
        logger.info(f"🐝 Launching swarm '{swarm_name}' with {len(tasks)} agents")
        
        # Create coroutines for all tasks
        coroutines = []
        for task in tasks:
            logger.debug(f" - Preparing agent: {task.name}")
            coroutines.append(self._safe_execute(task))
            
        # Execute in parallel
        results_list = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Map results back to agent names
        results_map = {}
        for i, task in enumerate(tasks):
            result = results_list[i]
            if isinstance(result, Exception):
                logger.error(f"❌ Agent '{task.name}' failed: {result}")
                results_map[task.name] = {"error": str(result), "success": False}
            else:
                logger.info(f"✅ Agent '{task.name}' completed successfully")
                results_map[task.name] = {"output": result, "success": True}
                
        return results_map

    async def _safe_execute(self, task: AgentTask) -> Any:
        """Execute a single agent task with error boundary"""
        try:
            return await task.func(*task.args, **task.kwargs)
        except Exception as e:
            logger.exception(f"Error in agent '{task.name}'")
            raise e

# Global instance
swarm = SwarmOrchestrator()
