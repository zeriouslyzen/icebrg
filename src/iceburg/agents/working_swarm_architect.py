"""
Working Swarm Architect - October 2025
=====================================

A fixed version that uses actual agent capabilities.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from ..micro_agent_swarm import get_swarm, MicroAgentSwarm
from ..specialized_agents import get_specialized_manager, SpecializedAgentManager

@dataclass
class WorkingSwarmResult:
    """Result from working swarm-based software generation"""
    architecture: Dict[str, Any]
    code_components: List[str] = field(default_factory=list)
    documentation: str = ""
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    agent_utilization: Dict[str, Any] = field(default_factory=dict)

class WorkingSwarmArchitect:
    """
    ICEBURG agent that uses the micro-agent swarm with CORRECT capabilities.
    """
    
    def __init__(self):
        self.swarm: Optional[MicroAgentSwarm] = None
        self.specialized_manager: Optional[SpecializedAgentManager] = None
        
    async def initialize(self):
        """Initialize the swarm architect"""
        self.swarm = await get_swarm()
        self.specialized_manager = await get_specialized_manager()

    async def generate_software_swarm(self, requirement: str, complexity: str = "medium") -> WorkingSwarmResult:
        """
        Generate software using the swarm with CORRECT capabilities.
        """
        
        start_time = time.time()
        
        # Use ACTUAL agent capabilities
        swarm_tasks = [
            {
                "type": "quick_analysis",
                "input": f"Quick analysis of: {requirement}",
                "requirements": ["quick_answers"],
                "priority": 9
            },
            {
                "type": "complex_reasoning",
                "input": f"Complex reasoning for: {requirement}",
                "requirements": ["complex_reasoning"],
                "priority": 8
            },
            {
                "type": "code_generation", 
                "input": f"Generate code for: {requirement}",
                "requirements": ["code_generation"],
                "priority": 7
            },
            {
                "type": "creative_documentation",
                "input": f"Create documentation for: {requirement}",
                "requirements": ["creative_writing"],
                "priority": 6
            },
            {
                "type": "data_analysis",
                "input": f"Analyze data requirements for: {requirement}",
                "requirements": ["data_analysis"],
                "priority": 5
            }
        ]
        
        # Process with swarm
        results = await self.swarm.process_parallel_tasks(swarm_tasks)
        
        # Combine results
        final_architecture = self._combine_swarm_results(results)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        swarm_status = self.swarm.get_swarm_status()
        
        result = WorkingSwarmResult(
            architecture=final_architecture,
            code_components=self._process_code_result(results[2] if len(results) > 2 else ""),
            documentation=self._process_documentation_result(results[3] if len(results) > 3 else ""),
            performance_metrics={
                "generation_time": generation_time,
                "swarm_tasks": len(swarm_tasks),
                "total_agents_used": swarm_status["total_agents"],
                "parallel_tasks": len(swarm_tasks)
            },
            agent_utilization=swarm_status["agents"]
        )
        
        return result
    
    def _combine_swarm_results(self, results) -> Dict[str, Any]:
        """Combine swarm results"""
        return {
            "name": "Working-Swarm Software",
            "description": "Software generated using working swarm intelligence",
            "quick_analysis": results[0] if len(results) > 0 else "",
            "complex_reasoning": results[1] if len(results) > 1 else "",
            "code": results[2] if len(results) > 2 else "",
            "documentation": results[3] if len(results) > 3 else "",
            "data_analysis": results[4] if len(results) > 4 else "",
            "generation_method": "working_swarm",
            "timestamp": time.time()
        }
    
    def _process_code_result(self, result) -> List[str]:
        """Process code generation result"""
        if isinstance(result, str) and result:
            return [{"component": "main", "language": "python", "type": "implementation", "code": result}]
        return []
    
    def _process_documentation_result(self, result) -> str:
        """Process documentation result"""
        return result if isinstance(result, str) else ""

# Global instance
_working_swarm_architect: Optional[WorkingSwarmArchitect] = None

async def get_working_swarm_architect() -> WorkingSwarmArchitect:
    """Get or create the global WorkingSwarmArchitect instance"""
    global _working_swarm_architect
    if _working_swarm_architect is None:
        _working_swarm_architect = WorkingSwarmArchitect()
        await _working_swarm_architect.initialize()
    return _working_swarm_architect

async def generate_software_with_working_swarm(requirement: str,
                                             complexity: str = "medium") -> WorkingSwarmResult:
    """Generate software using the working swarm architect"""
    architect = await get_working_swarm_architect()
    return await architect.generate_software_swarm(requirement, complexity)
