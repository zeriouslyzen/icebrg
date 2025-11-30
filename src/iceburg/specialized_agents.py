"""
ICEBURG Specialized Micro-Agents - October 2025
===============================================

Specialized micro-agents for different domains and tasks.
Each agent is optimized for specific use cases within ICEBURG.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .micro_agent_swarm import MicroAgentSwarm, SwarmTask
from .llm import chat_complete

@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    complexity_level: int  # 1-10
    speed_rating: int  # 1-10

class SpecializedAgentManager:
    """
    Manages specialized micro-agents for different domains.
    Each agent is optimized for specific tasks and can work in parallel.
    """
    
    def __init__(self):
        self.swarm = None
        self.agent_capabilities = self._define_agent_capabilities()
        self.domain_agents = self._create_domain_agents()
        
    def _define_agent_capabilities(self) -> Dict[str, AgentCapability]:
        """Define capabilities for each agent type"""
        return {
            "code_analysis": AgentCapability(
                name="Code Analysis",
                description="Analyzes code structure, patterns, and quality",
                input_types=["code", "repository", "file"],
                output_types=["analysis", "recommendations", "metrics"],
                complexity_level=8,
                speed_rating=7
            ),
            "research_synthesis": AgentCapability(
                name="Research Synthesis",
                description="Synthesizes research papers and scientific literature",
                input_types=["papers", "abstracts", "research_data"],
                output_types=["synthesis", "insights", "summary"],
                complexity_level=9,
                speed_rating=6
            ),
            "data_processing": AgentCapability(
                name="Data Processing",
                description="Processes and analyzes data sets",
                input_types=["csv", "json", "database", "api"],
                output_types=["processed_data", "insights", "visualizations"],
                complexity_level=7,
                speed_rating=8
            ),
            "creative_writing": AgentCapability(
                name="Creative Writing",
                description="Generates creative content and narratives",
                input_types=["prompts", "themes", "requirements"],
                output_types=["stories", "content", "narratives"],
                complexity_level=6,
                speed_rating=9
            ),
            "system_architecture": AgentCapability(
                name="System Architecture",
                description="Designs and analyzes system architectures",
                input_types=["requirements", "constraints", "specifications"],
                output_types=["architecture", "diagrams", "recommendations"],
                complexity_level=9,
                speed_rating=5
            ),
            "natural_language": AgentCapability(
                name="Natural Language Processing",
                description="Processes and understands natural language",
                input_types=["text", "speech", "documents"],
                output_types=["analysis", "translation", "summary"],
                complexity_level=8,
                speed_rating=7
            )
        }
    
    def _create_domain_agents(self) -> Dict[str, Dict[str, Any]]:
        """Create specialized agents for different domains"""
        return {
            "software_engineering": {
                "agents": ["code_analysis", "system_architecture", "creative_writing"],
                "primary_model": "gemma2:2b",
                "backup_model": "llama3.2:1b",
                "specializations": [
                    "code_review", "architecture_design", "documentation",
                    "testing_strategies", "performance_optimization"
                ]
            },
            "research_analysis": {
                "agents": ["research_synthesis", "data_processing", "natural_language"],
                "primary_model": "mini-ice:latest",
                "backup_model": "gemma2:2b",
                "specializations": [
                    "literature_review", "data_analysis", "hypothesis_generation",
                    "experimental_design", "statistical_analysis"
                ]
            },
            "creative_domains": {
                "agents": ["creative_writing", "natural_language", "data_processing"],
                "primary_model": "gemma2:2b",
                "backup_model": "tinyllama:1.1b",
                "specializations": [
                    "content_generation", "storytelling", "creative_problem_solving",
                    "artistic_analysis", "narrative_design"
                ]
            },
            "data_science": {
                "agents": ["data_processing", "research_synthesis", "system_architecture"],
                "primary_model": "llama3.2:1b",
                "backup_model": "gemma2:2b",
                "specializations": [
                    "data_analysis", "machine_learning", "statistical_modeling",
                    "visualization", "predictive_analytics"
                ]
            }
        }
    
    async def initialize(self):
        """Initialize the specialized agent manager"""
        self.swarm = MicroAgentSwarm(max_parallel_agents=8)
        await self.swarm.start_swarm()
    
    async def process_domain_task(self, domain: str, task_type: str, 
                                input_data: Any, requirements: List[str] = None) -> str:
        """Process a task using domain-specific agents"""
        if domain not in self.domain_agents:
            raise ValueError(f"Unknown domain: {domain}")
        
        domain_config = self.domain_agents[domain]
        
        # Add domain-specific requirements
        if requirements is None:
            requirements = []
        requirements.extend(domain_config["specializations"])
        
        # Submit task to swarm
        task_id = await self.swarm.submit_task(
            task_type=task_type,
            input_data=input_data,
            requirements=requirements,
            priority=7  # High priority for domain tasks
        )
        
        # Get result
        result = await self.swarm.get_task_result(task_id, timeout=60)
        return result
    
    async def process_parallel_domain_tasks(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Process multiple domain tasks in parallel"""
        swarm_tasks = []
        
        for task in tasks:
            domain = task.get("domain", "software_engineering")
            domain_config = self.domain_agents.get(domain, {})
            
            requirements = task.get("requirements", [])
            requirements.extend(domain_config.get("specializations", []))
            
            swarm_tasks.append({
                "type": task.get("type", "general"),
                "input": task.get("input", ""),
                "requirements": requirements,
                "priority": task.get("priority", 5)
            })
        
        return await self.swarm.process_parallel_tasks(swarm_tasks)
    
    def get_domain_capabilities(self, domain: str) -> Dict[str, Any]:
        """Get capabilities for a specific domain"""
        if domain not in self.domain_agents:
            return {}
        
        domain_config = self.domain_agents[domain]
        capabilities = []
        
        for agent_name in domain_config["agents"]:
            if agent_name in self.agent_capabilities:
                capabilities.append(self.agent_capabilities[agent_name])
        
        return {
            "domain": domain,
            "agents": domain_config["agents"],
            "capabilities": capabilities,
            "primary_model": domain_config["primary_model"],
            "backup_model": domain_config["backup_model"],
            "specializations": domain_config["specializations"]
        }
    
    def get_all_domains(self) -> List[str]:
        """Get list of all available domains"""
        return list(self.domain_agents.keys())
    
    async def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        if not self.swarm:
            return {}
        
        status = self.swarm.get_swarm_status()
        return {
            "total_agents": status["total_agents"],
            "active_agents": status["active_agents"],
            "available_agents": status["available_agents"],
            "total_tasks_processed": status["total_tasks_processed"],
            "average_response_time": status["average_response_time"],
            "agent_details": status["agents"]
        }

# Global specialized agent manager
_specialized_manager: Optional[SpecializedAgentManager] = None

async def get_specialized_manager() -> SpecializedAgentManager:
    """Get or create global specialized agent manager"""
    global _specialized_manager
    if _specialized_manager is None:
        _specialized_manager = SpecializedAgentManager()
        await _specialized_manager.initialize()
    return _specialized_manager

async def process_software_task(task_type: str, input_data: Any, 
                              requirements: List[str] = None) -> str:
    """Process a software engineering task"""
    manager = await get_specialized_manager()
    return await manager.process_domain_task("software_engineering", task_type, input_data, requirements)

async def process_research_task(task_type: str, input_data: Any, 
                              requirements: List[str] = None) -> str:
    """Process a research analysis task"""
    manager = await get_specialized_manager()
    return await manager.process_domain_task("research_analysis", task_type, input_data, requirements)

async def process_creative_task(task_type: str, input_data: Any, 
                              requirements: List[str] = None) -> str:
    """Process a creative task"""
    manager = await get_specialized_manager()
    return await manager.process_domain_task("creative_domains", task_type, input_data, requirements)

async def process_data_task(task_type: str, input_data: Any, 
                          requirements: List[str] = None) -> str:
    """Process a data science task"""
    manager = await get_specialized_manager()
    return await manager.process_domain_task("data_science", task_type, input_data, requirements)
