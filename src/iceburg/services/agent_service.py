"""
Agent Service
Microservice for agent management with dependency injection
"""

from typing import Any, Dict, Optional, List
from ..interfaces import IService, IAgent
from ..config import IceburgConfig


class AgentService(IService):
    """Service for agent management"""
    
    def __init__(self, config: IceburgConfig):
        self.name = "AgentService"
        self.config = config
        self.initialized = False
        self.running = False
        self.agents: Dict[str, IAgent] = {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the agent service"""
        self.config = config
        self.initialized = True
        return True
    
    def start(self) -> bool:
        """Start the agent service"""
        if not self.initialized:
            return False
        self.running = True
        return True
    
    def stop(self) -> bool:
        """Stop the agent service"""
        self.running = False
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health status"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "running": self.running,
            "healthy": self.initialized and self.running,
            "agents": list(self.agents.keys())
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return self.health_check()
    
    def register_agent(self, name: str, agent: IAgent) -> bool:
        """Register an agent with the service"""
        self.agents[name] = agent
        return True
    
    def run_agent(self, name: str, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Run an agent by name"""
        if name not in self.agents:
            raise ValueError(f"Agent {name} not found")
        return self.agents[name].run(query, context)
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    def find_agent_for_query(self, query: str) -> Optional[str]:
        """Find the best agent for a query"""
        for name, agent in self.agents.items():
            if agent.can_handle(query):
                return name
        return None

