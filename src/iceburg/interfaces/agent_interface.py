"""
Agent Interface Definitions
Defines contracts for agent execution in service-oriented architecture
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class IAgent(ABC):
    """Base interface for all agents in ICEBURG"""
    
    @abstractmethod
    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Run agent with query and optional context"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        pass
    
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Check if agent can handle the query"""
        pass


class AgentBase(IAgent):
    """Base implementation for agents"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        return self.capabilities
    
    def can_handle(self, query: str) -> bool:
        """Check if agent can handle the query"""
        # Default implementation - can be overridden
        return True
    
    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Run agent with query and optional context"""
        raise NotImplementedError("Subclasses must implement run method")

