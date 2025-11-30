"""
Protocol Interface Definitions
Defines contracts for protocol execution in microservices architecture
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable


class IProtocol(ABC):
    """Base interface for protocol execution"""
    
    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute protocol with query and optional context"""
        pass
    
    @abstractmethod
    def validate(self, query: str) -> bool:
        """Validate if protocol can handle the query"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of protocol capabilities"""
        pass


class ProtocolBase(IProtocol):
    """Base implementation for protocols"""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
    
    def validate(self, query: str) -> bool:
        """Validate if protocol can handle the query"""
        # Default implementation - can be overridden
        return True
    
    def get_capabilities(self) -> List[str]:
        """Get list of protocol capabilities"""
        return self.capabilities
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute protocol with query and optional context"""
        raise NotImplementedError("Subclasses must implement execute method")

