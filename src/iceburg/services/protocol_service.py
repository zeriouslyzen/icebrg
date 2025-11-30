"""
Protocol Service
Microservice for protocol execution with dependency injection
"""

from typing import Any, Dict, Optional, List
from ..interfaces import IService, IProtocol
from ..config import IceburgConfig


class ProtocolService(IService):
    """Service for protocol execution"""
    
    def __init__(self, config: IceburgConfig):
        self.name = "ProtocolService"
        self.config = config
        self.initialized = False
        self.running = False
        self.protocols: Dict[str, IProtocol] = {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the protocol service"""
        self.config = config
        self.initialized = True
        return True
    
    def start(self) -> bool:
        """Start the protocol service"""
        if not self.initialized:
            return False
        self.running = True
        return True
    
    def stop(self) -> bool:
        """Stop the protocol service"""
        self.running = False
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health status"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "running": self.running,
            "healthy": self.initialized and self.running,
            "protocols": list(self.protocols.keys())
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return self.health_check()
    
    def register_protocol(self, name: str, protocol: IProtocol) -> bool:
        """Register a protocol with the service"""
        self.protocols[name] = protocol
        return True
    
    def execute_protocol(self, name: str, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a protocol by name"""
        if name not in self.protocols:
            raise ValueError(f"Protocol {name} not found")
        return self.protocols[name].execute(query, context)
    
    def get_available_protocols(self) -> List[str]:
        """Get list of available protocols"""
        return list(self.protocols.keys())

