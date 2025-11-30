"""
Service Interface Definitions
Defines contracts for service-oriented architecture with dependency injection
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass


class IService(ABC):
    """Base interface for all services in ICEBURG"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the service with configuration"""
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """Start the service"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the service"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check service health status"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        pass


@dataclass
class ServiceBase(IService):
    """Base implementation for services with common functionality"""
    
    name: str
    config: Dict[str, Any]
    initialized: bool = False
    running: bool = False
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the service with configuration"""
        self.config = config
        self.initialized = True
        return True
    
    def start(self) -> bool:
        """Start the service"""
        if not self.initialized:
            return False
        self.running = True
        return True
    
    def stop(self) -> bool:
        """Stop the service"""
        self.running = False
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health status"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "running": self.running,
            "healthy": self.initialized and self.running
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return self.health_check()

