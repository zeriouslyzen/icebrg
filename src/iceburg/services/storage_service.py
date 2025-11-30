"""
Storage Service
Microservice for storage management with dependency injection
"""

from typing import Any, Dict, Optional, List
from ..interfaces import IService, IStorage
from ..config import IceburgConfig


class StorageService(IService):
    """Service for storage management"""
    
    def __init__(self, config: IceburgConfig):
        self.name = "StorageService"
        self.config = config
        self.initialized = False
        self.running = False
        self.storages: Dict[str, IStorage] = {}
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the storage service"""
        self.config = config
        self.initialized = True
        return True
    
    def start(self) -> bool:
        """Start the storage service"""
        if not self.initialized:
            return False
        self.running = True
        return True
    
    def stop(self) -> bool:
        """Stop the storage service"""
        self.running = False
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health status"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "running": self.running,
            "healthy": self.initialized and self.running,
            "storages": list(self.storages.keys())
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return self.health_check()
    
    def register_storage(self, name: str, storage: IStorage) -> bool:
        """Register a storage with the service"""
        self.storages[name] = storage
        return True
    
    def get_storage(self, name: str) -> Optional[IStorage]:
        """Get a storage by name"""
        return self.storages.get(name)
    
    def get_available_storages(self) -> List[str]:
        """Get list of available storages"""
        return list(self.storages.keys())

