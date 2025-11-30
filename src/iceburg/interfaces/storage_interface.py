"""
Storage Interface Definitions
Defines contracts for storage services in service-oriented architecture
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class IStorage(ABC):
    """Base interface for storage services"""
    
    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store value with key and optional metadata"""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value by key"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value by key"""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search storage with query"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass


class StorageBase(IStorage):
    """Base implementation for storage services"""
    
    def __init__(self, name: str):
        self.name = name
        self._storage: Dict[str, Any] = {}
    
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store value with key and optional metadata"""
        self._storage[key] = {"value": value, "metadata": metadata or {}}
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value by key"""
        if key in self._storage:
            return self._storage[key]["value"]
        return None
    
    def delete(self, key: str) -> bool:
        """Delete value by key"""
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search storage with query"""
        # Default implementation - can be overridden
        results = []
        for key, data in self._storage.items():
            if query.lower() in key.lower():
                results.append({"key": key, "value": data["value"], "metadata": data["metadata"]})
                if len(results) >= limit:
                    break
        return results
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._storage

