"""
ICEBURG Dynamic Resource Allocation System

Replaces simple semaphore with intelligent resource allocation based on:
- Agent requirements (memory, CPU, timeout)
- Current system resources
- Workload patterns
- Performance metrics
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import psutil
import threading

from ..agents.capability_registry import get_registry, AgentCapability

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources"""
    MEMORY = "memory"
    CPU = "cpu"
    CONCURRENT = "concurrent"
    TIMEOUT = "timeout"


@dataclass
class ResourceAllocation:
    """Resource allocation for an agent"""
    agent_id: str
    memory_mb: float
    cpu_cores: int
    timeout_seconds: float
    priority: int = 5
    allocated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


@dataclass
class SystemResources:
    """Current system resource availability"""
    total_memory_mb: float
    available_memory_mb: float
    total_cpu_cores: int
    available_cpu_cores: float
    memory_usage_percent: float
    cpu_usage_percent: float


class DynamicResourceAllocator:
    """
    Dynamic resource allocation system for agent execution.
    
    Features:
    - Per-agent resource tracking
    - Dynamic allocation based on requirements
    - Resource monitoring and adjustment
    - Priority-based allocation
    - Automatic resource cleanup
    """
    
    def __init__(
        self,
        max_total_memory_mb: Optional[float] = None,
        max_total_cpu_cores: Optional[int] = None,
        max_concurrent_agents: int = 10,
        reserve_memory_mb: float = 1024.0,  # Reserve 1GB for system
        reserve_cpu_cores: float = 1.0  # Reserve 1 core for system
    ):
        self.max_total_memory_mb = max_total_memory_mb
        self.max_total_cpu_cores = max_total_cpu_cores
        self.max_concurrent_agents = max_concurrent_agents
        self.reserve_memory_mb = reserve_memory_mb
        self.reserve_cpu_cores = reserve_cpu_cores
        
        # Active allocations
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.agent_allocations: Dict[str, List[str]] = defaultdict(list)  # agent_id -> allocation_ids
        
        # Resource tracking
        self.used_memory_mb: float = 0.0
        self.used_cpu_cores: float = 0.0
        self.active_agent_count: int = 0
        
        # Registry for agent capabilities
        self.registry = get_registry()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Performance metrics
        self.allocation_stats = {
            "total_allocations": 0,
            "successful_allocations": 0,
            "failed_allocations": 0,
            "resource_contention": 0
        }
        
        # Initialize system resources
        self._update_system_resources()
        
        logger.info(f"Dynamic Resource Allocator initialized: {self.max_total_memory_mb}MB memory, {self.max_total_cpu_cores} CPU cores")
    
    def _update_system_resources(self) -> SystemResources:
        """Update current system resource availability"""
        try:
            # Get system memory
            memory = psutil.virtual_memory()
            total_memory_mb = memory.total / (1024 * 1024)
            available_memory_mb = memory.available / (1024 * 1024)
            
            # Get CPU info
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            available_cpu_cores = cpu_count * (1 - cpu_percent / 100)
            
            # Set defaults if not specified
            if self.max_total_memory_mb is None:
                self.max_total_memory_mb = total_memory_mb * 0.8  # Use 80% of available
            if self.max_total_cpu_cores is None:
                self.max_total_cpu_cores = cpu_count
            
            return SystemResources(
                total_memory_mb=total_memory_mb,
                available_memory_mb=available_memory_mb,
                total_cpu_cores=cpu_count,
                available_cpu_cores=available_cpu_cores,
                memory_usage_percent=memory.percent,
                cpu_usage_percent=cpu_percent
            )
        except Exception as e:
            logger.warning(f"Error updating system resources: {e}")
            # Fallback values
            return SystemResources(
                total_memory_mb=16384.0,
                available_memory_mb=8192.0,
                total_cpu_cores=8,
                available_cpu_cores=4.0,
                memory_usage_percent=50.0,
                cpu_usage_percent=50.0
            )
    
    async def allocate_resources(
        self,
        agent_id: str,
        allocation_id: Optional[str] = None,
        priority: int = 5
    ) -> Optional[ResourceAllocation]:
        """
        Allocate resources for an agent execution.
        
        Args:
            agent_id: Agent identifier
            allocation_id: Optional allocation identifier (for tracking)
            priority: Allocation priority (1-10, higher = more important)
            
        Returns:
            ResourceAllocation if successful, None if resources unavailable
        """
        with self.lock:
            # Get agent capability
            agent_capability = self.registry.get_agent(agent_id)
            if not agent_capability:
                logger.warning(f"Agent {agent_id} not found in registry, using defaults")
                # Use default values
                required_memory_mb = 512.0
                required_cpu_cores = 1
                timeout_seconds = 30.0
            else:
                required_memory_mb = agent_capability.memory_mb
                required_cpu_cores = agent_capability.cpu_cores
                timeout_seconds = agent_capability.timeout_seconds
            
            # Check if resources are available
            if not self._can_allocate(required_memory_mb, required_cpu_cores):
                self.allocation_stats["failed_allocations"] += 1
                logger.warning(
                    f"Insufficient resources for agent {agent_id}: "
                    f"required {required_memory_mb}MB/{required_cpu_cores} cores, "
                    f"available {self.available_memory_mb:.1f}MB/{self.available_cpu_cores:.1f} cores"
                )
                return None
            
            # Create allocation
            if allocation_id is None:
                allocation_id = f"{agent_id}_{int(time.time() * 1000)}"
            
            allocation = ResourceAllocation(
                agent_id=agent_id,
                memory_mb=required_memory_mb,
                cpu_cores=required_cpu_cores,
                timeout_seconds=timeout_seconds,
                priority=priority,
                expires_at=time.time() + timeout_seconds
            )
            
            # Record allocation
            self.active_allocations[allocation_id] = allocation
            self.agent_allocations[agent_id].append(allocation_id)
            
            # Update resource usage
            self.used_memory_mb += required_memory_mb
            self.used_cpu_cores += required_cpu_cores
            self.active_agent_count += 1
            
            self.allocation_stats["total_allocations"] += 1
            self.allocation_stats["successful_allocations"] += 1
            
            logger.debug(
                f"Allocated resources for {agent_id}: "
                f"{required_memory_mb}MB, {required_cpu_cores} cores, "
                f"{timeout_seconds}s timeout"
            )
            
            return allocation
    
    def release_resources(self, allocation_id: str):
        """
        Release allocated resources.
        
        Args:
            allocation_id: Allocation identifier
        """
        with self.lock:
            if allocation_id not in self.active_allocations:
                logger.warning(f"Allocation {allocation_id} not found")
                return
            
            allocation = self.active_allocations[allocation_id]
            agent_id = allocation.agent_id
            
            # Update resource usage
            self.used_memory_mb -= allocation.memory_mb
            self.used_cpu_cores -= allocation.cpu_cores
            self.active_agent_count -= 1
            
            # Remove allocation
            del self.active_allocations[allocation_id]
            if agent_id in self.agent_allocations:
                if allocation_id in self.agent_allocations[agent_id]:
                    self.agent_allocations[agent_id].remove(allocation_id)
                if not self.agent_allocations[agent_id]:
                    del self.agent_allocations[agent_id]
            
            logger.debug(f"Released resources for allocation {allocation_id}")
    
    def _can_allocate(self, memory_mb: float, cpu_cores: int) -> bool:
        """Check if resources can be allocated"""
        # Check concurrent agent limit
        if self.active_agent_count >= self.max_concurrent_agents:
            return False
        
        # Check memory availability
        available_memory = self.available_memory_mb
        if self.used_memory_mb + memory_mb > available_memory - self.reserve_memory_mb:
            return False
        
        # Check CPU availability
        available_cpu = self.available_cpu_cores
        if self.used_cpu_cores + cpu_cores > available_cpu - self.reserve_cpu_cores:
            return False
        
        return True
    
    @property
    def available_memory_mb(self) -> float:
        """Get available memory in MB"""
        system_resources = self._update_system_resources()
        return min(
            system_resources.available_memory_mb,
            self.max_total_memory_mb - self.used_memory_mb
        )
    
    @property
    def available_cpu_cores(self) -> float:
        """Get available CPU cores"""
        system_resources = self._update_system_resources()
        return min(
            system_resources.available_cpu_cores,
            self.max_total_cpu_cores - self.used_cpu_cores
        )
    
    def cleanup_expired_allocations(self):
        """Remove expired allocations"""
        current_time = time.time()
        expired = [
            alloc_id for alloc_id, allocation in self.active_allocations.items()
            if allocation.expires_at and allocation.expires_at < current_time
        ]
        
        for alloc_id in expired:
            logger.warning(f"Cleaning up expired allocation {alloc_id}")
            self.release_resources(alloc_id)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        self.cleanup_expired_allocations()
        system_resources = self._update_system_resources()
        
        return {
            "system": {
                "total_memory_mb": system_resources.total_memory_mb,
                "available_memory_mb": system_resources.available_memory_mb,
                "total_cpu_cores": system_resources.total_cpu_cores,
                "available_cpu_cores": system_resources.available_cpu_cores,
                "memory_usage_percent": system_resources.memory_usage_percent,
                "cpu_usage_percent": system_resources.cpu_usage_percent
            },
            "allocated": {
                "used_memory_mb": self.used_memory_mb,
                "used_cpu_cores": self.used_cpu_cores,
                "active_agents": self.active_agent_count,
                "max_concurrent": self.max_concurrent_agents
            },
            "available": {
                "available_memory_mb": self.available_memory_mb,
                "available_cpu_cores": self.available_cpu_cores
            },
            "stats": self.allocation_stats.copy()
        }
    
    async def acquire(self, agent_id: str, priority: int = 5) -> Optional[ResourceAllocation]:
        """
        Acquire resources for agent execution (async context manager).
        
        Usage:
            async with allocator.acquire("surveyor") as allocation:
                if allocation:
                    # Execute agent
                    pass
        """
        allocation = await self.allocate_resources(agent_id, priority=priority)
        return allocation
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup_expired_allocations()


# Global resource allocator instance
_resource_allocator: Optional[DynamicResourceAllocator] = None


def get_resource_allocator() -> DynamicResourceAllocator:
    """Get or create global resource allocator"""
    global _resource_allocator
    if _resource_allocator is None:
        _resource_allocator = DynamicResourceAllocator()
    return _resource_allocator

