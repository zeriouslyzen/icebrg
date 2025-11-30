"""
Redis Coordinator for Distributed ICEBURG Processing
Implements cluster coordination, sharding, and distributed execution.
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import redis
    from redis import Redis, RedisCluster
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None
    RedisCluster = None

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    node_id: str
    host: str
    port: int
    status: WorkerStatus = WorkerStatus.IDLE
    load: float = 0.0  # 0.0 to 1.0
    capabilities: List[str] = field(default_factory=list)
    last_heartbeat: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskShard:
    """Represents a task shard for distributed processing."""
    shard_id: str
    task_type: str
    data: Dict[str, Any]
    priority: int = 0
    timeout: float = 300.0
    assigned_worker: Optional[str] = None
    created_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedICEBURG:
    """
    Distributed ICEBURG coordinator using Redis for cluster management.
    
    Features:
    - Redis cluster coordination
    - Intelligent task sharding
    - Load balancing across workers
    - Fault tolerance and recovery
    - Performance monitoring
    """
    
    def __init__(self, 
        redis_nodes: List[Tuple[str, int]] = None,
                 cluster_mode: bool = False,
                 max_workers: int = 10):
        """
        Initialize the distributed ICEBURG coordinator.
        
        Args:
            redis_nodes: List of (host, port) tuples for Redis nodes
            cluster_mode: Whether to use Redis cluster mode
            max_workers: Maximum number of workers to manage
        """
        self.redis_nodes = redis_nodes or [("os.getenv("HOST", "localhost")", 6379)]
        self.cluster_mode = cluster_mode
        self.max_workers = max_workers
        
        # Redis connection
        self.redis = None
        self.redis_connected = False
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_counter = 0
        
        # Task management
        self.task_queue = "iceburg:tasks"
        self.result_queue = "iceburg:results"
        self.worker_registry = "iceburg:workers"
        
        # Performance tracking
        self.performance_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_processing_time": 0.0,
            "cluster_utilization": 0.0
        }
        
        # Initialize Redis connection
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory coordination")
            return
        
        try:
            if self.cluster_mode:
                self.redis = RedisCluster(
                    startup_nodes=[{"host": host, "port": port} for host, port in self.redis_nodes],
                    decode_responses=True
                )
            else:
                # Use first node for single Redis instance
                host, port = self.redis_nodes[0]
                self.redis = Redis(host=host, port=port, decode_responses=True)
            
            # Test connection
            self.redis.ping()
            self.redis_connected = True
            logger.info(f"Connected to Redis {'cluster' if self.cluster_mode else 'instance'}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_connected = False
    
    async def register_worker(self, 
        host: str,
                            port: int, 
                            capabilities: List[str] = None,
                            metadata: Dict[str, Any] = None) -> str:
        """
        Register a new worker node.
        
        Args:
            host: Worker host
            port: Worker port
            capabilities: List of worker capabilities
            metadata: Additional worker metadata
            
        Returns:
            Worker ID
        """
        if not self.redis_connected:
            logger.warning("Redis not connected, cannot register worker")
            return None
        
        if capabilities is None:
            capabilities = []
        
        if metadata is None:
            metadata = {}
        
        worker_id = f"worker_{self.worker_counter}"
        self.worker_counter += 1
        
        worker = WorkerNode(
            node_id=worker_id,
            host=host,
            port=port,
            status=WorkerStatus.IDLE,
            capabilities=capabilities,
            last_heartbeat=time.time(),
            metadata=metadata
        )
        
        self.workers[worker_id] = worker
        
        # Register in Redis
        worker_data = {
            "node_id": worker_id,
            "host": host,
            "port": port,
            "status": worker.status.value,
            "load": worker.load,
            "capabilities": json.dumps(capabilities),
            "last_heartbeat": worker.last_heartbeat,
            "metadata": json.dumps(metadata)
        }
        
        self.redis.hset(f"{self.worker_registry}:{worker_id}", mapping=worker_data)
        self.redis.expire(f"{self.worker_registry}:{worker_id}", 300)  # 5 minute TTL
        
        logger.info(f"Registered worker {worker_id} at {host}:{port}")
        return worker_id
    
    async def distribute_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Distribute a query across the cluster.
        
        Args:
            query: Query to process
            context: Optional context
            
        Returns:
            Distributed processing results
        """
        if not self.redis_connected:
            logger.warning("Redis not connected, falling back to local processing")
            return await self._local_fallback(query, context)
        
        start_time = time.time()
        
        try:
            # Create task shards
            shards = await self._create_task_shards(query, context)
            
            # Distribute shards to workers
            futures = []
            for shard in shards:
                future = self._submit_shard_to_worker(shard)
                futures.append(future)
            
            # Wait for all shards to complete
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Synthesize results
            synthesis = await self._synthesize_results(results, query, context)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(len(shards), processing_time)
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Distributed processing failed: {e}")
            return await self._local_fallback(query, context)
    
    async def _create_task_shards(self, query: str, context: Dict[str, Any]) -> List[TaskShard]:
        """Create task shards for distributed processing."""
        shards = []
        
        # Determine sharding strategy based on query type
        if "research" in query.lower():
            # Research queries: shard by agent type
            agent_types = ["surveyor", "dissident", "synthesist", "oracle"]
            for agent_type in agent_types:
                shard = TaskShard(
                    shard_id=f"shard_{len(shards)}",
                    task_type="agent_processing",
                    data={
                        "query": query,
                        "context": context,
                        "agent_type": agent_type
                    },
                    priority=1,
                    timeout=300.0,
                    created_time=time.time()
                )
                shards.append(shard)
        
        elif "civilization" in query.lower():
            # Civilization queries: shard by simulation components
            components = ["world_state", "agent_society", "social_learning", "resource_economy"]
            for component in components:
                shard = TaskShard(
                    shard_id=f"shard_{len(shards)}",
                    task_type="civilization_processing",
                    data={
                        "query": query,
                        "context": context,
                        "component": component
                    },
                    priority=2,
                    timeout=600.0,
                    created_time=time.time()
                )
                shards.append(shard)
        
        else:
            # Default: single shard
            shard = TaskShard(
                shard_id="shard_0",
                task_type="general_processing",
                data={
                    "query": query,
                    "context": context
                },
                priority=0,
                timeout=300.0,
                created_time=time.time()
            )
            shards.append(shard)
        
        return shards
    
    async def _submit_shard_to_worker(self, shard: TaskShard) -> Dict[str, Any]:
        """Submit a task shard to an available worker."""
        # Find best worker for this shard
        worker_id = await self._select_worker_for_shard(shard)
        
        if not worker_id:
            raise Exception("No available workers")
        
        # Submit task to worker
        task_data = {
            "shard_id": shard.shard_id,
            "task_type": shard.task_type,
            "data": json.dumps(shard.data),
            "priority": shard.priority,
            "timeout": shard.timeout,
            "created_time": shard.created_time
        }
        
        # Add to task queue
        self.redis.lpush(self.task_queue, json.dumps(task_data))
        
        # Wait for result
        result = await self._wait_for_result(shard.shard_id, shard.timeout)
        
        return result
    
    async def _select_worker_for_shard(self, shard: TaskShard) -> Optional[str]:
        """Select the best worker for a task shard."""
        available_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker.status == WorkerStatus.IDLE and worker.load < 0.8
        ]
        
        if not available_workers:
            return None
        
        # Select worker with lowest load
        best_worker = min(available_workers, key=lambda w: self.workers[w].load)
        
        # Update worker status
        self.workers[best_worker].status = WorkerStatus.BUSY
        self.workers[best_worker].load += 0.1
        
        return best_worker
    
    async def _wait_for_result(self, shard_id: str, timeout: float) -> Dict[str, Any]:
        """Wait for a task result."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for result
            result_data = self.redis.lpop(f"{self.result_queue}:{shard_id}")
            if result_data:
                return json.loads(result_data)
            
            await asyncio.sleep(0.1)  # 100ms polling
        
        # Timeout
        return {
            "shard_id": shard_id,
            "status": "timeout",
            "error": f"Task timeout after {timeout}s"
        }
    
    async def _synthesize_results(self, 
        results: List[Dict[str, Any]],
                                query: str, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from distributed processing."""
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        if not valid_results:
            return {
                "status": "error",
                "error": "All distributed tasks failed",
                "query": query
            }
        
        # Synthesize based on result types
        synthesis = {
            "status": "success",
            "query": query,
            "shard_results": valid_results,
            "total_shards": len(results),
            "successful_shards": len(valid_results),
            "synthesis_quality": len(valid_results) / len(results) if results else 0.0
        }
        
        # Add specific synthesis based on task types
        if any("agent_processing" in r.get("task_type", "") for r in valid_results):
            synthesis["agent_synthesis"] = self._synthesize_agent_results(valid_results)
        
        if any("civilization_processing" in r.get("task_type", "") for r in valid_results):
            synthesis["civilization_synthesis"] = self._synthesize_civilization_results(valid_results)
        
        return synthesis
    
    def _synthesize_agent_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize agent processing results."""
        agent_outputs = {}
        
        for result in results:
            if result.get("task_type") == "agent_processing":
                agent_type = result.get("data", {}).get("agent_type")
                if agent_type:
                    agent_outputs[agent_type] = result.get("output", {})
        
        return {
            "agent_outputs": agent_outputs,
            "synthesis": "Combined agent analysis",
            "confidence": 0.8
        }
    
    def _synthesize_civilization_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize civilization processing results."""
        component_outputs = {}
        
        for result in results:
            if result.get("task_type") == "civilization_processing":
                component = result.get("data", {}).get("component")
                if component:
                    component_outputs[component] = result.get("output", {})
        
        return {
            "component_outputs": component_outputs,
            "synthesis": "Combined civilization analysis",
            "confidence": 0.8
        }
    
    async def _local_fallback(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to local processing when Redis is unavailable."""
        logger.info("Using local fallback processing")
        
        # Simple local processing
        return {
            "status": "fallback",
            "query": query,
            "result": "Processed locally due to Redis unavailability",
            "processing_time": 0.0
        }
    
    def _update_performance_stats(self, task_count: int, processing_time: float):
        """Update performance statistics."""
        self.performance_stats["total_tasks"] += task_count
        self.performance_stats["completed_tasks"] += task_count
        
        # Update average processing time
        total_tasks = self.performance_stats["total_tasks"]
        current_avg = self.performance_stats["average_processing_time"]
        new_avg = ((current_avg * (total_tasks - task_count)) + processing_time) / total_tasks
        self.performance_stats["average_processing_time"] = new_avg
        
        # Update cluster utilization
        active_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
        total_workers = len(self.workers)
        self.performance_stats["cluster_utilization"] = active_workers / total_workers if total_workers > 0 else 0.0
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        return {
            "redis_connected": self.redis_connected,
            "total_workers": len(self.workers),
            "active_workers": sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY),
            "idle_workers": sum(1 for w in self.workers.values() if w.status == WorkerStatus.IDLE),
            "offline_workers": sum(1 for w in self.workers.values() if w.status == WorkerStatus.OFFLINE),
            "performance_stats": self.performance_stats.copy(),
            "workers": {
                worker_id: {
                    "status": worker.status.value,
                    "load": worker.load,
                    "capabilities": worker.capabilities,
                    "last_heartbeat": worker.last_heartbeat
                }
                for worker_id, worker in self.workers.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources and connections."""
        if self.redis_connected:
            try:
                # Clear worker registry
                for worker_id in self.workers.keys():
                    self.redis.delete(f"{self.worker_registry}:{worker_id}")
                
                # Clear task queues
                self.redis.delete(self.task_queue)
                self.redis.delete(self.result_queue)
                
                self.redis.close()
                logger.info("Redis connection closed")
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


# Convenience functions for distributed processing
async def distribute_iceburg_query(query: str, 
    context: Dict[str, Any] = None,
                                 redis_nodes: List[Tuple[str, int]] = None) -> Dict[str, Any]:
    """
    Distribute an ICEBURG query across the cluster.
    
    Args:
        query: Query to process
        context: Optional context
        redis_nodes: Redis cluster nodes
        
    Returns:
        Distributed processing results
    """
    coordinator = DistributedICEBURG(redis_nodes=redis_nodes)
    
    try:
        result = await coordinator.distribute_query(query, context)
        return result
    finally:
        await coordinator.cleanup()


async def get_cluster_health(redis_nodes: List[Tuple[str, int]] = None) -> Dict[str, Any]:
    """
    Get cluster health status.
    
    Args:
        redis_nodes: Redis cluster nodes
        
    Returns:
        Cluster health information
    """
    coordinator = DistributedICEBURG(redis_nodes=redis_nodes)
    
    try:
        status = await coordinator.get_cluster_status()
        return status
    finally:
        await coordinator.cleanup()
