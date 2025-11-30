"""
Intelligent Load Balancer for ICEBURG Distributed Processing
Implements intelligent routing, circuit breaker, and adaptive load balancing.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import heapq

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    ADAPTIVE = "adaptive"


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    worker_id: str
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_response_time: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_health_check: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout_threshold: float = 30.0


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
    
    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record a successful request."""
        self.success_count += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
    
    def record_failure(self):
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.success_count = 0
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state


class IntelligentLoadBalancer:
    """
    Intelligent load balancer with adaptive routing and circuit breaker.
    
    Features:
    - Multiple load balancing strategies
    - Circuit breaker for fault tolerance
    - Adaptive load balancing based on performance
    - Health monitoring and auto-recovery
    - Intelligent routing based on query type
    """
    
    def __init__(self, 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
                 circuit_breaker_config: CircuitBreakerConfig = None):
        """
        Initialize the intelligent load balancer.
        
        Args:
            strategy: Load balancing strategy
            circuit_breaker_config: Circuit breaker configuration
        """
        self.strategy = strategy
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        
        # Worker management
        self.workers: Dict[str, WorkerMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Load balancing state
        self.round_robin_index = 0
        self.worker_weights: Dict[str, float] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.adaptive_weights: Dict[str, float] = {}
        
        # Health monitoring
        self.health_check_interval = 30.0  # seconds
        self.last_health_check = 0.0
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_trips": 0,
            "average_response_time": 0.0,
            "load_balancing_accuracy": 0.0
        }
    
    def add_worker(self, 
                  worker_id: str, 
                  weight: float = 1.0,
                  capabilities: List[str] = None,
                  metadata: Dict[str, Any] = None):
        """
        Add a worker to the load balancer.
        
        Args:
            worker_id: Unique worker identifier
            weight: Worker weight for weighted strategies
            capabilities: List of worker capabilities
            metadata: Additional worker metadata
        """
        if capabilities is None:
            capabilities = []
        
        if metadata is None:
            metadata = {}
        
        self.workers[worker_id] = WorkerMetrics(
            worker_id=worker_id,
            metadata=metadata
        )
        
        self.worker_weights[worker_id] = weight
        self.adaptive_weights[worker_id] = 1.0
        self.circuit_breakers[worker_id] = CircuitBreaker(self.circuit_breaker_config)
        
        logger.info(f"Added worker {worker_id} with weight {weight}")
    
    def remove_worker(self, worker_id: str):
        """Remove a worker from the load balancer."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.worker_weights[worker_id]
            del self.adaptive_weights[worker_id]
            del self.circuit_breakers[worker_id]
            logger.info(f"Removed worker {worker_id}")
    
    async def select_worker(self, 
                          query: str, 
                          context: Dict[str, Any] = None,
                          required_capabilities: List[str] = None) -> Optional[str]:
        """
        Select the best worker for a request.
        
        Args:
            query: Request query
            context: Optional context
            required_capabilities: Required worker capabilities
            
        Returns:
            Selected worker ID or None
        """
        if not self.workers:
            return None
        
        # Filter workers by capabilities
        available_workers = self._filter_workers_by_capabilities(required_capabilities)
        
        if not available_workers:
            logger.warning("No workers available with required capabilities")
            return None
        
        # Apply circuit breaker filtering
        healthy_workers = self._filter_healthy_workers(available_workers)
        
        if not healthy_workers:
            logger.warning("No healthy workers available")
            return None
        
        # Select worker based on strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_worker = self._round_robin_selection(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_worker = self._least_connections_selection(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected_worker = self._least_response_time_selection(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected_worker = self._weighted_round_robin_selection(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            selected_worker = self._adaptive_selection(healthy_workers, query, context)
        else:
            selected_worker = self._round_robin_selection(healthy_workers)
        
        if selected_worker:
            # Update worker metrics
            self.workers[selected_worker].active_connections += 1
            self.workers[selected_worker].total_requests += 1
        
        return selected_worker
    
    def _filter_workers_by_capabilities(self, required_capabilities: List[str]) -> List[str]:
        """Filter workers by required capabilities."""
        if not required_capabilities:
            return list(self.workers.keys())
        
        filtered_workers = []
        for worker_id, worker in self.workers.items():
            if all(cap in worker.metadata.get("capabilities", []) for cap in required_capabilities):
                filtered_workers.append(worker_id)
        
        return filtered_workers
    
    def _filter_healthy_workers(self, workers: List[str]) -> List[str]:
        """Filter workers that are healthy (circuit breaker closed)."""
        healthy_workers = []
        for worker_id in workers:
            circuit_breaker = self.circuit_breakers[worker_id]
            if circuit_breaker.can_execute():
                healthy_workers.append(worker_id)
        
        return healthy_workers
    
    def _round_robin_selection(self, workers: List[str]) -> str:
        """Round robin worker selection."""
        if not workers:
            return None
        
        selected_worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return selected_worker
    
    def _least_connections_selection(self, workers: List[str]) -> str:
        """Select worker with least active connections."""
        if not workers:
            return None
        
        return min(workers, key=lambda w: self.workers[w].active_connections)
    
    def _least_response_time_selection(self, workers: List[str]) -> str:
        """Select worker with least response time."""
        if not workers:
            return None
        
        return min(workers, key=lambda w: self.workers[w].average_response_time)
    
    def _weighted_round_robin_selection(self, workers: List[str]) -> str:
        """Weighted round robin selection."""
        if not workers:
            return None
        
        # Calculate total weight
        total_weight = sum(self.worker_weights[w] for w in workers)
        
        # Select based on weight
        import random
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for worker_id in workers:
            current_weight += self.worker_weights[worker_id]
            if random_value <= current_weight:
                return worker_id
        
        return workers[-1]  # Fallback
    
    def _adaptive_selection(self, 
                          workers: List[str], 
                          query: str, 
                          context: Dict[str, Any]) -> str:
        """Adaptive worker selection based on query type and performance."""
        if not workers:
            return None
        
        # Calculate adaptive scores
        scores = {}
        for worker_id in workers:
            score = self._calculate_adaptive_score(worker_id, query, context)
            scores[worker_id] = score
        
        # Select worker with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_adaptive_score(self, 
                                worker_id: str, 
                                query: str, 
                                context: Dict[str, Any]) -> float:
        """Calculate adaptive score for a worker."""
        worker = self.workers[worker_id]
        
        # Base score from performance metrics
        base_score = 1.0
        
        # Adjust for response time (lower is better)
        if worker.average_response_time > 0:
            response_score = 1.0 / (1.0 + worker.average_response_time)
        else:
            response_score = 1.0
        
        # Adjust for error rate (lower is better)
        error_score = 1.0 - worker.error_rate
        
        # Adjust for active connections (lower is better)
        connection_score = 1.0 / (1.0 + worker.active_connections)
        
        # Adjust for resource usage (lower is better)
        resource_score = 1.0 - (worker.cpu_usage + worker.memory_usage) / 2.0
        
        # Query-specific adjustments
        query_score = self._calculate_query_specific_score(worker_id, query, context)
        
        # Combine scores
        total_score = (
            base_score * 0.2 +
            response_score * 0.3 +
            error_score * 0.2 +
            connection_score * 0.1 +
            resource_score * 0.1 +
            query_score * 0.1
        )
        
        return total_score
    
    def _calculate_query_specific_score(self, 
                                     worker_id: str, 
                                     query: str, 
                                     context: Dict[str, Any]) -> float:
        """Calculate query-specific score for a worker."""
        # This could be enhanced with ML-based routing
        # For now, use simple heuristics
        
        worker = self.workers[worker_id]
        capabilities = worker.metadata.get("capabilities", [])
        
        # Boost score for specialized capabilities
        if "research" in query.lower() and "research" in capabilities:
            return 1.5
        elif "civilization" in query.lower() and "simulation" in capabilities:
            return 1.5
        elif "software" in query.lower() and "development" in capabilities:
            return 1.5
        
        return 1.0
    
    def record_request_result(self, 
                           worker_id: str, 
                           success: bool, 
                           response_time: float,
                           error: str = None):
        """
        Record the result of a request.
        
        Args:
            worker_id: Worker that processed the request
            success: Whether the request was successful
            response_time: Response time in seconds
            error: Error message if failed
        """
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        circuit_breaker = self.circuit_breakers[worker_id]
        
        # Update worker metrics
        worker.active_connections = max(0, worker.active_connections - 1)
        
        if success:
            worker.successful_requests += 1
            circuit_breaker.record_success()
        else:
            worker.failed_requests += 1
            circuit_breaker.record_failure()
            if circuit_breaker.get_state() == CircuitState.OPEN:
                self.stats["circuit_breaker_trips"] += 1
        
        # Update response time
        if worker.average_response_time == 0:
            worker.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            worker.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * worker.average_response_time
            )
        
        worker.last_response_time = response_time
        
        # Update error rate
        total_requests = worker.successful_requests + worker.failed_requests
        if total_requests > 0:
            worker.error_rate = worker.failed_requests / total_requests
        
        # Update global stats
        self.stats["total_requests"] += 1
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Update adaptive weights
        self._update_adaptive_weights(worker_id, success, response_time)
    
    def _update_adaptive_weights(self, worker_id: str, success: bool, response_time: float):
        """Update adaptive weights based on performance."""
        if worker_id not in self.adaptive_weights:
            return
        
        # Adjust weight based on performance
        if success and response_time < 1.0:  # Fast and successful
            self.adaptive_weights[worker_id] = min(2.0, self.adaptive_weights[worker_id] * 1.1)
        elif not success or response_time > 5.0:  # Slow or failed
            self.adaptive_weights[worker_id] = max(0.1, self.adaptive_weights[worker_id] * 0.9)
    
    async def health_check(self):
        """Perform health check on all workers."""
        current_time = time.time()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        
        for worker_id, worker in self.workers.items():
            # Simulate health check (in real implementation, this would ping workers)
            worker.last_health_check = current_time
            
            # Update circuit breaker state
            circuit_breaker = self.circuit_breakers[worker_id]
            if circuit_breaker.get_state() == CircuitState.OPEN:
                if current_time - worker.last_response_time > self.circuit_breaker_config.recovery_timeout:
                    circuit_breaker.state = CircuitState.HALF_OPEN
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        # Calculate success rate
        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = self.stats["successful_requests"] / self.stats["total_requests"]
        
        # Calculate average response time
        avg_response_time = 0.0
        if self.workers:
            response_times = [w.average_response_time for w in self.workers.values() if w.average_response_time > 0]
            if response_times:
                avg_response_time = statistics.mean(response_times)
        
        return {
            "strategy": self.strategy.value,
            "total_workers": len(self.workers),
            "healthy_workers": len([w for w in self.workers.values() if w.error_rate < 0.5]),
            "circuit_breaker_trips": self.stats["circuit_breaker_trips"],
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "worker_metrics": {
                worker_id: {
                    "active_connections": worker.active_connections,
                    "total_requests": worker.total_requests,
                    "success_rate": worker.successful_requests / max(1, worker.total_requests),
                    "average_response_time": worker.average_response_time,
                    "error_rate": worker.error_rate,
                    "circuit_state": self.circuit_breakers[worker_id].get_state().value
                }
                for worker_id, worker in self.workers.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        # Reset all circuit breakers
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.state = CircuitState.CLOSED
            circuit_breaker.failure_count = 0
            circuit_breaker.success_count = 0
        
        logger.info("Load balancer cleanup completed")


# Convenience functions
async def create_load_balancer(strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE) -> IntelligentLoadBalancer:
    """Create a new load balancer instance."""
    return IntelligentLoadBalancer(strategy=strategy)


async def balance_iceburg_request(query: str, 
                                context: Dict[str, Any] = None,
                                load_balancer: IntelligentLoadBalancer = None) -> Tuple[str, Dict[str, Any]]:
    """
    Balance an ICEBURG request across workers.
    
    Args:
        query: Request query
        context: Optional context
        load_balancer: Load balancer instance
        
    Returns:
        Tuple of (selected_worker_id, load_balancer_stats)
    """
    if load_balancer is None:
        load_balancer = await create_load_balancer()
    
    # Perform health check
    await load_balancer.health_check()
    
    # Select worker
    worker_id = await load_balancer.select_worker(query, context)
    
    # Get stats
    stats = load_balancer.get_load_balancer_stats()
    
    return worker_id, stats
