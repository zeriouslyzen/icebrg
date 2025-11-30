"""
Enhanced ICEBURG Protocol with Parallel Execution and Performance Optimization
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
import logging

from .parallel_execution import ParallelExecutionEngine, AgentTask, AgentStatus
from .caching.redis_intelligence import IntelligentCache, cached_compute
from .integration.reflexive_routing import ReflexiveRoutingSystem
from .config import load_config

logger = logging.getLogger(__name__)


class EnhancedICEBURGProtocol:
    """
    Enhanced ICEBURG protocol with parallel execution, caching, and intelligent routing.
    
    Features:
    - Parallel agent execution with dependency graphs
    - Intelligent caching with semantic similarity
    - Fast path routing for simple queries
    - Performance monitoring and optimization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced protocol."""
        self.config = load_config(config_path)
        self.cache = IntelligentCache()
        self.routing = ReflexiveRoutingSystem()
        self.parallel_engine = ParallelExecutionEngine()
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "fast_path_queries": 0,
            "parallel_executions": 0,
            "average_response_time": 0.0
        }
    
    async def process_query(self, 
                          query: str, 
                          context: Dict[str, Any] = None,
                          verbose: bool = False) -> Dict[str, Any]:
        """
        Process a query through the enhanced ICEBURG protocol.
        
        Args:
            query: Input query
            context: Optional context
            verbose: Enable verbose output
            
        Returns:
            Dictionary containing results and metadata
        """
        start_time = time.time()
        
        if context is None:
            context = {}
        
        # Update performance tracking
        self.performance_stats["total_queries"] += 1
        
        try:
            # 1. Check cache first
            cached_result = await self._check_cache(query, context)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                if verbose:
                    logger.info("Cache hit - returning cached result")
                return cached_result
            
            # 2. Route query for processing strategy
            routing_decision = self.routing.route_query(query, context)
            
            if routing_decision.route_type == "reflexive" and routing_decision.complexity_score < 0.3:
                # Fast path for simple queries
                result = await self._process_fast_path(query, context, verbose)
                self.performance_stats["fast_path_queries"] += 1
            else:
                # Full protocol execution
                result = await self._process_full_protocol(query, context, verbose)
                self.performance_stats["parallel_executions"] += 1
            
            # 3. Cache the result
            await self._cache_result(query, result, context)
            
            # 4. Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced protocol execution failed: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "processing_time": time.time() - start_time
            }
    
    async def _check_cache(self, query: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check cache for existing result."""
        try:
            # Use intelligent cache with semantic similarity
            cached_result = await self.cache.get_or_compute(
                query, 
                lambda q, c: None,  # Dummy function for cache lookup
                ttl=3600,
                context=context
            )
            return cached_result
        except Exception:
            return None
    
    async def _cache_result(self, query: str, result: Dict[str, Any], context: Dict[str, Any]):
        """Cache the result for future use."""
        try:
            await self.cache._cache_result(
                self.cache._get_embedding(query),
                result,
                ttl=3600
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    async def _process_fast_path(self, query: str, context: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
        """Process simple queries through fast path."""
        if verbose:
            logger.info("Processing through fast path")
        
        # Use reflexive routing for fast response
        reflexive_response = await self.routing.process_reflexive(query, context)
        
        return {
            "type": "fast_path",
            "response": reflexive_response.response,
            "confidence": reflexive_response.confidence,
            "escalation_recommended": reflexive_response.escalation_recommended,
            "processing_time": reflexive_response.processing_time,
            "metadata": {
                "route_type": "reflexive",
                "fast_path": True
            }
        }
    
    async def _process_full_protocol(self, query: str, context: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
        """Process complex queries through full protocol with parallel execution."""
        if verbose:
            logger.info("Processing through full protocol with parallel execution")
        
        # Create agent tasks for parallel execution
        agent_tasks = self._create_agent_tasks(query, context, verbose)
        
        # Execute agents in parallel
        results = await self.parallel_engine.execute_agents(
            agent_tasks, 
            query, 
            context,
            early_termination_threshold=0.3
        )
        
        # Synthesize results
        synthesis = await self._synthesize_results(results, query, context, verbose)
        
        return {
            "type": "full_protocol",
            "agent_results": {
                agent_name: result.result if result.status == AgentStatus.COMPLETED else None
                for agent_name, result in results.items()
            },
            "synthesis": synthesis,
            "metadata": {
                "route_type": "full_protocol",
                "parallel_execution": True,
                "agents_executed": len(results)
            }
        }
    
    def _create_agent_tasks(self, query: str, context: Dict[str, Any], verbose: bool) -> List[AgentTask]:
        """Create agent tasks for parallel execution."""
        from .agents.surveyor import run as surveyor
        from .agents import dissident
        from .agents.synthesist import run as synthesist
        from .agents.oracle import run as oracle
        
        # Define agent tasks with dependencies
        tasks = [
            AgentTask(
                name="surveyor",
                function=lambda q, c: surveyor(self.config, None, q, verbose),
                dependencies=[],
                timeout=300.0,
                priority=1,
                early_termination=True,
                metadata={"role": "consensus_research"}
            ),
            AgentTask(
                name="dissident",
                function=lambda q, c: dissident.run(self.config, q, verbose),
                dependencies=[],
                timeout=300.0,
                priority=1,
                early_termination=True,
                metadata={"role": "contrarian_analysis"}
            ),
            AgentTask(
                name="synthesist",
                function=lambda q, c: synthesist(self.config, q, verbose),
                dependencies=["surveyor", "dissident"],
                timeout=300.0,
                priority=2,
                metadata={"role": "synthesis"}
            ),
            AgentTask(
                name="oracle",
                function=lambda q, c: oracle(self.config, q, verbose),
                dependencies=["synthesist"],
                timeout=300.0,
                priority=3,
                metadata={"role": "principle_extraction"}
            )
        ]
        
        return tasks
    
    async def _synthesize_results(self, 
                                results: Dict[str, Any], 
                                query: str, 
                                context: Dict[str, Any], 
                                verbose: bool) -> Dict[str, Any]:
        """Synthesize results from parallel agent execution."""
        if verbose:
            logger.info("Synthesizing results from parallel execution")
        
        # Extract successful results
        successful_results = {
            agent_name: result.result 
            for agent_name, result in results.items() 
            if result.status == AgentStatus.COMPLETED and result.result is not None
        }
        
        # Create synthesis based on available results
        synthesis = {
            "query": query,
            "successful_agents": list(successful_results.keys()),
            "failed_agents": [
                agent_name for agent_name, result in results.items() 
                if result.status == AgentStatus.FAILED
            ],
            "results": successful_results,
            "synthesis_quality": len(successful_results) / len(results) if results else 0.0
        }
        
        return synthesis
    
    def _update_performance_metrics(self, response_time: float):
        """Update performance metrics."""
        total_queries = self.performance_stats["total_queries"]
        current_avg = self.performance_stats["average_response_time"]
        
        # Calculate rolling average
        new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        self.performance_stats["average_response_time"] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add cache statistics
        cache_stats = self.cache.get_cache_stats()
        stats.update({
            "cache_stats": cache_stats,
            "routing_stats": self.routing.get_routing_statistics(),
            "parallel_stats": self.parallel_engine.get_execution_stats()
        })
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "fast_path_queries": 0,
            "parallel_executions": 0,
            "average_response_time": 0.0
        }
        self.parallel_engine.reset_statistics()


# Convenience function for backward compatibility
async def enhanced_iceberg_protocol(query: str, 
                                   context: Dict[str, Any] = None,
                                   verbose: bool = False,
                                   config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced ICEBURG protocol with parallel execution and caching.
    
    Args:
        query: Input query
        context: Optional context
        verbose: Enable verbose output
        config_path: Optional config path
        
    Returns:
        Dictionary containing results and metadata
    """
    protocol = EnhancedICEBURGProtocol(config_path)
    return await protocol.process_query(query, context, verbose)
