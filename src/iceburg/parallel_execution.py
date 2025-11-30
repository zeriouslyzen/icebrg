"""
Enhanced Parallel Execution System for ICEBURG Protocol
Implements dependency graphs, early termination, and intelligent agent orchestration.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """Represents an agent task with dependencies and metadata."""
    name: str
    function: Callable
    dependencies: List[str]
    timeout: float = 300.0  # 5 minutes default
    priority: int = 0  # Higher number = higher priority
    early_termination: bool = False  # Can terminate early if conditions met
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionResult:
    """Result of agent execution."""
    agent_name: str
    status: AgentStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ParallelExecutionEngine:
    """
    Enhanced parallel execution engine with dependency graphs and early termination.
    
    Features:
    - Dependency graph execution
    - Early termination for simple queries
    - Intelligent timeout management
    - Result aggregation and synthesis
    - Performance monitoring
    """
    
    def __init__(self, max_concurrent: int = 4, default_timeout: float = 300.0):
        """
        Initialize the parallel execution engine.
        
        Args:
            max_concurrent: Maximum number of concurrent agent executions
            default_timeout: Default timeout for agent execution
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Execution tracking
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_results: Dict[str, ExecutionResult] = {}
        self.failed_agents: Set[str] = set()
        
        # Performance metrics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "early_terminations": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0
        }
    
    async def execute_agents(self, 
                           agent_tasks: List[AgentTask], 
                           query: str,
                           context: Dict[str, Any] = None,
                           early_termination_threshold: float = 0.3) -> Dict[str, ExecutionResult]:
        """
        Execute agents in parallel with dependency management.
        
        Args:
            agent_tasks: List of agent tasks to execute
            query: Original query
            context: Execution context
            early_termination_threshold: Complexity threshold for early termination
            
        Returns:
            Dictionary of agent results
        """
        if context is None:
            context = {}
        
        start_time = time.time()
        
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(agent_tasks)
            
            # Check for early termination opportunity
            if self._should_terminate_early(query, context, early_termination_threshold):
                logger.info("Early termination triggered - executing only essential agents")
                essential_agents = self._get_essential_agents(agent_tasks)
                return await self._execute_with_early_termination(essential_agents, query, context)
            
            # Execute with full dependency graph
            results = await self._execute_with_dependencies(agent_tasks, dependency_graph, query, context)
            
            # Update statistics
            execution_time = time.time() - start_time
            self._update_statistics(results, execution_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Return partial results if available
            return self.completed_results
    
    def _build_dependency_graph(self, agent_tasks: List[AgentTask]) -> Dict[str, List[str]]:
        """Build dependency graph from agent tasks."""
        graph = {}
        for task in agent_tasks:
            graph[task.name] = task.dependencies.copy()
        return graph
    
    def _should_terminate_early(self, query: str, context: Dict[str, Any], threshold: float) -> bool:
        """Determine if execution should terminate early based on query complexity."""
        # Simple heuristics for early termination
        query_lower = query.lower()
        
        # Check for simple patterns
        simple_patterns = [
            "what is", "define", "explain", "hello", "hi",
            "how are you", "help", "quick", "simple"
        ]
        
        if any(pattern in query_lower for pattern in simple_patterns):
            return True
        
        # Check context for complexity indicators
        if context.get("complexity_score", 0.0) < threshold:
            return True
        
        # Check query length
        if len(query.split()) < 10:
            return True
        
        return False
    
    def _get_essential_agents(self, agent_tasks: List[AgentTask]) -> List[AgentTask]:
        """Get essential agents for early termination."""
        # Priority order: surveyor first, then others based on priority
        essential = []
        
        # Add surveyor if available
        surveyor_task = next((task for task in agent_tasks if task.name == "surveyor"), None)
        if surveyor_task:
            essential.append(surveyor_task)
        
        # Add other high-priority agents
        high_priority = [task for task in agent_tasks if task.priority > 0]
        high_priority.sort(key=lambda x: x.priority, reverse=True)
        essential.extend(high_priority[:2])  # Limit to 2 additional agents
        
        return essential
    
    async def _execute_with_early_termination(self, 
                                            essential_agents: List[AgentTask],
                                            query: str,
                                            context: Dict[str, Any]) -> Dict[str, ExecutionResult]:
        """Execute only essential agents for early termination."""
        logger.info(f"Early termination: executing {len(essential_agents)} essential agents")
        
        # Execute essential agents in parallel
        tasks = []
        for agent_task in essential_agents:
            task = asyncio.create_task(
                self._execute_single_agent(agent_task, query, context)
            )
            tasks.append((agent_task.name, task))
        
        # Wait for completion
        results = {}
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
            except Exception as e:
                logger.error(f"Essential agent {agent_name} failed: {e}")
                results[agent_name] = ExecutionResult(
                    agent_name=agent_name,
                    status=AgentStatus.FAILED,
                    error=str(e)
                )
        
        return results
    
    async def _execute_with_dependencies(self, 
                                      agent_tasks: List[AgentTask],
                                      dependency_graph: Dict[str, List[str]],
                                      query: str,
                                      context: Dict[str, Any]) -> Dict[str, ExecutionResult]:
        """Execute agents with full dependency management."""
        # Create task mapping
        task_map = {task.name: task for task in agent_tasks}
        results = {}
        
        # Track completed dependencies
        completed = set()
        failed = set()
        
        while len(results) < len(agent_tasks):
            # Find agents ready to execute (dependencies satisfied and none failed)
            ready_agents = []
            for task in agent_tasks:
                if task.name in results:
                    continue
                
                # Check if all dependencies are completed
                deps = task.dependencies
                dependencies_satisfied = all(dep in completed for dep in deps)
                deps_failed = any(dep in failed for dep in deps)
                
                if dependencies_satisfied and not deps_failed:
                    ready_agents.append(task)
                elif deps_failed:
                    # Propagate dependency failure to this task
                    results[task.name] = ExecutionResult(
                        agent_name=task.name,
                        status=AgentStatus.FAILED,
                        error=f"Dependency failed: {[d for d in deps if d in failed]}"
                    )
                    completed.add(task.name)
                    failed.add(task.name)
            
            if not ready_agents:
                # No agents ready, check for deadlock
                remaining = [task.name for task in agent_tasks if task.name not in results]
                if remaining:
                    logger.warning(f"Potential deadlock with remaining agents: {remaining}")
                    # Force execute remaining agents
                    for task_name in remaining:
                        if task_name in task_map:
                            ready_agents.append(task_map[task_name])
            
            # Execute ready agents in parallel
            if ready_agents:
                concurrent_tasks = []
                for agent_task in ready_agents[:self.max_concurrent]:
                    task = asyncio.create_task(
                        self._execute_single_agent(agent_task, query, context)
                    )
                    concurrent_tasks.append((agent_task.name, task))
                
                # Wait for completion
                for agent_name, task in concurrent_tasks:
                    try:
                        result = await task
                        results[agent_name] = result
                        completed.add(agent_name)
                        
                        if result.status == AgentStatus.COMPLETED:
                            logger.info(f"Agent {agent_name} completed successfully")
                        else:
                            logger.warning(f"Agent {agent_name} failed: {result.error}")
                            failed.add(agent_name)
                            
                    except Exception as e:
                        logger.error(f"Agent {agent_name} execution error: {e}")
                        results[agent_name] = ExecutionResult(
                            agent_name=agent_name,
                            status=AgentStatus.FAILED,
                            error=str(e)
                        )
                        completed.add(agent_name)
                        failed.add(agent_name)
        
        return results
    
    async def _execute_single_agent(self, 
                                  agent_task: AgentTask, 
                                  query: str, 
                                  context: Dict[str, Any]) -> ExecutionResult:
        """Execute a single agent with timeout and error handling."""
        start_time = time.time()
        agent_name = agent_task.name
        
        try:
            logger.info(f"Executing agent: {agent_name}")
            
            # Allocate resources dynamically
            allocation = await self.resource_allocator.allocate_resources(
                agent_id=agent_name,
                priority=agent_task.metadata.get("priority", 5) if agent_task.metadata else 5
            )
            
            if not allocation:
                # Resources unavailable, fall back to semaphore
                logger.warning(f"Resource allocation failed for {agent_name}, using semaphore fallback")
                async with self.semaphore:
                    result = await asyncio.wait_for(
                        self._run_agent_function(agent_task, query, context),
                        timeout=agent_task.timeout
                    )
            else:
                try:
                    # Execute with timeout using allocated resources
                    result = await asyncio.wait_for(
                        self._run_agent_function(agent_task, query, context),
                        timeout=allocation.timeout_seconds
                    )
                finally:
                    # Release resources
                    self.resource_allocator.release_resources(allocation.agent_id + "_" + str(int(start_time * 1000)))
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    agent_name=agent_name,
                    status=AgentStatus.COMPLETED,
                    result=result,
                    execution_time=execution_time,
                    metadata=agent_task.metadata
                )
                
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"Agent {agent_name} timed out after {execution_time:.2f}s")
            return ExecutionResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                error=f"Timeout after {execution_time:.2f}s",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent {agent_name} failed: {e}")
            return ExecutionResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _run_agent_function(self, agent_task: AgentTask, query: str, context: Dict[str, Any]) -> Any:
        """Run the agent function with proper error handling."""
        # Simple retry policy via metadata: retries (int), backoff_s (float)
        retries = int((agent_task.metadata or {}).get("retries", 0))
        backoff_s = float((agent_task.metadata or {}).get("backoff_s", 0.0))
        attempt = 0
        while True:
            try:
                if asyncio.iscoroutinefunction(agent_task.function):
                    return await agent_task.function(query, context)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, agent_task.function, query, context)
            except Exception as e:
                attempt += 1
                if attempt > retries:
                    logger.error(f"Agent function error after {attempt} attempt(s): {e}")
                    raise
                delay = backoff_s * attempt if backoff_s > 0 else 0
                if delay > 0:
                    await asyncio.sleep(delay)
                logger.warning(f"Retrying agent {agent_task.name} (attempt {attempt}/{retries}) after error: {e}")
    
    def _update_statistics(self, results: Dict[str, ExecutionResult], total_time: float):
        """Update execution statistics."""
        self.execution_stats["total_executions"] += len(results)
        self.execution_stats["total_execution_time"] += total_time
        
        successful = sum(1 for r in results.values() if r.status == AgentStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == AgentStatus.FAILED)
        
        self.execution_stats["successful_executions"] += successful
        self.execution_stats["failed_executions"] += failed
        
        if self.execution_stats["total_executions"] > 0:
            self.execution_stats["average_execution_time"] = (
                self.execution_stats["total_execution_time"] / 
                self.execution_stats["total_executions"]
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    def reset_statistics(self):
        """Reset execution statistics."""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "early_terminations": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0
        }


# Convenience functions for common agent execution patterns
async def execute_surveyor_dissident_parallel(
    cfg, 
    vs, 
    query: str, 
    verbose: bool = False, 
    multimodal_input=None
) -> Tuple[Any, Any]:
    """Execute surveyor and dissident agents in parallel."""
    from .agents.surveyor import run as surveyor
    from .agents import dissident
    
    async def run_surveyor():
        return surveyor(cfg, vs, query, verbose=verbose, multimodal_input=multimodal_input)
    
    async def run_dissident():
        # Dissident doesn't need surveyor output for parallel execution
        # It will get it later if needed
        return dissident.run(cfg, query, "", verbose=verbose)
    
    # Execute in parallel
    surveyor_result, dissident_result = await asyncio.gather(
        run_surveyor(),
        run_dissident(),
        return_exceptions=True
    )
    
    # Handle exceptions
    if isinstance(surveyor_result, Exception):
        if verbose:
            print(f"[PARALLEL] Surveyor failed: {surveyor_result}")
        surveyor_result = None
    if isinstance(dissident_result, Exception):
        if verbose:
            print(f"[PARALLEL] Dissident failed: {dissident_result}")
        dissident_result = None
    
    return surveyor_result, dissident_result


async def execute_core_agents_parallel(cfg, query: str, verbose: bool = False) -> Dict[str, Any]:
    """Execute core ICEBURG agents in parallel with dependency management."""
    from .agents.surveyor import run as surveyor
    from .agents import dissident
    from .agents.synthesist import run as synthesist
    
    # Create agent tasks with dependencies
    agent_tasks = [
        AgentTask(
            name="surveyor",
            function=lambda q, c: surveyor(cfg, None, q, verbose),
            dependencies=[],
            timeout=300.0,
            priority=1
        ),
        AgentTask(
            name="dissident",
            function=lambda q, c: dissident.run(cfg, q, verbose),
            dependencies=[],
            timeout=300.0,
            priority=1
        ),
        AgentTask(
            name="synthesist",
            function=lambda q, c: synthesist(cfg, q, verbose),
            dependencies=["surveyor", "dissident"],
            timeout=300.0,
            priority=2
        )
    ]
    
    # Execute with parallel engine
    engine = ParallelExecutionEngine()
    results = await engine.execute_agents(agent_tasks, query)
    
    # Extract results
    return {
        agent_name: result.result if result.status == AgentStatus.COMPLETED else None
        for agent_name, result in results.items()
    }
