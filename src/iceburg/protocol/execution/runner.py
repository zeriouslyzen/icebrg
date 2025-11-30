from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Dict, List

from .agents.registry import get_agent_runner
from ..config import ProtocolConfig
from ..models import AgentResult, AgentTask, Query
from ...infrastructure.retry_manager import RetryManager, RetryConfig, RetryResult
# Parallel execution imports (optional, for future use)
# from ...parallel_execution import ParallelExecutionEngine, AgentTask as ParallelAgentTask, AgentStatus
from ..planner import get_parallelizable_groups

logger = logging.getLogger(__name__)


async def run_agent_tasks(tasks: List[AgentTask], query: Query, cfg: ProtocolConfig) -> List[AgentResult]:
    """
    Executes a list of agent tasks with parallel execution by default.
    Uses dependency graphs to execute independent agents in parallel.
    Falls back to sequential execution only if parallel execution fails.
    """
    # Use parallel execution by default (unless explicitly disabled)
    use_parallel = not getattr(cfg, 'force_sequential', False)
    
    if use_parallel and len(tasks) > 1:
        try:
            return await _run_agent_tasks_parallel(tasks, query, cfg)
        except Exception as e:
            logger.warning(f"Parallel execution failed, falling back to sequential: {e}")
            # Fall through to sequential execution
    
    # Sequential execution (fallback or when disabled)
    return await _run_agent_tasks_sequential(tasks, query, cfg)


async def _run_agent_tasks_parallel(tasks: List[AgentTask], query: Query, cfg: ProtocolConfig) -> List[AgentResult]:
    """
    Execute agents in parallel using dependency groups.
    Groups agents by dependency level and executes each group in parallel.
    """
    results_map: Dict[str, Any] = {}
    completed_results: List[AgentResult] = []
    
    # Initialize retry manager
    retry_config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=300.0
    )
    retry_manager = RetryManager(retry_config)
    
    # Get parallelizable groups
    groups = get_parallelizable_groups(tasks, cfg)
    
    if cfg.verbose:
        print(f"[RUNNER] Parallel execution: {len(tasks)} tasks grouped into {len(groups)} parallel groups")
    
    # Execute each group in parallel
    for group_idx, group in enumerate(groups):
        if cfg.verbose:
            print(f"[RUNNER] Executing group {group_idx + 1}/{len(groups)}: {[t.agent for t in group]}")
        
        # Execute all tasks in this group concurrently
        group_tasks = []
        for task in group:
            group_tasks.append(_execute_single_task_async(task, query, cfg, retry_manager, results_map))
        
        # Wait for all tasks in group to complete
        group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
        
        # Process results
        for task, result in zip(group, group_results):
            if isinstance(result, Exception):
                logger.error(f"Agent {task.agent} failed: {result}")
                agent_result = AgentResult(
                    agent=task.agent,
                    payload=None,
                    metadata={"success": False, "error": str(result)},
                    latency_ms=0.0
                )
            else:
                agent_result = result
                results_map[task.agent] = agent_result.payload
        
            completed_results.append(agent_result)
    
    return completed_results


async def _execute_single_task_async(
    task: AgentTask,
    query: Query,
    cfg: ProtocolConfig,
    retry_manager: RetryManager,
    results_map: Dict[str, Any]
) -> AgentResult:
    """Execute a single task asynchronously."""
    start_time = time.time()
    output: Any = None
    success = False
    error_message = None
    
    try:
        # Resolve dependencies
        requires = list(task.dependencies or [])
        if isinstance(task.payload, dict) and "requires" in task.payload:
            requires.extend(task.payload.get("requires", []))
        
        resolved_input_data: Dict[str, Any] = dict(task.payload or {})
        for dep in requires:
            if dep in results_map:
                resolved_input_data[f"{dep}_output"] = results_map[dep]
        resolved_input_data.setdefault("query", getattr(query, "text", ""))
        
        # Get agent runner
        agent_runner = get_agent_runner(task.agent)
        if agent_runner is None:
            raise RuntimeError(f"Agent not registered: {task.agent}")
        
        # Execute with retry
        async def execute_agent():
            if asyncio.iscoroutinefunction(agent_runner):
                return await agent_runner(cfg=cfg, **resolved_input_data)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: agent_runner(cfg=cfg, **resolved_input_data))
        
        retry_result: RetryResult = await retry_manager.execute_with_retry(
            execute_agent,
            operation_name=f"agent_{task.agent}"
        )
        
        if retry_result.success:
            output = retry_result.result
            success = True
        else:
            error_message = retry_result.final_error or "Agent execution failed"
            logger.error(f"Agent {task.agent} failed after {retry_result.attempts} attempts: {error_message}")
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error executing agent {task.agent}: {error_message}", exc_info=True)
        circuit_breaker = retry_manager.get_circuit_breaker(f"agent_{task.agent}")
        circuit_breaker.record_failure()
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    return AgentResult(
        agent=task.agent,
        payload=output,
        metadata={
            "input_data": resolved_input_data if 'resolved_input_data' in locals() else {},
            "success": success,
            "error": error_message,
            "circuit_breaker_state": retry_manager.get_operation_stats(f"agent_{task.agent}").get("circuit_breaker_state", "UNKNOWN"),
            "parallel_execution": True
        },
        latency_ms=latency_ms,
    )


async def _run_agent_tasks_sequential(tasks: List[AgentTask], query: Query, cfg: ProtocolConfig) -> List[AgentResult]:
    """
    Execute agents sequentially (fallback mode).
    Original sequential implementation for compatibility.
    """
    results_map: Dict[str, Any] = {}
    completed_results: List[AgentResult] = []
    
    # Initialize retry manager with circuit breakers
    retry_config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=300.0  # 5 minutes
    )
    retry_manager = RetryManager(retry_config)

    if cfg.verbose:
        print(f"[RUNNER] Sequential execution: {len(tasks)} tasks")

    for task in tasks:
        if cfg.verbose:
            try:
                print(f"[RUNNER] Executing task: {task.agent}")
            except Exception:
                print("[RUNNER] Executing task")

        start_time = time.time()
        output: Any = None
        success = False
        error_message = None

        try:
            # Resolve dependencies and build input payload
            requires = list(task.dependencies or [])
            if isinstance(task.payload, dict) and "requires" in task.payload:
                requires.extend(task.payload.get("requires") or [])

            resolved_input_data: Dict[str, Any] = dict(task.payload or {})
            for dep in requires:
                if dep in results_map:
                    resolved_input_data[f"{dep}_output"] = results_map[dep]
            # Allow agents to access raw query text if needed
            resolved_input_data.setdefault("query", getattr(query, "text", ""))

            # Get the agent runner (either direct or legacy adapter)
            agent_runner = get_agent_runner(task.agent)

            if agent_runner is None:
                raise RuntimeError(f"Agent not registered: {task.agent}")

            # Execute with circuit breaker and retry logic
            async def execute_agent():
                if asyncio.iscoroutinefunction(agent_runner):
                    return await agent_runner(cfg=cfg, **resolved_input_data)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: agent_runner(cfg=cfg, **resolved_input_data))
            
            # Execute with retry and circuit breaker
            retry_result: RetryResult = await retry_manager.execute_with_retry(
                execute_agent,
                operation_name=f"agent_{task.agent}"
            )
            
            if retry_result.success:
                output = retry_result.result
                success = True
            else:
                error_message = retry_result.final_error or "Agent execution failed"
                logger.error(f"Agent {task.agent} failed after {retry_result.attempts} attempts: {error_message}")
                
                # Check circuit breaker state
                circuit_breaker = retry_manager.get_circuit_breaker(f"agent_{task.agent}")
                if not circuit_breaker.can_execute():
                    logger.warning(f"Circuit breaker OPEN for agent {task.agent}")
                    error_message = f"Circuit breaker open: {error_message}"

        except Exception as e:
            error_message = str(e)
            logger.error(f"Error executing agent {task.agent}: {error_message}", exc_info=True)
            
            # Record failure in circuit breaker
            circuit_breaker = retry_manager.get_circuit_breaker(f"agent_{task.agent}")
            circuit_breaker.record_failure()

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        agent_result = AgentResult(
            agent=task.agent,
            payload=output,
            metadata={
                "input_data": resolved_input_data,
                "success": success,
                "error": error_message,
                "circuit_breaker_state": retry_manager.get_operation_stats(f"agent_{task.agent}").get("circuit_breaker_state", "UNKNOWN"),
                "parallel_execution": False  # Sequential execution
            },
            latency_ms=latency_ms,
        )
        completed_results.append(agent_result)
        results_map[task.agent] = output

    return completed_results
