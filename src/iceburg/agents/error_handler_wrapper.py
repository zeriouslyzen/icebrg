"""
Consistent Error Handling Wrapper for Agents

Provides standardized error handling, circuit breaker integration,
and retry logic for all agent executions.
"""

from typing import Callable, Any, Optional, Dict, TypeVar, Awaitable
from functools import wraps
import logging
import asyncio
from ..infrastructure.retry_manager import RetryManager, RetryConfig, RetryResult
from ..monitoring.error_handler import ErrorHandler, ErrorContext, ErrorCategory

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentErrorHandler:
    """
    Consistent error handling wrapper for agent execution.
    
    Provides:
    - Circuit breaker integration
    - Retry logic with exponential backoff
    - Error classification and recovery
    - Graceful degradation
    """
    
    def __init__(self):
        self.retry_manager = RetryManager(
            RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=30.0,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=300.0
            )
        )
        self.error_handler = ErrorHandler()
    
    async def execute_agent(
        self,
        agent_func: Callable[..., Awaitable[T] | T],
        agent_name: str,
        *args,
        **kwargs
    ) -> tuple[bool, T | None, Optional[str]]:
        """
        Execute agent function with consistent error handling.
        
        Args:
            agent_func: Agent function to execute
            agent_name: Name of agent for tracking
            *args: Arguments for agent function
            **kwargs: Keyword arguments for agent function
            
        Returns:
            Tuple of (success, result, error_message)
        """
        async def execute():
            if asyncio.iscoroutinefunction(agent_func):
                return await agent_func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: agent_func(*args, **kwargs))
        
        try:
            retry_result: RetryResult = await self.retry_manager.execute_with_retry(
                execute,
                operation_name=f"agent_{agent_name}"
            )
            
            if retry_result.success:
                return True, retry_result.result, None
            else:
                error_message = retry_result.final_error or "Agent execution failed"
                
                # Check circuit breaker state
                circuit_breaker = self.retry_manager.get_circuit_breaker(f"agent_{agent_name}")
                if not circuit_breaker.can_execute():
                    error_message = f"Circuit breaker open: {error_message}"
                
                logger.error(f"Agent {agent_name} failed after {retry_result.attempts} attempts: {error_message}")
                return False, None, error_message
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error executing agent {agent_name}: {error_message}", exc_info=True)
            
            # Record failure in circuit breaker
            circuit_breaker = self.retry_manager.get_circuit_breaker(f"agent_{agent_name}")
            circuit_breaker.record_failure()
            
            # Classify error and determine recovery action
            error_context = ErrorContext(
                component=agent_name,
                operation="execute",
                error_type=type(e).__name__,
                error_message=error_message,
                retry_count=0,
                max_retries=3
            )
            
            error_category = self.error_handler.classify_error(e)
            error_context.category = error_category
            
            recovery_action = self.error_handler.get_recovery_action(error_context)
            
            if recovery_action.action_type == "retry" and recovery_action.max_attempts > 0:
                logger.info(f"Retrying agent {agent_name} with recovery action: {recovery_action.action_type}")
                # Retry logic is handled by retry_manager
            
            return False, None, error_message
    
    def execute_agent_sync(
        self,
        agent_func: Callable[..., T],
        agent_name: str,
        *args,
        **kwargs
    ) -> tuple[bool, T | None, Optional[str]]:
        """
        Synchronous version of execute_agent.
        
        Args:
            agent_func: Agent function to execute
            agent_name: Name of agent for tracking
            *args: Arguments for agent function
            **kwargs: Keyword arguments for agent function
            
        Returns:
            Tuple of (success, result, error_message)
        """
        return asyncio.run(self.execute_agent(agent_func, agent_name, *args, **kwargs))


# Global error handler instance
_error_handler: Optional[AgentErrorHandler] = None


def get_agent_error_handler() -> AgentErrorHandler:
    """Get or create global agent error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = AgentErrorHandler()
    return _error_handler


def with_error_handling(agent_name: Optional[str] = None):
    """
    Decorator for consistent error handling in agent functions.
    
    Usage:
        @with_error_handling(agent_name="surveyor")
        async def run_surveyor(cfg, vs, query):
            # agent code
            return result
    """
    def decorator(func: Callable) -> Callable:
        nonlocal agent_name
        if agent_name is None:
            agent_name = func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = get_agent_error_handler()
            success, result, error = await error_handler.execute_agent(
                func,
                agent_name,
                *args,
                **kwargs
            )
            
            if success:
                return result
            else:
                # Return error result or raise exception based on agent preference
                logger.error(f"Agent {agent_name} failed: {error}")
                raise RuntimeError(f"Agent {agent_name} execution failed: {error}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = get_agent_error_handler()
            success, result, error = error_handler.execute_agent_sync(
                func,
                agent_name,
                *args,
                **kwargs
            )
            
            if success:
                return result
            else:
                logger.error(f"Agent {agent_name} failed: {error}")
                raise RuntimeError(f"Agent {agent_name} execution failed: {error}")
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

