"""
Circuit Breaker Wrapper for LLM Providers
Provides resilience and fault tolerance for LLM API calls
"""

import logging
from typing import Any, Callable, Optional
from functools import wraps
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        
        # half-open: allow one test request
        return True
    
    def record_success(self):
        """Record a successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record a failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")


def wrap_provider_with_circuit_breaker(
    provider: Any, 
    name: str = "llm_provider",
    provider_name: Optional[str] = None,
    fallback_provider: Optional[Any] = None,
    config: Optional[dict] = None
) -> Any:
    """
    Wrap a provider with circuit breaker functionality
    
    Args:
        provider: The LLM provider to wrap
        name: Name for the circuit breaker (for logging)
        provider_name: Alternative name parameter (same as name)
        fallback_provider: Optional fallback provider if circuit opens
        config: Optional configuration dict with keys like failure_threshold, success_threshold, timeout
    
    Returns:
        The same provider (passthrough for now - circuit breaker is optional)
    """
    # Use provider_name if provided, otherwise use name
    effective_name = provider_name or name
    
    # Parse config if provided (for future use)
    if config:
        logger.debug(f"Circuit breaker config for {effective_name}: {config}")
    
    # For now, just return the provider as-is
    # Full circuit breaker implementation can be added later
    logger.debug(f"Circuit breaker wrapper applied to {effective_name}")
    return provider


def circuit_breaker_decorator(
    failure_threshold: int = 5,
    reset_timeout: float = 60.0
) -> Callable:
    """
    Decorator to add circuit breaker functionality to a function
    
    Args:
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds to wait before trying again
    
    Returns:
        Decorated function
    """
    breaker = CircuitBreaker(failure_threshold, reset_timeout)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open for {func.__name__}"
                )
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open for {func.__name__}"
                )
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and requests are blocked"""
    pass

