"""
ICEBURG Retry Manager
Implements intelligent retry logic with exponential backoff and circuit breakers
"""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum


class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0  # 5 minutes


@dataclass
class RetryResult:
    """Result of retry operation"""
    success: bool
    result: Any
    attempts: int
    total_time: float
    errors: list[str]
    final_error: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, threshold: int = 5, timeout: float = 300.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = "OPEN"


class RetryManager:
    """Intelligent retry manager with circuit breaker support"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation"""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(
                threshold=self.config.circuit_breaker_threshold,
                timeout=self.config.circuit_breaker_timeout
            )
        return self.circuit_breakers[operation_name]
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.config.strategy == RetryStrategy.NO_RETRY:
            return 0
        
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        else:  # EXPONENTIAL_BACKOFF
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
        
        return delay
    
    async def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str = "operation",
        *args,
        **kwargs
    ) -> RetryResult:
        """
        Execute operation with retry logic and circuit breaker
        
        Args:
            operation: Function to execute
            operation_name: Name for circuit breaker tracking
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation
        
        Returns:
            RetryResult with success status and details
        """
        circuit_breaker = self.get_circuit_breaker(operation_name)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            return RetryResult(
                success=False,
                result=None,
                attempts=0,
                total_time=0,
                errors=[f"Circuit breaker OPEN for {operation_name}"],
                final_error="Circuit breaker is open"
            )
        
        start_time = time.time()
        errors = []
        last_error = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Success - record in circuit breaker
                circuit_breaker.record_success()
                
                total_time = time.time() - start_time
                
                if attempt > 1:
                    self.logger.info(f"Operation {operation_name} succeeded on attempt {attempt}")
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_time=total_time,
                    errors=errors
                )
                
            except Exception as e:
                last_error = str(e)
                errors.append(f"Attempt {attempt}: {last_error}")
                
                self.logger.warning(f"Operation {operation_name} failed on attempt {attempt}: {e}")
                
                # Record failure in circuit breaker
                circuit_breaker.record_failure()
                
                # If this is the last attempt, don't wait
                if attempt == self.config.max_retries:
                    break
                
                # Calculate delay and wait
                delay = self.calculate_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        
        # All retries failed
        total_time = time.time() - start_time
        
        return RetryResult(
            success=False,
            result=None,
            attempts=self.config.max_retries,
            total_time=total_time,
            errors=errors,
            final_error=last_error
        )
    
    def execute_sync_with_retry(
        self,
        operation: Callable,
        operation_name: str = "operation",
        *args,
        **kwargs
    ) -> RetryResult:
        """
        Synchronous version of execute_with_retry
        """
        return asyncio.run(self.execute_with_retry(operation, operation_name, *args, **kwargs))
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for an operation"""
        circuit_breaker = self.get_circuit_breaker(operation_name)
        
        return {
            "operation_name": operation_name,
            "circuit_breaker_state": circuit_breaker.state,
            "failure_count": circuit_breaker.failure_count,
            "last_failure_time": circuit_breaker.last_failure_time,
            "can_execute": circuit_breaker.can_execute()
        }
    
    def reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker for operation"""
        if operation_name in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(
                threshold=self.config.circuit_breaker_threshold,
                timeout=self.config.circuit_breaker_timeout
            )


# Global retry manager instance
_retry_manager = RetryManager()


async def retry_operation(
    operation: Callable,
    operation_name: str = "operation",
    max_retries: int = 3,
    base_delay: float = 1.0,
    *args,
    **kwargs
) -> RetryResult:
    """
    Convenience function for retrying operations
    
    Args:
        operation: Function to execute
        operation_name: Name for tracking
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        *args: Arguments for operation
        **kwargs: Keyword arguments for operation
    
    Returns:
        RetryResult with success status and details
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay
    )
    
    retry_manager = RetryManager(config)
    return await retry_manager.execute_with_retry(operation, operation_name, *args, **kwargs)


def retry_operation_sync(
    operation: Callable,
    operation_name: str = "operation",
    max_retries: int = 3,
    base_delay: float = 1.0,
    *args,
    **kwargs
) -> RetryResult:
    """
    Synchronous convenience function for retrying operations
    """
    return asyncio.run(retry_operation(operation, operation_name, max_retries, base_delay, *args, **kwargs))
