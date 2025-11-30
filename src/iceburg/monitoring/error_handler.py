"""
Comprehensive Error Handling and Recovery System for ICEBURG Autonomous System
"""

import asyncio
import logging
import traceback
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import functools


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    DATABASE = "database"
    COMPUTATION = "computation"
    MEMORY = "memory"
    CONCURRENCY = "concurrency"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RecoveryAction:
    """Recovery action to take for an error."""
    action_type: str  # retry, fallback, skip, abort
    delay: float = 0.0
    max_attempts: int = 1
    fallback_function: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Initialize recovery strategies
        self._init_recovery_strategies()
    
    def _init_recovery_strategies(self):
        """Initialize default recovery strategies."""
        self.recovery_strategies = {
            "network": RecoveryAction(
                action_type="retry",
                delay=1.0,
                max_attempts=3,
                metadata={"exponential_backoff": True}
            ),
            "database": RecoveryAction(
                action_type="retry",
                delay=0.5,
                max_attempts=5,
                metadata={"connection_pool": True}
            ),
            "computation": RecoveryAction(
                action_type="fallback",
                delay=0.0,
                max_attempts=1,
                metadata={"graceful_degradation": True}
            ),
            "memory": RecoveryAction(
                action_type="retry",
                delay=2.0,
                max_attempts=2,
                metadata={"gc_forced": True}
            ),
            "concurrency": RecoveryAction(
                action_type="retry",
                delay=0.1,
                max_attempts=3,
                metadata={"timeout_increase": True}
            ),
            "validation": RecoveryAction(
                action_type="skip",
                delay=0.0,
                max_attempts=1,
                metadata={"log_and_continue": True}
            ),
            "configuration": RecoveryAction(
                action_type="abort",
                delay=0.0,
                max_attempts=1,
                metadata={"requires_restart": True}
            )
        }
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error based on exception type and message."""
        error_name = type(error).__name__.lower()
        error_message = str(error).lower()
        
        if any(keyword in error_name or keyword in error_message for keyword in 
               ["connection", "network", "timeout", "socket", "http"]):
            return ErrorCategory.NETWORK
        elif any(keyword in error_name or keyword in error_message for keyword in 
                 ["database", "sqlite", "sql", "db", "connection"]):
            return ErrorCategory.DATABASE
        elif any(keyword in error_name or keyword in error_message for keyword in 
                 ["memory", "oom", "allocation", "buffer"]):
            return ErrorCategory.MEMORY
        elif any(keyword in error_name or keyword in error_message for keyword in 
                 ["concurrent", "async", "await", "task", "future"]):
            return ErrorCategory.CONCURRENCY
        elif any(keyword in error_name or keyword in error_message for keyword in 
                 ["validation", "invalid", "format", "type"]):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_name or keyword in error_message for keyword in 
                 ["config", "setting", "parameter", "option"]):
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN
    
    def determine_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Determine error severity based on context and error type."""
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorSeverity.HIGH
        elif context.retry_count >= context.max_retries:
            return ErrorSeverity.HIGH
        elif context.category in [ErrorCategory.NETWORK, ErrorCategory.DATABASE]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def should_retry(self, context: ErrorContext) -> bool:
        """Determine if operation should be retried."""
        if context.retry_count >= context.max_retries:
            return False
        
        if context.severity == ErrorSeverity.CRITICAL:
            return False
        
        if context.category == ErrorCategory.CONFIGURATION:
            return False
        
        # Check circuit breaker
        circuit_key = f"{context.component}_{context.operation}"
        if circuit_key in self.circuit_breakers:
            breaker = self.circuit_breakers[circuit_key]
            if breaker["state"] == "open":
                return False
        
        return True
    
    def get_recovery_action(self, context: ErrorContext) -> RecoveryAction:
        """Get recovery action for error context."""
        strategy_key = context.category.value
        if strategy_key in self.recovery_strategies:
            return self.recovery_strategies[strategy_key]
        else:
            return RecoveryAction(
                action_type="retry",
                delay=1.0,
                max_attempts=3
            )
    
    async def handle_error(self, 
                          error: Exception, 
                          context: ErrorContext,
                          original_function: Optional[Callable] = None) -> Any:
        """Handle error with appropriate recovery action."""
        # Update context
        context.severity = self.determine_severity(error, context)
        context.timestamp = time.time()
        
        # Log error
        self.logger.error(
            f"Error in {context.component}.{context.operation}: {error}",
            extra={
                "error_type": type(error).__name__,
                "severity": context.severity.value,
                "category": context.category.value,
                "retry_count": context.retry_count,
                "traceback": traceback.format_exc()
            }
        )
        
        # Add to error history
        self.error_history.append(context)
        
        # Get recovery action
        recovery = self.get_recovery_action(context)
        
        if recovery.action_type == "retry" and self.should_retry(context):
            context.retry_count += 1
            await asyncio.sleep(recovery.delay)
            
            if original_function:
                return await self.execute_with_error_handling(
                    original_function, 
                    context
                )
            else:
                raise error
        
        elif recovery.action_type == "fallback" and recovery.fallback_function:
            try:
                return await recovery.fallback_function()
            except Exception as fallback_error:
                self.logger.error(f"Fallback function failed: {fallback_error}")
                raise error
        
        elif recovery.action_type == "skip":
            self.logger.warning(f"Skipping {context.component}.{context.operation}")
            return None
        
        elif recovery.action_type == "abort":
            self.logger.critical(f"Aborting due to critical error in {context.component}.{context.operation}")
            raise error
        
        else:
            raise error
    
    async def execute_with_error_handling(self, 
                                        func: Callable, 
                                        context: ErrorContext,
                                        *args, **kwargs) -> Any:
        """Execute function with comprehensive error handling."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as error:
            return await self.handle_error(error, context, func)
    
    def with_error_handling(self, 
                           component: str, 
                           operation: str,
                           category: Optional[ErrorCategory] = None,
                           max_retries: int = 3):
        """Decorator for automatic error handling."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                error_category = category or self.classify_error(Exception())
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    timestamp=time.time(),
                    severity=ErrorSeverity.LOW,
                    category=error_category,
                    max_retries=max_retries
                )
                
                return await self.execute_with_error_handling(func, context, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def update_circuit_breaker(self, component: str, operation: str, success: bool):
        """Update circuit breaker state."""
        circuit_key = f"{component}_{operation}"
        
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                "state": "closed",
                "failure_count": 0,
                "success_count": 0,
                "last_failure": None
            }
        
        breaker = self.circuit_breakers[circuit_key]
        
        if success:
            breaker["success_count"] += 1
            breaker["failure_count"] = 0
            if breaker["state"] == "open":
                breaker["state"] = "half_open"
        else:
            breaker["failure_count"] += 1
            breaker["last_failure"] = time.time()
            
            # Open circuit if failure threshold exceeded
            if breaker["failure_count"] >= 5:
                breaker["state"] = "open"
                self.logger.warning(f"Circuit breaker opened for {circuit_key}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health metrics."""
        if not self.error_history:
            return {"total_errors": 0, "health_score": 100.0}
        
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        severity_counts = {}
        category_counts = {}
        
        for error in recent_errors:
            severity = error.severity.value
            category = error.category.value
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate health score (0-100)
        critical_errors = severity_counts.get("critical", 0)
        high_errors = severity_counts.get("high", 0)
        medium_errors = severity_counts.get("medium", 0)
        low_errors = severity_counts.get("low", 0)
        
        health_score = max(0, 100 - (critical_errors * 20 + high_errors * 10 + medium_errors * 5 + low_errors * 1))
        
        return {
            "total_errors": total_errors,
            "recent_errors": len(recent_errors),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "health_score": health_score,
            "circuit_breakers": len([b for b in self.circuit_breakers.values() if b["state"] == "open"])
        }


# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def with_error_handling(component: str, operation: str, **kwargs):
    """Convenience function for error handling decorator."""
    return get_global_error_handler().with_error_handling(component, operation, **kwargs)
