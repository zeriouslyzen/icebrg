"""
Circuit Breaker for Cascade Prevention
Prevents cascade failures by opening circuit when issues detected
"""

import logging
import time
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit open, reject requests
    HALF_OPEN = "half_open"  # Testing if issue is resolved


class CircuitBreaker:
    """Circuit breaker for cascade prevention."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize circuit breaker.
        
        Args:
            config: Configuration for circuit breaker
        """
        self.config = config or {}
        
        # Circuit breaker configuration
        self.failure_threshold = self.config.get("failure_threshold", 5)  # Open after 5 failures
        self.success_threshold = self.config.get("success_threshold", 2)  # Close after 2 successes
        self.timeout = self.config.get("timeout", 60.0)  # 60 seconds timeout
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        # Check if timeout has passed (for half-open state)
        if self.state == CircuitState.HALF_OPEN:
            if time.time() - self.last_state_change > self.timeout:
                # Timeout passed, try closing
                if self.success_count >= self.success_threshold:
                    self.close()
                else:
                    self.open()
        
        return self.state == CircuitState.OPEN
    
    def record_success(self):
        """Record successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.close()
        elif self.state == CircuitState.OPEN:
            # Transition to half-open
            self.state = CircuitState.HALF_OPEN
            self.success_count = 1
            self.last_state_change = time.time()
            logger.info("Circuit breaker transitioning to half-open state")
        else:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.open()
    
    def open(self):
        """Open circuit breaker."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            self.failure_count = 0
            logger.warning("Circuit breaker opened")
    
    def close(self):
        """Close circuit breaker."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()
            self.failure_count = 0
            self.success_count = 0
            logger.info("Circuit breaker closed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "is_open": self.is_open()
        }

