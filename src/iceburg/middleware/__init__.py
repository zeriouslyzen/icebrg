"""
Global Agent Middleware
Platform-wide middleware for hallucination detection and emergence tracking.
"""

from .global_agent_middleware import GlobalAgentMiddleware
from .middleware_registry import MiddlewareRegistry

__all__ = ['GlobalAgentMiddleware', 'MiddlewareRegistry']

