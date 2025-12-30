"""
Router module for ICEBURG v5
Request routing and query classification
"""

from .request_router import (
    RequestRouter,
    RoutingDecision,
    get_request_router
)

__all__ = [
    "RequestRouter",
    "RoutingDecision",
    "get_request_router"
]

