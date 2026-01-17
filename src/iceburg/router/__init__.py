"""
ICEBURG Routing Module
Classifies and routes queries to specialized handlers and expert models.
"""

from .request_router import RequestRouter, get_request_router, RoutingDecision
from .moe_router import MoERouter, get_moe_router, MoEDecision
from .local_rag_router import LocalRAGRouter

__all__ = [
    "RequestRouter",
    "get_request_router",
    "RoutingDecision",
    "MoERouter",
    "get_moe_router",
    "MoEDecision",
    "LocalRAGRouter"
]




