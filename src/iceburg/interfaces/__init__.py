"""
ICEBURG Interfaces Module
Defines clear API contracts and interfaces for service-oriented architecture
"""

from .service_interface import IService, ServiceBase
from .protocol_interface import IProtocol, ProtocolBase
from .agent_interface import IAgent, AgentBase
from .storage_interface import IStorage, StorageBase

__all__ = [
    "IService",
    "ServiceBase",
    "IProtocol",
    "ProtocolBase",
    "IAgent",
    "AgentBase",
    "IStorage",
    "StorageBase",
]

