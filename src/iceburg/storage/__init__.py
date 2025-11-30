"""
ICEBURG Local Persistence System
Comprehensive local storage for personality, information, and all interactions
Similar to browser storage but for long-term persistence on M4 Mac
"""

from .local_persistence import (
    LocalPersistence,
    PersonalityState,
    ConversationEntry,
    ResearchEntry
)

__all__ = [
    "LocalPersistence",
    "PersonalityState",
    "ConversationEntry",
    "ResearchEntry"
]
