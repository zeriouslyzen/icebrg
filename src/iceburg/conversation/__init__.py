"""
ICEBURG Conversation Evolution System
Builds conversation structure that accumulates knowledge
"""

from .conversation_structure_builder import ConversationStructureBuilder
from .proactive_gap_filling import ProactiveGapFilling

__all__ = [
    "ConversationStructureBuilder",
    "ProactiveGapFilling",
]

