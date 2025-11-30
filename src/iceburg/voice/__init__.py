"""
Iceberg Protocol - Voice Conversation Module

This module provides voice input/output capabilities for natural conversation
with the AI system, implementing the AI's own conversation optimization solution.
"""

from .real_voice_input import RealVoiceInputModule, RealVoiceInput
from .real_voice_output import RealVoiceOutputModule, RealVoiceOutput
from .real_voice_conversation import RealVoiceConversationSystem, RealVoiceConversation

__all__ = [
    "RealVoiceInputModule",
    "RealVoiceInput",
    "RealVoiceOutputModule", 
    "RealVoiceOutput",
    "RealVoiceConversationSystem",
    "RealVoiceConversation"
]

__version__ = "1.0.0"
__author__ = "Iceberg Protocol Team"
__description__ = "Voice conversation capabilities implementing AI's own optimization solution"
