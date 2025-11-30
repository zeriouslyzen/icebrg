"""
ICEBURG User Evolution System
Adapts and evolves with each user through conversation patterns
"""

from .user_profile_builder import UserProfileBuilder, UserProfile
from .conversation_pattern_learning import ConversationPatternLearning
from .adaptive_capability_discovery import AdaptiveCapabilityDiscovery

__all__ = [
    "UserProfileBuilder",
    "UserProfile",
    "ConversationPatternLearning",
    "AdaptiveCapabilityDiscovery",
]
