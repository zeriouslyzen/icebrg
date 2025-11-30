"""
User Profile Builder
Builds user profile from conversations and learns preferences
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

from ..memory.unified_memory import UnifiedMemory
from ..config import IceburgConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Represents a user profile"""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    domain_expertise: Dict[str, float] = field(default_factory=dict)
    conversation_patterns: Dict[str, Any] = field(default_factory=dict)
    total_conversations: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class UserProfileBuilder:
    """
    Builds user profile from conversations.
    
    Learns user preferences, interests, patterns, and adapts to user's domain expertise.
    Evolves with each conversation.
    """
    
    def __init__(self, cfg: Optional[IceburgConfig] = None):
        """
        Initialize user profile builder.
        
        Args:
            cfg: ICEBURG config (loads if None)
        """
        self.cfg = cfg or load_config()
        self.memory = UnifiedMemory(self.cfg)
        self.profiles: Dict[str, UserProfile] = {}
        self._profiles_path = Path(self.cfg.data_dir) / "profiles"
        self._profiles_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("User Profile Builder initialized")
    
    def build_user_profile(self, conversations: List[Dict[str, Any]], user_id: str = "default") -> UserProfile:
        """
        Build user profile from conversations.
        
        Args:
            conversations: List of conversation dictionaries
            user_id: User ID (default: "default")
            
        Returns:
            User profile
        """
        # Load existing profile or create new
        if user_id in self.profiles:
            profile = self.profiles[user_id]
        else:
            profile = UserProfile(user_id=user_id)
            self.profiles[user_id] = profile
        
        # Process conversations
        for conversation in conversations:
            # Learn preferences
            preferences = self.learn_user_preferences(conversation)
            profile.preferences.update(preferences)
            
            # Identify interests
            interests = self.identify_user_interests(conversation)
            for interest in interests:
                if interest not in profile.interests:
                    profile.interests.append(interest)
            
            # Adapt to domain
            domain_expertise = self.adapt_to_user_domain(conversation)
            for domain, expertise in domain_expertise.items():
                profile.domain_expertise[domain] = profile.domain_expertise.get(domain, 0.0) + expertise
        
        # Update profile
        profile.total_conversations += len(conversations)
        profile.last_updated = datetime.utcnow().isoformat()
        
        # Save profile
        self._save_profile(profile)
        
        logger.info(f"Built user profile for {user_id}: {len(profile.interests)} interests, "
                   f"{len(profile.domain_expertise)} domains, {profile.total_conversations} conversations")
        return profile
    
    def learn_user_preferences(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn user preferences from conversation.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Dictionary of learned preferences
        """
        preferences = {}
        
        query = conversation.get("query", "")
        response = conversation.get("response", "")
        metadata = conversation.get("metadata", {})
        
        # Learn response style preferences
        if "detailed" in query.lower() or "explain" in query.lower():
            preferences["response_style"] = "detailed"
        elif "brief" in query.lower() or "short" in query.lower():
            preferences["response_style"] = "brief"
        elif "example" in query.lower():
            preferences["response_style"] = "example_based"
        
        # Learn format preferences
        if "code" in query.lower():
            preferences["format"] = "code"
        elif "chart" in query.lower() or "graph" in query.lower():
            preferences["format"] = "visual"
        elif "table" in query.lower():
            preferences["format"] = "table"
        
        # Learn domain preferences
        domains = metadata.get("domains", [])
        if domains:
            preferences["preferred_domains"] = domains
        
        return preferences
    
    def identify_user_interests(self, conversation: Dict[str, Any]) -> List[str]:
        """
        Identify user interests from conversation.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            List of identified interests
        """
        interests = []
        
        query = conversation.get("query", "")
        metadata = conversation.get("metadata", {})
        
        # Extract domains as interests
        domains = metadata.get("domains", [])
        interests.extend(domains)
        
        # Extract keywords as interests
        query_lower = query.lower()
        interest_keywords = [
            "physics", "chemistry", "biology", "mathematics", "computer science",
            "astronomy", "psychology", "sociology", "economics", "philosophy",
            "astrology", "marketing", "data", "analysis", "research"
        ]
        
        for keyword in interest_keywords:
            if keyword in query_lower:
                if keyword not in interests:
                    interests.append(keyword)
        
        return interests
    
    def adapt_to_user_domain(self, conversation: Dict[str, Any]) -> Dict[str, float]:
        """
        Adapt to user's domain expertise.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Dictionary of domain expertise levels
        """
        domain_expertise = {}
        
        query = conversation.get("query", "")
        response = conversation.get("response", "")
        metadata = conversation.get("metadata", {})
        
        # Extract domains
        domains = metadata.get("domains", [])
        
        # Estimate expertise based on query complexity
        query_complexity = len(query.split()) / 10.0  # Simple heuristic
        
        for domain in domains:
            domain_expertise[domain] = min(query_complexity, 1.0)
        
        return domain_expertise
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        if user_id not in self.profiles:
            self._load_profile(user_id)
        return self.profiles.get(user_id)
    
    def _save_profile(self, profile: UserProfile) -> None:
        """Save user profile to disk."""
        profile_path = self._profiles_path / f"{profile.user_id}.json"
        
        try:
            data = {
                "user_id": profile.user_id,
                "preferences": profile.preferences,
                "interests": profile.interests,
                "domain_expertise": profile.domain_expertise,
                "conversation_patterns": profile.conversation_patterns,
                "total_conversations": profile.total_conversations,
                "created_at": profile.created_at,
                "last_updated": profile.last_updated
            }
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving profile {profile.user_id}: {e}")
    
    def _load_profile(self, user_id: str) -> None:
        """Load user profile from disk."""
        profile_path = self._profiles_path / f"{user_id}.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    profile = UserProfile(
                        user_id=data.get("user_id", user_id),
                        preferences=data.get("preferences", {}),
                        interests=data.get("interests", []),
                        domain_expertise=data.get("domain_expertise", {}),
                        conversation_patterns=data.get("conversation_patterns", {}),
                        total_conversations=data.get("total_conversations", 0),
                        created_at=data.get("created_at", datetime.utcnow().isoformat()),
                        last_updated=data.get("last_updated", datetime.utcnow().isoformat())
                    )
                    
                    self.profiles[user_id] = profile
            except Exception as e:
                logger.warning(f"Error loading profile {user_id}: {e}")

