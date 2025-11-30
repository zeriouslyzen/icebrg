"""
Local Persona Instance
User-specific persona instances that handle simple queries locally without hitting main process.
Based on always-on AI architecture patterns for local persona instances.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LocalPersonaInstance:
    """
    Local persona instance per user that handles simple queries locally.
    
    Architecture:
    - User-specific persona per user_id
    - Local knowledge base per user
    - Personal preferences and tuning
    - Instant responses for simple queries
    - Escalation to main process when needed
    """
    
    def __init__(self, user_id: str, config: Optional[Any] = None):
        self.user_id = user_id
        self.config = config
        self.persona: Dict[str, Any] = {}
        self.local_kb: Dict[str, Any] = {}
        self.preferences: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.response_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.stats = {
            "local_responses": 0,
            "escalations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0
        }
        
        # Load persona data
        self._load_persona()
        self._load_local_knowledge()
        self._load_preferences()
        
        logger.info(f"LocalPersonaInstance initialized for user: {user_id}")
    
    def _load_persona(self):
        """Load user-specific persona"""
        try:
            # Try to load from local persistence
            from ..storage.local_persistence import LocalPersistence
            persistence = LocalPersistence()
            
            # Load personality state (use load_personality which returns PersonalityState object)
            try:
                personality_obj = persistence.load_personality()
                # Convert PersonalityState to dict
                from dataclasses import asdict
                personality = asdict(personality_obj) if personality_obj else None
            except Exception:
                personality = None
            
            if personality:
                self.persona = {
                    "identity": personality.get("identity", "ICEBURG"),
                    "personality_traits": personality.get("personality_traits", {}),
                    "preferences": personality.get("preferences", {}),
                    "knowledge_base": personality.get("knowledge_base", {}),
                    "memory_context": personality.get("memory_context", {})
                }
            else:
                # Default persona
                self.persona = {
                    "identity": "ICEBURG",
                    "personality_traits": {},
                    "preferences": {},
                    "knowledge_base": {},
                    "memory_context": {}
                }
            
            logger.debug(f"Loaded persona for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error loading persona: {e}", exc_info=True)
            # Default persona
            self.persona = {
                "identity": "ICEBURG",
                "personality_traits": {},
                "preferences": {},
                "knowledge_base": {},
                "memory_context": {}
            }
    
    def _load_local_knowledge(self):
        """Load local knowledge base"""
        try:
            # Try to load from local persistence
            from ..storage.local_persistence import LocalPersistence
            persistence = LocalPersistence()
            
            # Load knowledge base (use load_knowledge if available, otherwise default)
            try:
                if hasattr(persistence, 'load_knowledge'):
                    knowledge = persistence.load_knowledge()
                elif hasattr(persistence, 'get_knowledge'):
                    knowledge = persistence.get_knowledge()
                else:
                    knowledge = None
            except Exception:
                knowledge = None
            
            if knowledge and isinstance(knowledge, dict):
                self.local_kb = knowledge
            else:
                self.local_kb = {}
            
            logger.debug(f"Loaded local knowledge base for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error loading local knowledge: {e}", exc_info=True)
            self.local_kb = {}
    
    def _load_preferences(self):
        """Load user preferences"""
        try:
            # Try to load from local persistence
            from ..storage.local_persistence import LocalPersistence
            persistence = LocalPersistence()
            
            # Load personality state for preferences (use load_personality if available, otherwise default)
            try:
                if hasattr(persistence, 'load_personality'):
                    personality = persistence.load_personality()
                elif hasattr(persistence, 'get_personality'):
                    personality = persistence.get_personality()
                else:
                    personality = None
            except Exception:
                personality = None
            
            if personality and isinstance(personality, dict):
                self.preferences = personality.get("preferences", {})
            else:
                self.preferences = {}
            
            logger.debug(f"Loaded preferences for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error loading preferences: {e}", exc_info=True)
            self.preferences = {}
    
    async def respond(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Respond using local persona (instant).
        
        Returns:
            Response dict if query can be answered locally, None if escalation needed
        """
        self.stats["total_queries"] += 1
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_response
            
            self.stats["cache_misses"] += 1
            
            # Check if query can be answered locally
            if self._can_answer_locally(query):
                # Generate local response
                response = await self._generate_local_response(query)
                
                # Only return if response was generated (not None)
                if response:
                    # Cache response
                    self.response_cache[cache_key] = response
                    
                    # Update conversation history
                    self.conversation_history.append({
                        "query": query,
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "source": "local_persona"
                    })
                    
                    self.stats["local_responses"] += 1
                    logger.debug(f"Local response generated for query: {query[:50]}...")
                    return response
                else:
                    # Response generation returned None - escalate
                    self.stats["escalations"] += 1
                    logger.debug(f"Query escalated (no local response): {query[:50]}...")
                    return None
            else:
                # Query too complex, need escalation
                self.stats["escalations"] += 1
                logger.debug(f"Query escalated to main process: {query[:50]}...")
                return None
                
        except Exception as e:
            logger.error(f"Error in local persona response: {e}", exc_info=True)
            # Escalate on error
            self.stats["escalations"] += 1
            return None
    
    def _can_answer_locally(self, query: str) -> bool:
        """Check if query can be answered locally"""
        query_lower = query.lower().strip()
        
        # Simple queries that can be answered locally
        simple_patterns = [
            "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
            "what is", "define", "explain", "tell me about",
            "how are you", "what can you do", "who are you"
        ]
        
        # Check if query matches simple patterns
        for pattern in simple_patterns:
            if pattern in query_lower:
                return True
        
        # Check if query is in local knowledge base
        if query_lower in self.local_kb:
            return True
        
        # Check if query is very short (likely simple)
        if len(query.split()) <= 3:
            return True
        
        # Otherwise, too complex for local response
        return False
    
    async def _generate_local_response(self, query: str) -> Dict[str, Any]:
        """Generate local response using persona and knowledge base - INSTANT"""
        query_lower = query.lower().strip()
        
        # Simple greeting responses - INSTANT
        if any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
            return {
                "response": "Hello! How can I help you today?",
                "source": "local_persona",
                "confidence": 1.0,
                "response_time": 0.0
            }
        
        # Simple goodbye responses - INSTANT
        if any(goodbye in query_lower for goodbye in ["bye", "goodbye", "thanks", "thank you"]):
            return {
                "response": "You're welcome! Feel free to ask if you need anything else.",
                "source": "local_persona",
                "confidence": 1.0,
                "response_time": 0.0
            }
        
        # Check local knowledge base - INSTANT
        if query_lower in self.local_kb:
            knowledge = self.local_kb[query_lower]
            return {
                "response": str(knowledge),
                "source": "local_kb",
                "confidence": 0.9,
                "response_time": 0.0
            }
        
        # For anything else, don't respond locally - escalate immediately
        # This prevents delays from trying to process complex queries locally
        return None
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def update_persona(self, updates: Dict[str, Any]):
        """Update persona with new information"""
        try:
            self.persona.update(updates)
            
            # Save to local persistence
            from ..storage.local_persistence import LocalPersistence
            persistence = LocalPersistence()
            
            # Update personality state
            personality = persistence.get_personality()
            if personality:
                personality.update(updates)
                persistence.save_personality(personality)
            
            logger.debug(f"Updated persona for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error updating persona: {e}", exc_info=True)
    
    def update_local_kb(self, key: str, value: Any):
        """Update local knowledge base"""
        try:
            self.local_kb[key] = value
            
            # Save to local persistence
            from ..storage.local_persistence import LocalPersistence
            persistence = LocalPersistence()
            persistence.save_knowledge(key, value)
            
            logger.debug(f"Updated local KB for user {self.user_id}: {key}")
            
        except Exception as e:
            logger.error(f"Error updating local KB: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persona statistics"""
        return {
            **self.stats,
            "user_id": self.user_id,
            "persona_loaded": bool(self.persona),
            "local_kb_size": len(self.local_kb),
            "conversation_history_size": len(self.conversation_history),
            "cache_size": len(self.response_cache)
        }

