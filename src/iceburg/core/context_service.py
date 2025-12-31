"""
ICEBURG Context Service

Centralized context management for all ICEBURG agents.
Provides runtime state, memory, and grounding to ensure agents have:
- Current date/time awareness
- Agent identity and capabilities
- User preferences and history
- Conversation context

Based on 2024-2025 industry best practices:
- Context Engineering (Anthropic, LangChain)
- Stateful agent architecture
- Multi-layer memory systems
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ContextService:
    """
    Centralized context management for all ICEBURG agents.
    
    Provides four core capabilities:
    1. Runtime Context - Current date/time, agent identity, system state
    2. Short-term Memory - Recent conversation history
    3. Long-term Memory - Cross-session user preferences and facts
    4. Interaction Storage - Persist all interactions for learning
    """
    
    def __init__(self, cfg=None):
        """
        Initialize Context Service.
        
        Args:
            cfg: ICEBURG configuration object
        """
        self.cfg = cfg
        self.local_persistence = None
        self.unified_memory = None
        
        # Initialize persistence layer
        try:
            from ..storage.local_persistence import LocalPersistence
            self.local_persistence = LocalPersistence()
            logger.info("✅ ContextService: LocalPersistence initialized")
        except Exception as e:
            logger.warning(f"ContextService: Could not initialize LocalPersistence: {e}")
        
        # Initialize unified memory
        try:
            from ..memory.unified_memory import UnifiedMemory
            self.unified_memory = UnifiedMemory(cfg) if cfg else None
            if self.unified_memory:
                logger.info("✅ ContextService: UnifiedMemory initialized")
        except Exception as e:
            logger.warning(f"ContextService: Could not initialize UnifiedMemory: {e}")
        
        logger.info("🎯 ContextService initialized")
    
    def get_runtime_context(
        self,
        agent_name: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get runtime context for an agent.
        
        Provides:
        - Current date/time in multiple formats
        - Agent identity (name, nickname, role)
        - User context (if user_id provided)
        - System state (active mode, capabilities)
        
        Args:
            agent_name: Name of the agent requesting context
            user_id: Optional user identifier
            conversation_id: Optional conversation identifier
            mode: Optional current mode (chat, research, etc.)
        
        Returns:
            Dictionary with runtime context
        """
        now = datetime.now()
        
        # Agent identity mapping
        agent_identities = {
            "secretary": {
                "name": "ICEBURG Secretary",
                "nickname": "ice",
                "role": "friendly assistant for the ICEBURG research platform",
                "capabilities": ["chat", "help", "navigation", "quick answers"]
            },
            "surveyor": {
                "name": "ICEBURG Surveyor",
                "nickname": "surveyor",
                "role": "gnostic research agent for deep analysis",
                "capabilities": ["research", "web search", "evidence gathering"]
            },
            "synthesist": {
                "name": "ICEBURG Synthesist",
                "nickname": "synth",
                "role": "evidence fusion specialist",
                "capabilities": ["synthesis", "pattern recognition", "integration"]
            },
            "dissident": {
                "name": "ICEBURG Dissident",
                "nickname": "diss",
                "role": "critical analysis and alternative perspectives",
                "capabilities": ["critique", "alternative views", "challenge assumptions"]
            },
            "oracle": {
                "name": "ICEBURG Oracle",
                "nickname": "oracle",
                "role": "evidence validation and truth verification",
                "capabilities": ["validation", "fact-checking", "evidence grading"]
            }
        }
        
        agent_info = agent_identities.get(agent_name.lower(), {
            "name": agent_name,
            "nickname": agent_name.lower(),
            "role": "ICEBURG agent",
            "capabilities": []
        })
        
        context = {
            # Time information
            "current_datetime": now.isoformat(),
            "current_date": now.strftime("%A, %B %d, %Y"),
            "current_time": now.strftime("%I:%M:%S %p %Z"),
            "current_date_short": now.strftime("%Y-%m-%d"),
            "current_time_24h": now.strftime("%H:%M:%S"),
            "timestamp": now.timestamp(),
            
            # Agent identity
            "agent_name": agent_info["name"],
            "agent_nickname": agent_info["nickname"],
            "agent_role": agent_info["role"],
            "agent_capabilities": agent_info["capabilities"],
            
            # Session info
            "user_id": user_id or "anonymous",
            "conversation_id": conversation_id or "new",
            "mode": mode or "chat",
            
            # System state
            "system_name": "ICEBURG",
            "system_version": "3.0",
            "system_status": "online"
        }
        
        return context
    
    def get_conversation_memory(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get short-term memory: recent conversation history.
        
        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of messages to retrieve
        
        Returns:
            List of conversation messages (most recent first)
        """
        if not self.local_persistence:
            logger.debug("ContextService: No LocalPersistence available for conversation memory")
            return []
        
        try:
            conversations = self.local_persistence.get_conversations(
                conversation_id=conversation_id,
                limit=limit
            )
            
            # Format for context injection
            memory = []
            for conv in conversations:
                memory.append({
                    "role": "user",
                    "content": conv.get("user_message", ""),
                    "timestamp": conv.get("timestamp", "")
                })
                memory.append({
                    "role": "assistant",
                    "content": conv.get("assistant_message", ""),
                    "timestamp": conv.get("timestamp", ""),
                    "agent": conv.get("agent_used", "unknown")
                })
            
            return memory
        
        except Exception as e:
            logger.warning(f"ContextService: Error retrieving conversation memory: {e}")
            return []
    
    def get_user_memory(
        self,
        user_id: str,
        query: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get long-term memory: cross-session user preferences and facts.
        
        Args:
            user_id: User identifier
            query: Optional query for semantic search
            k: Number of memories to retrieve
        
        Returns:
            List of relevant user memories
        """
        if not self.unified_memory:
            logger.debug("ContextService: No UnifiedMemory available for user memory")
            return []
        
        try:
            # Search user-specific namespace
            search_query = query or f"user:{user_id}"
            results = self.unified_memory.search(
                namespace="user_memories",
                query=search_query,
                k=k
            )
            
            # Filter for this user
            user_memories = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("user_id") == user_id:
                    user_memories.append({
                        "content": result.get("document", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "type": metadata.get("type", "fact"),
                        "importance": metadata.get("importance", 0.5)
                    })
            
            return user_memories
        
        except Exception as e:
            logger.warning(f"ContextService: Error retrieving user memory: {e}")
            return []
    
    def store_interaction(
        self,
        agent_name: str,
        query: str,
        response: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Store interaction in all memory systems.
        
        Args:
            agent_name: Name of the agent
            query: User query
            response: Agent response
            conversation_id: Optional conversation ID
            user_id: Optional user ID
            mode: Optional mode (chat, research, etc.)
            metadata: Optional additional metadata
        """
        # Store in LocalPersistence (conversation history)
        if self.local_persistence and conversation_id:
            try:
                from ..storage.local_persistence import ConversationEntry
                
                entry = ConversationEntry(
                    conversation_id=conversation_id,
                    user_message=query,
                    assistant_message=response,
                    agent_used=agent_name,
                    mode=mode or "chat",
                    timestamp=datetime.now().isoformat(),
                    metadata=metadata or {}
                )
                self.local_persistence.save_conversation(entry)
                logger.debug(f"ContextService: Stored interaction in LocalPersistence")
            
            except Exception as e:
                logger.warning(f"ContextService: Error storing in LocalPersistence: {e}")
        
        # Store in UnifiedMemory (long-term, user-specific)
        if self.unified_memory and user_id:
            try:
                memory_text = f"Q: {query}\nA: {response[:500]}"  # Truncate for storage
                self.unified_memory.index_texts(
                    namespace="user_memories",
                    texts=[memory_text],
                    metadatas=[{
                        "user_id": user_id,
                        "conversation_id": conversation_id or "",
                        "agent_name": agent_name,
                        "timestamp": datetime.now().isoformat(),
                        "type": "interaction",
                        "mode": mode or "chat"
                    }]
                )
                logger.debug(f"ContextService: Stored interaction in UnifiedMemory")
            
            except Exception as e:
                logger.warning(f"ContextService: Error storing in UnifiedMemory: {e}")
    
    def build_context_prompt(
        self,
        agent_name: str,
        query: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        mode: Optional[str] = None,
        include_conversation_history: bool = True,
        include_user_memory: bool = True
    ) -> str:
        """
        Build a complete context prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            query: User query
            user_id: Optional user ID
            conversation_id: Optional conversation ID
            mode: Optional mode
            include_conversation_history: Whether to include recent conversation
            include_user_memory: Whether to include user preferences
        
        Returns:
            Formatted context string ready for injection into system prompt
        """
        # Get runtime context
        runtime = self.get_runtime_context(
            agent_name=agent_name,
            user_id=user_id,
            conversation_id=conversation_id,
            mode=mode
        )
        
        # Build context sections
        context_parts = [
            f"CURRENT CONTEXT:",
            f"- Date: {runtime['current_date']}",
            f"- Time: {runtime['current_time']}",
            f"- Your Identity: {runtime['agent_name']} (nickname: {runtime['agent_nickname']})",
            f"- Your Role: {runtime['agent_role']}",
            f"- User: {runtime['user_id']}",
            f"- Mode: {runtime['mode']}",
            ""
        ]
        
        # Add conversation history if requested
        if include_conversation_history and conversation_id:
            conv_memory = self.get_conversation_memory(conversation_id, limit=6)
            if conv_memory:
                context_parts.append("RECENT CONVERSATION:")
                for msg in conv_memory[-6:]:  # Last 3 exchanges
                    role = "User" if msg["role"] == "user" else "You"
                    context_parts.append(f"{role}: {msg['content'][:200]}")
                context_parts.append("")
        
        # Add user memory if requested
        if include_user_memory and user_id:
            user_memory = self.get_user_memory(user_id, query=query, k=3)
            if user_memory:
                context_parts.append("USER PREFERENCES & HISTORY:")
                for mem in user_memory[:3]:
                    context_parts.append(f"- {mem['content'][:200]}")
                context_parts.append("")
        
        return "\n".join(context_parts)


# Singleton instance
_context_service_instance = None


def get_context_service(cfg=None) -> ContextService:
    """
    Get or create the global ContextService instance.
    
    Args:
        cfg: Optional ICEBURG configuration
    
    Returns:
        ContextService instance
    """
    global _context_service_instance
    
    if _context_service_instance is None:
        _context_service_instance = ContextService(cfg)
    
    return _context_service_instance
