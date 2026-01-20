"""
Investigation Context - Tracks active investigation for follow-up queries.
Enables context-aware dossier mode that doesn't re-run full pipeline.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from threading import Lock

from .storage import Investigation, get_investigation_store

logger = logging.getLogger(__name__)


@dataclass
class InvestigationContext:
    """
    Tracks an active investigation session.
    Used to provide context for follow-up queries.
    """
    investigation_id: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    investigation: Optional[Investigation] = None
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_query(self, query: str, query_type: str = "follow_up"):
        """Add a query to the history."""
        self.query_history.append({
            "query": query,
            "type": query_type,  # initial, follow_up, add_sources, analyze
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now().isoformat()
    
    def get_context_summary(self) -> str:
        """Get a summary of the investigation for LLM context."""
        if not self.investigation:
            return ""
        
        inv = self.investigation
        summary_parts = [
            f"## Active Investigation: {inv.metadata.title}",
            f"**Query:** {inv.metadata.query}",
            f"**Status:** {inv.metadata.status}",
            f"**Sources:** {inv.metadata.sources_count}",
            f"**Entities:** {inv.metadata.entities_count}",
            "",
            "### Executive Summary",
            inv.executive_summary[:500] + "..." if len(inv.executive_summary) > 500 else inv.executive_summary,
            "",
            "### Key Players",
        ]
        
        for player in inv.key_players[:5]:
            name = player.get("name", "Unknown")
            role = player.get("role", "")
            summary_parts.append(f"- **{name}**: {role}")
        
        if inv.alternative_narratives:
            summary_parts.append("")
            summary_parts.append("### Alternative Narratives")
            for i, narrative in enumerate(inv.alternative_narratives[:3], 1):
                summary_parts.append(f"{i}. {narrative.get('title', narrative.get('narrative', ''))[:100]}...")
        
        return "\n".join(summary_parts)
    
    def is_follow_up_query(self, query: str) -> bool:
        """
        Determine if a new query is a follow-up to the current investigation.
        Returns True if the query relates to the existing investigation.
        """
        if not self.investigation:
            return False
        
        query_lower = query.lower()
        
        # Check for explicit continuation phrases
        continuation_phrases = [
            "tell me more", "what about", "explain", "why", "how",
            "who else", "more details", "dig deeper", "add sources",
            "analyze", "compare", "related", "connection", "link",
            "follow up", "continue", "expand", "also", "and what about"
        ]
        if any(phrase in query_lower for phrase in continuation_phrases):
            return True
        
        # Check for entity references from investigation
        for player in self.investigation.key_players:
            name = player.get("name", "").lower()
            if name and name in query_lower:
                return True
        
        # Check for topic overlap with original query
        original_words = set(self.investigation.metadata.query.lower().split())
        query_words = set(query_lower.split())
        overlap = original_words & query_words
        # If more than 30% overlap, likely follow-up
        if len(overlap) > 0.3 * len(original_words):
            return True
        
        return False
    
    def get_pipeline_action(self, query: str) -> str:
        """
        Determine what pipeline action to take for a query.
        Returns: 'full', 'incremental', 'analyze', or 'chat'
        """
        query_lower = query.lower()
        
        # Check for specific action requests
        if any(phrase in query_lower for phrase in ["add sources", "more sources", "gather more", "find more"]):
            return "incremental"  # Just run gatherer + update synthesis
        
        if any(phrase in query_lower for phrase in ["analyze", "connection", "network", "relationship", "link between"]):
            return "analyze"  # Just run mapper on specific entities
        
        
        # Explicit new investigation requests
        if any(phrase in query_lower for phrase in ["new dossier", "fresh investigation", "start over", "different topic", "investigate"]):
            # If they simply say "investigate X", assume new dossier unless X is in current context
            if "investigate" in query_lower:
                if self.is_follow_up_query(query):
                    return "incremental"
                return "full"
            return "full"
        
        # Default to chat about existing investigation IF there is relevance
        if self.is_follow_up_query(query):
            return "chat"  # Use cached context, just LLM response
        
        # If no relevance to current investigation found, assume new topic
        # But for very short queries, default to chat to avoid accidental new dossiers
        if len(query.split()) < 4:
            return "chat"
            
        # Significant new query -> New Dossier
        return "full"


# Global active contexts by conversation_id
_active_contexts: Dict[str, InvestigationContext] = {}
_context_lock = Lock()


def get_active_context(conversation_id: str) -> Optional[InvestigationContext]:
    """Get the active investigation context for a conversation."""
    with _context_lock:
        return _active_contexts.get(conversation_id)


def set_active_context(
    conversation_id: str,
    investigation_id: str,
    investigation: Optional[Investigation] = None
) -> InvestigationContext:
    """Set or update the active investigation context for a conversation."""
    with _context_lock:
        # Load investigation if not provided
        if investigation is None:
            store = get_investigation_store()
            investigation = store.load(investigation_id)
        
        context = InvestigationContext(
            investigation_id=investigation_id,
            conversation_id=conversation_id,
            investigation=investigation
        )
        _active_contexts[conversation_id] = context
        logger.info(f"📍 Set active investigation context: {investigation_id} for conversation {conversation_id}")
        return context


def clear_active_context(conversation_id: str) -> bool:
    """Clear the active investigation context for a conversation."""
    with _context_lock:
        if conversation_id in _active_contexts:
            del _active_contexts[conversation_id]
            logger.info(f"🗑️ Cleared investigation context for conversation {conversation_id}")
            return True
        return False


def get_or_create_context(
    conversation_id: str,
    investigation_id: Optional[str] = None
) -> Optional[InvestigationContext]:
    """Get existing context or create new one if investigation_id provided."""
    context = get_active_context(conversation_id)
    if context:
        return context
    
    if investigation_id:
        return set_active_context(conversation_id, investigation_id)
    
    return None
