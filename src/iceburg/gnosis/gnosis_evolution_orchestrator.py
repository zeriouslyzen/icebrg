"""
Gnosis Evolution Orchestrator
Orchestrates all gnosis evolution components
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .universal_knowledge_accumulator import UniversalKnowledgeAccumulator
from .conversation_insight_extractor import ConversationInsightExtractor
from .gnosis_query import GnosisQuery
from ..discovery.dynamic_tool_usage import DynamicToolUsage
from ..awareness.matrix_reasoning import MatrixReasoning
from ..awareness.matrix_detection import MatrixDetection
from ..evolution.user_profile_builder import UserProfileBuilder
from ..truth.pattern_correlation_engine import PatternCorrelationEngine
from ..conversation.proactive_gap_filling import ProactiveGapFilling
from ..config import IceburgConfig, load_config

logger = logging.getLogger(__name__)


class GnosisEvolutionOrchestrator:
    """
    Orchestrates all gnosis evolution components.
    
    Coordinates conversation to knowledge accumulation, manages user evolution,
    and ensures complete knowledge building.
    """
    
    def __init__(self, cfg: Optional[IceburgConfig] = None):
        """
        Initialize gnosis evolution orchestrator.
        
        Args:
            cfg: ICEBURG config (loads if None)
        """
        self.cfg = cfg or load_config()
        
        # Initialize components
        self.accumulator = UniversalKnowledgeAccumulator(self.cfg)
        self.insight_extractor = ConversationInsightExtractor()
        self.gnosis_query = GnosisQuery(self.accumulator)
        self.dynamic_tool_usage = DynamicToolUsage()
        self.matrix_reasoning = MatrixReasoning()
        self.matrix_detection = MatrixDetection()
        self.user_profile_builder = UserProfileBuilder(self.cfg)
        self.pattern_correlation = PatternCorrelationEngine()
        self.gap_filling = ProactiveGapFilling()
        
        logger.info("Gnosis Evolution Orchestrator initialized")
    
    def orchestrate_conversation(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Orchestrate conversation through gnosis evolution system.
        
        Args:
            query: User query
            user_id: User ID (default: "default")
            
        Returns:
            Dictionary with orchestration results
        """
        result = {
            "query": query,
            "user_id": user_id,
            "tools_discovered": [],
            "matrices_identified": [],
            "knowledge_retrieved": [],
            "insights_extracted": [],
            "gaps_identified": [],
            "response": "",
            "gnosis_contribution": {}
        }
        
        # Discover computer capabilities
        try:
            tools_result = self.dynamic_tool_usage.use_computer_to_find_info(query)
            result["tools_discovered"] = tools_result.get("tools_used", [])
        except Exception as e:
            logger.warning(f"Error discovering tools: {e}")
        
        # Identify matrices
        try:
            matrices = self.matrix_detection.identify_underlying_matrices(query)
            result["matrices_identified"] = [m.matrix_id for m in matrices]
        except Exception as e:
            logger.warning(f"Error identifying matrices: {e}")
        
        # Query gnosis knowledge base
        try:
            knowledge_items = self.gnosis_query.query_complete_knowledge(query)
            result["knowledge_retrieved"] = [
                {
                    "knowledge_id": k.knowledge_id,
                    "content": k.content[:200],
                    "domains": k.domains
                }
                for k in knowledge_items[:5]
            ]
        except Exception as e:
            logger.warning(f"Error querying gnosis: {e}")
        
        # Identify knowledge gaps
        try:
            conversation = {"query": query, "response": "", "metadata": {}}
            gaps = self.gap_filling.identify_knowledge_gaps(conversation)
            result["gaps_identified"] = [g.gap_description for g in gaps]
        except Exception as e:
            logger.warning(f"Error identifying gaps: {e}")
        
        logger.info(f"Orchestrated conversation for user {user_id}: "
                   f"{len(result['tools_discovered'])} tools, "
                   f"{len(result['matrices_identified'])} matrices, "
                   f"{len(result['knowledge_retrieved'])} knowledge items")
        return result
    
    def accumulate_to_gnosis(self, conversation: Dict[str, Any]) -> None:
        """
        Accumulate conversation to gnosis knowledge base.
        
        Args:
            conversation: Conversation dictionary
        """
        try:
            # Extract insights
            insights = self.insight_extractor.extract_insights(conversation)
            
            # Accumulate to gnosis
            self.accumulator.accumulate_to_gnosis(insights)
            
            logger.info(f"Accumulated conversation to gnosis: {len(insights)} insights")
        except Exception as e:
            logger.error(f"Error accumulating to gnosis: {e}")
    
    def evolve_with_user(self, user_id: str, conversation: Dict[str, Any]) -> None:
        """
        Evolve system with user through conversation.
        
        Args:
            user_id: User ID
            conversation: Conversation dictionary
        """
        try:
            # Get or create user profile
            profile = self.user_profile_builder.get_profile(user_id)
            if profile is None:
                profile = self.user_profile_builder.build_user_profile([conversation], user_id)
            else:
                # Update profile with conversation
                profile = self.user_profile_builder.build_user_profile([conversation], user_id)
            
            logger.info(f"Evolved with user {user_id}: {profile.total_conversations} conversations")
        except Exception as e:
            logger.error(f"Error evolving with user: {e}")
    
    def ensure_complete_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Ensure complete knowledge for query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with complete knowledge context
        """
        try:
            # Get complete knowledge context
            context = self.gnosis_query.get_complete_knowledge_context(query)
            
            logger.info(f"Ensured complete knowledge for query: {context['total_items']} items, "
                       f"{len(context['domains'])} domains, {len(context['connections'])} connections")
            return context
        except Exception as e:
            logger.error(f"Error ensuring complete knowledge: {e}")
            return {"query": query, "knowledge_items": [], "domains": [], "connections": []}

