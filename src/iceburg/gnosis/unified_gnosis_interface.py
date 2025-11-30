"""
Unified Gnosis Interface
Unified interface for gnosis system
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .gnosis_evolution_orchestrator import GnosisEvolutionOrchestrator
from .gnosis_query import GnosisQuery
from ..discovery.dynamic_tool_usage import DynamicToolUsage
from ..awareness.matrix_reasoning import MatrixReasoning
from ..awareness.matrix_detection import MatrixDetection
from ..evolution.user_profile_builder import UserProfileBuilder
from ..knowledge.total_knowledge_accumulator import TotalKnowledgeAccumulator
from ..config import IceburgConfig, load_config

logger = logging.getLogger(__name__)


class UnifiedGnosisInterface:
    """
    Unified interface for gnosis system.
    
    Handles all gnosis queries, manages user evolution, and coordinates all components.
    """
    
    def __init__(self, cfg: Optional[IceburgConfig] = None):
        """
        Initialize unified gnosis interface.
        
        Args:
            cfg: ICEBURG config (loads if None)
        """
        self.cfg = cfg or load_config()
        self.orchestrator = GnosisEvolutionOrchestrator(self.cfg)
        self.gnosis_query = GnosisQuery(self.orchestrator.accumulator)
        self.dynamic_tool_usage = DynamicToolUsage()
        self.matrix_reasoning = MatrixReasoning()
        self.matrix_detection = MatrixDetection()
        self.user_profile_builder = UserProfileBuilder(self.cfg)
        self.total_knowledge = TotalKnowledgeAccumulator(self.cfg)
        
        logger.info("Unified Gnosis Interface initialized")
    
    def process_query(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process query through gnosis system.
        
        Args:
            query: User query
            user_id: User ID (default: "default")
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "query": query,
            "user_id": user_id,
            "computer_capabilities": {},
            "matrix_awareness": {},
            "gnosis_knowledge": {},
            "total_knowledge": {},
            "user_evolution": {},
            "response": ""
        }
        
        # Discover computer capabilities
        try:
            computer_result = self.discover_computer_capabilities(query)
            result["computer_capabilities"] = computer_result
        except Exception as e:
            logger.warning(f"Error discovering computer capabilities: {e}")
        
        # Use matrix awareness
        try:
            matrix_result = self.use_matrix_awareness(query)
            result["matrix_awareness"] = matrix_result
        except Exception as e:
            logger.warning(f"Error using matrix awareness: {e}")
        
        # Query gnosis
        try:
            gnosis_result = self.query_gnosis(query)
            result["gnosis_knowledge"] = gnosis_result
        except Exception as e:
            logger.warning(f"Error querying gnosis: {e}")
        
        # Decode total knowledge
        try:
            total_knowledge_result = self.total_knowledge.decode_complete_knowledge(query)
            result["total_knowledge"] = total_knowledge_result
        except Exception as e:
            logger.warning(f"Error decoding total knowledge: {e}")
        
        # Evolve with user
        try:
            conversation = {"query": query, "response": "", "metadata": {}}
            evolution_result = self.evolve_with_user(user_id, conversation)
            result["user_evolution"] = evolution_result
        except Exception as e:
            logger.warning(f"Error evolving with user: {e}")
        
        logger.info(f"Processed query for user {user_id}: {query[:50]}...")
        return result
    
    def discover_computer_capabilities(self, query: str) -> Dict[str, Any]:
        """
        Discover computer capabilities for query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with discovered capabilities
        """
        result = self.dynamic_tool_usage.use_computer_to_find_info(query)
        
        logger.info(f"Discovered {len(result.get('tools_used', []))} computer capabilities for query")
        return result
    
    def use_matrix_awareness(self, query: str) -> Dict[str, Any]:
        """
        Use matrix awareness for query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with matrix awareness results
        """
        # Identify matrices
        matrices = self.matrix_detection.identify_underlying_matrices(query)
        
        result = {
            "matrices_identified": [m.matrix_id for m in matrices],
            "matrix_count": len(matrices),
            "matrix_types": [m.matrix_type.value for m in matrices]
        }
        
        # Use matrix knowledge
        if matrices:
            matrix_knowledge = self.matrix_reasoning.use_matrix_knowledge(query, matrices[0])
            result["matrix_knowledge"] = matrix_knowledge
        
        logger.info(f"Used matrix awareness: {len(matrices)} matrices identified")
        return result
    
    def query_gnosis(self, query: str) -> Dict[str, Any]:
        """
        Query gnosis knowledge base.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with gnosis query results
        """
        # Get complete knowledge context
        context = self.gnosis_query.get_complete_knowledge_context(query)
        
        logger.info(f"Queried gnosis: {context['total_items']} items, {len(context['domains'])} domains")
        return context
    
    def evolve_with_user(self, user_id: str, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve system with user through conversation.
        
        Args:
            user_id: User ID
            conversation: Conversation dictionary
            
        Returns:
            Dictionary with evolution results
        """
        # Get or create user profile
        profile = self.user_profile_builder.get_profile(user_id)
        if profile is None:
            profile = self.user_profile_builder.build_user_profile([conversation], user_id)
        else:
            # Update profile with conversation
            profile = self.user_profile_builder.build_user_profile([conversation], user_id)
        
        result = {
            "user_id": user_id,
            "total_conversations": profile.total_conversations,
            "interests": profile.interests,
            "domain_expertise": profile.domain_expertise,
            "preferences": profile.preferences
        }
        
        logger.info(f"Evolved with user {user_id}: {profile.total_conversations} conversations")
        return result

