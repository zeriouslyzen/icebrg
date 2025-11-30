"""
ICEBURG Portal Architecture
Main entry point that opens "portals" to always-on system, routing queries to appropriate layers.
Based on always-on AI architecture patterns from frontier AI research.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .always_on_executor import AlwaysOnProtocolExecutor
from .pre_warmed_agent_pool import PreWarmedAgentPool
from .local_persona_instance import LocalPersonaInstance

logger = logging.getLogger(__name__)


class ICEBURGPortal:
    """
    Portal to always-on system that routes queries to appropriate layers.
    
    Architecture:
    - Open portal to always-on system
    - Route queries to appropriate layer:
      - Local persona (instant)
      - Pre-warmed agent (fast)
      - Full protocol (background)
    - Dynamic routing based on complexity
    - Escalation handling
    """
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.always_on_executor: Optional[AlwaysOnProtocolExecutor] = None
        self.agent_pool: Optional[PreWarmedAgentPool] = None
        self.user_personas: Dict[str, LocalPersonaInstance] = {}
        self.initialized = False
        
        # Performance tracking
        self.stats = {
            "portals_opened": 0,
            "local_persona_responses": 0,
            "pre_warmed_responses": 0,
            "full_protocol_responses": 0,
            "escalations": 0,
            "total_queries": 0
        }
        
        logger.info("ICEBURGPortal initialized")
    
    async def initialize(self):
        """Initialize portal with always-on executor and agent pool"""
        if self.initialized:
            logger.warning("Portal already initialized")
            return
        
        try:
            logger.info("Initializing ICEBURG Portal...")
            
            # Initialize always-on executor
            self.always_on_executor = AlwaysOnProtocolExecutor(self.config)
            await self.always_on_executor.start()
            
            # Initialize pre-warmed agent pool
            self.agent_pool = PreWarmedAgentPool(self.config)
            await self.agent_pool.warmup()
            
            self.initialized = True
            logger.info("ICEBURG Portal initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing portal: {e}", exc_info=True)
            raise
    
    async def shutdown(self):
        """Shutdown portal and cleanup resources"""
        try:
            logger.info("Shutting down ICEBURG Portal...")
            
            # Stop always-on executor
            if self.always_on_executor:
                await self.always_on_executor.stop()
            
            # Clear user personas
            self.user_personas.clear()
            
            self.initialized = False
            logger.info("ICEBURG Portal shut down successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down portal: {e}", exc_info=True)
    
    async def open_portal(self, user_id: str, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Open portal to always-on system and route query to appropriate layer.
        
        Args:
            user_id: User identifier
            query: User query
            context: Optional context
            
        Returns:
            Response dictionary with result and metadata
        """
        if not self.initialized:
            try:
                logger.warning("Portal not initialized, initializing now...")
                await self.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize portal: {e}", exc_info=True)
                # Return error response instead of crashing
                return {
                    "response": None,
                    "source": "error",
                    "confidence": 0.0,
                    "error": str(e),
                    "response_time": 0
                }
        
        self.stats["portals_opened"] += 1
        self.stats["total_queries"] += 1
        
        start_time = time.time()
        
        try:
            # Get or create user persona
            persona = self._get_or_create_persona(user_id)
            
            # Try local persona first (instant)
            local_response = await persona.respond(query)
            if local_response and local_response.get("confidence", 0) > 0.7:
                response_time = time.time() - start_time
                self.stats["local_persona_responses"] += 1
                logger.info(f"Local persona response in {response_time:.3f}s")
                return {
                    "response": local_response.get("response", ""),
                    "source": "local_persona",
                    "confidence": local_response.get("confidence", 1.0),
                    "response_time": response_time,
                    "metadata": {
                        "user_id": user_id,
                        "layer": "local_persona",
                        "cached": local_response.get("source") == "local_kb"
                    }
                }
            
            # Route to pre-warmed agent (fast)
            agent_response = await self._route_to_pre_warmed_agent(query, context)
            if agent_response:
                response_time = time.time() - start_time
                self.stats["pre_warmed_responses"] += 1
                logger.info(f"Pre-warmed agent response in {response_time:.3f}s")
                return {
                    "response": agent_response.get("response", ""),
                    "source": "pre_warmed_agent",
                    "confidence": agent_response.get("confidence", 0.8),
                    "response_time": response_time,
                    "metadata": {
                        "user_id": user_id,
                        "layer": "pre_warmed_agent",
                        "agent_id": agent_response.get("agent_id")
                    }
                }
            
            # Escalate to full protocol (background)
            protocol_response = await self._escalate_to_full_protocol(user_id, query, context)
            response_time = time.time() - start_time
            self.stats["full_protocol_responses"] += 1
            self.stats["escalations"] += 1
            logger.info(f"Full protocol response in {response_time:.3f}s")
            return {
                "response": protocol_response.get("response", ""),
                "source": "full_protocol",
                "confidence": protocol_response.get("confidence", 0.9),
                "response_time": response_time,
                "metadata": {
                    "user_id": user_id,
                    "layer": "full_protocol",
                    "escalated": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in portal: {e}", exc_info=True)
            # Fallback to full protocol
            return await self._escalate_to_full_protocol(user_id, query, context)
    
    def _get_or_create_persona(self, user_id: str) -> LocalPersonaInstance:
        """Get or create user persona instance"""
        if user_id not in self.user_personas:
            self.user_personas[user_id] = LocalPersonaInstance(user_id, self.config)
            logger.debug(f"Created persona instance for user: {user_id}")
        
        return self.user_personas[user_id]
    
    async def _route_to_pre_warmed_agent(self, query: str, context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Route query to pre-warmed agent"""
        if not self.agent_pool or not self.agent_pool.warmup_complete:
            return None
        
        try:
            # Determine which agent to use based on query
            agent_id = self._select_agent_for_query(query)
            
            # Get pre-warmed agent
            agent = self.agent_pool.get_agent(agent_id)
            if not agent:
                return None
            
            # For now, return a placeholder response
            # Actual implementation would call the agent
            return {
                "response": f"Pre-warmed agent {agent_id} processing query...",
                "agent_id": agent_id,
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Error routing to pre-warmed agent: {e}", exc_info=True)
            return None
    
    def _select_agent_for_query(self, query: str) -> str:
        """Select appropriate agent for query"""
        query_lower = query.lower()
        
        # Simple routing logic
        if any(word in query_lower for word in ["code", "program", "function", "class"]):
            return "code_gen"
        elif any(word in query_lower for word in ["analyze", "data", "statistics", "trend"]):
            return "analysis"
        elif any(word in query_lower for word in ["research", "study", "investigate"]):
            return "iceburg_custom"
        else:
            return "balanced"
    
    async def _escalate_to_full_protocol(self, user_id: str, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Escalate query to full protocol (background processing)"""
        try:
            # Import SystemIntegrator for full protocol processing
            from ..core.system_integrator import SystemIntegrator
            
            # Create system integrator instance
            system_integrator = SystemIntegrator()
            
            # Process query with full integration
            result = await system_integrator.process_query_with_full_integration(
                query=query,
                domain=context.get("domain") if context else None
            )
            
            # Extract response from result
            if result and result.get("results"):
                content = result.get("results", {}).get("content", "")
                if not content:
                    # Try to get from agent results
                    agent_results = result.get("results", {}).get("agent_results", {})
                    if agent_results:
                        # Get first available agent result
                        for agent_id, agent_result in agent_results.items():
                            if agent_result:
                                content = str(agent_result)
                                break
                
                if content:
                    return {
                        "response": content,
                        "confidence": 0.9,
                        "background_processing": False,
                        "full_protocol": True,
                        "result": result
                    }
            
            # Fallback response
            return {
                "response": "Processing your query with full protocol...",
                "confidence": 0.9,
                "background_processing": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error escalating to full protocol: {e}", exc_info=True)
            return {
                "response": "Error processing query. Please try again.",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get portal statistics"""
        stats = {
            **self.stats,
            "initialized": self.initialized,
            "user_personas_count": len(self.user_personas)
        }
        
        if self.always_on_executor:
            stats["always_on_executor"] = self.always_on_executor.get_stats()
        
        if self.agent_pool:
            stats["agent_pool"] = self.agent_pool.get_stats()
        
        return stats

