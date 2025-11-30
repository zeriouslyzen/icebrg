"""
Mixture of Experts Architecture
Implements MoE architecture for efficient parameter utilization.
Based on DeepSeek V3 MoE pattern (671B params, 37B active per token).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class ExpertRouter:
    """Router that selects relevant experts for a query"""
    
    def __init__(self):
        self.routing_cache: Dict[str, List[str]] = {}
        self.stats = {
            "routing_decisions": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def route(self, query: str, experts: Dict[str, Any]) -> List[str]:
        """Route query to relevant experts"""
        self.stats["routing_decisions"] += 1
        
        # Check cache
        cache_key = self._get_cache_key(query)
        if cache_key in self.routing_cache:
            self.stats["cache_hits"] += 1
            return self.routing_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Route based on query content
        query_lower = query.lower()
        selected_experts = []
        
        # Simple routing logic (can be enhanced with ML)
        if any(word in query_lower for word in ["code", "program", "function"]):
            if "code_expert" in experts:
                selected_experts.append("code_expert")
        
        if any(word in query_lower for word in ["research", "analyze", "study"]):
            if "research_expert" in experts:
                selected_experts.append("research_expert")
        
        if any(word in query_lower for word in ["math", "calculate", "equation"]):
            if "math_expert" in experts:
                selected_experts.append("math_expert")
        
        # Default: use first expert if none selected
        if not selected_experts and experts:
            selected_experts.append(list(experts.keys())[0])
        
        # Cache routing decision
        self.routing_cache[cache_key] = selected_experts
        
        return selected_experts
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()


class MixtureOfExperts:
    """
    Mixture of Experts architecture for efficient parameter utilization.
    
    Architecture:
    - Multiple expert models
    - Router selecting relevant experts
    - Only activate needed experts per query
    - Efficient resource utilization
    - Maintains performance with reduced computation
    """
    
    def __init__(self):
        self.experts: Dict[str, Any] = {}
        self.router = ExpertRouter()
        self.activation_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.stats = {
            "queries_processed": 0,
            "experts_activated": 0,
            "total_experts": 0,
            "avg_experts_per_query": 0.0,
            "efficiency_gain": 0.0
        }
        
        logger.info("MixtureOfExperts initialized")
    
    def add_expert(self, expert_id: str, expert: Any, specialization: str = "general"):
        """Add expert to MoE system"""
        try:
            self.experts[expert_id] = {
                "expert": expert,
                "specialization": specialization,
                "activation_count": 0,
                "last_activated": None
            }
            self.stats["total_experts"] = len(self.experts)
            logger.debug(f"Added expert {expert_id} with specialization {specialization}")
            
        except Exception as e:
            logger.error(f"Error adding expert {expert_id}: {e}", exc_info=True)
    
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process query using MoE architecture.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Result dictionary with expert outputs aggregated
        """
        start_time = datetime.now()
        
        try:
            # Route to relevant experts
            selected_expert_ids = self.router.route(query, self.experts)
            
            # Activate only needed experts
            expert_results = {}
            for expert_id in selected_expert_ids:
                if expert_id in self.experts:
                    expert_data = self.experts[expert_id]
                    expert = expert_data["expert"]
                    
                    # Activate expert
                    result = await self._activate_expert(expert, query, context)
                    expert_results[expert_id] = result
                    
                    # Update expert stats
                    expert_data["activation_count"] += 1
                    expert_data["last_activated"] = datetime.now().isoformat()
                    self.stats["experts_activated"] += 1
            
            # Aggregate expert outputs
            aggregated_result = self._aggregate_expert_outputs(expert_results)
            
            # Update stats
            self.stats["queries_processed"] += 1
            num_activated = len(selected_expert_ids)
            self.stats["avg_experts_per_query"] = (
                (self.stats["avg_experts_per_query"] * (self.stats["queries_processed"] - 1) + num_activated) /
                self.stats["queries_processed"]
            )
            
            # Calculate efficiency gain
            if self.stats["total_experts"] > 0:
                self.stats["efficiency_gain"] = (
                    1.0 - (self.stats["avg_experts_per_query"] / self.stats["total_experts"])
                ) * 100
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"MoE processing completed: {num_activated}/{self.stats['total_experts']} experts activated in {execution_time:.3f}s")
            
            return {
                "result": aggregated_result,
                "experts_activated": selected_expert_ids,
                "execution_time": execution_time,
                "efficiency_gain": self.stats["efficiency_gain"]
            }
            
        except Exception as e:
            logger.error(f"Error in MoE processing: {e}", exc_info=True)
            return {
                "result": None,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _activate_expert(self, expert: Any, query: str, context: Optional[Dict[str, Any]]) -> Any:
        """Activate a single expert"""
        try:
            # Check if expert is callable
            if callable(expert):
                if asyncio.iscoroutinefunction(expert):
                    result = await expert(query, context)
                else:
                    result = expert(query, context)
            else:
                # Expert is a dict or object, return as-is
                result = expert
            
            return result
            
        except Exception as e:
            logger.error(f"Error activating expert: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _aggregate_expert_outputs(self, expert_results: Dict[str, Any]) -> Any:
        """Aggregate outputs from multiple experts"""
        try:
            # Simple aggregation: combine all expert results
            aggregated = {
                "expert_results": expert_results,
                "count": len(expert_results),
                "timestamp": datetime.now().isoformat()
            }
            
            # If only one expert, return its result directly
            if len(expert_results) == 1:
                return list(expert_results.values())[0]
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating expert outputs: {e}", exc_info=True)
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MoE statistics"""
        return {
            **self.stats,
            "experts_count": len(self.experts),
            "router_stats": self.router.stats
        }

