"""
Middleware Analytics
Analytics and statistics for hallucination detection and emergence tracking.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from .middleware_registry import MiddlewareRegistry
from .hallucination_learning import HallucinationLearning
from .emergence_aggregator import EmergenceAggregator

logger = logging.getLogger(__name__)


class MiddlewareAnalytics:
    """
    Analytics system for middleware.
    
    Features:
    - Hallucination rate per agent
    - Common hallucination patterns
    - Domain-specific hallucinations
    - Detection accuracy over time
    - Emergence frequency and types
    - Agent contribution to emergence
    """
    
    def __init__(
        self,
        registry: MiddlewareRegistry,
        learning_system: Optional[HallucinationLearning] = None,
        emergence_aggregator: Optional[EmergenceAggregator] = None
    ):
        """
        Initialize analytics system.
        
        Args:
            registry: Middleware registry
            learning_system: Hallucination learning system
            emergence_aggregator: Emergence aggregator
        """
        self.registry = registry
        self.learning_system = learning_system
        self.emergence_aggregator = emergence_aggregator
        
        # Analytics cache
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)  # Cache for 5 minutes
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics statistics.
        
        Returns:
            Dictionary with all analytics
        """
        # Check cache
        if self._stats_cache and self._cache_timestamp:
            if datetime.now() - self._cache_timestamp < self._cache_ttl:
                return self._stats_cache
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "registry": self.registry.get_stats(),
            "hallucination": {},
            "emergence": {},
            "summary": {}
        }
        
        # Hallucination analytics
        if self.learning_system:
            try:
                pattern_stats = self.learning_system.get_pattern_stats()
                stats["hallucination"] = {
                    "total_patterns": pattern_stats.get("total_patterns", 0),
                    "patterns_by_agent": pattern_stats.get("patterns_by_agent", {}),
                    "patterns_by_type": pattern_stats.get("patterns_by_type", {}),
                    "last_updated": pattern_stats.get("last_updated")
                }
            except Exception as e:
                logger.debug(f"Could not get hallucination stats: {e}")
                stats["hallucination"] = {"error": str(e)}
        
        # Emergence analytics
        if self.emergence_aggregator:
            try:
                emergence_stats = self.emergence_aggregator.get_emergence_stats()
                recent_breakthroughs = self.emergence_aggregator.get_recent_breakthroughs(limit=10)
                stats["emergence"] = {
                    "total_events": emergence_stats.get("total_events", 0),
                    "events_by_agent": emergence_stats.get("events_by_agent", {}),
                    "events_by_type": emergence_stats.get("events_by_type", {}),
                    "breakthroughs_count": len(emergence_stats.get("breakthroughs", [])),
                    "recent_breakthroughs": recent_breakthroughs,
                    "last_updated": emergence_stats.get("last_updated")
                }
            except Exception as e:
                logger.debug(f"Could not get emergence stats: {e}")
                stats["emergence"] = {"error": str(e)}
        
        # Summary statistics
        stats["summary"] = {
            "total_agents": stats["registry"]["total_agents"],
            "enabled_agents": stats["registry"]["enabled_agents"],
            "hallucination_patterns": stats["hallucination"].get("total_patterns", 0),
            "emergence_events": stats["emergence"].get("total_events", 0),
            "breakthroughs": stats["emergence"].get("breakthroughs_count", 0)
        }
        
        # Cache results
        self._stats_cache = stats
        self._cache_timestamp = datetime.now()
        
        return stats
    
    def get_agent_analytics(self, agent_name: str) -> Dict[str, Any]:
        """
        Get analytics for a specific agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Agent-specific analytics
        """
        analytics = {
            "agent": agent_name,
            "middleware_enabled": self.registry.is_enabled(agent_name),
            "config": self.registry.get_config(agent_name),
            "hallucination": {},
            "emergence": {}
        }
        
        # Hallucination analytics for agent
        if self.learning_system:
            try:
                agent_patterns = self.learning_system.get_agent_patterns(agent_name)
                analytics["hallucination"] = agent_patterns
            except Exception as e:
                logger.debug(f"Could not get agent hallucination stats: {e}")
        
        # Emergence analytics for agent
        if self.emergence_aggregator:
            try:
                agent_emergence = self.emergence_aggregator.get_agent_emergence(agent_name)
                analytics["emergence"] = agent_emergence
            except Exception as e:
                logger.debug(f"Could not get agent emergence stats: {e}")
        
        return analytics
    
    def get_top_hallucination_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top hallucination patterns.
        
        Args:
            limit: Number of patterns to return
            
        Returns:
            List of top patterns
        """
        if not self.learning_system:
            return []
        
        try:
            pattern_stats = self.learning_system.get_pattern_stats()
            patterns_by_type = pattern_stats.get("patterns_by_type", {})
            
            # Sort by frequency
            sorted_patterns = sorted(
                patterns_by_type.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [
                {"pattern": pattern, "count": count}
                for pattern, count in sorted_patterns[:limit]
            ]
        except Exception as e:
            logger.debug(f"Could not get top patterns: {e}")
            return []
    
    def get_top_emergence_types(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top emergence types.
        
        Args:
            limit: Number of types to return
            
        Returns:
            List of top emergence types
        """
        if not self.emergence_aggregator:
            return []
        
        try:
            emergence_stats = self.emergence_aggregator.get_emergence_stats()
            events_by_type = emergence_stats.get("events_by_type", {})
            
            # Sort by frequency
            sorted_types = sorted(
                events_by_type.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [
                {"type": etype, "count": count}
                for etype, count in sorted_types[:limit]
            ]
        except Exception as e:
            logger.debug(f"Could not get top emergence types: {e}")
            return []

