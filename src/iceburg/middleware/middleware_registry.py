"""
Middleware Registry
Manages which agents use middleware and configuration.
"""

from typing import Dict, Any, List, Optional, Set
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class MiddlewareRegistry:
    """
    Registry for managing middleware configuration per agent.
    
    Features:
    - Auto-discovery of all agents
    - Per-agent enable/disable
    - Global enable/disable
    - Configuration management
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize middleware registry.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or Path("config/global_middleware_config.yaml")
        self.config = self._load_config()
        
        # Discover all agents
        self.known_agents = self._discover_agents()
        
        # Registry state
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self._initialize_agent_configs()
        
        logger.info(f"Middleware Registry initialized with {len(self.known_agents)} agents")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                logger.info(f"Loaded middleware config from {self.config_path}")
                return config
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using defaults.")
        
        # Default configuration
        return {
            "enable_global_middleware": True,
            "enable_hallucination_detection": True,
            "enable_emergence_tracking": True,
            "enable_learning": True,
            "per_agent_overrides": {}
        }
    
    def _discover_agents(self) -> Set[str]:
        """
        Auto-discover all agents in the system.
        
        Returns:
            Set of agent names
        """
        agents = set()
        
        # Known agents from capability registry
        try:
            from ..agents.capability_registry import AgentCapabilityRegistry
            registry = AgentCapabilityRegistry()
            agents.update(registry.agents.keys())
        except Exception as e:
            logger.debug(f"Could not load capability registry: {e}")
        
        # Common agents
        common_agents = {
            "secretary", "surveyor", "dissident", "synthesist", "oracle",
            "archaeologist", "scrutineer", "supervisor", "scribe", "weaver",
            "architect", "ide", "prompt_interpreter"
        }
        agents.update(common_agents)
        
        return agents
    
    def _initialize_agent_configs(self):
        """Initialize per-agent configurations."""
        global_config = {
            "enable_hallucination_detection": self.config.get("enable_hallucination_detection", True),
            "enable_emergence_tracking": self.config.get("enable_emergence_tracking", True),
            "enable_learning": self.config.get("enable_learning", True),
            "hallucination_threshold": self.config.get("hallucination_threshold", 0.15),
            "emergence_threshold": self.config.get("emergence_threshold", 0.6)
        }
        
        # Apply global config to all agents
        for agent_name in self.known_agents:
            self.agent_configs[agent_name] = global_config.copy()
        
        # Apply per-agent overrides
        overrides = self.config.get("per_agent_overrides", {}) or {}
        for agent_name, override_config in overrides.items():
            if agent_name in self.agent_configs:
                self.agent_configs[agent_name].update(override_config)
                logger.info(f"Applied override config for agent: {agent_name}")
    
    def is_enabled(self, agent_name: str) -> bool:
        """
        Check if middleware is enabled for an agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            True if middleware is enabled
        """
        if not self.config.get("enable_global_middleware", True):
            return False
        
        agent_config = self.agent_configs.get(agent_name, {})
        return agent_config.get("enable_hallucination_detection", True) or \
               agent_config.get("enable_emergence_tracking", True)
    
    def get_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for an agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Agent configuration dictionary
        """
        return self.agent_configs.get(agent_name, {
            "enable_hallucination_detection": True,
            "enable_emergence_tracking": True,
            "enable_learning": True,
            "hallucination_threshold": 0.15,
            "emergence_threshold": 0.6
        })
    
    def update_config(self, agent_name: str, config: Dict[str, Any]):
        """
        Update configuration for an agent.
        
        Args:
            agent_name: Agent name
            config: Configuration updates
        """
        if agent_name not in self.agent_configs:
            self.agent_configs[agent_name] = {}
        
        self.agent_configs[agent_name].update(config)
        logger.info(f"Updated config for agent: {agent_name}")
    
    def get_all_agents(self) -> List[str]:
        """Get list of all known agents."""
        return sorted(list(self.known_agents))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        enabled_count = sum(1 for agent in self.known_agents if self.is_enabled(agent))
        
        return {
            "total_agents": len(self.known_agents),
            "enabled_agents": enabled_count,
            "disabled_agents": len(self.known_agents) - enabled_count,
            "global_middleware_enabled": self.config.get("enable_global_middleware", True),
            "hallucination_detection_enabled": self.config.get("enable_hallucination_detection", True),
            "emergence_tracking_enabled": self.config.get("enable_emergence_tracking", True),
            "learning_enabled": self.config.get("enable_learning", True)
        }

