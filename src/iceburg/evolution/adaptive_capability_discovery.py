"""
Adaptive Capability Discovery
Discovers capabilities based on user needs and creates user-specific agents
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .user_profile_builder import UserProfile
from ..agents.dynamic_agent_factory import DynamicAgentFactory
from ..agents.capability_gap_detector import CapabilityGapDetector
from ..security.tool_inventory import ToolInventory

logger = logging.getLogger(__name__)


@dataclass
class Capability:
    """Represents a capability"""
    capability_id: str
    name: str
    description: str
    capability_type: str
    domains: List[str]
    complexity: float
    speed: float


@dataclass
class Agent:
    """Represents an agent"""
    agent_id: str
    name: str
    description: str
    agent_type: str
    capabilities: List[str]
    user_id: str
    created_at: str


class AdaptiveCapabilityDiscovery:
    """
    Discovers capabilities based on user's needs.
    
    Creates user-specific agents dynamically, adapts to user's domain expertise,
    and evolves capabilities with user.
    """
    
    def __init__(self, cfg: Optional[Any] = None):
        """
        Initialize adaptive capability discovery.
        
        Args:
            cfg: ICEBURG config (loads if None)
        """
        self.cfg = cfg
        self.agent_factory = DynamicAgentFactory(cfg)
        self.capability_gap_detector = CapabilityGapDetector(cfg)
        self.tool_inventory = ToolInventory(cfg=cfg)
        
        logger.info("Adaptive Capability Discovery initialized")
    
    def discover_user_capabilities(self, user_profile: UserProfile) -> List[Capability]:
        """
        Discover capabilities based on user's needs.
        
        Args:
            user_profile: User profile
            
        Returns:
            List of discovered capabilities
        """
        capabilities = []
        
        # Discover capabilities based on user interests
        for interest in user_profile.interests:
            capability = Capability(
                capability_id=f"cap_{interest}_{datetime.utcnow().timestamp()}",
                name=f"{interest.capitalize()} Capability",
                description=f"Capability for {interest} domain",
                capability_type="domain_specific",
                domains=[interest],
                complexity=0.5,
                speed=0.7
            )
            capabilities.append(capability)
        
        # Discover capabilities based on domain expertise
        for domain, expertise in user_profile.domain_expertise.items():
            if expertise > 0.5:  # High expertise
                capability = Capability(
                    capability_id=f"cap_{domain}_{datetime.utcnow().timestamp()}",
                    name=f"{domain.capitalize()} Expert Capability",
                    description=f"Expert-level capability for {domain}",
                    capability_type="expert",
                    domains=[domain],
                    complexity=0.8,
                    speed=0.9
                )
                capabilities.append(capability)
        
        logger.info(f"Discovered {len(capabilities)} capabilities for user {user_profile.user_id}")
        return capabilities
    
    def create_user_specific_agents(self, user_profile: UserProfile) -> List[Agent]:
        """
        Create user-specific agents dynamically.
        
        Args:
            user_profile: User profile
            
        Returns:
            List of created agents
        """
        agents = []
        
        # Create agents based on user interests
        for interest in user_profile.interests[:5]:  # Top 5 interests
            try:
                agent = self.agent_factory.create_agent(
                    agent_type=f"{interest}_specialist",
                    name=f"{interest.capitalize()} Specialist",
                    description=f"Specialized agent for {interest} domain",
                    capabilities=[interest]
                )
                
                if agent:
                    user_agent = Agent(
                        agent_id=f"agent_{interest}_{user_profile.user_id}",
                        name=agent.get("name", f"{interest} Specialist"),
                        description=agent.get("description", f"Specialized agent for {interest}"),
                        agent_type=agent.get("type", "specialist"),
                        capabilities=agent.get("capabilities", [interest]),
                        user_id=user_profile.user_id,
                        created_at=datetime.utcnow().isoformat()
                    )
                    agents.append(user_agent)
            except Exception as e:
                logger.warning(f"Error creating agent for {interest}: {e}")
        
        logger.info(f"Created {len(agents)} user-specific agents for user {user_profile.user_id}")
        return agents
    
    def adapt_to_user_domain(self, user_profile: UserProfile) -> Dict[str, Any]:
        """
        Adapt to user's domain expertise.
        
        Args:
            user_profile: User profile
            
        Returns:
            Dictionary with adaptation results
        """
        adaptation = {
            "domains": list(user_profile.domain_expertise.keys()),
            "expertise_levels": user_profile.domain_expertise.copy(),
            "adapted_capabilities": [],
            "adapted_agents": []
        }
        
        # Adapt capabilities to user's domain
        for domain, expertise in user_profile.domain_expertise.items():
            if expertise > 0.3:  # Some expertise
                adaptation["adapted_capabilities"].append({
                    "domain": domain,
                    "expertise": expertise,
                    "capability_level": "intermediate" if expertise > 0.5 else "beginner"
                })
        
        logger.info(f"Adapted to {len(adaptation['domains'])} domains for user {user_profile.user_id}")
        return adaptation
    
    def evolve_with_user(self, user_profile: UserProfile) -> Dict[str, Any]:
        """
        Evolve capabilities with user.
        
        Args:
            user_profile: User profile
            
        Returns:
            Dictionary with evolution results
        """
        evolution = {
            "user_id": user_profile.user_id,
            "total_conversations": user_profile.total_conversations,
            "evolved_capabilities": [],
            "evolved_agents": [],
            "evolution_trends": {}
        }
        
        # Analyze evolution trends
        if user_profile.total_conversations > 10:
            evolution["evolution_trends"]["mature_user"] = True
            evolution["evolution_trends"]["conversation_count"] = user_profile.total_conversations
        
        # Evolve capabilities based on usage
        for domain, expertise in user_profile.domain_expertise.items():
            if expertise > 0.5:
                evolution["evolved_capabilities"].append({
                    "domain": domain,
                    "expertise": expertise,
                    "evolution_stage": "advanced"
                })
        
        logger.info(f"Evolved capabilities for user {user_profile.user_id}: "
                   f"{len(evolution['evolved_capabilities'])} evolved capabilities")
        return evolution

