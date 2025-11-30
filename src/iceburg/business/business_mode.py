"""
ICEBURG Business Mode
Toggle between research mode and business mode
"""

import asyncio
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .agent_wallet import AgentWallet
from .payment_processor import PaymentProcessor, PaymentRequest

logger = logging.getLogger(__name__)

@dataclass
class BusinessModeConfig:
    """Configuration for business mode"""
    mode: str = "research"  # 'research', 'business', 'hybrid'
    platform_fee_percentage: float = 0.10
    minimum_service_price: float = 10.0
    auto_pricing: bool = True
    customer_interactions: bool = False
    revenue_tracking: bool = True

class BusinessMode:
    """Manages ICEBURG business mode operations"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("data/business/business_mode_config.json")
        self.config = BusinessModeConfig()
        self.payment_processor = PaymentProcessor()
        self.agent_wallets: Dict[str, AgentWallet] = {}
        self.trading_revenue = 0.0
        self.load_config()
        self.initialize_agent_wallets()
    
    def load_config(self):
        """Load business mode configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.config = BusinessModeConfig(**data)
                logger.info(f"Loaded business mode config: {self.config.mode}")
        except Exception as e:
            logger.warning(f"Failed to load business mode config: {e}")
    
    def save_config(self):
        """Save business mode configuration"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            logger.info(f"Saved business mode config: {self.config.mode}")
        except Exception as e:
            logger.warning(f"Failed to save business mode config: {e}")
    
    def initialize_agent_wallets(self):
        """Initialize wallets for all ICEBURG agents"""
        agent_types = ["surveyor", "dissident", "archaeologist", "oracle", "synthesist", "scribe", "trading"]
        
        for agent_type in agent_types:
            if agent_type not in self.agent_wallets:
                self.agent_wallets[agent_type] = AgentWallet(agent_type)
        
        logger.info(f"Initialized wallets for {len(self.agent_wallets)} agents")

    def update_trading_pnl(self, pnl: float) -> None:
        self.trading_revenue += pnl
        if "trading" in self.agent_wallets:
            wallet = self.agent_wallets["trading"]
            if pnl > 0:
                wallet.earn_money(pnl, "USDC", "Trading profit")
            else:
                wallet.spend_money(abs(pnl), "USDC", "Trading loss")
    
    def set_mode(self, mode: str) -> bool:
        """Set ICEBURG operating mode"""
        valid_modes = ["research", "business", "hybrid"]
        
        if mode not in valid_modes:
            logger.error(f"Invalid mode: {mode}. Valid modes: {valid_modes}")
            return False
        
        self.config.mode = mode
        
        # Configure mode-specific settings
        if mode == "research":
            self.config.customer_interactions = False
            self.config.auto_pricing = False
            logger.info("ICEBURG set to research mode - pure research, no commercial operations")
            
        elif mode == "business":
            self.config.customer_interactions = True
            self.config.auto_pricing = True
            logger.info("ICEBURG set to business mode - full commercial operations enabled")
            
        elif mode == "hybrid":
            self.config.customer_interactions = True
            self.config.auto_pricing = True
            logger.info("ICEBURG set to hybrid mode - research with optional monetization")
        
        self.save_config()
        return True
    
    def get_mode(self) -> str:
        """Get current operating mode"""
        return self.config.mode
    
    def is_business_mode_active(self) -> bool:
        """Check if business mode is active"""
        return self.config.mode in ["business", "hybrid"]
    
    def is_research_mode_active(self) -> bool:
        """Check if research mode is active"""
        return self.config.mode in ["research", "hybrid"]
    
    async def process_research_query(self, query: str, agent_type: str, 
                                   customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a research query with optional business layer"""
        try:
            # Always conduct research first
            research_result = await self._conduct_research(query, agent_type)
            
            # Add business layer if business mode is active
            if self.is_business_mode_active() and customer_id:
                business_result = await self._add_business_layer(
                    research_result, agent_type, customer_id
                )
                return business_result
            
            return research_result
            
        except Exception as e:
            logger.error(f"Failed to process research query: {e}")
            return {"error": str(e)}
    
    async def _conduct_research(self, query: str, agent_type: str) -> Dict[str, Any]:
        """Conduct pure research (unchanged from core ICEBURG)"""
        # This would integrate with existing ICEBURG research capabilities
        # For now, return a placeholder
        return {
            "query": query,
            "agent_type": agent_type,
            "research_result": f"Research conducted by {agent_type} agent",
            "timestamp": str(asyncio.get_event_loop().time()),
            "mode": "research"
        }
    
    async def _add_business_layer(self, research_result: Dict[str, Any], 
                                agent_type: str, customer_id: str) -> Dict[str, Any]:
        """Add business layer to research results"""
        try:
            # Calculate pricing
            price = await self._calculate_service_price(research_result, agent_type)
            
            # Create payment request
            payment_request = await self.payment_processor.create_payment_request(
                customer_id=customer_id,
                agent_id=agent_type,
                service_type=f"{agent_type}_research",
                amount=price,
                currency="USDC",
                description=f"Research service: {research_result.get('query', 'Unknown query')}"
            )
            
            # Add business information to result
            business_result = research_result.copy()
            business_result.update({
                "mode": "business",
                "price": price,
                "currency": "USDC",
                "payment_request_id": payment_request.request_id,
                "agent_character": self._get_agent_character(agent_type),
                "service_description": self._get_service_description(agent_type)
            })
            
            return business_result
            
        except Exception as e:
            logger.error(f"Failed to add business layer: {e}")
            return research_result
    
    async def _calculate_service_price(self, research_result: Dict[str, Any], 
                                     agent_type: str) -> float:
        """Calculate service price based on research complexity and agent type"""
        base_prices = {
            "surveyor": 1000.0,
            "dissident": 500.0,
            "archaeologist": 2000.0,
            "oracle": 1000.0,
            "synthesist": 800.0,
            "scribe": 500.0
        }
        
        base_price = base_prices.get(agent_type, 500.0)
        
        # Adjust price based on research complexity
        query = research_result.get("query", "")
        complexity_multiplier = 1.0
        
        if len(query) > 100:
            complexity_multiplier += 0.2
        if "analysis" in query.lower():
            complexity_multiplier += 0.3
        if "comprehensive" in query.lower():
            complexity_multiplier += 0.5
        
        final_price = base_price * complexity_multiplier
        
        # Ensure minimum price
        return max(final_price, self.config.minimum_service_price)
    
    def _get_agent_character(self, agent_type: str) -> Dict[str, str]:
        """Get agent character information"""
        characters = {
            "surveyor": {
                "name": "The Data Detective",
                "personality": "Methodical, thorough, always finds the truth",
                "signature_phrase": "The data never lies, and I never miss a clue"
            },
            "dissident": {
                "name": "The Contrarian",
                "personality": "Bold, independent, challenges conventional wisdom",
                "signature_phrase": "While others follow the herd, I chart my own path"
            },
            "archaeologist": {
                "name": "The Data Miner",
                "personality": "Adventurous, persistent, finds hidden gems",
                "signature_phrase": "The past holds the keys to the future"
            },
            "oracle": {
                "name": "The Fortune Teller",
                "personality": "Wise, mysterious, eerily accurate",
                "signature_phrase": "The future calls to me through the data streams"
            },
            "synthesist": {
                "name": "The Knowledge Weaver",
                "personality": "Creative, insightful, connects disparate ideas",
                "signature_phrase": "I weave the threads of knowledge into wisdom"
            },
            "scribe": {
                "name": "The Knowledge Keeper",
                "personality": "Organized, meticulous, preserves knowledge",
                "signature_phrase": "Knowledge preserved is wisdom gained"
            }
        }
        
        return characters.get(agent_type, {
            "name": "Unknown Agent",
            "personality": "Mysterious and capable",
            "signature_phrase": "I am here to serve"
        })
    
    def _get_service_description(self, agent_type: str) -> str:
        """Get service description for agent type"""
        descriptions = {
            "surveyor": "Market investigation and data analysis services",
            "dissident": "Contrarian analysis and alternative perspectives",
            "archaeologist": "Historical data excavation and pattern recognition",
            "oracle": "Market predictions and future trend analysis",
            "synthesist": "Knowledge synthesis and comprehensive reporting",
            "scribe": "Documentation and knowledge preservation services"
        }
        
        return descriptions.get(agent_type, "Specialized research and analysis services")
    
    async def get_agent_performance(self, agent_type: str) -> Dict[str, Any]:
        """Get agent performance metrics"""
        try:
            if agent_type not in self.agent_wallets:
                return {"error": f"Agent {agent_type} not found"}
            
            wallet = self.agent_wallets[agent_type]
            earnings = await self.payment_processor.get_agent_earnings(agent_type)
            
            return {
                "agent_type": agent_type,
                "wallet_balance": wallet.get_balance(),
                "earnings_summary": earnings,
                "character_info": self._get_agent_character(agent_type),
                "service_description": self._get_service_description(agent_type)
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent performance for {agent_type}: {e}")
            return {"error": str(e)}
    
    async def get_platform_summary(self) -> Dict[str, Any]:
        """Get platform performance summary"""
        try:
            platform_revenue = await self.payment_processor.get_platform_revenue()
            
            # Get all agent performance
            agent_performance = {}
            for agent_type in self.agent_wallets.keys():
                agent_performance[agent_type] = await self.get_agent_performance(agent_type)
            
            return {
                "current_mode": self.config.mode,
                "platform_revenue": platform_revenue,
                "agent_performance": agent_performance,
                "total_agents": len(self.agent_wallets),
                "active_agents": len([a for a in agent_performance.values() if not a.get("error")])
            }
            
        except Exception as e:
            logger.error(f"Failed to get platform summary: {e}")
            return {"error": str(e)}
