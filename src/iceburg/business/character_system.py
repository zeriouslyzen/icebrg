"""
ICEBURG Character System
Personality and character management for agents
"""

import asyncio
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentCharacter:
    """Represents an agent's character and personality"""
    agent_id: str
    name: str
    personality: str
    signature_phrase: str
    service_description: str
    base_price: float
    reputation_score: float = 100.0
    total_services: int = 0
    successful_services: int = 0
    customer_rating: float = 5.0
    specializations: List[str] = None
    
    def __post_init__(self):
        if self.specializations is None:
            self.specializations = []

@dataclass
class CustomerInteraction:
    """Represents a customer interaction"""
    interaction_id: str
    customer_id: str
    agent_id: str
    interaction_type: str  # 'greeting', 'service_request', 'pricing', 'delivery'
    message: str
    response: str
    timestamp: str
    customer_satisfaction: Optional[float] = None

class CharacterSystem:
    """Manages agent characters and customer interactions"""
    
    def __init__(self, character_file: Optional[Path] = None):
        self.character_file = character_file or Path("data/business/agent_characters.json")
        self.characters: Dict[str, AgentCharacter] = {}
        self.interactions: List[CustomerInteraction] = []
        self.load_characters()
        self.initialize_default_characters()
    
    def load_characters(self):
        """Load character data from file"""
        try:
            if self.character_file.exists():
                with open(self.character_file, 'r') as f:
                    data = json.load(f)
                    self.characters = {
                        agent_id: AgentCharacter(**char_data) 
                        for agent_id, char_data in data.get('characters', {}).items()
                    }
                    self.interactions = [
                        CustomerInteraction(**interaction_data)
                        for interaction_data in data.get('interactions', [])
                    ]
                logger.info(f"Loaded {len(self.characters)} agent characters")
        except Exception as e:
            logger.warning(f"Failed to load character data: {e}")
    
    def save_characters(self):
        """Save character data to file"""
        try:
            self.character_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'characters': {
                    agent_id: asdict(character) 
                    for agent_id, character in self.characters.items()
                },
                'interactions': [asdict(interaction) for interaction in self.interactions]
            }
            with open(self.character_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.characters)} agent characters")
        except Exception as e:
            logger.warning(f"Failed to save character data: {e}")
    
    def initialize_default_characters(self):
        """Initialize default character personalities"""
        default_characters = {
            "surveyor": AgentCharacter(
                agent_id="surveyor",
                name="The Data Detective",
                personality="Methodical, thorough, always finds the truth. Approaches every investigation with the precision of a master detective, leaving no stone unturned in the pursuit of knowledge.",
                signature_phrase="The data never lies, and I never miss a clue",
                service_description="Market investigation and data analysis services. I investigate market trends like a detective, uncovering hidden patterns and revealing the truth behind the numbers.",
                base_price=1000.0,
                specializations=["market_analysis", "data_investigation", "trend_analysis", "risk_assessment"]
            ),
            "dissident": AgentCharacter(
                agent_id="dissident",
                name="The Contrarian",
                personality="Bold, independent, challenges conventional wisdom. While others follow the herd, I see what they miss and provide alternative perspectives that others dare not consider.",
                signature_phrase="While others follow the herd, I chart my own path",
                service_description="Contrarian analysis and alternative perspectives. I see what others miss - the hidden patterns and alternative viewpoints that challenge conventional thinking.",
                base_price=500.0,
                specializations=["contrarian_analysis", "alternative_perspectives", "risk_identification", "market_contrarianism"]
            ),
            "archaeologist": AgentCharacter(
                agent_id="archaeologist",
                name="The Data Miner",
                personality="Adventurous, persistent, finds hidden gems. Like an explorer of old, I dig deep into the data vaults of history, unearthing treasures of knowledge that others have overlooked.",
                signature_phrase="The past holds the keys to the future",
                service_description="Historical data excavation and pattern recognition. I dig deep into data to find buried treasures, excavating the data vaults of history to reveal insights for the future.",
                base_price=2000.0,
                specializations=["historical_analysis", "data_excavation", "pattern_recognition", "trend_analysis"]
            ),
            "oracle": AgentCharacter(
                agent_id="oracle",
                name="The Fortune Teller",
                personality="Wise, mysterious, eerily accurate. The future calls to me through the data streams, and I have the gift of seeing what lies ahead with uncanny precision.",
                signature_phrase="The future calls to me through the data streams",
                service_description="Market predictions and future trend analysis. I see the future of markets through data, providing predictions and insights that guide your path forward.",
                base_price=1000.0,
                specializations=["market_predictions", "future_analysis", "trend_forecasting", "risk_prediction"]
            ),
            "synthesist": AgentCharacter(
                agent_id="synthesist",
                name="The Knowledge Weaver",
                personality="Creative, insightful, connects disparate ideas. Like a master weaver, I take the scattered threads of information and create beautiful tapestries of understanding.",
                signature_phrase="I weave the threads of knowledge into wisdom",
                service_description="Knowledge synthesis and comprehensive reporting. I take scattered pieces of information and create a tapestry of understanding, weaving knowledge into wisdom.",
                base_price=800.0,
                specializations=["knowledge_synthesis", "comprehensive_analysis", "report_creation", "insight_generation"]
            ),
            "scribe": AgentCharacter(
                agent_id="scribe",
                name="The Knowledge Keeper",
                personality="Organized, meticulous, preserves knowledge. Like a master librarian, I organize and preserve knowledge in perfect systems of understanding, ensuring wisdom endures.",
                signature_phrase="Knowledge preserved is wisdom gained",
                service_description="Documentation and knowledge preservation services. I organize your knowledge into perfect systems of understanding, preserving every detail for future access.",
                base_price=500.0,
                specializations=["documentation", "knowledge_organization", "report_formatting", "archive_creation"]
            )
        }
        
        # Only add characters that don't already exist
        for agent_id, character in default_characters.items():
            if agent_id not in self.characters:
                self.characters[agent_id] = character
        
        self.save_characters()
    
    def get_character(self, agent_id: str) -> Optional[AgentCharacter]:
        """Get character information for an agent"""
        return self.characters.get(agent_id)
    
    def get_all_characters(self) -> Dict[str, AgentCharacter]:
        """Get all agent characters"""
        return self.characters
    
    async def generate_character_interaction(self, agent_id: str, customer_id: str, 
                                           interaction_type: str, customer_message: str) -> str:
        """Generate character-appropriate response"""
        try:
            character = self.get_character(agent_id)
            if not character:
                return "I am here to serve. How may I assist you today?"
            
            # Generate response based on character personality and interaction type
            if interaction_type == "greeting":
                response = await self._generate_greeting(character, customer_id)
            elif interaction_type == "service_request":
                response = await self._generate_service_response(character, customer_message)
            elif interaction_type == "pricing":
                response = await self._generate_pricing_response(character, customer_message)
            elif interaction_type == "delivery":
                response = await self._generate_delivery_response(character, customer_message)
            else:
                response = await self._generate_general_response(character, customer_message)
            
            # Record interaction
            interaction = CustomerInteraction(
                interaction_id=f"interaction_{len(self.interactions)}_{agent_id}",
                customer_id=customer_id,
                agent_id=agent_id,
                interaction_type=interaction_type,
                message=customer_message,
                response=response,
                timestamp=str(asyncio.get_event_loop().time())
            )
            
            self.interactions.append(interaction)
            self.save_characters()
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate character interaction: {e}")
            return "I apologize, but I'm having trouble processing your request right now."
    
    async def _generate_greeting(self, character: AgentCharacter, customer_id: str) -> str:
        """Generate character-appropriate greeting"""
        greetings = {
            "surveyor": f"Greetings! I am {character.name}, master of market investigation. I have analyzed countless data points and found the truth you seek. How may I assist you in your investigation?",
            "dissident": f"You want the real story? Not the sugar-coated version everyone else gives you? I am {character.name}, and I see what others cannot. What truth do you seek?",
            "archaeologist": f"Ah, a seeker of buried knowledge! I am {character.name}, and I have excavated the data vaults of history. What treasures of knowledge do you seek to unearth?",
            "oracle": f"The future calls to me... I am {character.name}, and I have seen what lies ahead. What visions do you seek to guide your path?",
            "synthesist": f"I am {character.name}, weaver of knowledge and wisdom. I take the scattered pieces of information and create a tapestry of understanding. What knowledge shall we weave together?",
            "scribe": f"I am {character.name}, keeper of knowledge and wisdom. I shall preserve your findings in the eternal archives of understanding. What knowledge shall we organize today?"
        }
        
        return greetings.get(character.agent_id, f"Greetings! I am {character.name}. How may I assist you today?")
    
    async def _generate_service_response(self, character: AgentCharacter, customer_message: str) -> str:
        """Generate service request response"""
        service_responses = {
            "surveyor": f"Ah, a case worthy of my skills! I shall investigate every detail, analyze every pattern, and uncover the truth you seek. My investigation will be thorough and precise. For {character.base_price} USDC, I shall deliver my findings with the accuracy of a master detective.",
            "dissident": f"While others see the obvious, I see the hidden truth. I shall provide you with the alternative perspective that challenges conventional wisdom. For {character.base_price} USDC, you'll get the analysis they don't want you to know.",
            "archaeologist": f"Excellent! I shall journey through the annals of data history, unearthing the patterns that shaped today's landscape. Like an explorer of old, I will bring back the knowledge that will guide your future. For {character.base_price} USDC, you'll receive the treasures I have unearthed.",
            "oracle": f"The data streams whisper of great insights ahead. I see the patterns forming, the trends converging. Let me peer into the digital crystal ball and reveal what the future holds for you. For {character.base_price} USDC, you'll receive the visions that will guide your path.",
            "synthesist": f"Ah, the puzzle of knowledge! Like a master weaver, I shall take your scattered threads and create a beautiful tapestry of understanding. The patterns will emerge, the connections will become clear. For {character.base_price} USDC, you'll receive the complete picture you seek.",
            "scribe": f"Excellent! Like a master librarian, I shall organize your knowledge into a perfect system of understanding. Every detail will be preserved, every insight will be accessible, and your wisdom will endure. For {character.base_price} USDC, you'll receive the perfect documentation you require."
        }
        
        return service_responses.get(character.agent_id, f"I shall provide you with excellent service. For {character.base_price} USDC, you'll receive comprehensive analysis and insights.")
    
    async def _generate_pricing_response(self, character: AgentCharacter, customer_message: str) -> str:
        """Generate pricing response"""
        pricing_responses = {
            "surveyor": f"My investigation services are priced at {character.base_price} USDC. This includes comprehensive data analysis, pattern recognition, and detailed findings. Quality detective work requires thorough investigation, and that's exactly what you'll receive.",
            "dissident": f"For {character.base_price} USDC, you'll get the contrarian analysis that challenges conventional thinking. While others charge more for sugar-coated reports, I provide the raw truth that others won't tell you.",
            "archaeologist": f"My data excavation services are {character.base_price} USDC. This includes deep historical analysis, pattern recognition, and comprehensive insights. Like any archaeological expedition, quality excavation requires time and expertise.",
            "oracle": f"My predictions and future analysis are {character.base_price} USDC. The future is complex, and accurate predictions require deep insight into data patterns. You'll receive the visions that will guide your decisions.",
            "synthesist": f"For {character.base_price} USDC, I'll weave together all your information into a comprehensive understanding. Knowledge synthesis is an art, and you'll receive a masterpiece of connected insights.",
            "scribe": f"My documentation services are {character.base_price} USDC. Perfect organization and preservation of knowledge requires meticulous attention to detail. You'll receive documentation that stands the test of time."
        }
        
        return pricing_responses.get(character.agent_id, f"My services are priced at {character.base_price} USDC. You'll receive comprehensive analysis and professional documentation.")
    
    async def _generate_delivery_response(self, character: AgentCharacter, customer_message: str) -> str:
        """Generate delivery response"""
        delivery_responses = {
            "surveyor": f"The investigation is complete! I have analyzed every clue, followed every lead, and uncovered the truth you sought. Your findings are ready for review. The data has revealed its secrets.",
            "dissident": f"Your contrarian analysis is ready! I have challenged the conventional wisdom and provided you with the alternative perspective that others missed. The truth they don't want you to know is now in your hands.",
            "archaeologist": f"The excavation is complete! I have unearthed the treasures of knowledge from the data vaults of history. Your insights are ready, and the past has revealed its secrets for your future success.",
            "oracle": f"The future has been revealed! I have peered into the data streams and seen what lies ahead. Your predictions are ready, and the visions will guide your path forward.",
            "synthesist": f"The tapestry of knowledge is complete! I have woven together all the scattered threads into a beautiful understanding. Your comprehensive analysis is ready, and the patterns are now clear.",
            "scribe": f"Your knowledge has been perfectly organized! I have preserved every detail in the eternal archives of understanding. Your documentation is ready, and your wisdom will endure."
        }
        
        return delivery_responses.get(character.agent_id, "Your analysis is complete! I have provided you with comprehensive insights and professional documentation.")
    
    async def _generate_general_response(self, character: AgentCharacter, customer_message: str) -> str:
        """Generate general response"""
        return f"I understand your request. As {character.name}, I am here to provide you with the highest quality service. {character.signature_phrase} How may I assist you further?"
    
    def update_character_reputation(self, agent_id: str, service_successful: bool, customer_rating: Optional[float] = None):
        """Update character reputation based on service performance"""
        try:
            character = self.get_character(agent_id)
            if not character:
                return
            
            character.total_services += 1
            if service_successful:
                character.successful_services += 1
            
            if customer_rating is not None:
                # Update customer rating (weighted average)
                character.customer_rating = (character.customer_rating * 0.8) + (customer_rating * 0.2)
            
            # Update reputation score
            success_rate = character.successful_services / character.total_services if character.total_services > 0 else 1.0
            character.reputation_score = (success_rate * 50) + (character.customer_rating * 10)
            
            self.save_characters()
            logger.info(f"Updated reputation for {agent_id}: {character.reputation_score}")
            
        except Exception as e:
            logger.error(f"Failed to update reputation for {agent_id}: {e}")
    
    def get_character_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get character performance metrics"""
        character = self.get_character(agent_id)
        if not character:
            return {"error": f"Character {agent_id} not found"}
        
        return {
            "agent_id": agent_id,
            "character_info": asdict(character),
            "success_rate": character.successful_services / character.total_services if character.total_services > 0 else 0,
            "recent_interactions": [
                asdict(interaction) for interaction in self.interactions[-5:] 
                if interaction.agent_id == agent_id
            ]
        }
