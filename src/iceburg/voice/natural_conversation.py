#!/usr/bin/env python3
"""
Natural Conversation System for Jenny
Provides natural, responsive conversation flow
"""

import asyncio
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import threading

from .real_time_voice_detector import RealTimeVoiceDetector, VoiceEvent
from .enhanced_voice_output import EnhancedVoiceOutputModule
from .config_manager import VoiceConfigManager


@dataclass
class ConversationState:
    """Conversation state"""
    is_active: bool
    is_listening: bool
    is_speaking: bool
    last_interaction: datetime
    conversation_context: str
    user_present: bool
    attention_level: float


class NaturalConversationSystem:
    """Natural conversation system with real-time voice interaction"""
    
    def __init__(self, name: str = "Jenny"):
        self.name = name
        self.is_active = False
        
        # Core systems
        self.voice_detector = RealTimeVoiceDetector()
        self.voice_output = EnhancedVoiceOutputModule(VoiceConfigManager())
        
        # State
        self.state = ConversationState(
            is_active=False,
            is_listening=False,
            is_speaking=False,
            last_interaction=datetime.now(),
            conversation_context="idle",
            user_present=False,
            attention_level=0.0
        )
        
        # Conversation settings
        self.response_delay = 0.5  # Natural pause before responding
        self.interruption_threshold = 0.3  # Allow interruption after this time
        self.attention_timeout = 30.0  # Timeout for attention
        
        # Callbacks
        self.on_conversation_start: Optional[Callable[[], None]] = None
        self.on_conversation_end: Optional[Callable[[], None]] = None
        self.on_user_speaking: Optional[Callable[[str], None]] = None
        self.on_jenny_speaking: Optional[Callable[[str], None]] = None
        self.on_attention_needed: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Conversation history
        self.conversation_history = []
        self.max_history = 10
        
    
    async def initialize(self) -> bool:
        """Initialize the conversation system"""
        try:
            
            # Initialize voice detector
            if not self.voice_detector.initialize():
                return False
            
            # Set up voice detector callbacks
            self.voice_detector.set_callbacks(
                on_voice_detected=self._on_voice_detected,
                on_speech_start=self._on_speech_start,
                on_speech_end=self._on_speech_end,
                on_error=self._on_error
            )
            
            # Initialize voice output
            await self.voice_output.initialize()
            
            self.is_active = True
            self.state.is_active = True
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    async def start_conversation(self):
        """Start natural conversation"""
        if not self.is_active:
            await self.initialize()
        
        if self.state.is_listening:
            return
        
        self.state.is_listening = True
        self.voice_detector.start_listening()
        
        
        # Welcome message
        await self._speak(f"Hello! I'm {self.name}. I'm listening and ready to chat with you.")
        
        # Callback
        if self.on_conversation_start:
            self.on_conversation_start()
    
    async def stop_conversation(self):
        """Stop conversation"""
        self.state.is_listening = False
        self.voice_detector.stop_listening()
        
        
        # Goodbye message
        await self._speak("Goodbye! It was nice talking with you.")
        
        # Callback
        if self.on_conversation_end:
            self.on_conversation_end()
    
    def _on_voice_detected(self, event: VoiceEvent):
        """Handle detected voice input"""
        try:
            
            # Update state
            self.state.last_interaction = datetime.now()
            self.state.user_present = True
            self.state.attention_level = min(1.0, self.state.attention_level + 0.2)
            
            # Add to conversation history
            self.conversation_history.append({
                "type": "user",
                "text": event.text,
                "timestamp": event.timestamp,
                "confidence": event.confidence
            })
            
            # Keep history manageable
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
            
            # Process the input
            asyncio.create_task(self._process_user_input(event.text))
            
            # Callback
            if self.on_user_speaking:
                self.on_user_speaking(event.text)
                
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def _on_speech_start(self):
        """Handle speech start"""
        self.state.is_speaking = True
    
    def _on_speech_end(self):
        """Handle speech end"""
        self.state.is_speaking = False
    
    async def _process_user_input(self, text: str):
        """Process user input and generate response"""
        try:
            # Natural pause before responding
            await asyncio.sleep(self.response_delay)
            
            # Generate response based on input
            response = await self._generate_response(text)
            
            if response:
                await self._speak(response)
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    async def _generate_response(self, user_input: str) -> str:
        """Generate natural response to user input"""
        try:
            input_lower = user_input.lower()
            
            # Greeting responses
            if any(greeting in input_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
                return f"Hello! It's great to hear from you. How are you doing today?"
            
            # How are you responses
            elif any(phrase in input_lower for phrase in ["how are you", "how's it going", "how do you do"]):
                return "I'm doing wonderfully, thank you for asking! I'm here and ready to help you with whatever you need."
            
            # What can you do responses
            elif any(phrase in input_lower for phrase in ["what can you do", "what do you do", "help me", "what are you"]):
                return "I'm your AI companion! I can chat with you, help with your computer, answer questions, and just be here for conversation. What would you like to do?"
            
            # Computer interaction
            elif any(phrase in input_lower for phrase in ["click", "type", "scroll", "open", "close"]):
                return "I can help you with computer tasks! I can click, type, scroll, and open applications. Just tell me what you'd like me to do."
            
            # Screen analysis
            elif any(phrase in input_lower for phrase in ["what's on my screen", "screen", "see", "look"]):
                return "I can see your screen and help analyze what's on it. I'm monitoring your display to assist you better."
            
            # Thank you responses
            elif any(phrase in input_lower for phrase in ["thank you", "thanks", "appreciate"]):
                return "You're very welcome! I'm happy to help. Is there anything else you'd like to do?"
            
            # Goodbye responses
            elif any(phrase in input_lower for phrase in ["goodbye", "bye", "see you later", "talk to you later"]):
                return "Goodbye! It was wonderful talking with you. Feel free to come back anytime!"
            
            # Questions about Jenny
            elif any(phrase in input_lower for phrase in ["who are you", "what's your name", "introduce yourself"]):
                return f"I'm {self.name}, your AI companion! I'm here to help you with tasks, answer questions, and have natural conversations. I can see through your camera, monitor your screen, and interact with your computer."
            
            # Default conversational response
            else:
                # Use conversation history for context
                context = self._get_conversation_context()
                return f"That's interesting! {self._generate_contextual_response(user_input, context)}"
            
        except Exception as e:
            return "I'm sorry, I had trouble understanding that. Could you try saying it differently?"
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return "This is the start of our conversation."
        
        recent = self.conversation_history[-3:]  # Last 3 exchanges
        context = "Recent conversation: "
        for item in recent:
            context += f"{item['type']}: {item['text']} "
        
        return context
    
    def _generate_contextual_response(self, user_input: str, context: str) -> str:
        """Generate contextual response based on conversation history"""
        # Simple contextual responses
        if "computer" in user_input.lower() or "screen" in user_input.lower():
            return "I can definitely help you with computer tasks. What would you like me to do?"
        elif "help" in user_input.lower():
            return "I'm here to help! What do you need assistance with?"
        else:
            return "Tell me more about that. I'm listening and ready to help."
    
    async def _speak(self, text: str):
        """Make Jenny speak"""
        try:
            if self.state.is_speaking:
                # Wait for user to finish
                while self.state.is_speaking:
                    await asyncio.sleep(0.1)
            
            
            # Speak the response
            await self.voice_output.speak_response(text)
            
            # Add to conversation history
            self.conversation_history.append({
                "type": "jenny",
                "text": text,
                "timestamp": datetime.now(),
                "confidence": 1.0
            })
            
            # Keep history manageable
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
            
            # Callback
            if self.on_jenny_speaking:
                self.on_jenny_speaking(text)
                
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        if self.on_error:
            self.on_error(error)
    
    def set_callbacks(self,
                     on_conversation_start: Optional[Callable[[], None]] = None,
                     on_conversation_end: Optional[Callable[[], None]] = None,
                     on_user_speaking: Optional[Callable[[str], None]] = None,
                     on_jenny_speaking: Optional[Callable[[str], None]] = None,
                     on_attention_needed: Optional[Callable[[], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_conversation_start = on_conversation_start
        self.on_conversation_end = on_conversation_end
        self.on_user_speaking = on_user_speaking
        self.on_jenny_speaking = on_jenny_speaking
        self.on_attention_needed = on_attention_needed
        self.on_error = on_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get conversation system status"""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "state": {
                "is_listening": self.state.is_listening,
                "is_speaking": self.state.is_speaking,
                "user_present": self.state.user_present,
                "attention_level": self.state.attention_level,
                "conversation_context": self.state.conversation_context
            },
            "voice_detector": self.voice_detector.get_status(),
            "conversation_history_length": len(self.conversation_history)
        }
