#!/usr/bin/env python3
"""
Simple Jenny-ICEBURG Integration
Direct integration with ICEBURG's chat_complete function
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

# ICEBURG core imports
from .llm import chat_complete, OLLAMA_AVAILABLE
from .model_select import resolve_models_small

# Jenny 2025 imports
from .jenny_2025 import Jenny2025, JennyState


class JennySimpleICEBURG:
    """Simple Jenny 2025 with direct ICEBURG LLM integration"""
    
    def __init__(self, name: str = "Jenny"):
        self.name = name
        self.is_active = False
        
        # Initialize Jenny 2025
        self.jenny = Jenny2025(name)
        
        # ICEBURG configuration
        self.model = "llama3.2:1b"  # Default small model
        self.temperature = 0.7
        self.max_tokens = 1000
        self.system_prompt = f"""You are {name}, an advanced AI companion with access to ICEBURG's intelligence. You are helpful, friendly, and can assist with research, analysis, and conversation. You can see the user's screen, hear their voice, and remember conversations. Respond naturally and helpfully."""
        
        # State
        self.is_connected_to_icberg = False
        self.conversation_history = []
        
        # Callbacks
        self.on_jenny_awake: Optional[Callable[[], None]] = None
        self.on_jenny_sleep: Optional[Callable[[], None]] = None
        self.on_icberg_connected: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
    
    async def initialize(self) -> bool:
        """Initialize Jenny with simple ICEBURG integration"""
        try:
            
            # Initialize Jenny 2025
            jenny_success = await self.jenny.initialize()
            if not jenny_success:
                return False
            
            # Test ICEBURG connection
            icberg_success = await self._test_icberg_connection()
            if icberg_success:
                self.is_connected_to_icberg = True
            else:
            
            # Set up Jenny to use ICEBURG for responses
            self._setup_icberg_responses()
            
            # Set up callbacks
            self._setup_callbacks()
            
            self.is_active = True
            
            if self.is_connected_to_icberg:
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    async def _test_icberg_connection(self) -> bool:
        """Test ICEBURG connection"""
        try:
            if not OLLAMA_AVAILABLE:
                return False
            
            # Try to get available models
            try:
                import ollama
                models = ollama.list()
                available_models = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
                
                if not available_models:
                    return False
                
                # Try to find a suitable model
                self.model = available_models[0]  # Use first available model
                
            except Exception as e:
                return False
            
            # Test with a simple query
            test_response = chat_complete(
                model=self.model,
                prompt="Hello, this is a connection test. Please respond briefly.",
                system=self.system_prompt,
                temperature=self.temperature,
                context_tag="JENNY-ICEBURG-TEST"
            )
            
            if test_response and len(test_response.strip()) > 0:
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def _setup_icberg_responses(self):
        """Set up Jenny to use ICEBURG for response generation"""
        # Override Jenny's response generation methods
        original_generate_response = self.jenny._generate_contextual_response
        original_generate_empathetic = self.jenny._generate_empathetic_response
        
        async def icberg_generate_contextual_response(context_type: str, data: Any) -> str:
            """Generate responses using ICEBURG intelligence"""
            try:
                if not self.is_connected_to_icberg:
                    return await original_generate_response(context_type, data)
                
                # Build context from Jenny's state
                context = self._build_context(context_type, data)
                
                # Generate response using ICEBURG
                response = await self._generate_icberg_response(context)
                
                return response
                
            except Exception as e:
                # Fallback to original method
                return await original_generate_response(context_type, data)
        
        async def icberg_generate_empathetic_response(emotional_state) -> str:
            """Generate empathetic responses using ICEBURG intelligence"""
            try:
                if not self.is_connected_to_icberg:
                    return await original_generate_empathetic(emotional_state)
                
                # Build emotional context
                context = self._build_emotional_context(emotional_state)
                
                # Generate empathetic response using ICEBURG
                response = await self._generate_icberg_response(context)
                
                return response
                
            except Exception as e:
                # Fallback to original method
                return await original_generate_empathetic(emotional_state)
        
        # Replace Jenny's methods
        self.jenny._generate_contextual_response = icberg_generate_contextual_response
        self.jenny._generate_empathetic_response = icberg_generate_empathetic_response
    
    def _build_context(self, context_type: str, data: Any) -> str:
        """Build context for ICEBURG from Jenny's state"""
        try:
            context_parts = []
            
            # Add Jenny's current state
            context_parts.append(f"Jenny's current state:")
            context_parts.append(f"- Emotion: {self.jenny.state.current_emotion}")
            context_parts.append(f"- Attention level: {self.jenny.state.attention_level:.2f}")
            context_parts.append(f"- User present: {self.jenny.state.user_present}")
            context_parts.append(f"- Conversation context: {self.jenny.state.conversation_context}")
            
            # Add recent conversation history
            if self.jenny.conversation_history:
                context_parts.append("\\nRecent conversation:")
                for entry in self.jenny.conversation_history[-3:]:
                    context_parts.append(f"- {entry['type']}: {entry['text'][:100]}...")
            
            # Add context-specific information
            if context_type == "voice_input":
                context_parts.append(f"\\nUser voice input detected with confidence: {data.confidence:.2f}")
            elif context_type == "screen_change":
                context_parts.append(f"\\nScreen changed to: {data.active_window}")
            
            # Add memory context
            memories = self.jenny.memory.retrieve_memories(limit=3)
            if memories:
                context_parts.append("\\nRelevant memories:")
                for memory in memories:
                    context_parts.append(f"- {memory.content[:100]}...")
            
            return "\\n".join(context_parts)
            
        except Exception as e:
            return "Context building failed"
    
    def _build_emotional_context(self, emotional_state) -> str:
        """Build emotional context for ICEBURG"""
        try:
            context_parts = []
            
            context_parts.append(f"User's emotional state:")
            context_parts.append(f"- Emotion: {emotional_state.emotion}")
            context_parts.append(f"- Intensity: {emotional_state.intensity:.2f}")
            context_parts.append(f"- Confidence: {emotional_state.confidence:.2f}")
            context_parts.append(f"- Source: {emotional_state.source}")
            
            context_parts.append("\\nJenny should respond empathetically and helpfully.")
            
            return "\\n".join(context_parts)
            
        except Exception as e:
            return "Emotional context building failed"
    
    async def _generate_icberg_response(self, context: str) -> str:
        """Generate response using ICEBURG intelligence"""
        try:
            # Create prompt for ICEBURG
            prompt = f"""Based on the following context, generate a natural, helpful response as Jenny:

{context}

Please respond as Jenny, the AI companion, in a natural and helpful way. Keep responses conversational and appropriate to the context."""

            # Use ICEBURG's LLM
            response = chat_complete(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                temperature=self.temperature,
                context_tag="JENNY-ICEBURG"
            )
            
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "context": context,
                "response": response
            })
            
            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history.pop(0)
            
            return response.strip()
            
        except Exception as e:
            return "I'm having trouble processing that right now. Let me try again."
    
    def _setup_callbacks(self):
        """Set up all callbacks"""
        # Jenny callbacks
        self.jenny.set_callbacks(
            on_jenny_awake=lambda: self._on_jenny_awake(),
            on_jenny_sleep=lambda: self._on_jenny_sleep(),
            on_user_detected=lambda: self._on_user_detected(),
            on_emotion_changed=lambda emotion: self._on_emotion_changed(emotion),
            on_task_completed=lambda task: self._on_task_completed(task),
            on_error=lambda error: self._on_error(error)
        )
    
    async def wake_up(self):
        """Wake up Jenny with ICEBURG intelligence"""
        if not self.is_active:
            await self.initialize()
        
        # Wake up Jenny
        await self.jenny.wake_up()
        
        # Enhanced welcome message with ICEBURG capabilities
        if self.is_connected_to_icberg:
            await self._speak(f"Hello! I'm {self.name}, your AI companion with ICEBURG intelligence. I can help you with research, analysis, and complex reasoning. What would you like to explore today?")
        else:
            await self._speak(f"Hello! I'm {self.name}, your AI companion. I'm running in limited mode without ICEBURG intelligence. What can I help you with?")
    
    async def sleep(self):
        """Put Jenny to sleep"""
        await self.jenny.sleep()
    
    async def _speak(self, text: str):
        """Make Jenny speak"""
        await self.jenny._speak(text)
    
    # Callback handlers
    def _on_jenny_awake(self):
        """Handle Jenny awake"""
        if self.on_jenny_awake:
            self.on_jenny_awake()
    
    def _on_jenny_sleep(self):
        """Handle Jenny sleep"""
        if self.on_jenny_sleep:
            self.on_jenny_sleep()
    
    def _on_user_detected(self):
        """Handle user detection"""
    
    def _on_emotion_changed(self, emotion: str):
        """Handle emotion change"""
    
    def _on_task_completed(self, task: str):
        """Handle task completion"""
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        if self.on_error:
            self.on_error(error)
    
    # Public API
    def set_callbacks(self,
        on_jenny_awake: Optional[Callable[[], None]] = None,
                     on_jenny_sleep: Optional[Callable[[], None]] = None,
                     on_icberg_connected: Optional[Callable[[], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_jenny_awake = on_jenny_awake
        self.on_jenny_sleep = on_jenny_sleep
        self.on_icberg_connected = on_icberg_connected
        self.on_error = on_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        jenny_status = self.jenny.get_status()
        
        return {
            "name": self.name,
            "is_active": self.is_active,
            "is_connected_to_icberg": self.is_connected_to_icberg,
            "model": self.model,
            "temperature": self.temperature,
            "ollama_available": OLLAMA_AVAILABLE,
            "jenny": jenny_status,
            "conversation_history_length": len(self.conversation_history)
        }
    
    async def shutdown(self):
        """Shutdown complete system"""
        try:
            
            # Shutdown Jenny
            await self.jenny.shutdown()
            
            self.is_active = False
            
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)

