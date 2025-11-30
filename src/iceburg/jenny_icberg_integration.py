#!/usr/bin/env python3
"""
Jenny-ICEBURG Integration
Connects Jenny 2025 to ICEBURG's core intelligence, LLM models, and research protocols
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
from .protocol import iceberg_protocol
from .config import load_config, IceburgConfig
from .model_select import resolve_models

# Jenny 2025 imports
from .jenny_2025 import Jenny2025, JennyState


@dataclass
class ICEBURGIntelligence:
    """ICEBURG intelligence configuration"""
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    context_window: str
    processing_mode: str


class JennyICEBURGIntegration:
    """Integration between Jenny 2025 and ICEBURG's intelligence"""
    
    def __init__(self, jenny: Jenny2025):
        self.jenny = jenny
        self.is_connected = False
        
        # ICEBURG configuration
        self.config = load_config()
        self.intelligence = ICEBURGIntelligence(
            model="llama3.2:1b",  # Default model
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are Jenny, an advanced AI companion integrated with ICEBURG's intelligence. You can help with research, analysis, and complex reasoning.",
            context_window="8k",
            processing_mode="standard"
        )
        
        # Integration state
        self.conversation_context = []
        self.research_history = []
        self.active_research = None
        
        # Callbacks
        self.on_research_started: Optional[Callable[[str], None]] = None
        self.on_research_completed: Optional[Callable[[str], None]] = None
        self.on_intelligence_connected: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
    
    async def connect_to_icberg(self) -> bool:
        """Connect Jenny to ICEBURG's intelligence"""
        try:
            
            # Check if Ollama is available
            if not OLLAMA_AVAILABLE:
                return False
            
            # Test connection with a simple query
            test_response = await self._test_icberg_connection()
            if not test_response:
                return False
            
            # Set up Jenny's response generation to use ICEBURG
            self._setup_icberg_response_generation()
            
            self.is_connected = True
            
            
            # Callback
            if self.on_intelligence_connected:
                self.on_intelligence_connected()
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    async def _test_icberg_connection(self) -> bool:
        """Test ICEBURG connection with a simple query"""
        try:
            test_query = "Hello, this is a connection test. Please respond briefly."
            
            response = chat_complete(
                model=self.intelligence.model,
                prompt=test_query,
                system=self.intelligence.system_prompt,
                temperature=self.intelligence.temperature,
                context_tag="JENNY-ICEBURG-TEST"
            )
            
            if response and len(response.strip()) > 0:
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def _setup_icberg_response_generation(self):
        """Set up Jenny to use ICEBURG for response generation"""
        # Override Jenny's response generation methods
        original_generate_response = self.jenny._generate_contextual_response
        original_generate_empathetic = self.jenny._generate_empathetic_response
        
        async def icberg_generate_contextual_response(context_type: str, data: Any) -> str:
            """Generate responses using ICEBURG intelligence"""
            try:
                # Build context from Jenny's state and memory
                context = self._build_icberg_context(context_type, data)
                
                # Generate response using ICEBURG
                response = await self._generate_icberg_response(context)
                
                return response
                
            except Exception as e:
                # Fallback to original method
                return await original_generate_response(context_type, data)
        
        async def icberg_generate_empathetic_response(emotional_state) -> str:
            """Generate empathetic responses using ICEBURG intelligence"""
            try:
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
    
    def _build_icberg_context(self, context_type: str, data: Any) -> str:
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
                model=self.intelligence.model,
                prompt=prompt,
                system=self.intelligence.system_prompt,
                temperature=self.intelligence.temperature,
                max_tokens=self.intelligence.max_tokens,
                context_tag="JENNY-ICEBURG"
            )
            
            # Store in conversation history
            self.conversation_context.append({
                "timestamp": datetime.now(),
                "context": context,
                "response": response
            })
            
            # Keep history manageable
            if len(self.conversation_context) > 20:
                self.conversation_context.pop(0)
            
            return response.strip()
            
        except Exception as e:
            return "I'm having trouble processing that right now. Let me try again."
    
    async def start_research(self, query: str, fast: bool = False, hybrid: bool = False) -> str:
        """Start ICEBURG research for Jenny"""
        try:
            if not self.is_connected:
                return "I'm not connected to ICEBURG's research capabilities right now."
            
            
            # Callback
            if self.on_research_started:
                self.on_research_started(query)
            
            # Use ICEBURG protocol for research
            research_result = iceberg_protocol(
                initial_query=query,
                fast=fast,
                hybrid=hybrid,
                verbose=False,
                evidence_strict=False,
                project_id=f"jenny_research_{int(time.time())}"
            )
            
            # Store research in history
            self.research_history.append({
                "timestamp": datetime.now(),
                "query": query,
                "result": research_result,
                "mode": "fast" if fast else ("hybrid" if hybrid else "standard")
            })
            
            # Keep history manageable
            if len(self.research_history) > 10:
                self.research_history.pop(0)
            
            
            # Callback
            if self.on_research_completed:
                self.on_research_completed(query)
            
            return research_result
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return f"I encountered an error while researching '{query}': {str(e)}"
    
    def get_research_history(self) -> List[Dict[str, Any]]:
        """Get research history"""
        return self.research_history.copy()
    
    def get_conversation_context(self) -> List[Dict[str, Any]]:
        """Get conversation context"""
        return self.conversation_context.copy()
    
    def set_intelligence_config(self, 
                              model: str = None,
                              temperature: float = None,
                              max_tokens: int = None,
                              system_prompt: str = None):
        """Update ICEBURG intelligence configuration"""
        try:
            if model:
                self.intelligence.model = model
            if temperature is not None:
                self.intelligence.temperature = temperature
            if max_tokens:
                self.intelligence.max_tokens = max_tokens
            if system_prompt:
                self.intelligence.system_prompt = system_prompt
            
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def set_callbacks(self,
                     on_research_started: Optional[Callable[[str], None]] = None,
                     on_research_completed: Optional[Callable[[str], None]] = None,
                     on_intelligence_connected: Optional[Callable[[], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_research_started = on_research_started
        self.on_research_completed = on_research_completed
        self.on_intelligence_connected = on_intelligence_connected
        self.on_error = on_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "is_connected": self.is_connected,
            "intelligence": {
                "model": self.intelligence.model,
                "temperature": self.intelligence.temperature,
                "max_tokens": self.intelligence.max_tokens,
                "context_window": self.intelligence.context_window,
                "processing_mode": self.intelligence.processing_mode
            },
            "conversation_context_length": len(self.conversation_context),
            "research_history_length": len(self.research_history),
            "ollama_available": OLLAMA_AVAILABLE
        }
    
    async def disconnect(self):
        """Disconnect from ICEBURG"""
        try:
            self.is_connected = False
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)

