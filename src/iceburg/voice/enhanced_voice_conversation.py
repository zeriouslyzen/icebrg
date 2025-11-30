#!/usr/bin/env python3
"""
Enhanced Voice Conversation System with streaming and advanced capabilities
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime

from .config_manager import VoiceConfigManager, VoiceSystemConfig
from .enhanced_voice_input import EnhancedVoiceInputModule, VoiceInput, AudioChunk
from .enhanced_voice_output import EnhancedVoiceOutputModule, VoiceOutput, AudioChunk as OutputAudioChunk


@dataclass
class ConversationContext:
    """Conversation context and history"""
    session_id: str
    start_time: datetime
    messages: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    current_topic: str
    emotion_state: str
    attention_level: float


@dataclass
class ConversationMetrics:
    """Conversation performance metrics"""
    total_messages: int
    average_response_time: float
    user_satisfaction: float
    system_performance: float
    error_rate: float


class EnhancedVoiceConversationSystem:
    """Enhanced voice conversation system with streaming and advanced features"""
    
    def __init__(self, config_manager: Optional[VoiceConfigManager] = None):
        self.config_manager = config_manager or VoiceConfigManager()
        self.config = self.config_manager.get_config()
        
        # Core components
        self.voice_input = EnhancedVoiceInputModule(self.config_manager)
        self.voice_output = EnhancedVoiceOutputModule(self.config_manager)
        
        # Conversation state
        self.is_active = False
        self.current_session = None
        self.conversation_context = None
        self.metrics = ConversationMetrics(0, 0.0, 0.0, 0.0, 0.0)
        
        # Callbacks
        self.on_conversation_start: Optional[Callable[[], None]] = None
        self.on_conversation_end: Optional[Callable[[], None]] = None
        self.on_user_input: Optional[Callable[[VoiceInput], None]] = None
        self.on_system_response: Optional[Callable[[VoiceOutput], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Performance tracking
        self.response_times = []
        self.error_count = 0
        self.total_interactions = 0
        
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup internal callbacks"""
        # Voice input callbacks
        self.voice_input.set_callbacks(
            on_audio_chunk=self._on_audio_chunk,
            on_transcription=self._on_transcription,
            on_error=self._on_error
        )
        
        # Voice output callbacks
        self.voice_output.set_callbacks(
            on_audio_chunk=self._on_output_audio_chunk,
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            on_error=self._on_error
        )
    
    def set_callbacks(self,
        on_conversation_start: Optional[Callable[[], None]] = None,
                     on_conversation_end: Optional[Callable[[], None]] = None,
                     on_user_input: Optional[Callable[[VoiceInput], None]] = None,
                     on_system_response: Optional[Callable[[VoiceOutput], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set conversation callbacks"""
        self.on_conversation_start = on_conversation_start
        self.on_conversation_end = on_conversation_end
        self.on_user_input = on_user_input
        self.on_system_response = on_system_response
        self.on_error = on_error
    
    async def start_conversation(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        if self.is_active:
            await self.end_conversation()
        
        # Generate session ID
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        # Initialize conversation context
        self.conversation_context = ConversationContext(
            session_id=session_id,
            start_time=datetime.now(),
            messages=[],
            user_preferences={},
            current_topic="general",
            emotion_state="neutral",
            attention_level=1.0
        )
        
        # Start voice input streaming
        await self.voice_input.start_streaming()
        
        self.is_active = True
        
        # Callback
        if self.on_conversation_start:
            self.on_conversation_start()
        
        return session_id
    
    async def end_conversation(self) -> None:
        """End current conversation session"""
        if not self.is_active:
            return
        
        # Stop voice input streaming
        await self.voice_input.stop_streaming()
        
        # Calculate final metrics
        self._calculate_metrics()
        
        # Save conversation history
        self._save_conversation_history()
        
        self.is_active = False
        
        # Callback
        if self.on_conversation_end:
            self.on_conversation_end()
        
    
    async def process_voice_input(self, text_input: str) -> Optional[VoiceOutput]:
        """Process voice input and generate response"""
        if not self.is_active:
            return None
        
        start_time = time.time()
        
        try:
            # Add user message to context
            user_message = {
                "type": "user",
                "text": text_input,
                "timestamp": datetime.now(),
                "session_id": self.conversation_context.session_id
            }
            self.conversation_context.messages.append(user_message)
            
            # Generate response using ICEBURG's intelligence
            response_text = await self._generate_response(text_input)
            
            # Add system message to context
            system_message = {
                "type": "system",
                "text": response_text,
                "timestamp": datetime.now(),
                "session_id": self.conversation_context.session_id
            }
            self.conversation_context.messages.append(system_message)
            
            # Speak response
            voice_output = await self.voice_output.speak_response(response_text)
            
            # Track performance
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.total_interactions += 1
            
            # Callback
            if self.on_system_response:
                self.on_system_response(voice_output)
            
            return voice_output
            
        except Exception as e:
            self.error_count += 1
            if self.on_error:
                self.on_error(e)
            return None
    
    async def process_streaming_input(self, text_input: str) -> Optional[VoiceOutput]:
        """Process voice input with streaming response"""
        if not self.is_active:
            return None
        
        start_time = time.time()
        
        try:
            # Add user message to context
            user_message = {
                "type": "user",
                "text": text_input,
                "timestamp": datetime.now(),
                "session_id": self.conversation_context.session_id
            }
            self.conversation_context.messages.append(user_message)
            
            # Generate response using ICEBURG's intelligence
            response_text = await self._generate_response(text_input)
            
            # Add system message to context
            system_message = {
                "type": "system",
                "text": response_text,
                "timestamp": datetime.now(),
                "session_id": self.conversation_context.session_id
            }
            self.conversation_context.messages.append(system_message)
            
            # Speak response with streaming
            voice_output = await self.voice_output.speak_streaming(response_text)
            
            # Track performance
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.total_interactions += 1
            
            # Callback
            if self.on_system_response:
                self.on_system_response(voice_output)
            
            return voice_output
            
        except Exception as e:
            self.error_count += 1
            if self.on_error:
                self.on_error(e)
            return None
    
    async def _generate_response(self, text_input: str) -> str:
        """Generate response using ICEBURG's intelligence"""
        # This would integrate with ICEBURG's core intelligence system
        # For now, we'll use a simple response generator
        
        # Analyze input for emotion and context
        emotion = self._analyze_emotion(text_input)
        topic = self._analyze_topic(text_input)
        
        # Update conversation context
        self.conversation_context.emotion_state = emotion
        self.conversation_context.current_topic = topic
        
        # Generate contextual response
        response = self._generate_contextual_response(text_input, emotion, topic)
        
        return response
    
    def _analyze_emotion(self, text: str) -> str:
        """Analyze emotion in text"""
        # Simple emotion analysis
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["happy", "great", "awesome", "excellent"]):
            return "happy"
        elif any(word in text_lower for word in ["sad", "terrible", "awful", "horrible"]):
            return "sad"
        elif any(word in text_lower for word in ["angry", "mad", "furious", "annoyed"]):
            return "angry"
        elif any(word in text_lower for word in ["excited", "thrilled", "amazing", "incredible"]):
            return "excited"
        else:
            return "neutral"
    
    def _analyze_topic(self, text: str) -> str:
        """Analyze topic in text"""
        # Simple topic analysis
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["weather", "temperature", "rain", "sunny"]):
            return "weather"
        elif any(word in text_lower for word in ["time", "clock", "schedule", "appointment"]):
            return "time"
        elif any(word in text_lower for word in ["help", "assist", "support", "problem"]):
            return "assistance"
        elif any(word in text_lower for word in ["research", "study", "analyze", "investigate"]):
            return "research"
        else:
            return "general"
    
    def _generate_contextual_response(self, text: str, emotion: str, topic: str) -> str:
        """Generate contextual response"""
        # This would integrate with ICEBURG's intelligence system
        # For now, we'll use simple contextual responses
        
        responses = {
            "weather": {
                "happy": "I'm glad you're asking about the weather! It's a beautiful day for research.",
                "sad": "I understand the weather might be affecting your mood. Let me help you with something else.",
                "neutral": "The weather is an interesting topic. Would you like me to research current conditions?",
                "excited": "Weather research is fascinating! I can help you analyze atmospheric patterns."
            },
            "time": {
                "happy": "Time management is crucial for productivity! How can I help you organize your schedule?",
                "sad": "I understand time can be stressful. Let me help you prioritize your tasks.",
                "neutral": "Time is a fundamental concept. Would you like me to help you with scheduling?",
                "excited": "Time optimization is exciting! I can help you maximize your productivity."
            },
            "assistance": {
                "happy": "I'm here to help! What specific assistance do you need?",
                "sad": "I'm sorry you're having difficulties. Let me help you resolve this issue.",
                "neutral": "I'm ready to assist you. What can I help you with?",
                "excited": "I love helping! What exciting challenge can we tackle together?"
            },
            "research": {
                "happy": "Research is my passion! What would you like to investigate?",
                "sad": "Research can be challenging, but I'm here to help you through it.",
                "neutral": "Research is a systematic process. What topic interests you?",
                "excited": "Research is thrilling! Let's dive deep into your topic of interest."
            },
            "general": {
                "happy": "I'm excited to help you! What would you like to explore?",
                "sad": "I'm here to support you. What can I help you with?",
                "neutral": "I'm ready to assist you. What's on your mind?",
                "excited": "I'm thrilled to help! What exciting topic shall we explore?"
            }
        }
        
        return responses.get(topic, responses["general"]).get(emotion, "I'm here to help you. What would you like to know?")
    
    def _on_audio_chunk(self, chunk: AudioChunk):
        """Handle incoming audio chunk"""
        # Real-time audio processing
        pass
    
    def _on_transcription(self, voice_input: VoiceInput):
        """Handle voice transcription"""
        # Callback
        if self.on_user_input:
            self.on_user_input(voice_input)

        # Process the transcription with proper async handling
        try:
            # Try to get the running event loop
            loop = asyncio.get_running_loop()
            loop.create_task(self.process_voice_input(voice_input.transcription))
        except RuntimeError:
            # No running event loop, run the coroutine synchronously
            try:
                asyncio.run(self.process_voice_input(voice_input.transcription))
            except Exception as e:
    
    def _on_output_audio_chunk(self, chunk: OutputAudioChunk):
        """Handle outgoing audio chunk"""
        # Real-time audio output processing
        pass
    
    def _on_speech_start(self):
        """Handle speech start"""
        pass
    
    def _on_speech_end(self):
        """Handle speech end"""
        pass
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        self.error_count += 1
        if self.on_error:
            self.on_error(error)
    
    def _calculate_metrics(self):
        """Calculate conversation metrics"""
        if self.response_times:
            self.metrics.average_response_time = sum(self.response_times) / len(self.response_times)
        
        self.metrics.total_messages = len(self.conversation_context.messages)
        self.metrics.error_rate = self.error_count / max(self.total_interactions, 1)
        self.metrics.system_performance = 1.0 - self.metrics.error_rate
        
        # User satisfaction would be calculated based on feedback
        self.metrics.user_satisfaction = 0.8  # Placeholder
    
    def _save_conversation_history(self):
        """Save conversation history"""
        if not self.conversation_context:
            return
        
        try:
            # Create conversation record
            conversation_record = {
                "session_id": self.conversation_context.session_id,
                "start_time": self.conversation_context.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "messages": self.conversation_context.messages,
                "metrics": {
                    "total_messages": self.metrics.total_messages,
                    "average_response_time": self.metrics.average_response_time,
                    "error_rate": self.metrics.error_rate,
                    "system_performance": self.metrics.system_performance
                }
            }
            
            # Save to file
            filename = f"conversation_{self.conversation_context.session_id}.json"
            # Use relative path to avoid iCloud sync issues
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent.parent
            logs_dir = project_root / "data" / "conversation_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            filepath = logs_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(conversation_record, f, indent=2)
            
            
        except Exception as e:
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "is_active": self.is_active,
            "session_id": self.conversation_context.session_id if self.conversation_context else None,
            "total_interactions": self.total_interactions,
            "error_count": self.error_count,
            "metrics": {
                "total_messages": self.metrics.total_messages,
                "average_response_time": self.metrics.average_response_time,
                "error_rate": self.metrics.error_rate,
                "system_performance": self.metrics.system_performance
            },
            "voice_input": self.voice_input.get_status(),
            "voice_output": self.voice_output.get_status()
        }
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config_manager.update_config(section, updates)
        self.config = self.config_manager.get_config()
        
        # Update components
        self.voice_input.update_config(section, updates)
        self.voice_output.update_config(section, updates)
        
    
    def get_available_voices(self) -> List[str]:
        """Get available voices"""
        return self.voice_output.get_available_voices()
    
    def set_voice(self, voice: str) -> bool:
        """Set voice"""
        return self.voice_output.set_voice(voice)
    
    def set_engine(self, engine: str) -> bool:
        """Set TTS engine"""
        return self.voice_output.set_engine(engine)
