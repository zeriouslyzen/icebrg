#!/usr/bin/env python3
"""
Jenny 2025 - Complete Ambient AI Companion
Integrates all advanced capabilities: VAD, macOS accessibility, emotional intelligence, and memory
"""

import asyncio
import time
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
import queue

# Core systems
from .voice.advanced_vad import AdvancedVAD, VoiceFrame
from .voice.enhanced_voice_output import EnhancedVoiceOutputModule
from .voice.config_manager import VoiceConfigManager
from .vision.macos_accessibility import MacOSAccessibilityEngine, ScreenAnalysis, UIElement
from .memory.emotional_memory import EmotionalMemorySystem, EmotionalState, MemoryEntry


@dataclass
class JennyState:
    """Complete Jenny state"""
    is_active: bool
    is_listening: bool
    is_speaking: bool
    is_seeing: bool
    is_thinking: bool
    current_emotion: str
    attention_level: float
    user_present: bool
    last_interaction: datetime
    conversation_context: str
    active_tasks: List[str]


class Jenny2025:
    """Complete 2025 Jenny - Your Ambient AI Companion"""
    
    def __init__(self, name: str = "Jenny"):
        self.name = name
        self.is_active = False
        
        # Core systems
        from .voice.enhanced_voice_input import EnhancedVoiceInputModule
        self.voice_input = EnhancedVoiceInputModule(VoiceConfigManager())
        self.voice_output = EnhancedVoiceOutputModule(VoiceConfigManager())
        self.accessibility = MacOSAccessibilityEngine()
        self.memory = EmotionalMemorySystem()
        
        # State
        self.state = JennyState(
            is_active=False,
            is_listening=False,
            is_speaking=False,
            is_seeing=False,
            is_thinking=False,
            current_emotion="neutral",
            attention_level=0.0,
            user_present=False,
            last_interaction=datetime.now(),
            conversation_context="idle",
            active_tasks=[]
        )
        
        # Integration
        self.integration_queue = queue.Queue()
        self.processing_thread = None
        
        # Callbacks
        self.on_jenny_awake: Optional[Callable[[], None]] = None
        self.on_jenny_sleep: Optional[Callable[[], None]] = None
        self.on_user_detected: Optional[Callable[[], None]] = None
        self.on_emotion_changed: Optional[Callable[[str], None]] = None
        self.on_task_completed: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Conversation
        self.conversation_history = []
        self.max_history = 20
        
    
    async def initialize(self) -> bool:
        """Initialize all Jenny systems"""
        try:
            
            # Initialize voice input
            if not self.voice_input.initialize():
                return False
            
            # Initialize voice output
            await self.voice_output.initialize()
            
            # Initialize accessibility
            if not self.accessibility.initialize():
                return False
            
            # Initialize memory system
            if not self.memory.initialize():
                return False
            
            # Set up system callbacks
            self._setup_system_callbacks()
            
            # Start integration processing
            self.processing_thread = threading.Thread(target=self._integration_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.is_active = True
            self.state.is_active = True
            
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    def _setup_system_callbacks(self):
        """Set up callbacks for all systems"""
        # Voice input callbacks
        self.voice_input.set_callbacks(
            on_transcription=self._on_voice_transcription,
            on_error=self._on_error
        )
        
        # Accessibility callbacks
        self.accessibility.set_callbacks(
            on_screen_changed=self._on_screen_changed,
            on_ui_element_detected=self._on_ui_element_detected,
            on_window_changed=self._on_window_changed,
            on_error=self._on_error
        )
        
        # Memory callbacks
        self.memory.set_callbacks(
            on_emotion_detected=self._on_emotion_detected,
            on_memory_created=self._on_memory_created,
            on_memory_retrieved=self._on_memory_retrieved,
            on_user_profile_updated=self._on_user_profile_updated,
            on_error=self._on_error
        )
    
    async def wake_up(self):
        """Wake up Jenny - start all systems"""
        if not self.is_active:
            await self.initialize()
        
        # Start VAD
        self.voice_input.start_listening()
        self.state.is_listening = True
        
        # Start screen monitoring
        await self.accessibility.start_screen_monitoring()
        self.state.is_seeing = True
        
        
        # Welcome message
        await self._speak(f"Hello! I'm {self.name}, your 2025 AI companion. I'm now fully awake and ready to help you with anything you need.")
        
        # Callback
        if self.on_jenny_awake:
            self.on_jenny_awake()
    
    async def sleep(self):
        """Put Jenny to sleep - stop all systems"""
        # Stop VAD
        self.vad.stop_detection()
        self.state.is_listening = False
        
        # Stop screen monitoring
        await self.accessibility.stop_monitoring()
        self.state.is_seeing = False
        
        
        # Goodbye message
        await self._speak("Goodbye! I'll be here when you need me. Sweet dreams!")
        
        # Callback
        if self.on_jenny_sleep:
            self.on_jenny_sleep()
    
    def _integration_worker(self):
        """Integration processing worker thread"""
        while self.is_active:
            try:
                # Process integration queue
                event = self.integration_queue.get(timeout=1)
                
                if event:
                    asyncio.create_task(self._process_integration_event(event))
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
    
    async def _process_integration_event(self, event: Dict[str, Any]):
        """Process integration events"""
        try:
            event_type = event.get("type")
            
            if event_type == "voice_detected":
                await self._handle_voice_input(event["data"])
            elif event_type == "screen_changed":
                await self._handle_screen_change(event["data"])
            elif event_type == "emotion_detected":
                await self._handle_emotion_change(event["data"])
            elif event_type == "memory_created":
                await self._handle_memory_creation(event["data"])
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    # System event handlers
    def _on_voice_transcription(self, voice_input):
        """Handle voice transcription"""
        try:
            
            self.state.user_present = True
            self.state.attention_level = min(1.0, self.state.attention_level + 0.1)
            self.state.last_interaction = datetime.now()
            
            # Add to integration queue
            self.integration_queue.put({
                "type": "voice_detected",
                "data": voice_input
            })
            
            # Callback
            if self.on_user_detected:
                self.on_user_detected()
            
        except Exception as e:
    
    def _on_speech_start(self):
        """Handle speech start"""
        self.state.is_speaking = True
    
    def _on_speech_end(self):
        """Handle speech end"""
        self.state.is_speaking = False
    
    def _on_screen_changed(self, analysis: ScreenAnalysis):
        """Handle screen changes"""
        try:
            
            # Add to integration queue
            self.integration_queue.put({
                "type": "screen_changed",
                "data": analysis
            })
            
        except Exception as e:
    
    def _on_ui_element_detected(self, element: UIElement):
        """Handle UI element detection"""
        try:
            
        except Exception as e:
    
    def _on_window_changed(self, window_name: str):
        """Handle window changes"""
        try:
            
        except Exception as e:
    
    def _on_emotion_detected(self, emotional_state: EmotionalState):
        """Handle emotion detection"""
        try:
            
            # Update state
            self.state.current_emotion = emotional_state.emotion
            
            # Add to integration queue
            self.integration_queue.put({
                "type": "emotion_detected",
                "data": emotional_state
            })
            
            # Callback
            if self.on_emotion_changed:
                self.on_emotion_changed(emotional_state.emotion)
            
        except Exception as e:
    
    def _on_memory_created(self, memory: MemoryEntry):
        """Handle memory creation"""
        try:
            
            # Add to integration queue
            self.integration_queue.put({
                "type": "memory_created",
                "data": memory
            })
            
        except Exception as e:
    
    def _on_memory_retrieved(self, memory: MemoryEntry):
        """Handle memory retrieval"""
        try:
            
        except Exception as e:
    
    def _on_user_profile_updated(self, profile):
        """Handle user profile updates"""
        try:
            
        except Exception as e:
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        if self.on_error:
            self.on_error(error)
    
    # Integration event handlers
    async def _handle_voice_input(self, voice_input):
        """Handle voice input with full context"""
        try:
            self.state.is_thinking = True
            
            # Get the actual transcribed text
            if hasattr(voice_input, 'transcription'):
                text = voice_input.transcription
                confidence = voice_input.confidence
            elif hasattr(voice_input, 'text'):
                text = voice_input.text
                confidence = voice_input.confidence
            else:
                text = str(voice_input)
                confidence = 0.5
            
            
            # Store memory of voice input
            self.memory.store_memory(
                content=f"User said: {text}",
                category="interaction",
                context="voice_detection"
            )
            
            # Generate contextual response based on what was actually said
            response = await self._generate_contextual_response("voice_input", {
                "text": text,
                "confidence": confidence
            })
            
            if response:
                await self._speak(response)
            
            self.state.is_thinking = False
            
        except Exception as e:
            self.state.is_thinking = False
    
    async def _handle_screen_change(self, analysis: ScreenAnalysis):
        """Handle screen changes with context"""
        try:
            # Store memory of screen change
            self.memory.store_memory(
                content=f"Screen changed to {analysis.active_window}",
                category="activity",
                context="screen_monitoring"
            )
            
            # Analyze screen content for proactive assistance
            await self._analyze_screen_for_assistance(analysis)
            
        except Exception as e:
    
    async def _handle_emotion_change(self, emotional_state: EmotionalState):
        """Handle emotion changes"""
        try:
            # Update conversation context based on emotion
            if emotional_state.emotion in ["happy", "excited"]:
                self.state.conversation_context = "positive"
            elif emotional_state.emotion in ["sad", "angry"]:
                self.state.conversation_context = "supportive"
            else:
                self.state.conversation_context = "neutral"
            
            # Generate empathetic response
            response = await self._generate_empathetic_response(emotional_state)
            
            if response:
                await self._speak(response)
            
        except Exception as e:
    
    async def _handle_memory_creation(self, memory: MemoryEntry):
        """Handle memory creation"""
        try:
            # Update conversation context based on memory
            if memory.category == "preference":
                self.state.conversation_context = "personalized"
            elif memory.category == "task":
                self.state.conversation_context = "productive"
            
        except Exception as e:
    
    # Response generation
    async def _generate_contextual_response(self, context_type: str, data: Any) -> str:
        """Generate contextual responses"""
        try:
            if context_type == "voice_input":
                # Get the actual text that was said
                if isinstance(data, dict) and "text" in data:
                    text = data["text"].lower()
                    confidence = data.get("confidence", 0.5)
                    
                    
                    # Respond to specific things the user said
                    if "hello" in text or "hi" in text:
                        return "Hello! It's great to hear from you. How can I help you today?"
                    elif "goodbye" in text or "bye" in text or "good night" in text:
                        return "Goodbye! It was nice talking with you. Have a wonderful day!"
                    elif "how are you" in text:
                        return "I'm doing great, thank you for asking! I'm here and ready to help you with anything you need."
                    elif "what can you do" in text:
                        return "I can help you with many things! I can see your screen, hear your voice, remember our conversations, and assist with research and analysis. What would you like to explore?"
                    elif "help" in text:
                        return "I'm here to help! I can assist with computer tasks, research, analysis, or just have a conversation. What do you need help with?"
                    elif "thank you" in text:
                        return "You're very welcome! I'm happy to help. Is there anything else I can do for you?"
                    elif "yes" in text:
                        return "Great! I'm glad I could help. What would you like to do next?"
                    elif "no" in text:
                        return "No problem at all. I'm here whenever you need me. What can I help you with?"
                    else:
                        return f"I heard you say '{data['text']}'. That's interesting! Can you tell me more about that?"
                else:
                    # Retrieve relevant memories
                    memories = self.memory.retrieve_memories(limit=5)
                    
                    # Generate response based on context and memory
                    if self.state.current_emotion == "happy":
                        return "I can hear the happiness in your voice! What's making you so cheerful today?"
                    elif self.state.current_emotion == "sad":
                        return "I notice you might be feeling down. I'm here to listen and help however I can."
                    else:
                        return "I'm listening! What would you like to talk about or work on together?"
            
            return "I'm here and ready to help!"
            
        except Exception as e:
            return "I'm here and ready to help!"
    
    async def _generate_empathetic_response(self, emotional_state: EmotionalState) -> str:
        """Generate empathetic responses"""
        try:
            emotion = emotional_state.emotion
            intensity = emotional_state.intensity
            
            if emotion == "happy" and intensity > 0.7:
                return "You sound really happy! I love hearing that joy in your voice. What's bringing you such happiness?"
            elif emotion == "sad" and intensity > 0.6:
                return "I can sense you might be feeling sad. Remember, I'm here for you. Would you like to talk about what's on your mind?"
            elif emotion == "excited" and intensity > 0.8:
                return "Wow, you sound really excited! I'm excited to hear what's got you so energized!"
            elif emotion == "calm" and intensity > 0.5:
                return "You seem very calm and peaceful right now. That's wonderful. How can I help you maintain this peaceful state?"
            
            return "I'm here with you, whatever you're feeling."
            
        except Exception as e:
            return "I'm here with you."
    
    async def _analyze_screen_for_assistance(self, analysis: ScreenAnalysis):
        """Analyze screen for proactive assistance"""
        try:
            # Look for opportunities to help
            if "error" in analysis.text_content.lower():
                await self._speak("I notice there might be an error on your screen. Would you like me to help you troubleshoot it?")
            elif "loading" in analysis.text_content.lower():
                await self._speak("I see something is loading. I can help you with other tasks while you wait.")
            elif analysis.active_window == "Safari":
                await self._speak("I see you're browsing the web. I can help you find information or assist with research.")
            
        except Exception as e:
    
    async def _speak(self, text: str):
        """Make Jenny speak"""
        try:
            if self.state.is_speaking:
                # Wait for user to finish
                while self.state.is_speaking:
                    await asyncio.sleep(0.1)
            
            
            # Speak the response
            await self.voice_output.speak_response(text)
            
            # Store memory of response
            self.memory.store_memory(
                content=f"Jenny said: {text}",
                category="conversation",
                context="response"
            )
            
            # Add to conversation history
            self.conversation_history.append({
                "type": "jenny",
                "text": text,
                "timestamp": datetime.now(),
                "emotion": self.state.current_emotion
            })
            
            # Keep history manageable
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    # Public API
    def set_callbacks(self,
        on_jenny_awake: Optional[Callable[[], None]] = None,
                     on_jenny_sleep: Optional[Callable[[], None]] = None,
                     on_user_detected: Optional[Callable[[], None]] = None,
                     on_emotion_changed: Optional[Callable[[str], None]] = None,
                     on_task_completed: Optional[Callable[[str], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_jenny_awake = on_jenny_awake
        self.on_jenny_sleep = on_jenny_sleep
        self.on_user_detected = on_user_detected
        self.on_emotion_changed = on_emotion_changed
        self.on_task_completed = on_task_completed
        self.on_error = on_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete Jenny status"""
        return {
            "name": self.name,
            "state": {
                "is_active": self.state.is_active,
                "is_listening": self.state.is_listening,
                "is_speaking": self.state.is_speaking,
                "is_seeing": self.state.is_seeing,
                "is_thinking": self.state.is_thinking,
                "current_emotion": self.state.current_emotion,
                "attention_level": self.state.attention_level,
                "user_present": self.state.user_present,
                "conversation_context": self.state.conversation_context
            },
            "systems": {
                "vad": self.vad.get_status(),
                "accessibility": self.accessibility.get_status(),
                "memory": self.memory.get_status()
            },
            "conversation_history_length": len(self.conversation_history)
        }
    
    async def shutdown(self):
        """Shutdown Jenny completely"""
        try:
            
            # Stop all systems
            self.vad.stop_detection()
            await self.accessibility.stop_monitoring()
            self.memory.cleanup()
            
            # Cleanup
            self.is_active = False
            self.state.is_active = False
            
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
