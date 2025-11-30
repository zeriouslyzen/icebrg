#!/usr/bin/env python3
"""
Ambient AI Companion - Jenny
Integrates voice, vision, and computer interaction for a true Jarvis-like experience
"""

import asyncio
import time
import json
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

from .voice.enhanced_voice_conversation import EnhancedVoiceConversationSystem
from .voice.config_manager import VoiceConfigManager
from .vision.computer_vision_engine import ComputerVisionEngine, VisionFrame, ComputerState


@dataclass
class CompanionState:
    """Ambient AI companion state"""
    is_active: bool
    is_listening: bool
    is_seeing: bool
    is_interacting: bool
    current_context: str
    user_present: bool
    attention_level: float
    last_interaction: datetime


class AmbientAICompanion:
    """Jenny - Your ambient AI companion with vision and interaction"""
    
    def __init__(self, name: str = "Jenny"):
        self.name = name
        self.is_active = False
        
        # Core systems
        self.voice_system = None
        self.vision_engine = None
        self.config_manager = VoiceConfigManager()
        
        # State
        self.state = CompanionState(
            is_active=False,
            is_listening=False,
            is_seeing=False,
            is_interacting=False,
            current_context="idle",
            user_present=False,
            attention_level=0.0,
            last_interaction=datetime.now()
        )
        
        # Interaction queue
        self.interaction_queue = queue.Queue()
        
        # Callbacks
        self.on_user_detected: Optional[Callable[[], None]] = None
        self.on_attention_needed: Optional[Callable[[str], None]] = None
        self.on_computer_interaction: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
    
    async def initialize(self) -> bool:
        """Initialize all systems"""
        try:
            
            # Initialize voice system with Jenny
            self.voice_system = EnhancedVoiceConversationSystem(self.config_manager)
            await self.voice_system.start_conversation()
            
            # Set Jenny as the voice
            self.voice_system.set_voice("en-US-JennyNeural")
            
            # Initialize vision engine
            self.vision_engine = ComputerVisionEngine()
            
            # Set up vision callbacks
            self.vision_engine.set_callbacks(
                on_frame_processed=self._on_frame_processed,
                on_face_detected=self._on_face_detected,
                on_object_detected=self._on_object_detected,
                on_screen_changed=self._on_screen_changed,
                on_error=self._on_error
            )
            
            # Start vision systems
            await self.vision_engine.start_camera_vision()
            await self.vision_engine.start_screen_monitoring()
            
            self.is_active = True
            self.state.is_active = True
            self.state.is_listening = True
            self.state.is_seeing = True
            
            
            # Welcome message
            await self._speak("Hello! I'm Jenny, your AI companion. I can see you through the camera, monitor your screen, and help you with your computer. How can I assist you today?")
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    async def _speak(self, text: str) -> None:
        """Make Jenny speak"""
        try:
            if self.voice_system:
                # Use the voice output directly for speaking
                if hasattr(self.voice_system, 'voice_output'):
                    await self.voice_system.voice_output.speak_response(text)
                else:
                    # Fallback to processing as voice input
                    await self.voice_system.process_voice_input(text)
        except Exception as e:
    
    def _on_frame_processed(self, frame: VisionFrame):
        """Handle processed vision frame"""
        try:
            # Update attention level based on what we see
            if frame.faces_detected:
                self.state.user_present = True
                self.state.attention_level = min(1.0, self.state.attention_level + 0.1)
                
                # If user just appeared, greet them
                if self.state.attention_level > 0.5 and not self.state.is_interacting:
                    asyncio.create_task(self._handle_user_attention())
            else:
                self.state.user_present = False
                self.state.attention_level = max(0.0, self.state.attention_level - 0.05)
            
            # Analyze screen content
            if frame.frame_type == "screen" and frame.text_detected:
                self._analyze_screen_content(frame.text_detected)
                
        except Exception as e:
    
    def _on_face_detected(self, faces: List[Dict]):
        """Handle face detection"""
        try:
            if faces:
                
                # Check if user is looking at screen
                for face in faces:
                    confidence = face.get("confidence", 0)
                    if confidence > 0.7:
                        self.state.user_present = True
                        self.state.attention_level = confidence
                        
        except Exception as e:
    
    def _on_object_detected(self, objects: List[Dict]):
        """Handle object detection"""
        try:
            if objects:
                
        except Exception as e:
    
    def _on_screen_changed(self, content: str):
        """Handle screen content changes"""
        try:
            
        except Exception as e:
    
    def _analyze_screen_content(self, text_elements: List[Dict]):
        """Analyze what's on the screen"""
        try:
            # This would analyze the screen content and provide insights
            # For now, just log that we're monitoring
            pass
            
        except Exception as e:
    
    async def _handle_user_attention(self):
        """Handle when user needs attention"""
        try:
            if self.state.attention_level > 0.7 and not self.state.is_interacting:
                self.state.is_interacting = True
                
                # Proactive assistance
                await self._speak("I notice you're at your computer. Is there anything I can help you with?")
                
                # Callback
                if self.on_attention_needed:
                    self.on_attention_needed("user_attention")
                
        except Exception as e:
    
    async def process_voice_command(self, command: str) -> str:
        """Process voice commands for computer interaction"""
        try:
            command_lower = command.lower()
            
            # Computer interaction commands
            if "click" in command_lower:
                # Extract coordinates or element to click
                response = await self._handle_click_command(command)
                
            elif "type" in command_lower or "write" in command_lower:
                # Extract text to type
                response = await self._handle_type_command(command)
                
            elif "scroll" in command_lower:
                # Handle scroll commands
                response = await self._handle_scroll_command(command)
                
            elif "screenshot" in command_lower or "capture" in command_lower:
                # Take screenshot
                response = await self._handle_screenshot_command()
                
            elif "what's on my screen" in command_lower:
                # Analyze screen content
                response = await self._handle_screen_analysis()
                
            elif "open" in command_lower:
                # Open applications
                response = await self._handle_open_command(command)
                
            elif "close" in command_lower:
                # Close applications
                response = await self._handle_close_command(command)
                
            else:
                # General conversation
                response = await self._handle_general_conversation(command)
            
            return response
            
        except Exception as e:
            return "I'm sorry, I had trouble processing that command."
    
    async def _handle_click_command(self, command: str) -> str:
        """Handle click commands"""
        try:
            # Simple click at center for now
            if self.vision_engine:
                screen_width, screen_height = pyautogui.size()
                x, y = screen_width // 2, screen_height // 2
                
                if self.vision_engine.click_at_position(x, y):
                    return "I clicked at the center of your screen."
                else:
                    return "I couldn't click at that location."
            else:
                return "I don't have access to click on your screen right now."
                
        except Exception as e:
            return f"I had trouble clicking: {str(e)}"
    
    async def _handle_type_command(self, command: str) -> str:
        """Handle type commands"""
        try:
            # Extract text to type (simplified)
            if "type" in command.lower():
                text_start = command.lower().find("type") + 5
                text_to_type = command[text_start:].strip()
                
                if self.vision_engine and text_to_type:
                    if self.vision_engine.type_text(text_to_type):
                        return f"I typed: {text_to_type}"
                    else:
                        return "I couldn't type that text."
                else:
                    return "I need to know what text to type."
            else:
                return "I didn't understand what you want me to type."
                
        except Exception as e:
            return f"I had trouble typing: {str(e)}"
    
    async def _handle_scroll_command(self, command: str) -> str:
        """Handle scroll commands"""
        try:
            if "up" in command.lower():
                clicks = 3
            elif "down" in command.lower():
                clicks = -3
            else:
                clicks = -3  # Default down
            
            if self.vision_engine:
                if self.vision_engine.scroll(clicks):
                    direction = "up" if clicks > 0 else "down"
                    return f"I scrolled {direction}."
                else:
                    return "I couldn't scroll."
            else:
                return "I don't have access to scroll right now."
                
        except Exception as e:
            return f"I had trouble scrolling: {str(e)}"
    
    async def _handle_screenshot_command(self) -> str:
        """Handle screenshot commands"""
        try:
            if self.vision_engine:
                screenshot = self.vision_engine.take_screenshot()
                if screenshot.size > 0:
                    return "I took a screenshot of your screen."
                else:
                    return "I couldn't take a screenshot."
            else:
                return "I don't have access to take screenshots right now."
                
        except Exception as e:
            return f"I had trouble taking a screenshot: {str(e)}"
    
    async def _handle_screen_analysis(self) -> str:
        """Handle screen analysis requests"""
        try:
            # This would analyze the current screen content
            return "I can see your screen, but I'm still learning to analyze its content in detail. What would you like me to help you with?"
            
        except Exception as e:
            return f"I had trouble analyzing your screen: {str(e)}"
    
    async def _handle_open_command(self, command: str) -> str:
        """Handle open application commands"""
        try:
            # Extract application name
            if "open" in command.lower():
                app_start = command.lower().find("open") + 5
                app_name = command[app_start:].strip()
                
                if app_name:
                    # Use Spotlight to open applications
                    if self.vision_engine:
                        self.vision_engine.press_key("cmd+space")  # Open Spotlight
                        time.sleep(0.5)
                        self.vision_engine.type_text(app_name)
                        time.sleep(0.5)
                        self.vision_engine.press_key("enter")
                        
                        return f"I'm opening {app_name} for you."
                    else:
                        return f"I can't open {app_name} right now."
                else:
                    return "What application would you like me to open?"
            else:
                return "I didn't understand what you want me to open."
                
        except Exception as e:
            return f"I had trouble opening the application: {str(e)}"
    
    async def _handle_close_command(self, command: str) -> str:
        """Handle close application commands"""
        try:
            if self.vision_engine:
                self.vision_engine.press_key("cmd+q")  # Close current application
                return "I closed the current application."
            else:
                return "I can't close applications right now."
                
        except Exception as e:
            return f"I had trouble closing the application: {str(e)}"
    
    async def _handle_general_conversation(self, command: str) -> str:
        """Handle general conversation"""
        try:
            # Use ICEBURG's intelligence for general conversation
            if self.voice_system:
                response = await self.voice_system.process_voice_input(command)
                if response:
                    return response.text_input
                else:
                    return "I'm here to help. What would you like to know?"
            else:
                return "I'm here to help. What would you like to know?"
                
        except Exception as e:
            return f"I had trouble processing that: {str(e)}"
    
    def set_callbacks(self,
        on_user_detected: Optional[Callable[[], None]] = None,
                     on_attention_needed: Optional[Callable[[str], None]] = None,
                     on_computer_interaction: Optional[Callable[[str], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_user_detected = on_user_detected
        self.on_attention_needed = on_attention_needed
        self.on_computer_interaction = on_computer_interaction
        self.on_error = on_error
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        if self.on_error:
            self.on_error(error)
    
    async def shutdown(self):
        """Shutdown Jenny"""
        try:
            
            if self.voice_system:
                await self.voice_system.end_conversation()
            
            if self.vision_engine:
                await self.vision_engine.stop_vision()
            
            self.is_active = False
            self.state.is_active = False
            
            
        except Exception as e:
    
    def get_status(self) -> Dict[str, Any]:
        """Get Jenny's current status"""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "state": {
                "is_listening": self.state.is_listening,
                "is_seeing": self.state.is_seeing,
                "is_interacting": self.state.is_interacting,
                "user_present": self.state.user_present,
                "attention_level": self.state.attention_level,
                "current_context": self.state.current_context
            },
            "voice_system": self.voice_system.get_status() if self.voice_system else None,
            "vision_engine": self.vision_engine.get_status() if self.vision_engine else None
        }
