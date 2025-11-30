#!/usr/bin/env python3
"""
Complete Jenny-ICEBURG Integration
Jenny 2025 with full ICEBURG intelligence integration
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

# Jenny 2025 imports
from .jenny_2025 import Jenny2025, JennyState
from .jenny_icberg_integration import JennyICEBURGIntegration


class JennyICEBURGComplete:
    """Complete Jenny 2025 with ICEBURG intelligence integration"""
    
    def __init__(self, name: str = "Jenny"):
        self.name = name
        self.is_active = False
        
        # Initialize Jenny 2025
        self.jenny = Jenny2025(name)
        
        # Initialize ICEBURG integration
        self.icberg_integration = JennyICEBURGIntegration(self.jenny)
        
        # State
        self.is_connected_to_icberg = False
        
        # Callbacks
        self.on_jenny_awake: Optional[Callable[[], None]] = None
        self.on_jenny_sleep: Optional[Callable[[], None]] = None
        self.on_icberg_connected: Optional[Callable[[], None]] = None
        self.on_research_started: Optional[Callable[[str], None]] = None
        self.on_research_completed: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
    
    async def initialize(self) -> bool:
        """Initialize complete Jenny-ICEBURG system"""
        try:
            
            # Initialize Jenny 2025
            jenny_success = await self.jenny.initialize()
            if not jenny_success:
                return False
            
            # Connect to ICEBURG intelligence
            icberg_success = await self.icberg_integration.connect_to_icberg()
            if not icberg_success:
            else:
                self.is_connected_to_icberg = True
            
            # Set up callbacks
            self._setup_callbacks()
            
            self.is_active = True
            
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
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
        
        # ICEBURG integration callbacks
        self.icberg_integration.set_callbacks(
            on_research_started=lambda query: self._on_research_started(query),
            on_research_completed=lambda query: self._on_research_completed(query),
            on_intelligence_connected=lambda: self._on_icberg_connected(),
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
    
    async def start_research(self, query: str, fast: bool = False, hybrid: bool = False) -> str:
        """Start ICEBURG research"""
        if not self.is_connected_to_icberg:
            return "I'm not connected to ICEBURG's research capabilities right now."
        
        return await self.icberg_integration.start_research(query, fast, hybrid)
    
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
    
    def _on_research_started(self, query: str):
        """Handle research started"""
        if self.on_research_started:
            self.on_research_started(query)
    
    def _on_research_completed(self, query: str):
        """Handle research completed"""
        if self.on_research_completed:
            self.on_research_completed(query)
    
    def _on_icberg_connected(self):
        """Handle ICEBURG connection"""
        if self.on_icberg_connected:
            self.on_icberg_connected()
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        if self.on_error:
            self.on_error(error)
    
    # Public API
    def set_callbacks(self,
        on_jenny_awake: Optional[Callable[[], None]] = None,
                     on_jenny_sleep: Optional[Callable[[], None]] = None,
                     on_icberg_connected: Optional[Callable[[], None]] = None,
                     on_research_started: Optional[Callable[[str], None]] = None,
                     on_research_completed: Optional[Callable[[str], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_jenny_awake = on_jenny_awake
        self.on_jenny_sleep = on_jenny_sleep
        self.on_icberg_connected = on_icberg_connected
        self.on_research_started = on_research_started
        self.on_research_completed = on_research_completed
        self.on_error = on_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        jenny_status = self.jenny.get_status()
        icberg_status = self.icberg_integration.get_status()
        
        return {
            "name": self.name,
            "is_active": self.is_active,
            "is_connected_to_icberg": self.is_connected_to_icberg,
            "jenny": jenny_status,
            "icberg_integration": icberg_status
        }
    
    async def shutdown(self):
        """Shutdown complete system"""
        try:
            
            # Disconnect from ICEBURG
            if self.is_connected_to_icberg:
                await self.icberg_integration.disconnect()
            
            # Shutdown Jenny
            await self.jenny.shutdown()
            
            self.is_active = False
            
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)

