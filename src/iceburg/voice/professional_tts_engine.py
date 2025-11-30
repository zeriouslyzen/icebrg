#!/usr/bin/env python3
"""
Professional TTS Engine using pyttsx3 for high-quality voice synthesis
"""

import pyttsx3
import threading
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from .config_manager import VoiceSystemConfig


@dataclass
class VoiceOutput:
    """Professional voice output data"""
    text_input: str
    duration: float
    timestamp: datetime
    voice_model: str
    quality: str
    source: str
    streaming: bool
    chunk_count: int
    total_chunks: int


class ProfessionalTTSEngine:
    """Professional TTS engine with high-quality voice synthesis"""
    
    def __init__(self, config: VoiceSystemConfig):
        self.config = config
        self.engine = None
        self.is_speaking = False
        self.is_streaming = False
        self.current_voice = None
        
        # Callbacks
        self.on_speech_start: Optional[callable] = None
        self.on_speech_end: Optional[callable] = None
        self.on_error: Optional[callable] = None
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the TTS engine"""
        try:
            # Initialize pyttsx3 engine
            self.engine = pyttsx3.init()
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            self.available_voices = [voice.id for voice in voices] if voices else []
            
            # Set default voice (prefer high-quality voices)
            self._set_best_voice()
            
            # Configure engine properties
            self._configure_engine()
            
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def _set_best_voice(self):
        """Set the best available voice"""
        # Prefer high-quality voices in order of preference
        preferred_voices = [
            'com.apple.speech.synthesis.voice.samantha.premium',  # Premium Samantha
            'com.apple.speech.synthesis.voice.samantha',          # Regular Samantha
            'com.apple.speech.synthesis.voice.alex.premium',      # Premium Alex
            'com.apple.speech.synthesis.voice.alex',              # Regular Alex
            'com.apple.speech.synthesis.voice.victoria',          # Victoria
            'com.apple.speech.synthesis.voice.daniel',            # Daniel
            'com.apple.speech.synthesis.voice.karen',             # Karen
        ]
        
        # Find the best available voice
        for preferred_voice in preferred_voices:
            if preferred_voice in self.available_voices:
                self.current_voice = preferred_voice
                self.engine.setProperty('voice', preferred_voice)
                break
        
        # Fallback to first available voice
        if not self.current_voice and self.available_voices:
            self.current_voice = self.available_voices[0]
            self.engine.setProperty('voice', self.available_voices[0])
    
    def _configure_engine(self):
        """Configure engine properties for best quality"""
        try:
            # Set speech rate (words per minute)
            rate = self.config.tts.rate
            self.engine.setProperty('rate', rate)
            
            # Set volume (0.0 to 1.0)
            volume = self.config.tts.volume
            self.engine.setProperty('volume', volume)
            
            # Note: pyttsx3 doesn't support pitch directly on macOS
            # The pitch is controlled by the voice selection
            
            
        except Exception as e:
    
    def speak(self, text: str, voice: Optional[str] = None) -> VoiceOutput:
        """Speak text with high quality"""
        try:
            self.is_speaking = True
            
            # Callback
            if self.on_speech_start:
                self.on_speech_start()
            
            # Set voice if specified
            if voice and voice in self.available_voices:
                self.engine.setProperty('voice', voice)
                current_voice = voice
            else:
                current_voice = self.current_voice
            
            # Calculate estimated duration
            word_count = len(text.split())
            duration = (word_count / self.config.tts.rate) * 60  # Convert to seconds
            
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
            
            # Create voice output
            voice_output = VoiceOutput(
                text_input=text,
                duration=duration,
                timestamp=datetime.now(),
                voice_model=current_voice,
                quality="professional",
                source="pyttsx3",
                streaming=False,
                chunk_count=1,
                total_chunks=1
            )
            
            self.is_speaking = False
            
            # Callback
            if self.on_speech_end:
                self.on_speech_end()
            
            return voice_output
            
        except Exception as e:
            self.is_speaking = False
            if self.on_error:
                self.on_error(e)
            
            return VoiceOutput(
                text_input=text,
                duration=0,
                timestamp=datetime.now(),
                voice_model=voice or self.current_voice,
                quality="error",
                source="pyttsx3",
                streaming=False,
                chunk_count=0,
                total_chunks=0
            )
    
    def speak_streaming(self, text: str, voice: Optional[str] = None) -> VoiceOutput:
        """Speak text with streaming (chunked) delivery"""
        try:
            self.is_speaking = True
            self.is_streaming = True
            
            # Callback
            if self.on_speech_start:
                self.on_speech_start()
            
            # Set voice if specified
            if voice and voice in self.available_voices:
                self.engine.setProperty('voice', voice)
                current_voice = voice
            else:
                current_voice = self.current_voice
            
            # Split text into chunks for streaming
            words = text.split()
            chunk_size = self.config.tts.chunk_size
            chunks = []
            
            for i in range(0, len(words), chunk_size):
                chunk_text = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk_text)
            
            # Speak each chunk
            total_duration = 0
            for i, chunk in enumerate(chunks):
                if not self.is_streaming:
                    break
                
                # Calculate chunk duration
                word_count = len(chunk.split())
                chunk_duration = (word_count / self.config.tts.rate) * 60
                total_duration += chunk_duration
                
                # Speak chunk
                self.engine.say(chunk)
                self.engine.runAndWait()
                
                # Small pause between chunks
                if i < len(chunks) - 1:
                    time.sleep(0.1)
            
            # Create voice output
            voice_output = VoiceOutput(
                text_input=text,
                duration=total_duration,
                timestamp=datetime.now(),
                voice_model=current_voice,
                quality="professional_streaming",
                source="pyttsx3",
                streaming=True,
                chunk_count=len(chunks),
                total_chunks=len(chunks)
            )
            
            self.is_speaking = False
            self.is_streaming = False
            
            # Callback
            if self.on_speech_end:
                self.on_speech_end()
            
            return voice_output
            
        except Exception as e:
            self.is_speaking = False
            self.is_streaming = False
            if self.on_error:
                self.on_error(e)
            
            return VoiceOutput(
                text_input=text,
                duration=0,
                timestamp=datetime.now(),
                voice_model=voice or self.current_voice,
                quality="error",
                source="pyttsx3",
                streaming=False,
                chunk_count=0,
                total_chunks=0
            )
    
    def speak_with_emotion(self, text: str, emotion: str = "neutral") -> VoiceOutput:
        """Speak with emotional inflection"""
        # Adjust voice and rate based on emotion
        emotion_config = self._get_emotion_config(emotion)
        
        # Temporarily adjust settings
        original_rate = self.engine.getProperty('rate')
        original_voice = self.engine.getProperty('voice')
        
        try:
            # Apply emotion settings
            self.engine.setProperty('rate', emotion_config['rate'])
            if emotion_config['voice'] in self.available_voices:
                self.engine.setProperty('voice', emotion_config['voice'])
            
            # Speak with emotion
            result = self.speak(text)
            result.quality = f"emotional_{emotion}"
            
            return result
            
        finally:
            # Restore original settings
            self.engine.setProperty('rate', original_rate)
            self.engine.setProperty('voice', original_voice)
    
    def _get_emotion_config(self, emotion: str) -> Dict[str, Any]:
        """Get configuration for emotional speech"""
        emotion_configs = {
            "happy": {
                "rate": int(self.config.tts.rate * 1.1),  # Slightly faster
                "voice": "com.apple.speech.synthesis.voice.samantha"
            },
            "sad": {
                "rate": int(self.config.tts.rate * 0.8),  # Slower
                "voice": "com.apple.speech.synthesis.voice.alex"
            },
            "angry": {
                "rate": int(self.config.tts.rate * 1.2),  # Faster
                "voice": "com.apple.speech.synthesis.voice.fred"
            },
            "excited": {
                "rate": int(self.config.tts.rate * 1.3),  # Much faster
                "voice": "com.apple.speech.synthesis.voice.victoria"
            },
            "calm": {
                "rate": int(self.config.tts.rate * 0.9),  # Slightly slower
                "voice": "com.apple.speech.synthesis.voice.alex"
            },
            "neutral": {
                "rate": self.config.tts.rate,
                "voice": self.current_voice
            }
        }
        
        return emotion_configs.get(emotion, emotion_configs["neutral"])
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        return self.available_voices
    
    def set_voice(self, voice: str) -> bool:
        """Set current voice"""
        if voice in self.available_voices:
            self.current_voice = voice
            self.engine.setProperty('voice', voice)
            return True
        else:
            return False
    
    def set_rate(self, rate: int) -> None:
        """Set speech rate"""
        self.engine.setProperty('rate', rate)
        self.config.tts.rate = rate
    
    def set_volume(self, volume: float) -> None:
        """Set volume"""
        volume = max(0.0, min(1.0, volume))  # Clamp to 0-1
        self.engine.setProperty('volume', volume)
        self.config.tts.volume = volume
    
    def stop(self) -> None:
        """Stop current speech"""
        if self.engine:
            self.engine.stop()
        self.is_speaking = False
        self.is_streaming = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "is_speaking": self.is_speaking,
            "is_streaming": self.is_streaming,
            "current_voice": self.current_voice,
            "available_voices": len(self.available_voices),
            "rate": self.engine.getProperty('rate') if self.engine else 0,
            "volume": self.engine.getProperty('volume') if self.engine else 0
        }
    
    def __del__(self):
        """Cleanup"""
        if self.engine:
            self.engine.stop()
