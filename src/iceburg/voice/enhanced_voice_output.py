#!/usr/bin/env python3
"""
Enhanced Voice Output Module with streaming and advanced TTS capabilities
"""

import asyncio
import threading
import queue
import time
import subprocess
import tempfile
import os
import json
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime
import pyaudio
import wave
import numpy as np

from .config_manager import VoiceConfigManager, VoiceSystemConfig
from .professional_tts_engine import ProfessionalTTSEngine
from .advanced_tts_engine import AdvancedTTSEngine


@dataclass
class VoiceOutput:
    """Enhanced voice output data"""
    text_input: str
    audio_data: bytes
    duration: float
    timestamp: datetime
    voice_model: str
    quality: str
    source: str
    streaming: bool
    chunk_count: int
    total_chunks: int


@dataclass
class AudioChunk:
    """Audio chunk for streaming"""
    data: bytes
    chunk_index: int
    total_chunks: int
    timestamp: datetime


class TTSEngine:
    """Text-to-Speech engine interface"""
    
    def __init__(self, config: VoiceSystemConfig):
        self.config = config
    
    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """Synthesize text to audio"""
        raise NotImplementedError
    
    async def synthesize_streaming(self, text: str, voice: str = None) -> List[bytes]:
        """Synthesize text to streaming audio chunks"""
        raise NotImplementedError
    
    def get_available_voices(self) -> List[str]:
        """Get available voices"""
        raise NotImplementedError


class MacOSSayEngine(TTSEngine):
    """macOS built-in TTS engine"""
    
    def __init__(self, config: VoiceSystemConfig):
        super().__init__(config)
        self.available_voices = self._get_available_voices()
    
    def _get_available_voices(self) -> List[str]:
        """Get available macOS voices"""
        try:
            result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
            voices = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    voice_name = line.split()[0]
                    voices.append(voice_name)
            return voices
        except Exception as e:
            return ["Alex", "Victoria", "Daniel", "Samantha"]
    
    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """Synthesize text to audio using macOS say"""
        voice = voice or self.config.tts.voice
        
        try:
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # Generate speech
            cmd = [
                'say',
                '-v', voice,
                '-r', str(self.config.tts.rate),
                '-o', temp_file_path,
                text
            ]
            
            subprocess.run(cmd, check=True)
            
            # Read audio file
            with open(temp_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_file_path)
            
            return audio_data
            
        except Exception as e:
            return b""
    
    async def synthesize_streaming(self, text: str, voice: str = None) -> List[bytes]:
        """Synthesize text to streaming audio chunks"""
        # For macOS say, we'll simulate streaming by chunking the text
        voice = voice or self.config.tts.voice
        
        # Split text into chunks
        words = text.split()
        chunk_size = self.config.tts.chunk_size
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_text = ' '.join(words[i:i + chunk_size])
            audio_data = await self.synthesize(chunk_text, voice)
            chunks.append(audio_data)
        
        return chunks
    
    def get_available_voices(self) -> List[str]:
        """Get available voices"""
        return self.available_voices


class PyAudioEngine(TTSEngine):
    """PyAudio-based TTS engine for real-time streaming"""
    
    def __init__(self, config: VoiceSystemConfig):
        super().__init__(config)
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 22050
        self.channels = 1
        self.chunk_size = 1024
    
    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """Synthesize text to audio"""
        # This would integrate with a more advanced TTS engine
        # For now, we'll use a placeholder
        return b""
    
    async def synthesize_streaming(self, text: str, voice: str = None) -> List[bytes]:
        """Synthesize text to streaming audio chunks"""
        # This would provide real-time streaming TTS
        # For now, we'll use a placeholder
        return []
    
    def get_available_voices(self) -> List[str]:
        """Get available voices"""
        return ["default"]


class EnhancedVoiceOutputModule:
    """Enhanced voice output with streaming and advanced TTS"""
    
    def __init__(self, config_manager: Optional[VoiceConfigManager] = None):
        self.config_manager = config_manager or VoiceConfigManager()
        self.config = self.config_manager.get_config()
        
        self.voice_counter = 0
        self.is_speaking = False
        self.is_streaming = False
        self.current_voice = self.config.tts.voice
        
        # TTS engines
        self.tts_engines = {
            "macos_say": MacOSSayEngine(self.config),
            "pyaudio": PyAudioEngine(self.config),
            "professional": ProfessionalTTSEngine(self.config),
            "advanced": AdvancedTTSEngine(self.config)
        }
        self.current_engine = self.tts_engines.get(self.config.tts.engine)
        
        # Streaming components
        self.audio_queue = queue.Queue()
        self.streaming_thread = None
        self.audio_thread = None
        
        # Audio output
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Callbacks
        self.on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        self._initialize_audio()
    
    async def initialize(self):
        """Initialize the voice output system"""
        try:
            # System is already initialized in constructor
            self.is_active = True
            
        except Exception as e:
            raise
    
    def _initialize_audio(self):
        """Initialize audio output"""
        try:
            # Skip PyAudio initialization for now to avoid static
            # We'll use direct macOS say command instead
            self.stream = None
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def set_callbacks(self,
                     on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None,
                     on_speech_start: Optional[Callable[[], None]] = None,
                     on_speech_end: Optional[Callable[[], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions for real-time processing"""
        self.on_audio_chunk = on_audio_chunk
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_error = on_error
    
    async def speak_response(self, text_response: str, voice_model: str = None) -> VoiceOutput:
        """Convert text response to speech"""
        voice_id = self._generate_voice_id()
        voice = voice_model or self.current_voice
        
        try:
            self.is_speaking = True
            
            # Callback
            if self.on_speech_start:
                self.on_speech_start()
            
            # Synthesize audio (handle different engine types)
            if hasattr(self.current_engine, 'synthesize'):
                audio_data = await self.current_engine.synthesize(text_response, voice)
            elif hasattr(self.current_engine, 'speak'):
                # For professional TTS engine
                if asyncio.iscoroutinefunction(self.current_engine.speak):
                    voice_output = await self.current_engine.speak(text_response, voice)
                else:
                    voice_output = self.current_engine.speak(text_response, voice)
                audio_data = voice_output.audio_data if hasattr(voice_output, 'audio_data') else b""
            else:
                audio_data = b""
            
            # Calculate duration (approximate)
            duration = len(audio_data) / (22050 * 2)  # 16-bit audio
            
            # Create voice output
            voice_output = VoiceOutput(
                text_input=text_response,
                audio_data=audio_data,
                duration=duration,
                timestamp=datetime.now(),
                voice_model=voice,
                quality="high",
                source=self.config.tts.engine,
                streaming=False,
                chunk_count=1,
                total_chunks=1
            )
            
            # Play audio using the current engine
            if hasattr(self.current_engine, 'speak'):
                # Use TTS engine (handle both sync and async)
                if asyncio.iscoroutinefunction(self.current_engine.speak):
                    voice_output = await self.current_engine.speak(text_response, voice)
                else:
                    voice_output = self.current_engine.speak(text_response, voice)
                duration = voice_output.duration
            else:
                # Fallback to direct say command
                try:
                    subprocess.run(['say', '-v', voice, '-r', str(self.config.tts.rate), text_response], check=True)
                except Exception as e:
                    # Fallback to PyAudio if needed
                    if self.stream and audio_data:
                        self.stream.write(audio_data)
            
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
                text_input=text_response,
                audio_data=b"",
                duration=0,
                timestamp=datetime.now(),
                voice_model=voice,
                quality="error",
                source=self.config.tts.engine,
                streaming=False,
                chunk_count=0,
                total_chunks=0
            )
    
    async def speak_streaming(self, text_response: str, voice_model: str = None) -> VoiceOutput:
        """Convert text response to streaming speech"""
        voice_id = self._generate_voice_id()
        voice = voice_model or self.current_voice
        
        try:
            self.is_speaking = True
            self.is_streaming = True
            
            # Callback
            if self.on_speech_start:
                self.on_speech_start()
            
            # Synthesize streaming audio
            audio_chunks = await self.current_engine.synthesize_streaming(text_response, voice)
            
            # Start streaming thread
            self.streaming_thread = threading.Thread(
                target=self._streaming_worker,
                args=(audio_chunks,)
            )
            self.streaming_thread.start()
            
            # Calculate total duration
            total_audio = b"".join(audio_chunks)
            duration = len(total_audio) / (22050 * 2)
            
            # Create voice output
            voice_output = VoiceOutput(
                text_input=text_response,
                audio_data=total_audio,
                duration=duration,
                timestamp=datetime.now(),
                voice_model=voice,
                quality="high",
                source=self.config.tts.engine,
                streaming=True,
                chunk_count=len(audio_chunks),
                total_chunks=len(audio_chunks)
            )
            
            # Wait for streaming to complete
            self.streaming_thread.join()
            
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
                text_input=text_response,
                audio_data=b"",
                duration=0,
                timestamp=datetime.now(),
                voice_model=voice,
                quality="error",
                source=self.config.tts.engine,
                streaming=False,
                chunk_count=0,
                total_chunks=0
            )
    
    def _streaming_worker(self, audio_chunks: List[bytes]):
        """Worker thread for streaming audio"""
        try:
            for i, chunk in enumerate(audio_chunks):
                if not self.is_streaming:
                    break
                
                # Create audio chunk
                audio_chunk = AudioChunk(
                    data=chunk,
                    chunk_index=i,
                    total_chunks=len(audio_chunks),
                    timestamp=datetime.now()
                )
                
                # Play audio chunk using direct say command
                try:
                    # For streaming, we'll use direct say command per chunk
                    subprocess.run(['say', '-v', voice, '-r', str(self.config.tts.rate), chunk_text], check=True)
                except Exception as e:
                    # Fallback to PyAudio if needed
                    if self.stream and chunk:
                        self.stream.write(chunk)
                
                # Callback
                if self.on_audio_chunk:
                    self.on_audio_chunk(audio_chunk)
                
                # Small delay between chunks
                time.sleep(0.1)
                
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    async def speak_with_emotion(self, text_response: str, emotion: str = "neutral") -> VoiceOutput:
        """Speak with emotional inflection"""
        # Adjust voice parameters based on emotion
        voice_params = self._get_emotion_voice_params(emotion)
        
        # Temporarily update voice parameters
        original_voice = self.current_voice
        original_rate = self.config.tts.rate
        original_pitch = self.config.tts.pitch
        
        # Apply emotion parameters
        self.current_voice = voice_params.get("voice", original_voice)
        self.config.tts.rate = voice_params.get("rate", original_rate)
        self.config.tts.pitch = voice_params.get("pitch", original_pitch)
        
        try:
            # Speak with emotion
            result = await self.speak_response(text_response)
            result.quality = f"emotional_{emotion}"
            return result
        finally:
            # Restore original parameters
            self.current_voice = original_voice
            self.config.tts.rate = original_rate
            self.config.tts.pitch = original_pitch
    
    def _get_emotion_voice_params(self, emotion: str) -> Dict[str, Any]:
        """Get voice parameters for emotion"""
        emotion_params = {
            "happy": {"rate": 220, "pitch": 1.2, "voice": "Samantha"},
            "sad": {"rate": 150, "pitch": 0.8, "voice": "Alex"},
            "angry": {"rate": 250, "pitch": 1.1, "voice": "Fred"},
            "excited": {"rate": 280, "pitch": 1.3, "voice": "Victoria"},
            "calm": {"rate": 180, "pitch": 0.9, "voice": "Alex"},
            "neutral": {"rate": 200, "pitch": 1.0, "voice": "Alex"}
        }
        
        return emotion_params.get(emotion, emotion_params["neutral"])
    
    def get_available_voices(self) -> List[str]:
        """Get available voices"""
        if self.current_engine:
            return self.current_engine.get_available_voices()
        return []
    
    def set_voice(self, voice: str) -> bool:
        """Set current voice"""
        available_voices = self.get_available_voices()
        if voice in available_voices:
            self.current_voice = voice
            self.config.tts.voice = voice
            return True
        else:
            return False
    
    def set_engine(self, engine: str) -> bool:
        """Set TTS engine"""
        if engine in self.tts_engines:
            self.current_engine = self.tts_engines[engine]
            self.config.tts.engine = engine
            return True
        else:
            return False
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config_manager.update_config(section, updates)
        self.config = self.config_manager.get_config()
        
        # Reinitialize if needed
        if section == "tts":
            self._initialize_audio()
        
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "is_speaking": self.is_speaking,
            "is_streaming": self.is_streaming,
            "current_voice": self.current_voice,
            "current_engine": self.config.tts.engine,
            "available_voices": self.get_available_voices(),
            "config": {
                "engine": self.config.tts.engine,
                "voice": self.config.tts.voice,
                "rate": self.config.tts.rate,
                "volume": self.config.tts.volume,
                "pitch": self.config.tts.pitch,
                "streaming": self.config.tts.streaming
            }
        }
    
    def _generate_voice_id(self) -> str:
        """Generate unique voice ID"""
        self.voice_counter += 1
        return f"voice_{self.voice_counter}_{int(time.time())}"
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
