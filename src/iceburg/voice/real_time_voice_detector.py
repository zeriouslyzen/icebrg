#!/usr/bin/env python3
"""
Real-Time Voice Detection for Natural Conversation
Continuously listens and responds to voice input naturally
"""

import asyncio
import threading
import time
import speech_recognition as sr
import whisper
import tempfile
import os
import numpy as np
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import queue
import pyaudio
import wave
import io


@dataclass
class VoiceEvent:
    """Voice event data"""
    text: str
    confidence: float
    timestamp: datetime
    source: str
    is_final: bool


class RealTimeVoiceDetector:
    """Real-time voice detection with natural conversation flow"""
    
    def __init__(self):
        self.is_listening = False
        self.is_processing = False
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Voice activity detection
        self.energy_threshold = 300
        self.pause_threshold = 0.8
        self.phrase_timeout = 3.0
        self.phrase_time_limit = 5.0
        
        # Whisper model
        self.whisper_model = None
        self.use_whisper = True
        
        # Callbacks
        self.on_voice_detected: Optional[Callable[[VoiceEvent], None]] = None
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        
        # Threads
        self.listening_thread = None
        self.processing_thread = None
        
        # State
        self.last_activity = time.time()
        self.is_speaking = False
        
    
    def initialize(self) -> bool:
        """Initialize the voice detector"""
        try:
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Set energy threshold for voice activity detection
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = self.pause_threshold
            self.recognizer.phrase_timeout = self.phrase_timeout
            self.recognizer.phrase_time_limit = self.phrase_time_limit
            
            # Load Whisper model
            if self.use_whisper:
                try:
                    import warnings
                    # Suppress FP16 warning for CPU-only execution
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")
                        self.whisper_model = whisper.load_model("base", device="cpu")
                except Exception as e:
                    self.use_whisper = False
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    def start_listening(self):
        """Start continuous voice detection"""
        if self.is_listening:
            return
        
        self.is_listening = True
        
        # Start listening thread
        self.listening_thread = threading.Thread(target=self._listening_worker)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    
    def stop_listening(self):
        """Stop voice detection"""
        self.is_listening = False
    
    def _listening_worker(self):
        """Main listening worker thread"""
        while self.is_listening:
            try:
                # Listen for audio
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source, 
                        timeout=1, 
                        phrase_time_limit=self.phrase_time_limit
                    )
                
                if audio:
                    
                    # Notify speech start
                    if self.on_speech_start and not self.is_speaking:
                        self.on_speech_start()
                        self.is_speaking = True
                    
                    # Add to processing queue
                    self.processing_queue.put(audio)
                    self.last_activity = time.time()
                
            except sr.WaitTimeoutError:
                # No speech detected, check if we should end current speech
                if self.is_speaking and (time.time() - self.last_activity) > self.phrase_timeout:
                    if self.on_speech_end:
                        self.on_speech_end()
                    self.is_speaking = False
                continue
                
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                time.sleep(0.1)
    
    def _processing_worker(self):
        """Audio processing worker thread"""
        while self.is_listening:
            try:
                # Get audio from queue
                audio = self.processing_queue.get(timeout=1)
                
                if audio:
                    # Process audio
                    self._process_audio(audio)
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
    
    def _process_audio(self, audio):
        """Process audio and extract text"""
        try:
            # Try Google Speech Recognition first (faster)
            try:
                text = self.recognizer.recognize_google(audio)
                confidence = 0.9  # Google doesn't provide confidence
                
                if text.strip():
                    event = VoiceEvent(
                        text=text.strip(),
                        confidence=confidence,
                        timestamp=datetime.now(),
                        source="google_speech",
                        is_final=True
                    )
                    
                    
                    if self.on_voice_detected:
                        self.on_voice_detected(event)
                    
                    return
                    
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
            
            # Fallback to Whisper if available
            if self.use_whisper and self.whisper_model:
                try:
                    # Save audio to temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file_path = temp_file.name
                    
                    # Save audio data
                    with open(temp_file_path, "wb") as f:
                        f.write(audio.get_wav_data())
                    
                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(temp_file_path)
                    text = result["text"].strip()
                    
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    
                    if text:
                        event = VoiceEvent(
                            text=text,
                            confidence=0.95,
                            timestamp=datetime.now(),
                            source="whisper",
                            is_final=True
                        )
                        
                        
                        if self.on_voice_detected:
                            self.on_voice_detected(event)
                    
                except Exception as e:
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def set_callbacks(self,
        on_voice_detected: Optional[Callable[[VoiceEvent], None]] = None,
                     on_speech_start: Optional[Callable[[], None]] = None,
                     on_speech_end: Optional[Callable[[], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_voice_detected = on_voice_detected
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_error = on_error
    
    def set_sensitivity(self, energy_threshold: int = 300, pause_threshold: float = 0.8):
        """Adjust voice detection sensitivity"""
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status"""
        return {
            "is_listening": self.is_listening,
            "is_processing": self.is_processing,
            "is_speaking": self.is_speaking,
            "use_whisper": self.use_whisper,
            "energy_threshold": self.energy_threshold,
            "pause_threshold": self.pause_threshold,
            "last_activity": self.last_activity
        }
