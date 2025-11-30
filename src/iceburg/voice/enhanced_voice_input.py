#!/usr/bin/env python3
"""
Enhanced Voice Input Module with streaming and advanced audio processing
"""

import asyncio
import threading
import queue
import time
import tempfile
import os
import numpy as np
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime
import speech_recognition as sr
import whisper
import pyaudio
import wave
from scipy import signal
from scipy.io import wavfile

from .config_manager import VoiceConfigManager, VoiceSystemConfig


@dataclass
class AudioChunk:
    """Audio chunk data"""
    data: bytes
    timestamp: datetime
    sample_rate: int
    channels: int
    duration: float


@dataclass
class VoiceInput:
    """Enhanced voice input data"""
    audio_data: bytes
    duration: float
    timestamp: datetime
    transcription: str
    confidence: float
    source: str
    language: str
    emotions: Dict[str, float]
    noise_level: float
    quality_score: float


class AudioProcessor:
    """Advanced audio processing utilities"""
    
    def __init__(self, config: VoiceSystemConfig):
        self.config = config
    
    def apply_noise_reduction(self, audio_data: bytes, sample_rate: int) -> bytes:
        """Apply noise reduction to audio data"""
        if not self.config.voice.noise_reduction:
            return audio_data
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply noise reduction using spectral subtraction
            # This is a simplified implementation
            noise_reduced = self._spectral_subtraction(audio_array, sample_rate)
            
            # Convert back to bytes
            return noise_reduced.astype(np.int16).tobytes()
        except Exception as e:
            return audio_data
    
    def _spectral_subtraction(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Spectral subtraction for noise reduction"""
        # Simple high-pass filter to remove low-frequency noise
        nyquist = sample_rate / 2
        cutoff = 300  # Hz
        normalized_cutoff = cutoff / nyquist
        
        # Design Butterworth high-pass filter
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def apply_echo_cancellation(self, audio_data: bytes, sample_rate: int) -> bytes:
        """Apply echo cancellation"""
        if not self.config.voice.echo_cancellation:
            return audio_data
        
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Simple echo cancellation using adaptive filtering
            # This is a basic implementation
            echo_cancelled = self._adaptive_echo_cancellation(audio_array)
            
            return echo_cancelled.astype(np.int16).tobytes()
        except Exception as e:
            return audio_data
    
    def _adaptive_echo_cancellation(self, audio: np.ndarray) -> np.ndarray:
        """Adaptive echo cancellation"""
        # Simple delay-based echo cancellation
        # In a real implementation, this would be more sophisticated
        delay_samples = int(0.1 * len(audio) / 100)  # 10% delay
        if delay_samples > 0:
            echo_cancelled = audio - np.roll(audio, delay_samples) * 0.3
            return echo_cancelled
        return audio
    
    def apply_auto_gain_control(self, audio_data: bytes, sample_rate: int) -> bytes:
        """Apply automatic gain control"""
        if not self.config.voice.auto_gain_control:
            return audio_data
        
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Target RMS level for speech detection (much higher for Whisper compatibility)
            target_rms = 1000.0

            if rms > 0:
                gain = target_rms / rms
                gain = np.clip(gain, 0.1, 20.0)  # Increased max gain for better speech detection
                
                # Apply gain
                audio_array = audio_array.astype(np.float32) * gain
                audio_array = np.clip(audio_array, -32768, 32767)
            
            return audio_array.astype(np.int16).tobytes()
        except Exception as e:
            return audio_data
    
    def calculate_quality_score(self, audio_data: bytes, sample_rate: int) -> float:
        """Calculate audio quality score"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate various quality metrics
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            snr = self._calculate_snr(audio_array)
            dynamic_range = np.max(audio_array) - np.min(audio_array)
            
            # Normalize metrics to 0-1 scale
            rms_score = min(rms / 10000, 1.0)  # Normalize RMS
            snr_score = min(snr / 30, 1.0)     # Normalize SNR
            dynamic_score = min(dynamic_range / 65536, 1.0)  # Normalize dynamic range
            
            # Weighted average
            quality_score = (rms_score * 0.4 + snr_score * 0.4 + dynamic_score * 0.2)
            
            return min(max(quality_score, 0.0), 1.0)
        except Exception as e:
            return 0.5
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simple SNR calculation
        signal_power = np.mean(audio.astype(np.float32) ** 2)
        
        # Estimate noise as high-frequency components
        noise = audio - signal.medfilt(audio, kernel_size=5)
        noise_power = np.mean(noise.astype(np.float32) ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return max(snr, 0)
        return 0


class EnhancedVoiceInputModule:
    """Enhanced voice input with streaming and advanced processing"""
    
    def __init__(self, config_manager: Optional[VoiceConfigManager] = None):
        self.config_manager = config_manager or VoiceConfigManager()
        self.config = self.config_manager.get_config()
        
        self.voice_counter = 0
        self.is_listening = False
        self.is_streaming = False
        self.current_session = None
        
        # Audio processing
        self.audio_processor = AudioProcessor(self.config)
        
        # Streaming components
        self.audio_queue = queue.Queue()
        self.streaming_thread = None
        self.processing_thread = None
        
        # Speech recognition
        self.recognizer = None
        self.microphone = None
        self.whisper_model = None
        
        # Callbacks
        self.on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None
        self.on_transcription: Optional[Callable[[VoiceInput], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize speech recognition and audio components"""
        try:
            # Initialize speech recognizer
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Configure recognizer
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.8
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Initialize Whisper model if enabled
            if self.config.voice.use_whisper:
                try:
                    import warnings
                    # Suppress FP16 warning for CPU-only execution
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

                        # Try to load the specified model, fallback to 'base' if not found
                        try:
                            self.whisper_model = whisper.load_model(
                                self.config.voice.whisper_model,
                                device="cpu"
                            )
                        except Exception as model_error:
                            self.whisper_model = whisper.load_model(
                                "base",
                                device="cpu"
                            )
                except Exception as e:
                    self.whisper_model = None
            
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def set_callbacks(self, 
                     on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None,
                     on_transcription: Optional[Callable[[VoiceInput], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions for real-time processing"""
        self.on_audio_chunk = on_audio_chunk
        self.on_transcription = on_transcription
        self.on_error = on_error
    
    async def start_streaming(self) -> None:
        """Start continuous audio streaming"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_worker)
        self.processing_thread = threading.Thread(target=self._processing_worker)
        
        self.streaming_thread.start()
        self.processing_thread.start()
        
    
    async def stop_streaming(self) -> None:
        """Stop continuous audio streaming"""
        self.is_streaming = False
        
        if self.streaming_thread:
            self.streaming_thread.join()
        if self.processing_thread:
            self.processing_thread.join()
        
    
    def _streaming_worker(self):
        """Worker thread for continuous audio streaming"""
        try:
            with self.microphone as source:
                while self.is_streaming:
                    try:
                        # Listen for audio
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        
                        # Create audio chunk
                        chunk = AudioChunk(
                            data=audio.get_wav_data(),
                            timestamp=datetime.now(),
                            sample_rate=self.config.voice.sample_rate,
                            channels=1,
                            duration=len(audio.get_wav_data()) / (self.config.voice.sample_rate * 2)
                        )
                        
                        # Add to queue
                        self.audio_queue.put(chunk)
                        
                        # Callback
                        if self.on_audio_chunk:
                            self.on_audio_chunk(chunk)
                            
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        if self.on_error:
                            self.on_error(e)
                        break
                        
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def _processing_worker(self):
        """Worker thread for audio processing"""
        while self.is_streaming:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=1)
                
                # Process audio
                processed_audio = self._process_audio_chunk(chunk)
                
                # Transcribe
                transcription = self._transcribe_audio(processed_audio)
                
                if transcription and transcription.confidence >= self.config.voice.confidence_threshold:
                    # Callback
                    if self.on_transcription:
                        self.on_transcription(transcription)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
    
    def _process_audio_chunk(self, chunk: AudioChunk) -> bytes:
        """Process audio chunk with advanced filtering"""
        audio_data = chunk.data
        
        # Apply noise reduction
        audio_data = self.audio_processor.apply_noise_reduction(audio_data, chunk.sample_rate)
        
        # Apply echo cancellation
        audio_data = self.audio_processor.apply_echo_cancellation(audio_data, chunk.sample_rate)
        
        # Apply auto gain control
        audio_data = self.audio_processor.apply_auto_gain_control(audio_data, chunk.sample_rate)
        
        return audio_data
    
    def _transcribe_audio(self, audio_data: bytes) -> Optional[VoiceInput]:
        """Transcribe audio using Whisper or Google Speech"""
        temp_file_path = None
        try:
            # Create temporary WAV file with proper header
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file_path = temp_file.name

            # Write WAV file with proper header
            with wave.open(temp_file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.voice.sample_rate)
                wav_file.writeframes(audio_data)

            try:
                if self.config.voice.use_whisper and self.whisper_model:
                    # Use Whisper for transcription
                    result = self.whisper_model.transcribe(temp_file_path)
                    transcription = result["text"].strip()
                    confidence = 0.95  # Whisper doesn't provide confidence scores
                    source = "local_whisper"
                else:
                    # Use Google Speech Recognition
                    with sr.AudioFile(temp_file_path) as source:
                        audio = self.recognizer.record(source)
                    
                    transcription = self.recognizer.recognize_google(audio)
                    confidence = 0.95
                    source = "google_speech"
                
                # Calculate quality metrics
                quality_score = self.audio_processor.calculate_quality_score(audio_data, self.config.voice.sample_rate)
                noise_level = self._calculate_noise_level(audio_data)
                
                # Create voice input
                voice_input = VoiceInput(
                    audio_data=audio_data,
                    duration=len(audio_data) / (self.config.voice.sample_rate * 2),
                    timestamp=datetime.now(),
                    transcription=transcription,
                    confidence=confidence,
                    source=source,
                    language="en",
                    emotions={},  # Would be populated by emotion detection
                    noise_level=noise_level,
                    quality_score=quality_score
                )
                
                return voice_input
                
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            # Clean up temporary file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            return None
    
    def _calculate_noise_level(self, audio_data: bytes) -> float:
        """Calculate noise level in audio"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate noise as high-frequency components
            noise = audio_array - signal.medfilt(audio_array, kernel_size=5)
            noise_level = np.sqrt(np.mean(noise.astype(np.float32) ** 2))
            
            # Normalize to 0-1 scale
            return min(noise_level / 10000, 1.0)
        except:
            return 0.5
    
    async def listen_once(self, timeout: int = 5) -> Optional[VoiceInput]:
        """Listen for a single voice input"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
                # Process audio
                processed_audio = self._process_audio_chunk(AudioChunk(
                    data=audio.get_wav_data(),
                    timestamp=datetime.now(),
                    sample_rate=self.config.voice.sample_rate,
                    channels=1,
                    duration=len(audio.get_wav_data()) / (self.config.voice.sample_rate * 2)
                ))
                
                # Transcribe
                transcription = self._transcribe_audio(processed_audio)
                
                if transcription:
                    return transcription
                else:
                    return None
                    
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return None
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration and reload components"""
        self.config_manager.update_config(section, updates)
        self.config = self.config_manager.get_config()
        
        # Reinitialize components if needed
        if section == "voice_system":
            self._initialize_components()
        
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "is_listening": self.is_listening,
            "is_streaming": self.is_streaming,
            "whisper_available": self.whisper_model is not None,
            "queue_size": self.audio_queue.qsize(),
            "config": {
                "use_whisper": self.config.voice.use_whisper,
                "whisper_model": self.config.voice.whisper_model,
                "sample_rate": self.config.voice.sample_rate,
                "confidence_threshold": self.config.voice.confidence_threshold
            }
        }
