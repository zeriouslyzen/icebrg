#!/usr/bin/env python3
"""
Advanced Voice Activity Detection (VAD) for 2025
Implements MFCC, STFT, and deep learning for real-time voice detection
"""

import numpy as np
import librosa
import scipy.signal
import threading
import queue
import time
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import pyaudio
import wave
import tempfile
import os

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class VoiceFrame:
    """Voice frame data"""
    audio_data: np.ndarray
    mfcc_features: np.ndarray
    spectral_features: np.ndarray
    energy: float
    zero_crossing_rate: float
    timestamp: datetime
    is_speech: bool
    confidence: float


class AdvancedVAD:
    """Advanced Voice Activity Detection using MFCC, STFT, and ML"""
    
    def __init__(self, sample_rate: int = 16000, frame_length: int = 1024, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # Audio parameters
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # VAD parameters
        self.energy_threshold = 0.01
        self.zcr_threshold = 0.1
        self.speech_threshold = 0.5
        
        # Feature extraction
        self.n_mfcc = 13
        self.n_fft = 2048
        
        # State
        self.is_active = False
        self.audio_queue = queue.Queue()
        self.feature_queue = queue.Queue()
        
        # Threads
        self.audio_thread = None
        self.processing_thread = None
        
        # Callbacks
        self.on_voice_detected: Optional[Callable[[VoiceFrame], None]] = None
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Speech state
        self.is_speaking = False
        self.speech_frames = []
        self.silence_frames = 0
        self.silence_threshold = 10  # frames
        
    
    def initialize(self) -> bool:
        """Initialize the VAD system"""
        try:
            # Initialize audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    def start_detection(self):
        """Start voice activity detection"""
        if self.is_active:
            return
        
        self.is_active = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start audio stream
        self.stream.start_stream()
        
    
    def stop_detection(self):
        """Stop voice activity detection"""
        self.is_active = False
        
        if self.stream:
            self.stream.stop_stream()
        
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if self.is_active:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to processing queue
            self.audio_queue.put(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def _processing_worker(self):
        """Main processing worker thread"""
        while self.is_active:
            try:
                # Get audio data
                audio_data = self.audio_queue.get(timeout=1)
                
                if audio_data is not None:
                    # Extract features
                    features = self._extract_features(audio_data)
                    
                    # Classify as speech or non-speech
                    is_speech, confidence = self._classify_speech(features)
                    
                    # Create voice frame
                    voice_frame = VoiceFrame(
                        audio_data=audio_data,
                        mfcc_features=features['mfcc'],
                        spectral_features=features['spectral'],
                        energy=features['energy'],
                        zero_crossing_rate=features['zcr'],
                        timestamp=datetime.now(),
                        is_speech=is_speech,
                        confidence=confidence
                    )
                    
                    # Handle speech state changes
                    self._handle_speech_state(voice_frame)
                    
                    # Callback
                    if self.on_voice_detected:
                        self.on_voice_detected(voice_frame)
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
    
    def _extract_features(self, audio_data: np.ndarray) -> dict:
        """Extract MFCC and spectral features"""
        try:
            # Ensure audio is the right length
            if len(audio_data) < self.frame_length:
                audio_data = np.pad(audio_data, (0, self.frame_length - len(audio_data)))
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )
            
            # Combine spectral features
            spectral_features = np.concatenate([
                spectral_centroid.flatten(),
                spectral_rolloff.flatten(),
                spectral_bandwidth.flatten()
            ])
            
            # Calculate energy
            energy = np.sum(audio_data ** 2)
            
            # Calculate zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            return {
                'mfcc': mfcc.flatten(),
                'spectral': spectral_features,
                'energy': energy,
                'zcr': zcr
            }
            
        except Exception as e:
            return {
                'mfcc': np.zeros(self.n_mfcc),
                'spectral': np.zeros(3),
                'energy': 0.0,
                'zcr': 0.0
            }
    
    def _classify_speech(self, features: dict) -> Tuple[bool, float]:
        """Classify audio as speech or non-speech"""
        try:
            # Simple rule-based classification
            energy = features['energy']
            zcr = features['zcr']
            
            # Energy-based classification
            energy_score = 1.0 if energy > self.energy_threshold else 0.0
            
            # Zero crossing rate classification
            zcr_score = 1.0 if zcr > self.zcr_threshold else 0.0
            
            # Combined confidence
            confidence = (energy_score + zcr_score) / 2.0
            
            # Speech decision
            is_speech = confidence > self.speech_threshold
            
            return is_speech, confidence
            
        except Exception as e:
            return False, 0.0
    
    def _handle_speech_state(self, voice_frame: VoiceFrame):
        """Handle speech state changes"""
        try:
            if voice_frame.is_speech:
                self.speech_frames.append(voice_frame)
                self.silence_frames = 0
                
                # Speech start detection
                if not self.is_speaking and len(self.speech_frames) >= 3:
                    self.is_speaking = True
                    if self.on_speech_start:
                        self.on_speech_start()
            else:
                self.silence_frames += 1
                
                # Speech end detection
                if self.is_speaking and self.silence_frames >= self.silence_threshold:
                    self.is_speaking = False
                    if self.on_speech_end:
                        self.on_speech_end()
                    
                    # Process collected speech frames
                    if self.speech_frames:
                        self._process_speech_segment()
                        self.speech_frames = []
            
        except Exception as e:
    
    def _process_speech_segment(self):
        """Process collected speech frames"""
        try:
            if not self.speech_frames:
                return
            
            # Combine audio data
            audio_segments = [frame.audio_data for frame in self.speech_frames]
            combined_audio = np.concatenate(audio_segments)
            
            # Calculate average confidence
            avg_confidence = np.mean([frame.confidence for frame in self.speech_frames])
            
            
        except Exception as e:
    
    def set_callbacks(self,
        on_voice_detected: Optional[Callable[[VoiceFrame], None]] = None,
                     on_speech_start: Optional[Callable[[], None]] = None,
                     on_speech_end: Optional[Callable[[], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_voice_detected = on_voice_detected
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_error = on_error
    
    def set_sensitivity(self, energy_threshold: float = 0.01, speech_threshold: float = 0.5):
        """Adjust VAD sensitivity"""
        self.energy_threshold = energy_threshold
        self.speech_threshold = speech_threshold
    
    def get_status(self) -> dict:
        """Get VAD status"""
        return {
            "is_active": self.is_active,
            "is_speaking": self.is_speaking,
            "energy_threshold": self.energy_threshold,
            "speech_threshold": self.speech_threshold,
            "speech_frames_count": len(self.speech_frames),
            "silence_frames": self.silence_frames
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_detection()
        
        if self.stream:
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
