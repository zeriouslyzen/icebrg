"""
Physiological Frequency Synthesizer
Generate specific frequencies to influence physiological states and relaxation
"""

import asyncio
import numpy as np
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)

@dataclass
class FrequencyProfile:
    """Frequency profile for physiological state modulation"""
    name: str
    frequency: float
    amplitude: float
    duration: float
    waveform: str  # 'sine', 'square', 'triangle', 'sawtooth'
    harmonics: List[float]
    binaural_beat: Optional[float] = None

@dataclass
class PhysiologicalState:
    """Target physiological state"""
    state_name: str
    primary_frequency: float
    secondary_frequencies: List[float]
    binaural_beat: float
    description: str

class PhysiologicalFrequencySynthesizer:
    """
    Synthesize frequencies to influence physiological states
    """
    
    def __init__(self):
        self.audio_active = False
        self.current_profile = None
        self.audio_thread = None
        self.pyaudio_instance = None
        self.stream = None
        
        # Audio parameters
        self.sample_rate = 44100
        self.channels = 2  # Stereo for binaural beats
        self.chunk_size = 1024
        
        # Consciousness state profiles
        self.consciousness_profiles = {
            'deep_meditation': PhysiologicalState(
                state_name='deep_meditation',
                primary_frequency=4.0,  # Theta
                secondary_frequencies=[2.0, 6.0],
                binaural_beat=4.0,
                description='Deep meditation and subconscious access'
            ),
            'focused_attention': PhysiologicalState(
                state_name='focused_attention',
                primary_frequency=40.0,  # Gamma
                secondary_frequencies=[20.0, 60.0],
                binaural_beat=40.0,
                description='High focus and insight states'
            ),
            'creative_flow': PhysiologicalState(
                state_name='creative_flow',
                primary_frequency=10.0,  # Alpha
                secondary_frequencies=[8.0, 12.0],
                binaural_beat=10.0,
                description='Creative flow and inspiration'
            ),
            'earth_sync': PhysiologicalState(
                state_name='earth_sync',
                primary_frequency=7.83,  # Schumann resonance
                secondary_frequencies=[14.3, 20.8, 27.3, 33.8],
                binaural_beat=7.83,
                description='Synchronization with Earth frequencies'
            ),
            'insight_breakthrough': PhysiologicalState(
                state_name='insight_breakthrough',
                primary_frequency=100.0,  # High gamma
                secondary_frequencies=[40.0, 80.0, 120.0],
                binaural_beat=100.0,
                description='Breakthrough insights and epiphanies'
            ),
            'icberg_sync': PhysiologicalState(
                state_name='icberg_sync',
                primary_frequency=432.0,  # A4 tuning
                secondary_frequencies=[216.0, 864.0, 1296.0],
                binaural_beat=432.0,
                description='Synchronization with ICEBURG processing'
            )
        }
        
        # Brainwave frequency bands
        self.brainwave_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0)
        }
        
    async def start_audio_synthesis(self) -> None:
        """Start audio synthesis system"""
        try:
            if not PYAUDIO_AVAILABLE:
                logger.warning("🎵 PyAudio not available - audio synthesis disabled")
                self.audio_active = False
                return
                
            logger.info("🎵 Starting consciousness frequency synthesis...")
            self.audio_active = True
            
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Start audio synthesis thread
            self.audio_thread = threading.Thread(target=self._audio_synthesis_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            logger.info("✅ Consciousness frequency synthesis started")
            
        except Exception as e:
            logger.error(f"Error starting audio synthesis: {e}")
            self.audio_active = False
            
    async def stop_audio_synthesis(self) -> None:
        """Stop audio synthesis"""
        try:
            self.audio_active = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                
            if self.audio_thread:
                self.audio_thread.join(timeout=2)
                
            logger.info("🎵 Consciousness frequency synthesis stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio synthesis: {e}")
            
    def set_consciousness_state(self, state_name: str) -> bool:
        """Set target consciousness state"""
        try:
            if state_name not in self.consciousness_profiles:
                logger.error(f"Unknown consciousness state: {state_name}")
                return False
                
            profile = self.consciousness_profiles[state_name]
            
            # Create frequency profile
            self.current_profile = FrequencyProfile(
                name=state_name,
                frequency=profile.primary_frequency,
                amplitude=0.3,  # Safe amplitude
                duration=0.0,  # Continuous
                waveform='sine',
                harmonics=profile.secondary_frequencies,
                binaural_beat=profile.binaural_beat
            )
            
            logger.info(f"🧠 Set consciousness state to: {state_name} ({profile.description})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting consciousness state: {e}")
            return False
            
    def generate_custom_frequency(self, frequency: float, amplitude: float = 0.3, 
                                waveform: str = 'sine', binaural_beat: Optional[float] = None) -> bool:
        """Generate custom frequency"""
        try:
            self.current_profile = FrequencyProfile(
                name='custom',
                frequency=frequency,
                amplitude=amplitude,
                duration=0.0,
                waveform=waveform,
                harmonics=[],
                binaural_beat=binaural_beat
            )
            
            logger.info(f"🎵 Generated custom frequency: {frequency} Hz")
            return True
            
        except Exception as e:
            logger.error(f"Error generating custom frequency: {e}")
            return False
            
    def _audio_synthesis_loop(self) -> None:
        """Main audio synthesis loop"""
        try:
            # Open audio stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Generate audio samples
            sample_count = 0
            
            while self.audio_active:
                if self.current_profile:
                    # Generate audio chunk
                    audio_chunk = self._generate_audio_chunk(self.current_profile, sample_count)
                    
                    # Write to audio stream
                    self.stream.write(audio_chunk.tobytes())
                    
                    sample_count += self.chunk_size
                else:
                    # Generate silence
                    silence = np.zeros((self.chunk_size, self.channels), dtype=np.float32)
                    self.stream.write(silence.tobytes())
                    
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in audio synthesis loop: {e}")
            
    def _generate_audio_chunk(self, profile: FrequencyProfile, sample_offset: int) -> np.ndarray:
        """Generate audio chunk for given profile"""
        try:
            # Generate time array
            t = np.arange(sample_offset, sample_offset + self.chunk_size) / self.sample_rate
            
            # Initialize audio buffer
            audio_buffer = np.zeros((self.chunk_size, self.channels), dtype=np.float32)
            
            # Generate primary frequency
            primary_wave = self._generate_waveform(
                profile.frequency, profile.amplitude, t, profile.waveform
            )
            
            # Generate harmonics
            harmonic_waves = []
            for harmonic_freq in profile.harmonics:
                harmonic_wave = self._generate_waveform(
                    harmonic_freq, profile.amplitude * 0.3, t, profile.waveform
                )
                harmonic_waves.append(harmonic_wave)
                
            # Combine all waves
            combined_wave = primary_wave
            for harmonic_wave in harmonic_waves:
                combined_wave += harmonic_wave
                
            # Apply binaural beat if specified
            if profile.binaural_beat:
                left_freq = profile.frequency
                right_freq = profile.frequency + profile.binaural_beat
                
                left_wave = self._generate_waveform(left_freq, profile.amplitude, t, profile.waveform)
                right_wave = self._generate_waveform(right_freq, profile.amplitude, t, profile.waveform)
                
                audio_buffer[:, 0] = left_wave  # Left channel
                audio_buffer[:, 1] = right_wave  # Right channel
            else:
                # Mono to stereo
                audio_buffer[:, 0] = combined_wave
                audio_buffer[:, 1] = combined_wave
                
            return audio_buffer
            
        except Exception as e:
            logger.error(f"Error generating audio chunk: {e}")
            return np.zeros((self.chunk_size, self.channels), dtype=np.float32)
            
    def _generate_waveform(self, frequency: float, amplitude: float, 
                          time_array: np.ndarray, waveform: str) -> np.ndarray:
        """Generate specific waveform"""
        try:
            if waveform == 'sine':
                return amplitude * np.sin(2 * np.pi * frequency * time_array)
            elif waveform == 'square':
                return amplitude * np.sign(np.sin(2 * np.pi * frequency * time_array))
            elif waveform == 'triangle':
                return amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * time_array))
            elif waveform == 'sawtooth':
                return amplitude * (2 / np.pi) * np.arctan(np.tan(np.pi * frequency * time_array))
            else:
                return amplitude * np.sin(2 * np.pi * frequency * time_array)
                
        except Exception as e:
            logger.error(f"Error generating waveform: {e}")
            return np.zeros_like(time_array)
            
    def get_available_states(self) -> List[str]:
        """Get list of available consciousness states"""
        return list(self.consciousness_profiles.keys())
        
    def get_current_state(self) -> Optional[str]:
        """Get current consciousness state"""
        if self.current_profile:
            return self.current_profile.name
        return None
        
    def get_synthesis_status(self) -> Dict[str, Any]:
        """Get synthesis system status"""
        return {
            'audio_active': self.audio_active,
            'current_state': self.get_current_state(),
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'available_states': self.get_available_states(),
            'stream_active': self.stream is not None and self.stream.is_active() if self.stream else False
        }
