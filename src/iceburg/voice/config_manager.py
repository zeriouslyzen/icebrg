#!/usr/bin/env python3
"""
Voice System Configuration Manager
Handles dynamic configuration loading and validation
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class VoiceConfig:
    """Voice system configuration"""
    enabled: bool = True
    use_whisper: bool = True
    whisper_model: str = "whisper-1"
    fallback_to_google: bool = True
    sample_rate: int = 16000
    chunk_size: int = 1024
    ambient_mode: bool = True
    touch_activation: bool = True
    context_retention: int = 10
    max_audio_duration: int = 30
    confidence_threshold: float = 0.7
    noise_reduction: bool = True
    echo_cancellation: bool = True
    auto_gain_control: bool = True


@dataclass
class TTSConfig:
    """Text-to-Speech configuration"""
    engine: str = "macos_say"
    voice: str = "Alex"
    rate: int = 200
    volume: float = 0.8
    pitch: float = 1.0
    streaming: bool = True
    chunk_size: int = 100
    buffer_size: int = 1000
    real_time: bool = True


@dataclass
class SensorConfig:
    """M1 sensor configuration"""
    enabled: bool = True
    touch_detection: bool = True
    physiological_monitoring: bool = True
    background_monitoring: bool = True
    emergence_detection: bool = True
    sample_rate: int = 1
    buffer_size: int = 1000
    sensitivity: float = 0.5
    threshold: float = 0.7


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    max_cpu_usage: int = 25
    max_memory_usage: int = 500
    optimize_for_speed: bool = True
    enable_caching: bool = True
    cache_size: int = 100
    parallel_processing: bool = True
    thread_pool_size: int = 4


@dataclass
class PrivacyConfig:
    """Privacy and security configuration"""
    local_processing_only: bool = True
    no_cloud_storage: bool = True
    encrypt_sensitive_data: bool = True
    data_retention_days: int = 30
    user_consent_required: bool = True
    anonymize_data: bool = True


@dataclass
class AdvancedConfig:
    """Advanced features configuration"""
    enable_emotion_detection: bool = False
    enable_gesture_recognition: bool = False
    enable_eye_tracking: bool = False
    multi_language_support: bool = False
    voice_cloning: bool = False
    predictive_responses: bool = False
    adaptive_learning: bool = True
    context_awareness: bool = True


@dataclass
class VoiceSystemConfig:
    """Complete voice system configuration"""
    voice: VoiceConfig
    tts: TTSConfig
    sensors: SensorConfig
    performance: PerformanceConfig
    privacy: PrivacyConfig
    advanced: AdvancedConfig


class VoiceConfigManager:
    """Manages voice system configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Use relative path to avoid iCloud sync issues
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent.parent
        project_config = project_root / "config" / "voice_config.json"
        if project_config.exists():
            return str(project_config)
        
        # Fallback to user home directory
        home_config = os.path.expanduser("~/.iceburg/voice_config.json")
        os.makedirs(os.path.dirname(home_config), exist_ok=True)
        return home_config
    
    def _load_config(self) -> VoiceSystemConfig:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return self._parse_config(config_data)
            except Exception as e:
        
        # Create default configuration
        default_config = self._create_default_config()
        self._save_config(default_config)
        return default_config
    
    def _parse_config(self, config_data: Dict[str, Any]) -> VoiceSystemConfig:
        """Parse configuration data into structured format"""
        return VoiceSystemConfig(
            voice=VoiceConfig(**config_data.get('voice_system', {})),
            tts=TTSConfig(**config_data.get('tts', {})),
            sensors=SensorConfig(**config_data.get('m1_sensors', {})),
            performance=PerformanceConfig(**config_data.get('performance', {})),
            privacy=PrivacyConfig(**config_data.get('privacy', {})),
            advanced=AdvancedConfig(**config_data.get('advanced', {}))
        )
    
    def _create_default_config(self) -> VoiceSystemConfig:
        """Create default configuration"""
        return VoiceSystemConfig(
            voice=VoiceConfig(),
            tts=TTSConfig(),
            sensors=SensorConfig(),
            performance=PerformanceConfig(),
            privacy=PrivacyConfig(),
            advanced=AdvancedConfig()
        )
    
    def _save_config(self, config: VoiceSystemConfig) -> None:
        """Save configuration to file"""
        try:
            config_dict = {
                'voice_system': asdict(config.voice),
                'tts': asdict(config.tts),
                'm1_sensors': asdict(config.sensors),
                'performance': asdict(config.performance),
                'privacy': asdict(config.privacy),
                'advanced': asdict(config.advanced)
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
        except Exception as e:
    
    def get_config(self) -> VoiceSystemConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section"""
        if hasattr(self.config, section):
            section_config = getattr(self.config, section)
            for key, value in updates.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
            
            self._save_config(self.config)
        else:
    
    def reload_config(self) -> None:
        """Reload configuration from file"""
        self.config = self._load_config()
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Validate voice configuration
        if self.config.voice.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            issues.append("Invalid sample rate. Must be 8000, 16000, 22050, 44100, or 48000")
        
        if self.config.voice.chunk_size <= 0:
            issues.append("Chunk size must be positive")
        
        if not 0 <= self.config.voice.confidence_threshold <= 1:
            issues.append("Confidence threshold must be between 0 and 1")
        
        # Validate TTS configuration
        if not 0 <= self.config.tts.volume <= 1:
            issues.append("TTS volume must be between 0 and 1")
        
        if not 0.5 <= self.config.tts.pitch <= 2.0:
            issues.append("TTS pitch must be between 0.5 and 2.0")
        
        # Validate performance configuration
        if self.config.performance.max_cpu_usage <= 0 or self.config.performance.max_cpu_usage > 100:
            issues.append("Max CPU usage must be between 1 and 100")
        
        if self.config.performance.thread_pool_size <= 0:
            issues.append("Thread pool size must be positive")
        
        return issues
    
    def get_available_voices(self) -> List[str]:
        """Get list of available TTS voices"""
        # This would be dynamically detected from the system
        return [
            "Alex", "Victoria", "Daniel", "Samantha", "Tom", "Karen",
            "Fred", "Ralph", "Nick", "Bells", "Whisper", "Trinoids",
            "Agnes", "Albert", "Bad News", "Bahh", "Bells", "Boing",
            "Bubbles", "Cellos", "Deranged", "Good News", "Hysterical",
            "Pipe Organ", "Princess", "Ralph", "Trinoids", "Whisper",
            "Zarvox"
        ]
    
    def get_available_whisper_models(self) -> List[str]:
        """Get list of available Whisper models"""
        return [
            "whisper-1",  # Fastest, good for real-time
            "whisper-2",  # Balanced speed/accuracy
            "whisper-3",  # Most accurate, slower
            "whisper-4",  # Latest model
            "whisper-5"   # Experimental
        ]
    
    def export_config(self, export_path: str) -> None:
        """Export configuration to specified path"""
        try:
            config_dict = {
                'voice_system': asdict(self.config.voice),
                'tts': asdict(self.config.tts),
                'm1_sensors': asdict(self.config.sensors),
                'performance': asdict(self.config.performance),
                'privacy': asdict(self.config.privacy),
                'advanced': asdict(self.config.advanced)
            }
            
            with open(export_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
        except Exception as e:
    
    def import_config(self, import_path: str) -> None:
        """Import configuration from specified path"""
        try:
            with open(import_path, 'r') as f:
                config_data = json.load(f)
            
            self.config = self._parse_config(config_data)
            self._save_config(self.config)
        except Exception as e:
