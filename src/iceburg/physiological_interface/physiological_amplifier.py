"""
Physiological State Detector
Real-time physiological state analysis using MacBook sensors
Detects heart rate variability, breathing patterns, stress levels, and micro-movements
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json

from .sensor_interface import MacSensorInterface, BrainwaveReading
from .frequency_synthesizer import PhysiologicalFrequencySynthesizer
from .earth_connection import EarthConnectionInterface, EarthFrequencyReading

logger = logging.getLogger(__name__)

@dataclass
class PhysiologicalState:
    """Physiological state analysis using MacBook sensors"""
    timestamp: datetime
    heart_rate_variability: float  # Detected via accelerometer pulse analysis
    breathing_rate: float  # Detected via accelerometer chest movement
    stress_level: float  # Based on micro-movement patterns
    relaxation_state: str  # 'stressed', 'neutral', 'relaxed'
    micro_movement_intensity: float  # Overall movement activity
    sensor_quality: float  # Quality of sensor readings
    analysis_confidence: float  # Confidence in physiological analysis

@dataclass
class AmplificationProfile:
    """Consciousness amplification profile"""
    name: str
    target_brainwave: str
    earth_sync: bool
    icberg_sync: bool
    frequency_modulation: Dict[str, float]
    description: str

class PhysiologicalStateDetector:
    """
    Physiological state detection using MacBook sensors
    Analyzes heart rate variability, breathing patterns, stress levels, and micro-movements
    Note: This system cannot detect brainwaves - MacBook sensors lack the required sensitivity
    """
    
    def __init__(self):
        self.amplification_active = False
        self.consciousness_history = []
        
        # Initialize subsystems
        self.sensor_interface = MacSensorInterface()
        self.frequency_synthesizer = PhysiologicalFrequencySynthesizer()
        self.earth_connection = EarthConnectionInterface()
        
        # Amplification profiles
        self.amplification_profiles = {
            'deep_meditation': AmplificationProfile(
                name='deep_meditation',
                target_brainwave='theta',
                earth_sync=True,
                icberg_sync=True,
                frequency_modulation={
                    'primary': 4.0,  # Theta
                    'earth_sync': 7.83,  # Schumann
                    'icberg_sync': 432.0  # A4 tuning
                },
                description='Deep meditation with Earth and ICEBURG synchronization'
            ),
            'creative_flow': AmplificationProfile(
                name='creative_flow',
                target_brainwave='alpha',
                earth_sync=True,
                icberg_sync=True,
                frequency_modulation={
                    'primary': 10.0,  # Alpha
                    'earth_sync': 7.83,
                    'icberg_sync': 432.0
                },
                description='Creative flow state with enhanced inspiration'
            ),
            'insight_breakthrough': AmplificationProfile(
                name='insight_breakthrough',
                target_brainwave='gamma',
                earth_sync=True,
                icberg_sync=True,
                frequency_modulation={
                    'primary': 40.0,  # Gamma
                    'earth_sync': 7.83,
                    'icberg_sync': 432.0
                },
                description='Breakthrough insights and epiphanies'
            ),
            'earth_connection': AmplificationProfile(
                name='earth_connection',
                target_brainwave='alpha',
                earth_sync=True,
                icberg_sync=False,
                frequency_modulation={
                    'primary': 7.83,  # Schumann resonance
                    'harmonic_1': 14.3,
                    'harmonic_2': 20.8
                },
                description='Direct connection to Earth frequencies'
            ),
            'icberg_sync': AmplificationProfile(
                name='icberg_sync',
                target_brainwave='beta',
                earth_sync=False,
                icberg_sync=True,
                frequency_modulation={
                    'primary': 432.0,  # A4 tuning
                    'harmonic_1': 216.0,
                    'harmonic_2': 864.0
                },
                description='Synchronization with ICEBURG processing'
            )
        }
        
        # Current state
        self.current_profile = None
        self.current_consciousness_state = None
        
    async def start_consciousness_amplification(self) -> None:
        """Start the complete consciousness amplification system"""
        try:
            logger.info("🧠 Starting consciousness amplification system...")
            self.amplification_active = True
            
            # Start all subsystems
            await asyncio.gather(
                self.sensor_interface.start_sensor_monitoring(),
                self.frequency_synthesizer.start_audio_synthesis(),
                self.earth_connection.start_earth_connection()
            )
            
            # Start main amplification loop
            await self._consciousness_amplification_loop()
            
        except Exception as e:
            logger.error(f"Error starting consciousness amplification: {e}")
            self.amplification_active = False
            
    async def stop_consciousness_amplification(self) -> None:
        """Stop consciousness amplification system"""
        try:
            self.amplification_active = False
            
            # Stop all subsystems
            await asyncio.gather(
                self.sensor_interface.stop_sensor_monitoring(),
                self.frequency_synthesizer.stop_audio_synthesis(),
                self.earth_connection.stop_earth_connection()
            )
            
            logger.info("🧠 Consciousness amplification system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping consciousness amplification: {e}")
            
    async def _consciousness_amplification_loop(self) -> None:
        """Main consciousness amplification loop"""
        while self.amplification_active:
            try:
                # Get current consciousness state
                physiological_state = await self._assess_physiological_state()
                
                if physiological_state:
                    self.current_consciousness_state = physiological_state
                    self.consciousness_history.append(physiological_state)
                    
                    # Keep history manageable
                    if len(self.consciousness_history) > 1000:
                        self.consciousness_history = self.consciousness_history[-500:]
                        
                    # Log physiological state (no amplification for now)
                    logger.info(f"Physiological State: {physiological_state.relaxation_state}, "
                              f"Stress: {physiological_state.stress_level:.2f}, "
                              f"HRV: {physiological_state.heart_rate_variability:.2f}, "
                              f"Confidence: {physiological_state.analysis_confidence:.2f}")
                    
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in consciousness amplification loop: {e}")
                await asyncio.sleep(1)
                
    async def _assess_physiological_state(self) -> Optional[PhysiologicalState]:
        """Assess current physiological state using MacBook sensors"""
        try:
            # Get sensor data for physiological analysis
            sensor_data = self.sensor_interface.get_recent_sensor_data()
            
            # Analyze heart rate variability from accelerometer data
            hrv = self._analyze_heart_rate_variability(sensor_data)
            
            # Analyze breathing patterns from accelerometer data
            breathing_rate = self._analyze_breathing_patterns(sensor_data)
            
            # Analyze stress levels from micro-movements
            stress_level = self._analyze_stress_from_movements(sensor_data)
            
            # Determine relaxation state
            relaxation_state = self._determine_relaxation_state(stress_level, hrv)
            
            # Calculate micro-movement intensity
            movement_intensity = self._calculate_movement_intensity(sensor_data)
            
            # Assess sensor quality
            sensor_quality = self._assess_sensor_quality(sensor_data)
            
            # Calculate analysis confidence
            analysis_confidence = self._calculate_analysis_confidence(sensor_data)
            
            return PhysiologicalState(
                timestamp=datetime.now(),
                heart_rate_variability=hrv,
                breathing_rate=breathing_rate,
                stress_level=stress_level,
                relaxation_state=relaxation_state,
                micro_movement_intensity=movement_intensity,
                sensor_quality=sensor_quality,
                analysis_confidence=analysis_confidence
            )
            
        except Exception as e:
            logger.error(f"Error assessing consciousness state: {e}")
            return None
            
    def _calculate_consciousness_level(self, brainwave_state: str, 
                                     earth_quality: float, icberg_sync: float) -> float:
        """Calculate overall consciousness level"""
        try:
            # Map brainwave states to consciousness levels
            brainwave_levels = {
                'deep_sleep': 0.1,
                'light_sleep': 0.2,
                'meditation': 0.8,
                'focused': 0.7,
                'insight': 0.9,
                'unknown': 0.5,
                'error': 0.0
            }
            
            brainwave_level = brainwave_levels.get(brainwave_state, 0.5)
            
            # Combine all factors
            overall_level = (
                brainwave_level * 0.4 +
                earth_quality * 0.3 +
                icberg_sync * 0.3
            )
            
            return min(1.0, max(0.0, overall_level))
            
        except Exception as e:
            logger.error(f"Error calculating consciousness level: {e}")
            return 0.5
    
    # New legitimate physiological analysis methods
    
    def _analyze_heart_rate_variability(self, sensor_data: Dict[str, Any]) -> float:
        """Analyze heart rate variability from accelerometer data"""
        try:
            if not sensor_data or 'accelerometer' not in sensor_data:
                return 0.0
                
            # Extract accelerometer data
            accel_data = sensor_data['accelerometer']
            if len(accel_data) < 100:  # Need sufficient data
                return 0.0
                
            # Calculate magnitude variations (proxy for pulse detection)
            magnitudes = []
            for reading in accel_data[-100:]:  # Last 100 readings
                if isinstance(reading, dict) and 'x' in reading:
                    mag = np.sqrt(reading['x']**2 + reading['y']**2 + reading['z']**2)
                    magnitudes.append(mag)
            
            if len(magnitudes) < 50:
                return 0.0
                
            # Calculate heart rate variability (simplified)
            # In reality, this would require more sophisticated pulse detection
            magnitude_std = np.std(magnitudes)
            hrv_score = min(magnitude_std * 10, 1.0)  # Normalize to 0-1
            
            return float(hrv_score)
            
        except Exception as e:
            logger.error(f"Error analyzing heart rate variability: {e}")
            return 0.0
    
    def _analyze_breathing_patterns(self, sensor_data: Dict[str, Any]) -> float:
        """Analyze breathing patterns from accelerometer data"""
        try:
            if not sensor_data or 'accelerometer' not in sensor_data:
                return 0.0
                
            accel_data = sensor_data['accelerometer']
            if len(accel_data) < 50:
                return 0.0
                
            # Extract Z-axis data (vertical movement for breathing)
            z_values = []
            for reading in accel_data[-50:]:
                if isinstance(reading, dict) and 'z' in reading:
                    z_values.append(reading['z'])
            
            if len(z_values) < 20:
                return 0.0
                
            # Calculate breathing rate from Z-axis variations
            z_std = np.std(z_values)
            breathing_rate = min(z_std * 5, 1.0)  # Normalize to 0-1
            
            return float(breathing_rate)
            
        except Exception as e:
            logger.error(f"Error analyzing breathing patterns: {e}")
            return 0.0
    
    def _analyze_stress_from_movements(self, sensor_data: Dict[str, Any]) -> float:
        """Analyze stress levels from micro-movement patterns"""
        try:
            if not sensor_data or 'accelerometer' not in sensor_data:
                return 0.0
                
            accel_data = sensor_data['accelerometer']
            if len(accel_data) < 30:
                return 0.0
                
            # Calculate movement variability (higher = more stressed)
            movements = []
            for i in range(1, len(accel_data[-30:])):
                prev = accel_data[-30:][i-1]
                curr = accel_data[-30:][i]
                if isinstance(prev, dict) and isinstance(curr, dict):
                    movement = np.sqrt(
                        (curr['x'] - prev['x'])**2 + 
                        (curr['y'] - prev['y'])**2 + 
                        (curr['z'] - prev['z'])**2
                    )
                    movements.append(movement)
            
            if not movements:
                return 0.0
                
            # Higher movement variability indicates stress
            movement_variability = np.std(movements)
            stress_level = min(movement_variability * 20, 1.0)  # Normalize to 0-1
            
            return float(stress_level)
            
        except Exception as e:
            logger.error(f"Error analyzing stress from movements: {e}")
            return 0.0
    
    def _determine_relaxation_state(self, stress_level: float, hrv: float) -> str:
        """Determine relaxation state based on stress and HRV"""
        try:
            # Combine stress level and HRV for relaxation assessment
            relaxation_score = (1.0 - stress_level) * 0.7 + hrv * 0.3
            
            if relaxation_score > 0.7:
                return 'relaxed'
            elif relaxation_score > 0.4:
                return 'neutral'
            else:
                return 'stressed'
                
        except Exception as e:
            logger.error(f"Error determining relaxation state: {e}")
            return 'unknown'
    
    def _calculate_movement_intensity(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate overall micro-movement intensity"""
        try:
            if not sensor_data or 'accelerometer' not in sensor_data:
                return 0.0
                
            accel_data = sensor_data['accelerometer']
            if len(accel_data) < 10:
                return 0.0
                
            # Calculate average movement intensity
            total_movement = 0.0
            count = 0
            
            for reading in accel_data[-10:]:
                if isinstance(reading, dict):
                    movement = np.sqrt(reading['x']**2 + reading['y']**2 + reading['z']**2)
                    total_movement += movement
                    count += 1
            
            if count == 0:
                return 0.0
                
            avg_movement = total_movement / count
            intensity = min(avg_movement / 10.0, 1.0)  # Normalize to 0-1
            
            return float(intensity)
            
        except Exception as e:
            logger.error(f"Error calculating movement intensity: {e}")
            return 0.0
    
    def _assess_sensor_quality(self, sensor_data: Dict[str, Any]) -> float:
        """Assess quality of sensor readings"""
        try:
            if not sensor_data:
                return 0.0
                
            quality_score = 0.0
            sensor_count = 0
            
            # Check accelerometer quality
            if 'accelerometer' in sensor_data and sensor_data['accelerometer']:
                accel_data = sensor_data['accelerometer']
                if len(accel_data) > 10:
                    quality_score += 0.4
                sensor_count += 1
            
            # Check gyroscope quality
            if 'gyroscope' in sensor_data and sensor_data['gyroscope']:
                gyro_data = sensor_data['gyroscope']
                if len(gyro_data) > 10:
                    quality_score += 0.3
                sensor_count += 1
            
            # Check magnetometer quality
            if 'magnetometer' in sensor_data and sensor_data['magnetometer']:
                mag_data = sensor_data['magnetometer']
                if len(mag_data) > 10:
                    quality_score += 0.3
                sensor_count += 1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing sensor quality: {e}")
            return 0.0
    
    def _calculate_analysis_confidence(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate confidence in physiological analysis"""
        try:
            # Base confidence on sensor quality and data availability
            sensor_quality = self._assess_sensor_quality(sensor_data)
            
            # Additional confidence factors
            data_availability = 0.0
            if sensor_data:
                available_sensors = sum(1 for sensor in ['accelerometer', 'gyroscope', 'magnetometer'] 
                                      if sensor in sensor_data and sensor_data[sensor])
                data_availability = available_sensors / 3.0
            
            # Combine factors
            confidence = (sensor_quality * 0.7 + data_availability * 0.3)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {e}")
            return 0.0
            
    def _determine_recommended_frequency(self, brainwave_state: str, 
                                       earth_quality: float, icberg_sync: float) -> float:
        """Determine recommended frequency for amplification"""
        try:
            # Base frequencies for different states
            base_frequencies = {
                'deep_sleep': 2.0,  # Delta
                'light_sleep': 4.0,  # Theta
                'meditation': 7.83,  # Schumann
                'focused': 40.0,  # Gamma
                'insight': 100.0,  # High gamma
                'unknown': 10.0,  # Alpha
                'error': 7.83  # Schumann
            }
            
            base_freq = base_frequencies.get(brainwave_state, 10.0)
            
            # Adjust based on Earth connection
            if earth_quality > 0.7:
                base_freq = 7.83  # Prioritize Schumann resonance
                
            # Adjust based on ICEBURG sync
            if icberg_sync > 0.7:
                base_freq = 432.0  # Prioritize A4 tuning
                
            return base_freq
            
        except Exception as e:
            logger.error(f"Error determining recommended frequency: {e}")
            return 7.83
            
    async def _adjust_amplification(self, physiological_state: PhysiologicalState) -> None:
        """Adjust amplification based on current consciousness state"""
        try:
            # Determine if we should change the current profile
            if not self.current_profile or self._should_change_profile(physiological_state):
                new_profile = self._select_optimal_profile(physiological_state)
                if new_profile:
                    await self._activate_profile(new_profile)
                    
        except Exception as e:
            logger.error(f"Error adjusting amplification: {e}")
            
    def _should_change_profile(self, physiological_state: PhysiologicalState) -> bool:
        """Determine if we should change the current amplification profile"""
        try:
            if not self.current_profile:
                return True
                
            # Change profile if consciousness level changes significantly
            if len(self.consciousness_history) > 1:
                previous_state = self.consciousness_history[-2]
                level_change = abs(
                    consciousness_state.overall_consciousness_level - 
                    previous_state.overall_consciousness_level
                )
                
                if level_change > 0.3:  # Significant change
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error determining profile change: {e}")
            return False
            
    def _select_optimal_profile(self, physiological_state: PhysiologicalState) -> Optional[AmplificationProfile]:
        """Select optimal amplification profile based on consciousness state"""
        try:
            # Use physiological state for profile selection
            stress_level = physiological_state.stress_level
            relaxation_state = physiological_state.relaxation_state
            hrv = physiological_state.heart_rate_variability
            
            # Select profile based on physiological state
            if relaxation_state == 'relaxed' and stress_level < 0.3:
                return self.amplification_profiles.get('deep_meditation', None)
            elif relaxation_state == 'neutral' and stress_level < 0.6:
                return self.amplification_profiles.get('focused_work', None)
            elif relaxation_state == 'stressed' and stress_level > 0.7:
                return self.amplification_profiles.get('stress_relief', None)
            else:
                return self.amplification_profiles.get('creative_flow', None)
                
        except Exception as e:
            logger.error(f"Error selecting optimal profile: {e}")
            return None
            
    async def _activate_profile(self, profile: AmplificationProfile) -> None:
        """Activate an amplification profile"""
        try:
            self.current_profile = profile
            
            # Set frequency synthesizer state
            self.frequency_synthesizer.set_consciousness_state(profile.name)
            
            logger.info(f"🧠 Activated consciousness profile: {profile.name} - {profile.description}")
            
        except Exception as e:
            logger.error(f"Error activating profile: {e}")
            
    def set_amplification_profile(self, profile_name: str) -> bool:
        """Manually set amplification profile"""
        try:
            if profile_name not in self.amplification_profiles:
                logger.error(f"Unknown amplification profile: {profile_name}")
                return False
                
            profile = self.amplification_profiles[profile_name]
            asyncio.create_task(self._activate_profile(profile))
            return True
            
        except Exception as e:
            logger.error(f"Error setting amplification profile: {e}")
            return False
            
    def get_current_consciousness_state(self) -> Optional[PhysiologicalState]:
        """Get current consciousness state"""
        return self.current_consciousness_state
        
    def get_amplification_status(self) -> Dict[str, Any]:
        """Get amplification system status"""
        return {
            'amplification_active': self.amplification_active,
            'current_profile': self.current_profile.name if self.current_profile else None,
            'current_consciousness_state': self.current_consciousness_state.brainwave_state if self.current_consciousness_state else None,
            'consciousness_level': self.current_consciousness_state.overall_consciousness_level if self.current_consciousness_state else 0.0,
            'earth_connection_quality': self.current_consciousness_state.earth_connection_quality if self.current_consciousness_state else 0.0,
            'icberg_sync_level': self.current_consciousness_state.icberg_sync_level if self.current_consciousness_state else 0.0,
            'recommended_frequency': self.current_consciousness_state.recommended_frequency if self.current_consciousness_state else 7.83,
            'available_profiles': list(self.amplification_profiles.keys()),
            'consciousness_history_count': len(self.consciousness_history)
        }
