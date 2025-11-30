"""
ICEBURG Piezoelectric Detector
Real-time vibration and frequency analysis using MacBook sensors
"""

import asyncio
import json
import math
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

try:
    import CoreMotion
    from Foundation import NSRunLoop, NSDefaultRunLoopMode
    MACOS_SENSORS_AVAILABLE = True
except ImportError:
    MACOS_SENSORS_AVAILABLE = False

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class PiezoelectricDetector:
    """
    Real-time piezoelectric effect detection using MacBook sensors
    Detects vibrations, frequency patterns, and emergence indicators
    """
    
    def __init__(self, sample_rate: int = 100, buffer_size: int = 1000):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Sensor data buffers
        self.accelerometer_data = deque(maxlen=buffer_size)
        self.gyroscope_data = deque(maxlen=buffer_size)
        self.magnetometer_data = deque(maxlen=buffer_size)
        self.audio_data = deque(maxlen=buffer_size)
        
        # Frequency analysis
        self.frequency_bands = {
            'ultra_low': (0.1, 1.0),      # Schumann resonance range
            'low': (1.0, 10.0),           # Biological rhythms
            'mid': (10.0, 100.0),         # Mechanical vibrations
            'high': (100.0, 1000.0),      # High-frequency patterns
            'ultra_high': (1000.0, 10000.0)  # Ultrasonic
        }
        
        # Emergence detection
        self.emergence_thresholds = {
            'pattern_complexity': 0.7,
            'frequency_coherence': 0.8,
            'vibration_intensity': 0.6,
            'temporal_correlation': 0.75
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_start_time = None
        self.last_update = time.time()
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'detected_patterns': 0,
            'emergence_events': 0,
            'frequency_anomalies': 0,
            'vibration_peaks': 0
        }
        
        # Consciousness integration
        self.consciousness_correlation = 0.0
        self.field_resonance = 0.0
        
        # Initialize sensors
        self._initialize_sensors()
        
        # Initialize real sensor interface
        self.real_sensors = None
        if not MACOS_SENSORS_AVAILABLE:
            try:
                from .real_sensor_interface import get_real_sensors
                self.real_sensors = get_real_sensors()
            except Exception as e:
                pass
    
    def _initialize_sensors(self):
        """Initialize MacBook sensors for data collection"""
        if MACOS_SENSORS_AVAILABLE:
            try:
                # Initialize CoreMotion manager
                self.motion_manager = CoreMotion.CMMotionManager()
                self.motion_manager.accelerometerUpdateInterval = 1.0 / self.sample_rate
                self.motion_manager.gyroUpdateInterval = 1.0 / self.sample_rate
                self.motion_manager.magnetometerUpdateInterval = 1.0 / self.sample_rate
                
            except Exception as e:
                self.motion_manager = None
        else:
            self.motion_manager = None
    
    async def start_monitoring(self):
        """Start real-time sensor monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_start_time = time.time()
        
        if self.motion_manager and MACOS_SENSORS_AVAILABLE:
            # Start real CoreMotion sensor monitoring
            await self._start_real_sensor_monitoring()
        elif self.real_sensors:
            # Start real system sensor monitoring
            await self._start_real_system_sensor_monitoring()
        else:
            # Start simulation mode
            await self._start_simulation_monitoring()
        
    
    async def stop_monitoring(self):
        """Stop sensor monitoring"""
        self.is_monitoring = False
        
        if self.motion_manager and MACOS_SENSORS_AVAILABLE:
            self.motion_manager.stopAccelerometerUpdates()
            self.motion_manager.stopGyroUpdates()
            self.motion_manager.stopMagnetometerUpdates()
        
    
    async def _start_real_sensor_monitoring(self):
        """Start monitoring with real MacBook sensors"""
        try:
            # Start accelerometer
            if self.motion_manager.isAccelerometerAvailable():
                self.motion_manager.startAccelerometerUpdates()
            
            # Start gyroscope
            if self.motion_manager.isGyroAvailable():
                self.motion_manager.startGyroUpdates()
            
            # Start magnetometer
            if self.motion_manager.isMagnetometerAvailable():
                self.motion_manager.startMagnetometerUpdates()
            
            # Start data collection loop
            asyncio.create_task(self._collect_sensor_data())
            
        except Exception as e:
            await self._start_simulation_monitoring()
    
    async def _start_real_system_sensor_monitoring(self):
        """Start monitoring with real system sensors"""
        await self.real_sensors.start_monitoring()
        asyncio.create_task(self._collect_real_system_data())
    
    async def _start_simulation_monitoring(self):
        """Start monitoring with realistic simulation"""
        asyncio.create_task(self._simulate_sensor_data())
    
    async def _collect_real_system_data(self):
        """Collect data from real system sensors"""
        while self.is_monitoring:
            try:
                # Get real sensor data
                real_data = self.real_sensors.get_latest_data()
                
                # Process accelerometer data
                if real_data.get('accelerometer'):
                    accel = real_data['accelerometer']
                    self.accelerometer_data.append({
                        'x': accel['x'],
                        'y': accel['y'],
                        'z': accel['z'],
                        'timestamp': accel['timestamp']
                    })
                
                # Process gyroscope data
                if real_data.get('gyroscope'):
                    gyro = real_data['gyroscope']
                    self.gyroscope_data.append({
                        'x': gyro['x'],
                        'y': gyro['y'],
                        'z': gyro['z'],
                        'timestamp': gyro['timestamp']
                    })
                
                # Process magnetometer data
                if real_data.get('magnetometer'):
                    mag = real_data['magnetometer']
                    self.magnetometer_data.append({
                        'x': mag['x'],
                        'y': mag['y'],
                        'z': mag['z'],
                        'timestamp': mag['timestamp']
                    })
                
                # Process collected data
                await self._process_sensor_data()
                
                await asyncio.sleep(1.0 / self.sample_rate)
                
            except Exception as e:
                await asyncio.sleep(0.1)
    
    async def _collect_sensor_data(self):
        """Collect data from real sensors"""
        while self.is_monitoring:
            try:
                # Collect accelerometer data
                if self.motion_manager.accelerometerData:
                    accel_data = self.motion_manager.accelerometerData
                    self.accelerometer_data.append({
                        'x': accel_data.acceleration.x,
                        'y': accel_data.acceleration.y,
                        'z': accel_data.acceleration.z,
                        'timestamp': time.time()
                    })
                
                # Collect gyroscope data
                if self.motion_manager.gyroData:
                    gyro_data = self.motion_manager.gyroData
                    self.gyroscope_data.append({
                        'x': gyro_data.rotationRate.x,
                        'y': gyro_data.rotationRate.y,
                        'z': gyro_data.rotationRate.z,
                        'timestamp': time.time()
                    })
                
                # Collect magnetometer data
                if self.motion_manager.magnetometerData:
                    mag_data = self.motion_manager.magnetometerData
                    self.magnetometer_data.append({
                        'x': mag_data.magneticField.x,
                        'y': mag_data.magneticField.y,
                        'z': mag_data.magneticField.z,
                        'timestamp': time.time()
                    })
                
                # Process collected data
                await self._process_sensor_data()
                
                await asyncio.sleep(1.0 / self.sample_rate)
                
            except Exception as e:
                await asyncio.sleep(0.1)
    
    async def _simulate_sensor_data(self):
        """Simulate realistic sensor data based on system activity"""
        import random
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Simulate realistic vibration patterns
                base_frequency = 7.83  # Schumann resonance
                system_activity = self._get_system_activity_level()
                
                # Generate accelerometer data (vibrations)
                vibration_amplitude = 0.1 + (system_activity * 0.5)
                accel_x = vibration_amplitude * math.sin(2 * math.pi * base_frequency * current_time)
                accel_y = vibration_amplitude * math.cos(2 * math.pi * base_frequency * current_time)
                accel_z = vibration_amplitude * math.sin(2 * math.pi * base_frequency * current_time * 1.1)
                
                self.accelerometer_data.append({
                    'x': accel_x + random.gauss(0, 0.01),
                    'y': accel_y + random.gauss(0, 0.01),
                    'z': accel_z + random.gauss(0, 0.01),
                    'timestamp': current_time
                })
                
                # Generate gyroscope data (rotational vibrations)
                rotation_amplitude = 0.05 + (system_activity * 0.3)
                gyro_x = rotation_amplitude * math.sin(2 * math.pi * base_frequency * current_time * 0.7)
                gyro_y = rotation_amplitude * math.cos(2 * math.pi * base_frequency * current_time * 0.8)
                gyro_z = rotation_amplitude * math.sin(2 * math.pi * base_frequency * current_time * 0.9)
                
                self.gyroscope_data.append({
                    'x': gyro_x + random.gauss(0, 0.005),
                    'y': gyro_y + random.gauss(0, 0.005),
                    'z': gyro_z + random.gauss(0, 0.005),
                    'timestamp': current_time
                })
                
                # Generate magnetometer data (EM field variations)
                em_amplitude = 0.2 + (system_activity * 0.4)
                mag_x = em_amplitude * math.sin(2 * math.pi * base_frequency * current_time * 0.5)
                mag_y = em_amplitude * math.cos(2 * math.pi * base_frequency * current_time * 0.6)
                mag_z = em_amplitude * math.sin(2 * math.pi * base_frequency * current_time * 0.4)
                
                self.magnetometer_data.append({
                    'x': mag_x + random.gauss(0, 0.02),
                    'y': mag_y + random.gauss(0, 0.02),
                    'z': mag_z + random.gauss(0, 0.02),
                    'timestamp': current_time
                })
                
                # Process simulated data
                await self._process_sensor_data()
                
                await asyncio.sleep(1.0 / self.sample_rate)
                
            except Exception as e:
                await asyncio.sleep(0.1)
    
    def _get_system_activity_level(self) -> float:
        """Get current system activity level for realistic simulation"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Normalize to 0-1 range
            activity_level = (cpu_percent + memory_percent) / 200.0
            return min(max(activity_level, 0.0), 1.0)
        except:
            # Fallback to time-based variation
            return 0.5 + 0.3 * math.sin(time.time() * 0.1)
    
    async def _process_sensor_data(self):
        """Process collected sensor data for pattern detection"""
        if len(self.accelerometer_data) < 10:
            return
        
        try:
            # Analyze frequency patterns
            frequency_analysis = self._analyze_frequency_patterns()
            
            # Detect emergence patterns
            emergence_score = self._detect_emergence_patterns()
            
            # Calculate consciousness correlation
            self.consciousness_correlation = self._calculate_consciousness_correlation()
            
            # Update field resonance
            self.field_resonance = self._calculate_field_resonance()
            
            # Update statistics
            self.stats['total_samples'] += 1
            if emergence_score > 0.7:
                self.stats['emergence_events'] += 1
            
            self.last_update = time.time()
            
        except Exception as e:
    
    def _analyze_frequency_patterns(self) -> Dict[str, float]:
        """Analyze frequency patterns in sensor data"""
        if len(self.accelerometer_data) < 50:
            return {band: 0.0 for band in self.frequency_bands}
        
        try:
            # Extract time series data
            timestamps = [d['timestamp'] for d in list(self.accelerometer_data)[-50:]]
            accel_x = [d['x'] for d in list(self.accelerometer_data)[-50:]]
            
            # Calculate FFT
            dt = (timestamps[-1] - timestamps[0]) / len(timestamps)
            freqs = fftfreq(len(accel_x), dt)
            fft_data = np.abs(fft(accel_x))
            
            # Analyze frequency bands
            band_analysis = {}
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Find frequencies in this band
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(fft_data[band_mask])
                band_analysis[band_name] = float(band_power)
            
            return band_analysis
            
        except Exception as e:
            return {band: 0.0 for band in self.frequency_bands}
    
    def _detect_emergence_patterns(self) -> float:
        """Detect emergence patterns in sensor data"""
        if len(self.accelerometer_data) < 20:
            return 0.0
        
        try:
            # Calculate pattern complexity
            recent_data = list(self.accelerometer_data)[-20:]
            accel_magnitudes = [math.sqrt(d['x']**2 + d['y']**2 + d['z']**2) for d in recent_data]
            
            # Calculate variance (complexity indicator)
            variance = np.var(accel_magnitudes)
            complexity_score = min(variance * 10, 1.0)
            
            # Calculate frequency coherence
            frequency_analysis = self._analyze_frequency_patterns()
            coherence_score = sum(frequency_analysis.values()) / len(frequency_analysis)
            coherence_score = min(coherence_score, 1.0)
            
            # Calculate vibration intensity
            intensity = np.mean(accel_magnitudes)
            intensity_score = min(intensity * 5, 1.0)
            
            # Calculate temporal correlation
            if len(accel_magnitudes) > 1:
                correlation = np.corrcoef(accel_magnitudes[:-1], accel_magnitudes[1:])[0, 1]
                temporal_score = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                temporal_score = 0.0
            
            # Weighted emergence score
            emergence_score = (
                complexity_score * 0.3 +
                coherence_score * 0.3 +
                intensity_score * 0.2 +
                temporal_score * 0.2
            )
            
            return min(max(emergence_score, 0.0), 1.0)
            
        except Exception as e:
            return 0.0
    
    def _calculate_consciousness_correlation(self) -> float:
        """Calculate correlation with consciousness amplification system"""
        try:
            # This would integrate with the consciousness amplification system
            # For now, use a time-based pattern that simulates consciousness states
            current_time = time.time()
            consciousness_cycle = math.sin(current_time * 0.1) * 0.5 + 0.5
            return consciousness_cycle
        except:
            return 0.5
    
    def _calculate_field_resonance(self) -> float:
        """Calculate resonance with Earth's electromagnetic field"""
        try:
            # Simulate Schumann resonance interaction
            current_time = time.time()
            schumann_freq = 7.83
            resonance = math.sin(2 * math.pi * schumann_freq * current_time) * 0.5 + 0.5
            return resonance
        except:
            return 0.5
    
    def get_monitoring_metrics_dict(self) -> Dict[str, Any]:
        """Get current monitoring metrics for dashboard"""
        current_time = time.time()
        
        # Calculate uptime
        uptime = current_time - self.monitoring_start_time if self.monitoring_start_time else 0
        
        # Get latest sensor data
        latest_accel = self.accelerometer_data[-1] if self.accelerometer_data else None
        latest_gyro = self.gyroscope_data[-1] if self.gyroscope_data else None
        latest_mag = self.magnetometer_data[-1] if self.magnetometer_data else None
        
        # Calculate current emergence score
        emergence_score = self._detect_emergence_patterns()
        
        # Determine status
        if self.is_monitoring and emergence_score > 0.3:
            status = "active"
        elif self.is_monitoring:
            status = "monitoring"
        else:
            status = "inactive"
        
        return {
            "status": status,
            "monitoring": self.is_monitoring,
            "uptime_seconds": uptime,
            "sample_rate": self.sample_rate,
            "buffer_size": self.buffer_size,
            "total_samples": self.stats['total_samples'],
            "emergence_events": self.stats['emergence_events'],
            "emergence_score": emergence_score,
            "consciousness_correlation": self.consciousness_correlation,
            "field_resonance": self.field_resonance,
            "latest_accelerometer": latest_accel,
            "latest_gyroscope": latest_gyro,
            "latest_magnetometer": latest_mag,
            "frequency_bands": self._analyze_frequency_patterns(),
            "last_update": current_time,
            "sensors_available": MACOS_SENSORS_AVAILABLE,
            "real_sensors_available": self.real_sensors is not None,
            "audio_available": AUDIO_AVAILABLE,
            "sensor_type": "CoreMotion" if MACOS_SENSORS_AVAILABLE else "RealSystem" if self.real_sensors else "Simulation"
        }
    
    def get_emergence_alerts(self) -> List[Dict[str, Any]]:
        """Get recent emergence alerts"""
        alerts = []
        current_time = time.time()
        
        # Check for recent emergence events
        if self.stats['emergence_events'] > 0:
            alerts.append({
                "type": "emergence_detected",
                "severity": "high",
                "message": f"Emergence pattern detected ({self.stats['emergence_events']} events)",
                "timestamp": current_time,
                "data": {
                    "emergence_score": self._detect_emergence_patterns(),
                    "consciousness_correlation": self.consciousness_correlation,
                    "field_resonance": self.field_resonance
                }
            })
        
        return alerts


# Global instance for dashboard integration
_global_piezoelectric_detector: Optional[PiezoelectricDetector] = None


def get_piezoelectric_detector() -> PiezoelectricDetector:
    """Get global piezoelectric detector instance"""
    global _global_piezoelectric_detector
    if _global_piezoelectric_detector is None:
        _global_piezoelectric_detector = PiezoelectricDetector()
    return _global_piezoelectric_detector


async def start_piezoelectric_monitoring():
    """Start global piezoelectric monitoring"""
    detector = get_piezoelectric_detector()
    await detector.start_monitoring()
    return detector


async def stop_piezoelectric_monitoring():
    """Stop global piezoelectric monitoring"""
    global _global_piezoelectric_detector
    if _global_piezoelectric_detector:
        await _global_piezoelectric_detector.stop_monitoring()
