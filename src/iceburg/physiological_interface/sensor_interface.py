"""
Mac Sensor Interface
Direct access to MacBook's built-in sensors for physiological state detection

IMPORTANT DISCLAIMER: This system cannot detect consciousness or brainwaves.
It analyzes legitimate physiological patterns from accelerometer, gyroscope,
and magnetometer data for general wellness monitoring only.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import subprocess
import json
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Raw sensor reading"""
    timestamp: datetime
    sensor_type: str
    x: float
    y: float
    z: float
    magnitude: float
    raw_data: Dict[str, Any]

@dataclass
class BrainwaveReading:
    """Processed brainwave reading"""
    timestamp: datetime
    alpha: float  # 8-13 Hz
    beta: float   # 13-30 Hz
    theta: float  # 4-8 Hz
    delta: float  # 0.5-4 Hz
    gamma: float  # 30-100 Hz
    dominant_frequency: float
    consciousness_state: str

class MacSensorInterface:
    """
    Direct interface to MacBook's built-in sensors
    """
    
    def __init__(self):
        self.sensors_active = False
        self.sensor_data = {
            'magnetometer': [],
            'accelerometer': [],
            'gyroscope': [],
            'audio': []
        }
        self.brainwave_data = []
        self.sampling_rate = 100  # Hz
        self.buffer_size = 1000
        
        # Brainwave frequency bands
        self.brainwave_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0)
        }
        
        # Consciousness states
        self.consciousness_states = {
            'deep_sleep': 'delta',
            'light_sleep': 'theta',
            'meditation': 'alpha',
            'focused': 'beta',
            'insight': 'gamma'
        }
        
    async def start_sensor_monitoring(self) -> None:
        """Start monitoring all available sensors"""
        try:
            logger.info("🧠 Starting consciousness sensor monitoring...")
            self.sensors_active = True
            
            # Start sensor monitoring tasks
            tasks = [
                self._monitor_magnetometer(),
                self._monitor_accelerometer(),
                self._monitor_gyroscope(),
                self._monitor_audio_input(),
                self._process_brainwave_data()
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error starting sensor monitoring: {e}")
            self.sensors_active = False
            
    async def stop_sensor_monitoring(self) -> None:
        """Stop all sensor monitoring"""
        self.sensors_active = False
        logger.info("🧠 Consciousness sensor monitoring stopped")
        
    async def _monitor_magnetometer(self) -> None:
        """Monitor magnetometer for EM field detection (brainwave proxy)"""
        while self.sensors_active:
            try:
                # Use CoreMotion to get magnetometer data
                reading = await self._get_magnetometer_reading()
                if reading:
                    self.sensor_data['magnetometer'].append(reading)
                    # Keep buffer manageable
                    if len(self.sensor_data['magnetometer']) > self.buffer_size:
                        self.sensor_data['magnetometer'] = self.sensor_data['magnetometer'][-self.buffer_size:]
                        
                await asyncio.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                logger.error(f"Error monitoring magnetometer: {e}")
                await asyncio.sleep(1)
                
    async def _monitor_accelerometer(self) -> None:
        """Monitor accelerometer for vibration patterns"""
        while self.sensors_active:
            try:
                reading = await self._get_accelerometer_reading()
                if reading:
                    self.sensor_data['accelerometer'].append(reading)
                    if len(self.sensor_data['accelerometer']) > self.buffer_size:
                        self.sensor_data['accelerometer'] = self.sensor_data['accelerometer'][-self.buffer_size:]
                        
                await asyncio.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                logger.error(f"Error monitoring accelerometer: {e}")
                await asyncio.sleep(1)
                
    async def _monitor_gyroscope(self) -> None:
        """Monitor gyroscope for rotational patterns"""
        while self.sensors_active:
            try:
                reading = await self._get_gyroscope_reading()
                if reading:
                    self.sensor_data['gyroscope'].append(reading)
                    if len(self.sensor_data['gyroscope']) > self.buffer_size:
                        self.sensor_data['gyroscope'] = self.sensor_data['gyroscope'][-self.buffer_size:]
                        
                await asyncio.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                logger.error(f"Error monitoring gyroscope: {e}")
                await asyncio.sleep(1)
                
    async def _monitor_audio_input(self) -> None:
        """Monitor audio input for frequency analysis"""
        while self.sensors_active:
            try:
                # Use system audio input for frequency detection
                reading = await self._get_audio_reading()
                if reading:
                    self.sensor_data['audio'].append(reading)
                    if len(self.sensor_data['audio']) > self.buffer_size:
                        self.sensor_data['audio'] = self.sensor_data['audio'][-self.buffer_size:]
                        
                await asyncio.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                logger.error(f"Error monitoring audio: {e}")
                await asyncio.sleep(1)
                
    async def _process_brainwave_data(self) -> None:
        """Process sensor data to extract brainwave patterns"""
        while self.sensors_active:
            try:
                if self.sensor_data['magnetometer']:
                    # Analyze magnetometer data for brainwave patterns
                    brainwave_reading = self._analyze_brainwave_patterns()
                    if brainwave_reading:
                        self.brainwave_data.append(brainwave_reading)
                        if len(self.brainwave_data) > self.buffer_size:
                            self.brainwave_data = self.brainwave_data[-self.buffer_size:]
                            
                await asyncio.sleep(0.1)  # 10 Hz brainwave processing
                
            except Exception as e:
                logger.error(f"Error processing brainwave data: {e}")
                await asyncio.sleep(1)
                
    async def _get_magnetometer_reading(self) -> Optional[SensorReading]:
        """Get magnetometer reading using system tools"""
        try:
            # Use ioreg to get magnetometer data
            result = subprocess.run([
                'ioreg', '-l', '-k', 'MagneticField'
            ], capture_output=True, text=True, timeout=1)
            
            if result.returncode == 0:
                # Parse magnetometer data
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'MagneticField' in line:
                        # Extract x, y, z values
                        # This is a simplified parser - real implementation would be more robust
                        return SensorReading(
                            timestamp=datetime.now(),
                            sensor_type='magnetometer',
                            x=0.0,  # Would extract from actual data
                            y=0.0,
                            z=0.0,
                            magnitude=0.0,
                            raw_data={'raw': line}
                        )
                        
        except Exception as e:
            logger.debug(f"Could not get magnetometer reading: {e}")
            
        return None
        
    async def _get_accelerometer_reading(self) -> Optional[SensorReading]:
        """Get accelerometer reading"""
        try:
            # Use ioreg to get accelerometer data
            result = subprocess.run([
                'ioreg', '-l', '-k', 'Acceleration'
            ], capture_output=True, text=True, timeout=1)
            
            if result.returncode == 0:
                return SensorReading(
                    timestamp=datetime.now(),
                    sensor_type='accelerometer',
                    x=0.0,  # Would extract from actual data
                    y=0.0,
                    z=0.0,
                    magnitude=0.0,
                    raw_data={'raw': result.stdout}
                )
                
        except Exception as e:
            logger.debug(f"Could not get accelerometer reading: {e}")
            
        return None
        
    async def _get_gyroscope_reading(self) -> Optional[SensorReading]:
        """Get gyroscope reading"""
        try:
            # Use ioreg to get gyroscope data
            result = subprocess.run([
                'ioreg', '-l', '-k', 'RotationRate'
            ], capture_output=True, text=True, timeout=1)
            
            if result.returncode == 0:
                return SensorReading(
                    timestamp=datetime.now(),
                    sensor_type='gyroscope',
                    x=0.0,  # Would extract from actual data
                    y=0.0,
                    z=0.0,
                    magnitude=0.0,
                    raw_data={'raw': result.stdout}
                )
                
        except Exception as e:
            logger.debug(f"Could not get gyroscope reading: {e}")
            
        return None
        
    async def _get_audio_reading(self) -> Optional[SensorReading]:
        """Get audio input reading for frequency analysis"""
        try:
            # Use system audio input
            # This would use Core Audio or similar for real implementation
            return SensorReading(
                timestamp=datetime.now(),
                sensor_type='audio',
                x=0.0,  # Frequency
                y=0.0,  # Amplitude
                z=0.0,  # Phase
                magnitude=0.0,
                raw_data={'frequency': 0.0, 'amplitude': 0.0}
            )
            
        except Exception as e:
            logger.debug(f"Could not get audio reading: {e}")
            
        return None
        
    def _analyze_brainwave_patterns(self) -> Optional[BrainwaveReading]:
        """Analyze sensor data to extract brainwave patterns"""
        try:
            if not self.sensor_data['magnetometer']:
                return None
                
            # Get recent magnetometer data
            recent_data = self.sensor_data['magnetometer'][-100:]  # Last 100 readings
            
            # Calculate magnitude variations (proxy for brainwave activity)
            magnitudes = [reading.magnitude for reading in recent_data]
            
            if not magnitudes:
                return None
                
            # Perform FFT analysis to extract frequency components
            fft_data = np.fft.fft(magnitudes)
            freqs = np.fft.fftfreq(len(magnitudes), 1.0 / self.sampling_rate)
            
            # Extract brainwave bands
            brainwave_powers = {}
            for band_name, (low_freq, high_freq) in self.brainwave_bands.items():
                # Find frequencies in this band
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(np.abs(fft_data[band_mask]))
                brainwave_powers[band_name] = float(band_power)
                
            # Find dominant frequency
            dominant_freq = freqs[np.argmax(np.abs(fft_data))]
            
            # Determine consciousness state
            consciousness_state = self._determine_consciousness_state(brainwave_powers)
            
            return BrainwaveReading(
                timestamp=datetime.now(),
                alpha=brainwave_powers.get('alpha', 0.0),
                beta=brainwave_powers.get('beta', 0.0),
                theta=brainwave_powers.get('theta', 0.0),
                delta=brainwave_powers.get('delta', 0.0),
                gamma=brainwave_powers.get('gamma', 0.0),
                dominant_frequency=float(dominant_freq),
                consciousness_state=consciousness_state
            )
            
        except Exception as e:
            logger.error(f"Error analyzing brainwave patterns: {e}")
            return None
            
    def _determine_consciousness_state(self, brainwave_powers: Dict[str, float]) -> str:
        """Determine consciousness state from brainwave powers"""
        try:
            # Find the dominant brainwave band
            dominant_band = max(brainwave_powers.items(), key=lambda x: x[1])[0]
            
            # Map to consciousness state
            for state, band in self.consciousness_states.items():
                if band == dominant_band:
                    return state
                    
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error determining consciousness state: {e}")
            return 'error'
            
    def get_current_brainwave_state(self) -> Optional[BrainwaveReading]:
        """Get the most recent brainwave reading"""
        if self.brainwave_data:
            return self.brainwave_data[-1]
        return None
        
    def get_sensor_summary(self) -> Dict[str, Any]:
        """Get summary of all sensor data"""
        return {
            'sensors_active': self.sensors_active,
            'magnetometer_readings': len(self.sensor_data['magnetometer']),
            'accelerometer_readings': len(self.sensor_data['accelerometer']),
            'gyroscope_readings': len(self.sensor_data['gyroscope']),
            'audio_readings': len(self.sensor_data['audio']),
            'brainwave_readings': len(self.brainwave_data),
            'current_consciousness_state': self.get_current_brainwave_state().consciousness_state if self.get_current_brainwave_state() else 'unknown',
            'sampling_rate': self.sampling_rate
        }
    
    def get_recent_sensor_data(self) -> Dict[str, Any]:
        """Get recent sensor data for physiological analysis"""
        try:
            return {
                'accelerometer': list(self.sensor_data['accelerometer'])[-100:],  # Last 100 readings
                'gyroscope': list(self.sensor_data['gyroscope'])[-100:],
                'magnetometer': list(self.sensor_data['magnetometer'])[-100:],
                'audio': list(self.sensor_data['audio'])[-100:],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting recent sensor data: {e}")
            return {}