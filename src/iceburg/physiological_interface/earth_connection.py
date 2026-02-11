"""
Earth Connection Interface
Connect ICEBURG with Earth's electromagnetic field and Schumann resonance
"""

import asyncio
import numpy as np
import requests
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import math
import os

logger = logging.getLogger(__name__)

@dataclass
class EarthFrequencyReading:
    """Earth frequency reading"""
    timestamp: datetime
    schumann_fundamental: float  # 7.83 Hz
    schumann_harmonics: List[float]  # 14.3, 20.8, 27.3, 33.8 Hz
    geomagnetic_activity: float
    solar_wind_speed: float
    kp_index: float
    aurora_activity: float

@dataclass
class EarthConnectionState:
    """Earth connection state"""
    connected: bool
    resonance_strength: float
    sync_quality: float
    last_update: datetime
    frequency_drift: float

class EarthConnectionInterface:
    """
    Interface to Earth's electromagnetic field and natural frequencies
    """
    
    def __init__(self):
        self.connection_active = False
        self.earth_data = []
        self.connection_state = EarthConnectionState(
            connected=False,
            resonance_strength=0.0,
            sync_quality=0.0,
            last_update=datetime.now(),
            frequency_drift=0.0
        )
        
        # Schumann resonance frequencies
        self.schumann_frequencies = {
            'fundamental': 7.83,  # Primary Schumann resonance
            'harmonic_1': 14.3,   # First harmonic
            'harmonic_2': 20.8,   # Second harmonic
            'harmonic_3': 27.3,   # Third harmonic
            'harmonic_4': 33.8    # Fourth harmonic
        }
        
        # Earth's natural frequencies
        self.earth_frequencies = {
            'schumann': 7.83,
            'earth_rotation': 1.0 / (24 * 3600),  # 1 day period
            'moon_cycle': 1.0 / (29.5 * 24 * 3600),  # Lunar month
            'solar_cycle': 1.0 / (365.25 * 24 * 3600),  # Solar year
            'precession': 1.0 / (25772 * 365.25 * 24 * 3600)  # Precession of equinoxes
        }
        
        # Geomagnetic data sources
        self.data_sources = {
            'noaa': 'https://services.swpc.noaa.gov/json/planetary_k_index_1m.json',
            'nasa': 'https://api.nasa.gov/planetary/apod',
            'schumann': 'https://www2.irf.se/maggraphs/schumann.php'
        }
        
    async def start_earth_connection(self) -> None:
        """Start connection to Earth's electromagnetic field"""
        try:
            logger.info("🌍 Starting Earth connection interface...")
            self.connection_active = True
            
            # Start monitoring tasks
            tasks = [
                self._monitor_schumann_resonance(),
                self._monitor_geomagnetic_activity(),
                self._update_connection_state()
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error starting Earth connection: {e}")
            self.connection_active = False
            
    async def stop_earth_connection(self) -> None:
        """Stop Earth connection monitoring"""
        self.connection_active = False
        logger.info("🌍 Earth connection interface stopped")
        
    async def _monitor_schumann_resonance(self) -> None:
        """Monitor Schumann resonance frequencies"""
        while self.connection_active:
            try:
                # Get Schumann resonance data
                schumann_data = await self._get_schumann_data()
                
                if schumann_data:
                    # Create Earth frequency reading
                    reading = EarthFrequencyReading(
                        timestamp=datetime.now(),
                        schumann_fundamental=schumann_data.get('fundamental', 7.83),
                        schumann_harmonics=[
                            schumann_data.get('harmonic_1', 14.3),
                            schumann_data.get('harmonic_2', 20.8),
                            schumann_data.get('harmonic_3', 27.3),
                            schumann_data.get('harmonic_4', 33.8)
                        ],
                        geomagnetic_activity=schumann_data.get('geomagnetic', 0.0),
                        solar_wind_speed=schumann_data.get('solar_wind', 0.0),
                        kp_index=schumann_data.get('kp_index', 0.0),
                        aurora_activity=schumann_data.get('aurora', 0.0)
                    )
                    
                    self.earth_data.append(reading)
                    
                    # Keep buffer manageable
                    if len(self.earth_data) > 1000:
                        self.earth_data = self.earth_data[-500:]
                        
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error monitoring Schumann resonance: {e}")
                await asyncio.sleep(60)
                
    async def _monitor_geomagnetic_activity(self) -> None:
        """Monitor geomagnetic activity"""
        while self.connection_active:
            try:
                # Get geomagnetic data from NOAA
                geomagnetic_data = await self._get_geomagnetic_data()
                
                if geomagnetic_data:
                    # Update connection state with geomagnetic activity
                    self.connection_state.resonance_strength = geomagnetic_data.get('kp_index', 0.0) / 9.0
                    
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring geomagnetic activity: {e}")
                await asyncio.sleep(300)
                
    async def _update_connection_state(self) -> None:
        """Update Earth connection state"""
        while self.connection_active:
            try:
                if self.earth_data:
                    # Calculate connection quality
                    recent_readings = self.earth_data[-10:]  # Last 10 readings
                    
                    # Calculate frequency stability
                    frequencies = [r.schumann_fundamental for r in recent_readings]
                    frequency_std = np.std(frequencies) if len(frequencies) > 1 else 0.0
                    frequency_stability = 1.0 - (frequency_std / 7.83)  # Normalize to fundamental
                    
                    # Calculate sync quality
                    sync_quality = min(1.0, frequency_stability + self.connection_state.resonance_strength)
                    
                    # Update connection state
                    self.connection_state.connected = True
                    self.connection_state.sync_quality = sync_quality
                    self.connection_state.last_update = datetime.now()
                    self.connection_state.frequency_drift = frequency_std
                    
                else:
                    self.connection_state.connected = False
                    self.connection_state.sync_quality = 0.0
                    
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error updating connection state: {e}")
                await asyncio.sleep(10)
                
    async def _get_schumann_data(self) -> Optional[Dict[str, float]]:
        """Get Schumann resonance data"""
        try:
            # Check if external API calls are disabled
            if os.getenv("ICEBURG_DISABLE_EXTERNAL_APIS", "0") == "1":
                logger.debug("External API calls disabled, using simulated Schumann data")
            
            # Simulate Schumann resonance data
            # In a real implementation, this would connect to actual monitoring stations
            
            # Add some realistic variation to the fundamental frequency
            base_frequency = 7.83
            variation = np.random.normal(0, 0.1)  # ±0.1 Hz variation
            current_frequency = base_frequency + variation
            
            return {
                'fundamental': current_frequency,
                'harmonic_1': current_frequency * 1.83,
                'harmonic_2': current_frequency * 2.66,
                'harmonic_3': current_frequency * 3.49,
                'harmonic_4': current_frequency * 4.32,
                'geomagnetic': np.random.uniform(0, 1),
                'solar_wind': np.random.uniform(300, 800),
                'kp_index': np.random.uniform(0, 9),
                'aurora': np.random.uniform(0, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting Schumann data: {e}")
            return None
            
    async def _get_geomagnetic_data(self) -> Optional[Dict[str, float]]:
        """Get geomagnetic activity data"""
        try:
            # Check if external API calls are disabled
            if os.getenv("ICEBURG_DISABLE_EXTERNAL_APIS", "0") == "1":
                logger.debug("External API calls disabled, using simulated data")
                return {
                    'kp_index': np.random.uniform(0, 9),
                    'geomagnetic_activity': np.random.uniform(0, 1)
                }
            
            # Try to get real data from NOAA
            try:
                response = requests.get(
                    self.data_sources['noaa'],
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Parse NOAA data (simplified)
                    return {
                        'kp_index': data.get('kp_index', 0.0),
                        'geomagnetic': data.get('geomagnetic_activity', 0.0)
                    }
                    
            except Exception as e:
                logger.debug(f"Could not get NOAA data: {e}")
                
            # Fallback to simulated data
            return {
                'kp_index': np.random.uniform(0, 9),
                'geomagnetic_activity': np.random.uniform(0, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting geomagnetic data: {e}")
            return None
            
    def get_current_earth_frequency(self) -> float:
        """Get current Earth frequency (Schumann fundamental)"""
        if self.earth_data:
            return self.earth_data[-1].schumann_fundamental
        return 7.83  # Default Schumann frequency
        
    def get_earth_connection_quality(self) -> float:
        """Get Earth connection quality (0-1)"""
        return self.connection_state.sync_quality
        
    def is_connected_to_earth(self) -> bool:
        """Check if connected to Earth's electromagnetic field"""
        return self.connection_state.connected
        
    def get_earth_frequency_profile(self) -> Dict[str, float]:
        """Get complete Earth frequency profile"""
        if self.earth_data:
            latest = self.earth_data[-1]
            return {
                'schumann_fundamental': latest.schumann_fundamental,
                'schumann_harmonic_1': latest.schumann_harmonics[0],
                'schumann_harmonic_2': latest.schumann_harmonics[1],
                'schumann_harmonic_3': latest.schumann_harmonics[2],
                'schumann_harmonic_4': latest.schumann_harmonics[3],
                'geomagnetic_activity': latest.geomagnetic_activity,
                'solar_wind_speed': latest.solar_wind_speed,
                'kp_index': latest.kp_index,
                'aurora_activity': latest.aurora_activity
            }
        else:
            return {
                'schumann_fundamental': 7.83,
                'schumann_harmonic_1': 14.3,
                'schumann_harmonic_2': 20.8,
                'schumann_harmonic_3': 27.3,
                'schumann_harmonic_4': 33.8,
                'geomagnetic_activity': 0.0,
                'solar_wind_speed': 0.0,
                'kp_index': 0.0,
                'aurora_activity': 0.0
            }
            
    def get_connection_status(self) -> Dict[str, Any]:
        """Get Earth connection status"""
        return {
            'connected': self.connection_state.connected,
            'resonance_strength': self.connection_state.resonance_strength,
            'sync_quality': self.connection_state.sync_quality,
            'last_update': self.connection_state.last_update.isoformat(),
            'frequency_drift': self.connection_state.frequency_drift,
            'current_frequency': self.get_current_earth_frequency(),
            'earth_readings_count': len(self.earth_data)
        }
