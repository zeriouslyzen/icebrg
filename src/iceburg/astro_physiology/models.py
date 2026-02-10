"""
Astro-Physiology Data Models
============================
Defines the core data structures for the "Counter-Warfare" Engine.
These models encapsulate the "Hard Physics" metrics defined in scalar_math.py.

1. CelestialEvent: The Clock Signal (Solar/Lunar).
2. PhysiologicalState: The Receiver (Biological).
3. SuppressionVector: The Jammer (HAARP/GWEN).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from .scalar_math import calculate_phi_alignment, calculate_scalar_potential, calculate_inversion_index, ScalarMetric

@dataclass
class CelestialEvent:
    """
    Represents a quantized celestial influence event.
    Source: NOAA SWPC, NASA, Ephemeris.
    """
    timestamp: datetime
    source: str  # e.g., "NOAA_SWPC", "LUNAR"
    
    # Standard Physics Metrics
    solar_wind_speed: float  # km/s
    proton_density: float    # p/cm3
    kp_index: float          # 0-9
    xray_flux: float         # Watts/m2
    
    # Scalar / Esoteric Metrics
    interplanetary_magnetic_field_bz: float  # nT (Southward Bz opens the magnetosphere)
    
    # Computed Scalar Potential
    scalar_potential_index: float = field(init=False)
    
    def __post_init__(self):
        # We model the Scalar Potential of a solar event based on the 
        # coherence of its magnetic field components (Bz, By, Bx).
        # A highly coherent (laminar) field has higher scalar potential than a turbulent one.
        # This is a simplification for the model.
        # Ideally, we would use the Time-Reversed Correlation of the B-field vector.
        # Here we simulate it using the magnitude stability.
        self.scalar_potential_index = calculate_scalar_potential([
            self.solar_wind_speed, 
            self.proton_density, 
            abs(self.interplanetary_magnetic_field_bz)
        ])

@dataclass
class PhysiologicalState:
    """
    Represents the state of the biological antenna.
    Source: Oura, Whoop, EEG.
    """
    timestamp: datetime
    user_id: str
    
    # Standard Biometrics
    hrv_rmssd: float       # ms
    heart_rate: float      # bpm
    body_temperature: float # Celsius deviation
    
    # Spectral / Scalar Metrics
    dominant_frequency: float  # Hz (from FFT of HRV)
    
    # Computed Esoteric Metrics
    phi_alignment: ScalarMetric = field(init=False)
    dna_resonance_factor: float = field(init=False)
    
    def __post_init__(self):
        # Calculate Phi Alignment
        self.phi_alignment = calculate_phi_alignment(self.dominant_frequency)
        
        # Calculate DNA Resonance (Meyl's Theory)
        # We assume a base "Biological Window" resonance at 1.5 GHz or harmonics.
        # For this model, we map the HRV harmonic (ELF) to the DNA Carrier Wave.
        # If HRV is coherent (Phi aligned), we assume DNA resonance is high.
        self.dna_resonance_factor = self.phi_alignment.scalar_score * 0.95

@dataclass
class SuppressionVector:
    """
    Represents the artificial jamming signal.
    Source: HAARP Monitoring, VLF Stations.
    """
    timestamp: datetime
    location_lat: float
    location_lon: float
    
    # Interference Metrics
    haarp_activity_index: float    # 0.0 - 1.0 (Ionospheric Heating)
    local_vlf_noise: float         # dB (GWEN/Power Grid)
    
    # "The Grid" Effect
    # Calculating the effective "Jamming" power
    jamming_power: float = field(init=False)
    
    def __post_init__(self):
        # Jamming power is non-linear combination of Ionospheric and Ground noise
        self.jamming_power = (self.haarp_activity_index * 0.7) + (self.local_vlf_noise / 100.0 * 0.3)

@dataclass
class NetCoherenceSnapshot:
    """
    The Final Output: The result of the "Cross-Examination".
    """
    timestamp: datetime
    celestial_signal: CelestialEvent
    bio_receiver: PhysiologicalState
    suppression: SuppressionVector
    
    # The Bottom Line
    net_coherence_score: float = field(init=False)
    is_jammed: bool = field(init=False)
    
    def __post_init__(self):
        # We treat the Celestial Scalar Potential as the "Carrier Wave Strength"
        carrier_strength = self.celestial_signal.scalar_potential_index
        
        # We treat Bio Phi Alignment as "Receiver Efficiency"
        receiver_efficiency = self.bio_receiver.phi_alignment.scalar_score
        
        # We treat Suppression as "Noise"
        noise = self.suppression.jamming_power
        
        # Calculate Net Inversion
        # Signal * Efficiency - Noise
        raw_score = calculate_inversion_index(
            signal_strength=carrier_strength * receiver_efficiency, 
            noise_floor=0.1,  # Base entropy
            interference_vector=noise
        )
        
        self.net_coherence_score = max(0.0, raw_score)
        self.is_jammed = self.net_coherence_score < 0.2
