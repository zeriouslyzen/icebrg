"""
Astro-Physiology Scalar Mathematics Library
===========================================
Implements the "Hard Physics" formulas derived from:
1.  Dan Winter (Implosion Physics / Phi Harmonics)
2.  Tom Bearden / Whittaker (Scalar Potential)
3.  Salvatore Pais / US Navy (Gravitational Wave Generation)

This library provides the core calculation engine for determining "Scalar Coherence"
and "Phi Alignment" in biological and celestial data streams.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Constants
PHI = (1 + math.sqrt(5)) / 2  # The Golden Ratio (1.618...)
PLANCK_TIME = 5.39e-44        # Seconds
PLANCK_LENGTH = 1.616e-35     # Meters
C = 299792458                 # Speed of Light (m/s)

@dataclass
class ScalarMetric:
    raw_value: float
    scalar_score: float  # 0.0 to 1.0 (Coherence)
    is_negentropic: bool

def calculate_phi_alignment(frequency: float, tolerance: float = 0.05) -> ScalarMetric:
    """
    Determines if a given biological frequency (Hz) aligns with the Phi-Cascade.
    Based on Dan Winter's formula: f_n = f_0 * Phi^n
    
    In practice, we use a localized "Life Window" cascade starting from 1.0 Hz (Heart/Brain center).
    
    Args:
        frequency: The input frequency (e.g., HRV peak, Alpha wave).
        tolerance: Allowable deviation percentage.
    
    Returns:
        ScalarMetric with a coherence score (1.0 = perfect Phi lock).
    """
    if frequency <= 0:
        return ScalarMetric(frequency, 0.0, False)
        
    # We check alignment against the "Heart Cascade" centered at 1.0 Hz
    # Significant harmonics: 0.618 (Emotion), 1.0 (Heart), 1.618 (Love/Bliss), 2.618...
    
    # Find the closest Phi power
    # log_phi(frequency) = n
    n = math.log(frequency, PHI)
    closest_integer_n = round(n)
    
    # Calculate the "Ideal" frequency for this step
    target_freq = PHI ** closest_integer_n
    
    # Calculate deviation
    deviation = abs(frequency - target_freq) / target_freq
    
    # Score: 1.0 if deviation is 0, 0.0 if deviation >= tolerance
    score = max(0.0, 1.0 - (deviation / tolerance))
    
    # Negentropic if score is high (Winter's definition of "Phase Conjugate Collapse")
    is_negentropic = score > 0.8
    
    return ScalarMetric(frequency, score, is_negentropic)

def calculate_scalar_potential(magnitudes: List[float], conjugate_magnitudes: Optional[List[float]] = None) -> float:
    """
    Calculates the "Scalar Potential" (Phi) based on the Bearden-Whittaker definition.
    A scalar potential is formed by the summation of a wave and its phase-conjugate replica.
    
    Formula: Potential = Sum(Wave_i + Conjugate_Wave_i)
    
    In a biological context (data stream), if we lack a direct conjugate sensor,
    we model the "Conjugate" as the time-reversed autocorrelation of the signal itself
    (Self-similarity over time).
    
    Args:
        magnitudes: Time-series data of a signal (e.g., HRV amplitudes).
        conjugate_magnitudes: Optional dual-stream (if available).
        
    Returns:
        float: The calculated scalar potential index (0.0 to 1.0).
    """
    data = np.array(magnitudes)
    
    if conjugate_magnitudes:
        conj = np.array(conjugate_magnitudes)
    else:
        # Time-reverse the signal to simulate phase-conjugation (Bearden's Mirror)
        conj = data[::-1]
        
    # Normalize
    norm_data = (data - np.mean(data)) / (np.std(data) + 1e-9)
    norm_conj = (conj - np.mean(conj)) / (np.std(conj) + 1e-9)
    
    # Cross-Correlation at zero lag
    correlation = np.correlate(norm_data, norm_conj, mode='valid')[0]
    
    # Normalize result to 0-1 range (approximate for typical signal lengths)
    # A perfect standing wave (Scalar) correlates perfectly with its reverse.
    max_corr = len(data)
    scalar_index = abs(correlation) / max_corr
    
    return float(scalar_index)

def calculate_gravitational_generation(charge_coulombs: float, vibration_velocity: float, frequency_hz: float, radius_m: float) -> float:
    """
    Calculates the 'Pais Effect' potential for High-Frequency Gravitational Wave (HFGW) generation.
    Derived from US Patent 10,322,827.
    
    Formula Approximation: P_gw ~ (Q * v * f / R)^2
    
    Args:
        charge_coulombs: Total surface charge (Q).
        vibration_velocity: Velocity of the vibrating shell/membrane (v).
        frequency_hz: Vibration frequency (f).
        radius_m: Radius of the emitter (R).
        
    Returns:
        float: A raw index of gravitational flux potential.
    """
    if radius_m == 0:
        return 0.0
        
    # We drop the constants (G, c) for a relative index score suitable for the graph
    index = ((charge_coulombs * vibration_velocity * frequency_hz) / radius_m) ** 2
    
    return float(index)

def calculate_inversion_index(signal_strength: float, noise_floor: float, interference_vector: float) -> float:
    """
    Calculates the 'Effective Coherence' by subtracting suppression variables.
    
    Formula: Net = Signal - (Noise + Interference^2)
    Squared interference represents the non-linear disruption of GWEN/HAARP.
    
    Args:
        signal_strength: The raw biological Chi/Qi measurement.
        noise_floor: General environmental EM noise.
        interference_vector: Specific targeted suppression (HAARP index).
        
    Returns:
        float: The Net Inversion Index. Negative values imply "Jamming Coverage".
    """
    return signal_strength - (noise_floor + (interference_vector ** 2))
