"""
Planetary Ephemeris Integration
===============================
Fetches real-time planetary positions (Heliocentric) from NASA JPL Horizons.
Used to calculate "Planetary Resonance" based on tidal forces and angular aspects.

Key Massive Oscillators:
- Jupiter (SPK-ID: 599)
- Saturn (SPK-ID: 699)
- Mars (SPK-ID: 499)
- Venus (SPK-ID: 299)
- Mercury (SPK-ID: 199)
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math

logger = logging.getLogger(__name__)

PLANET_IDS = {
    "Mercury": "199",
    "Venus": "299",
    "Mars": "499",
    "Jupiter": "599",
    "Saturn": "699"
}

# Masses in kg (approx)
PLANET_MASSES = {
    "Mercury": 3.301e23,
    "Venus": 4.867e24,
    "Mars": 6.39e23,
    "Jupiter": 1.898e27,
    "Saturn": 5.683e26
}

class EphemerisIngestor:
    def __init__(self):
        # Correct REST API endpoint
        self.base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"

    def fetch_planetary_positions(self, timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """
        Fetches heliocentric longitude and distance for major planets.
        """
        results = {}
        # JPL requires YYYY-MM-DD HH:MM
        # We need to quote the time strings for the query parameters
        start_time = timestamp.strftime("'%Y-%m-%d %H:%M'")
        stop_time = (timestamp + timedelta(minutes=1)).strftime("'%Y-%m-%d %H:%M'")
        
        for name, planet_id in PLANET_IDS.items():
            try:
                # We use Vectors to get heli-centric coordinates (X, Y, Z)
                params = {
                    "format": "json",
                    "COMMAND": planet_id,
                    "CENTER": "'500@10'", # Sun center
                    "EPHEM_TYPE": "VECTORS",
                    "START_TIME": start_time,
                    "STOP_TIME": stop_time,
                    "STEP_SIZE": "'1d'"
                }
                
                response = requests.get(self.base_url, params=params, timeout=15)
                if response.status_code != 200:
                    logger.error(f"JPL API Error {response.status_code} for {name}")
                    continue
                    
                data = response.json()
                
                if "result" in data:
                    # Parse the vector output between $$SOE and $$EOE
                    # Typically looks like: JDTDB, Calendar Date, X, Y, Z, VX, VY, VZ, LT, RG
                    if "$$SOE" not in data["result"]:
                        logger.error(f"Could not find start of ephemeris for {name}")
                        continue
                        
                    vector_str = data["result"].split("$$SOE")[1].split("$$EOE")[0].strip()
                    
                    try:
                        # JPL Vector table format (default) uses X = ... Y = ... Z = ...
                        # We extract by searching for the labels.
                        def extract_val(label, text):
                            part = text.split(label)[1].strip().split()[0]
                            return float(part.replace(",", ""))

                        x_km = extract_val("X =", vector_str)
                        y_km = extract_val("Y =", vector_str)
                        z_km = extract_val("Z =", vector_str)
                        rg_km = extract_val("RG=", vector_str)
                        
                        # Convert KM to AU (1 AU = 149,597,870.7 km)
                        AU_KM = 149597870.7
                        x = x_km / AU_KM
                        y = y_km / AU_KM
                        z = z_km / AU_KM
                        distance = rg_km / AU_KM
                        
                        # Calculate Longitude (Ecliptic)
                        longitude = math.degrees(math.atan2(y, x)) % 360
                        
                        results[name] = {
                            "longitude": longitude,
                            "distance_au": distance,
                            "mass_kg": PLANET_MASSES[name]
                        }
                    except (IndexError, ValueError) as e:
                        logger.error(f"Parsing error for {name}: {e}. Raw: {vector_str}")
                        continue
                else:
                    logger.error(f"Error fetching data for {name}: No result in JSON")
                    
            except Exception as e:
                logger.error(f"Ephemeris error for {name}: {e}")
        
        return results

    def calculate_resonance_index(self, planetary_data: Dict[str, Dict[str, float]]) -> float:
        """
        Calculates a 'Planetary Resonance Index' (0.0 to 1.0).
        Logic:
        1. Tidal Force Contribution (M / r^3) - Weighting the influence of each body.
        2. Aspect Coherence - Checking for harmonic angular relationships.
        """
        if not planetary_data:
            return 0.5 # Baseline
            
        # 1. Calculate Tidal Forces and Total Weight
        # We calculate relative tidal force: Mass / Distance^3
        influences = {}
        total_tidal_force = 1e-20 # Avoid div by zero
        
        for name, data in planetary_data.items():
            # Distance is in AU, Mass in kg
            # Force ~ M / d^3
            force = data["mass_kg"] / (data["distance_au"] ** 3)
            influences[name] = force
            total_tidal_force += force

        # 2. Aspect Coherence with Tidal Weighting
        # We check the angles between all bodies, weighted by their combined tidal influence
        primary_bodies = list(planetary_data.keys())
        weighted_aspect_score = 0.0
        total_weight = 0.0
        
        for i in range(len(primary_bodies)):
            for j in range(i + 1, len(primary_bodies)):
                p1 = primary_bodies[i]
                p2 = primary_bodies[j]
                
                # Combined weight of this pair is the product of their relative tidal forces
                # This ensures Jupiter-Venus aspects matter more than Mercury-Mars
                weight = (influences[p1] * influences[p2]) / (total_tidal_force ** 2)
                
                angle = abs(planetary_data[p1]["longitude"] - planetary_data[p2]["longitude"]) % 180
                
                # Resonance is high at 0, 60, 120 (Harmonic)
                # Stress is high at 90, 180 (Inversion)
                # Phase = angle * 3 makes 0, 120, 240 local maxima
                coherence = 0.5 + 0.5 * math.cos(math.radians(angle * 3))
                
                weighted_aspect_score += (coherence * weight)
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
            
        # Normalize weighted score
        # The raw weighted score will be small since weights are products of fractions.
        # We normalize by the total weight to get a 0-1 resonance index.
        normalized_resonance = weighted_aspect_score / total_weight
        
        return normalized_resonance
