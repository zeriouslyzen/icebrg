"""
ICEBURG TCM-Planetary Integration
Traditional Chinese Medicine organ-planet correlations with time-of-day effects
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime, time
import math


@dataclass
class TCMOrganClock:
    """TCM organ clock - peak activity times for each organ"""
    organ: str
    peak_hour: int  # 0-23
    element: str
    planet: str
    emotion: str
    function: str


@dataclass
class PlanetaryOrganState:
    """State of organ-planet association at a given time"""
    organ: str
    planet: str
    time_of_day: datetime
    activity_level: float  # 0.0-1.0
    gravitational_influence: float
    electromagnetic_influence: float
    health_indicator: str  # "strong", "moderate", "weak"


class TCMPlanetaryIntegration:
    """
    TCM-Planetary Integration System
    
    Philosophy: Traditional Chinese Medicine organ-planet associations
    with time-of-day effects and gravitational/electromagnetic influences
    """
    
    def __init__(self):
        self.organ_clock: Dict[str, TCMOrganClock] = {}
        self.planetary_associations: Dict[str, Dict[str, Any]] = {}
        self._initialize_organ_clock()
        self._initialize_planetary_associations()
    
    def _initialize_organ_clock(self):
        """Initialize TCM organ clock (24-hour cycle)"""
        # TCM organ clock - each organ has peak activity at specific times
        self.organ_clock = {
            "liver": TCMOrganClock(
                organ="liver",
                peak_hour=1,  # 1-3 AM
                element="wood",
                planet="moon",
                emotion="anger",
                function="detoxification_and_planning"
            ),
            "lung": TCMOrganClock(
                organ="lung",
                peak_hour=3,  # 3-5 AM
                element="metal",
                planet="venus",
                emotion="grief",
                function="respiration_and_elimination"
            ),
            "large_intestine": TCMOrganClock(
                organ="large_intestine",
                peak_hour=5,  # 5-7 AM
                element="metal",
                planet="mercury",
                emotion="grief",
                function="elimination"
            ),
            "stomach": TCMOrganClock(
                organ="stomach",
                peak_hour=7,  # 7-9 AM
                element="earth",
                planet="jupiter",
                emotion="worry",
                function="digestion_and_nourishment"
            ),
            "spleen": TCMOrganClock(
                organ="spleen",
                peak_hour=9,  # 9-11 AM
                element="earth",
                planet="mercury",
                emotion="worry",
                function="digestion_and_thought"
            ),
            "heart": TCMOrganClock(
                organ="heart",
                peak_hour=11,  # 11 AM-1 PM
                element="fire",
                planet="sun",
                emotion="joy",
                function="circulation_and_consciousness"
            ),
            "small_intestine": TCMOrganClock(
                organ="small_intestine",
                peak_hour=13,  # 1-3 PM
                element="fire",
                planet="sun",
                emotion="joy",
                function="absorption_and_separation"
            ),
            "bladder": TCMOrganClock(
                organ="bladder",
                peak_hour=15,  # 3-5 PM
                element="water",
                planet="saturn",
                emotion="fear",
                function="fluid_balance"
            ),
            "kidney": TCMOrganClock(
                organ="kidney",
                peak_hour=17,  # 5-7 PM
                element="water",
                planet="saturn",
                emotion="fear",
                function="filtration_and_willpower"
            ),
            "pericardium": TCMOrganClock(
                organ="pericardium",
                peak_hour=19,  # 7-9 PM
                element="fire",
                planet="sun",
                emotion="joy",
                function="circulation_protection"
            ),
            "triple_warmer": TCMOrganClock(
                organ="triple_warmer",
                peak_hour=21,  # 9-11 PM
                element="fire",
                planet="sun",
                emotion="joy",
                function="temperature_regulation"
            ),
            "gallbladder": TCMOrganClock(
                organ="gallbladder",
                peak_hour=23,  # 11 PM-1 AM
                element="wood",
                planet="mars",
                emotion="anger",
                function="decision_making_and_courage"
            )
        }
    
    def _initialize_planetary_associations(self):
        """Initialize planetary-organ associations"""
        self.planetary_associations = {
            "sun": {
                "organs": ["heart", "small_intestine", "pericardium", "triple_warmer"],
                "element": "fire",
                "emotion": "joy",
                "gravitational_influence": 0.1,  # Solar gravitational effects
                "electromagnetic_influence": 100.0,  # Solar wind, radiation
                "peak_time": time(12, 0)  # Noon
            },
            "moon": {
                "organs": ["liver"],
                "element": "wood",
                "emotion": "anger",
                "gravitational_influence": 0.5,  # Tidal effects
                "electromagnetic_influence": 0.1,
                "peak_time": time(1, 0)  # 1 AM
            },
            "mars": {
                "organs": ["gallbladder"],
                "element": "wood",
                "emotion": "anger",
                "gravitational_influence": 0.05,
                "electromagnetic_influence": 0.05,
                "peak_time": time(23, 0)  # 11 PM
            },
            "mercury": {
                "organs": ["spleen", "large_intestine"],
                "element": "earth",
                "emotion": "worry",
                "gravitational_influence": 0.01,
                "electromagnetic_influence": 0.01,
                "peak_time": time(9, 0)  # 9 AM
            },
            "jupiter": {
                "organs": ["stomach"],
                "element": "earth",
                "emotion": "worry",
                "gravitational_influence": 0.3,
                "electromagnetic_influence": 0.5,
                "peak_time": time(7, 0)  # 7 AM
            },
            "venus": {
                "organs": ["lung"],
                "element": "metal",
                "emotion": "grief",
                "gravitational_influence": 0.02,
                "electromagnetic_influence": 0.02,
                "peak_time": time(3, 0)  # 3 AM
            },
            "saturn": {
                "organs": ["kidney", "bladder"],
                "element": "water",
                "emotion": "fear",
                "gravitational_influence": 0.3,
                "electromagnetic_influence": 0.3,
                "peak_time": time(17, 0)  # 5 PM
            }
        }
    
    def get_organ_activity(self, organ: str, current_time: datetime) -> float:
        """
        Get organ activity level at current time
        
        Args:
            organ: Organ name
            current_time: Current datetime
        
        Returns:
            Activity level (0.0-1.0)
        """
        if organ not in self.organ_clock:
            return 0.5  # Default activity
        
        clock = self.organ_clock[organ]
        current_hour = current_time.hour
        
        # Calculate activity based on distance from peak hour
        # Activity peaks at peak_hour and is lowest 12 hours later
        hour_diff = abs(current_hour - clock.peak_hour)
        if hour_diff > 12:
            hour_diff = 24 - hour_diff
        
        # Activity follows cosine curve (peaks at peak_hour)
        activity = 0.5 + 0.5 * math.cos(math.pi * hour_diff / 12)
        
        return max(0.0, min(1.0, activity))
    
    def get_planetary_influence(self, planet: str, current_time: datetime) -> Dict[str, float]:
        """
        Get planetary influence at current time
        
        Args:
            planet: Planet name
            current_time: Current datetime
        
        Returns:
            Dictionary with gravitational and electromagnetic influences
        """
        if planet not in self.planetary_associations:
            return {"gravitational": 0.0, "electromagnetic": 0.0}
        
        association = self.planetary_associations[planet]
        
        # Base influences
        gravitational = association["gravitational_influence"]
        electromagnetic = association["electromagnetic_influence"]
        
        # Time-of-day modulation (influence stronger at peak time)
        peak_time = association["peak_time"]
        current_time_only = current_time.time()
        
        # Calculate time difference
        time_diff = abs((current_time_only.hour * 60 + current_time_only.minute) - 
                       (peak_time.hour * 60 + peak_time.minute))
        if time_diff > 720:  # 12 hours
            time_diff = 1440 - time_diff
        
        # Modulate influence based on time (stronger at peak time)
        time_modulation = 0.5 + 0.5 * math.cos(math.pi * time_diff / 720)
        
        gravitational *= time_modulation
        electromagnetic *= time_modulation
        
        return {
            "gravitational": gravitational,
            "electromagnetic": electromagnetic,
            "time_modulation": time_modulation
        }
    
    def get_organ_planetary_state(self, organ: str, current_time: datetime) -> PlanetaryOrganState:
        """
        Get organ-planet state at current time
        
        Args:
            organ: Organ name
            current_time: Current datetime
        
        Returns:
            PlanetaryOrganState with activity and influences
        """
        if organ not in self.organ_clock:
            return PlanetaryOrganState(
                organ=organ,
                planet="unknown",
                time_of_day=current_time,
                activity_level=0.5,
                gravitational_influence=0.0,
                electromagnetic_influence=0.0,
                health_indicator="unknown"
            )
        
        clock = self.organ_clock[organ]
        planet = clock.planet
        
        # Get organ activity
        activity = self.get_organ_activity(organ, current_time)
        
        # Get planetary influence
        planetary_influence = self.get_planetary_influence(planet, current_time)
        
        # Calculate health indicator
        if activity > 0.8:
            health_indicator = "strong"
        elif activity > 0.5:
            health_indicator = "moderate"
        else:
            health_indicator = "weak"
        
        return PlanetaryOrganState(
            organ=organ,
            planet=planet,
            time_of_day=current_time,
            activity_level=activity,
            gravitational_influence=planetary_influence["gravitational"],
            electromagnetic_influence=planetary_influence["electromagnetic"],
            health_indicator=health_indicator
        )
    
    def get_all_organs_state(self, current_time: datetime) -> Dict[str, PlanetaryOrganState]:
        """Get state of all organs at current time"""
        states = {}
        for organ in self.organ_clock.keys():
            states[organ] = self.get_organ_planetary_state(organ, current_time)
        return states
    
    def get_optimal_times(self, organ: str) -> List[int]:
        """
        Get optimal times for organ activity
        
        Args:
            organ: Organ name
        
        Returns:
            List of optimal hours (0-23)
        """
        if organ not in self.organ_clock:
            return []
        
        clock = self.organ_clock[organ]
        peak_hour = clock.peak_hour
        
        # Optimal times: peak hour ± 2 hours
        optimal_hours = []
        for i in range(-2, 3):
            hour = (peak_hour + i) % 24
            optimal_hours.append(hour)
        
        return optimal_hours
    
    def get_organ_planet_correlation(self, organ: str) -> Dict[str, Any]:
        """Get organ-planet correlation data"""
        if organ not in self.organ_clock:
            return {}
        
        clock = self.organ_clock[organ]
        planet = clock.planet
        
        if planet not in self.planetary_associations:
            return {}
        
        association = self.planetary_associations[planet]
        
        return {
            "organ": organ,
            "planet": planet,
            "element": clock.element,
            "emotion": clock.emotion,
            "function": clock.function,
            "peak_hour": clock.peak_hour,
            "gravitational_influence": association["gravitational_influence"],
            "electromagnetic_influence": association["electromagnetic_influence"]
        }


# Global TCM-planetary integration instance
_tcm_planetary: Optional[TCMPlanetaryIntegration] = None

def get_tcm_planetary_integration() -> TCMPlanetaryIntegration:
    """Get or create the global TCM-planetary integration instance"""
    global _tcm_planetary
    if _tcm_planetary is None:
        _tcm_planetary = TCMPlanetaryIntegration()
    return _tcm_planetary

