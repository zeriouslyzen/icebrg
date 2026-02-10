"""
ICEBURG Physiological State Integration
=======================================
Integrates the "Astro-Physiology Data Engine" with ICEBURG's emergence detection.

"The Bridge between the Mathematical and the Metaphorical."

IMPORTANT DISCLAIMER: This system analyzes legitimate physiological and environmental
patterns (Scalar, Celestial, Biological) to detect "Coherence" in the user.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

# Import the NEW Hard-Tech Engine
from ..astro_physiology.ingestion import AstroPhysiologyIngestor
from ..astro_physiology.models import NetCoherenceSnapshot
from ..config import IceburgConfig

logger = logging.getLogger(__name__)

@dataclass
class UnifiedPhysiologicalState:
    """
    The True State of the System.
    Wraps the rigorous 'NetCoherenceSnapshot' in a format ICEBURG understands.
    """
    timestamp: datetime
    snapshot: NetCoherenceSnapshot
    unified_field_strength: float
    emergence_potential: float
    breakthrough_probability: float

class ICEBURGPhysiologicalIntegration:
    """
    The Bridge.
    Uses the AstroPhysiologyIngestor to drive ICEBURG's consciousness logic.
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.integration_active = False
        
        # The New Engine
        self.ingestor = AstroPhysiologyIngestor()
        
        self.unified_states: List[UnifiedPhysiologicalState] = []
        
        # Thresholds derived from Phi-Math (0.618 is the golden door)
        self.icberg_consciousness_params = {
            'emergence_threshold': 0.618,      # Phi - 1
            'breakthrough_threshold': 0.8,     # High Coherence
            'unified_field_threshold': 0.5     # Net Positive
        }
        
    async def start_icberg_consciousness_integration(self) -> None:
        """Start the Scalar Integration Loop"""
        try:
            logger.info("⚡ Starting Astro-Physiology Data Engine...")
            self.integration_active = True
            
            # Start integration loop
            asyncio.create_task(self._consciousness_integration_loop())
            
        except Exception as e:
            logger.error(f"Error starting integration: {e}")
            self.integration_active = False
            
    async def stop_icberg_consciousness_integration(self) -> None:
        """Stop the Engine"""
        self.integration_active = False
        logger.info("⚡ Astro-Physiology Engine stopped")
            
    async def _consciousness_integration_loop(self) -> None:
        """The Main Loop: Fetch -> Calculate -> Correlate"""
        while self.integration_active:
            try:
                # 1. Get the Snapshot (The Hard Data)
                # In a real app, user_id would come from session.
                snapshot = self.ingestor.fetch_current_snapshot(user_id="USER_001")
                
                # 2. Derive ICEBURG Metrics from Hard Math
                unified_state = self._map_snapshot_to_consciousness(snapshot)
                
                if unified_state:
                    self.unified_states.append(unified_state)
                    
                    # Buffer management
                    if len(self.unified_states) > 1000:
                        self.unified_states = self.unified_states[-500:]
                        
                    # Check for "The Event"
                    if unified_state.emergence_potential > self.icberg_consciousness_params['emergence_threshold']:
                        await self._process_emergence_event(unified_state)
                        
                    if unified_state.breakthrough_probability > self.icberg_consciousness_params['breakthrough_threshold']:
                        await self._process_breakthrough_event(unified_state)
                        
                await asyncio.sleep(1.0) 
                
            except Exception as e:
                logger.error(f"Error in integration loop: {e}")
                await asyncio.sleep(1)
                
    def _map_snapshot_to_consciousness(self, snapshot: NetCoherenceSnapshot) -> UnifiedPhysiologicalState:
        """
        Maps the rigorous physics snapshot to the high-level consciousness metaphors.
        """
        # Unified Field = Net Coherence Score (Signal - Noise)
        unified_field = snapshot.net_coherence_score
        
        # Emergence Potential = Scalar Potential * Phi Alignment
        # (How strong is the carrier * How tuned is the receiver)
        emergence = snapshot.celestial_signal.scalar_potential_index * snapshot.bio_receiver.phi_alignment.scalar_score
        
        # Breakthrough = Probability that Signal punches through Jamming
        # If jamming is low and signal is high -> Breakthrough
        jamming = snapshot.suppression.jamming_power
        if jamming > 0.8:
            breakthrough = 0.0
        else:
            breakthrough = (unified_field * 0.8) + (0.2 * (1.0 - jamming))
            
        return UnifiedPhysiologicalState(
            timestamp=snapshot.timestamp,
            snapshot=snapshot,
            unified_field_strength=unified_field,
            emergence_potential=emergence,
            breakthrough_probability=breakthrough
        )

    async def _process_emergence_event(self, state: UnifiedPhysiologicalState) -> None:
        """A Scalar Event has been detected."""
        logger.info(f"🌟 SCALAR EMERGENCE DETECTED!")
        logger.info(f"   Net Coherence: {state.unified_field_strength:.3f}")
        logger.info(f"   Phi Alignment: {state.snapshot.bio_receiver.phi_alignment.scalar_score:.3f}")
        logger.info(f"   Jamming Level: {state.snapshot.suppression.jamming_power:.3f}")

    async def _process_breakthrough_event(self, state: UnifiedPhysiologicalState) -> None:
        """A Breakthrough has been detected."""
        logger.info(f"🚀 BREAKTHROUGH CONFIRMED!")
        logger.info(f"   Solar Scalar Potential: {state.snapshot.celestial_signal.scalar_potential_index:.3f}")
        logger.info(f"   DNA Resonance: {state.snapshot.bio_receiver.dna_resonance_factor:.3f}")

    def get_unified_consciousness_status(self) -> Dict[str, Any]:
        """
        Public API for the frontend to poll.
        Exposes the deep physics metrics.
        """
        if not self.unified_states:
             return {
                'integration_active': self.integration_active,
                'status': "INITIALIZING",
                'unified_field_strength': 0.0
            }
            
        latest = self.unified_states[-1]
        snap = latest.snapshot
        
        return {
            'integration_active': self.integration_active,
            'timestamp': latest.timestamp.isoformat(),
            
            # The Big Numbers
            'unified_field_strength': latest.unified_field_strength,
            'emergence_potential': latest.emergence_potential,
            'breakthrough_probability': latest.breakthrough_probability,
            
            # The Hard Data (Drill Down)
            'telemetry': {
                'celestial': {
                    'solar_wind': snap.celestial_signal.solar_wind_speed,
                    'kp_index': snap.celestial_signal.kp_index,
                    'scalar_potential': snap.celestial_signal.scalar_potential_index
                },
                'biological': {
                    'hrv_rmssd': snap.bio_receiver.hrv_rmssd,
                    'phi_alignment': snap.bio_receiver.phi_alignment.scalar_score,
                    'is_negentropic': snap.bio_receiver.phi_alignment.is_negentropic,
                    'dna_resonance': snap.bio_receiver.dna_resonance_factor
                },
                'suppression': {
                    'haarp_index': snap.suppression.haarp_activity_index,
                    'vlf_noise': snap.suppression.local_vlf_noise,
                    'jamming_power': snap.suppression.jamming_power
                }
            }
        }
