"""
Bio-Reactor: Physiological Amplification Engine
===============================================
Implements the "Breathing Algorithm" to ramp user into Gamma state.
Based on Tummo (Inner Fire) and Pranayama retention mechanics.

Logic:
1.  **Ignition (Hyper-Oxygenation)**: Rapid, deep breathing to lower CO2 and increase O2. Alkalizes blood.
2.  **Retention (Hypoxia)**: Breath hold. CO2 builds up, triggering "Survival Mode" -> Adrenaline + DMT? implies Gamma burst.
3.  **Implosion (Gamma)**: The recovery breath. Squeezing the charge into the pineal/brain center.

This module acts as a State Machine driven by `PhysiologicalState` feedback.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .models import PhysiologicalState

logger = logging.getLogger(__name__)

class ReactorState(Enum):
    IDLE = "IDLE"
    IGNITION = "IGNITION"       # Phase 1: Hyper-Oxygenation (Charging)
    RETENTION = "RETENTION"     # Phase 2: Hypoxia (The Void)
    IMPLOSION = "IMPLOSION"     # Phase 3: Recovery Squeeze (The Spike)
    RECOVERY = "RECOVERY"       # Phase 4: Stabilization

@dataclass
class ReactorInstruction:
    """Instructions for the User Interface"""
    state: ReactorState
    instruction_text: str
    target_breathing_rate: float # breaths per minute
    duration_remaining: float    # seconds
    current_intensity: float     # 0.0 to 1.0 (Feedback Visuals)

class BioReactor:
    def __init__(self):
        self.current_state = ReactorState.IDLE
        self.state_start_time = datetime.utcnow()
        self.session_start_time = None
        self.current_cycle = 0
        
        # Configuration
        self.ignition_duration = 30  # breaths (approx 90 seconds)
        self.retention_target = 60.0 # seconds (increases per cycle)
        self.implosion_duration = 15.0 # seconds
        self.recovery_duration = 30.0 # seconds

    def start_session(self):
        """Initiates the Bio-Reactor sequence."""
        self.current_state = ReactorState.IGNITION
        self.state_start_time = datetime.utcnow()
        self.session_start_time = datetime.utcnow()
        self.current_cycle = 1 # Start at Cycle 1
        logger.info("☢️ BIO-REACTOR INITIATED: Cycle 1 - IGNITION")

    def stop_session(self):
        """Aborts the sequence."""
        self.current_state = ReactorState.IDLE
        logger.info("BIO-REACTOR ABORTED.")

    def process_bio_feedback(self, bio_state: PhysiologicalState) -> ReactorInstruction:
        """
        Main Loop: transitions states based on time and bio-feedback.
        Returns instructions for the frontend/user.
        """
        now = datetime.utcnow()
        elapsed = (now - self.state_start_time).total_seconds()
        
        instruction_text = ""
        target_rate = 0.0
        intensity = 0.0
        duration_remaining = 0.0
        
        # --- STATE MACHINE ---
        
        if self.current_state == ReactorState.IDLE:
            instruction_text = "Ready to Initialize. Press Start."
            intensity = 0.0
            duration_remaining = 0.0

        elif self.current_state == ReactorState.IGNITION:
            # Phase 1: CHARGE
            # Goal: High Intensity Breathing
            target_rate = 30.0 # Fast breathing
            intensity = min(elapsed / 60.0, 1.0) # Ramp up visual intensity
            instruction_text = "Breathe DEEP. Rhythm. Charge."
            duration_remaining = max(0.0, 60.0 - elapsed)
            
            # Transition Logic: Time-based for now (or breath count if we had it)
            # Simulating 30 breaths ~ 60 seconds for this prototype
            if elapsed > 60.0: 
                self._transition_to(ReactorState.RETENTION)

        elif self.current_state == ReactorState.RETENTION:
            # Phase 2: THE VOID
            # Goal: Absolute stillness. 
            target_rate = 0.0
            instruction_text = "STOP. Hold. Go Deep into the Void."
            duration_remaining = max(0.0, self.retention_target - elapsed)
            
            # Bio-Feedback: Lower heart rate is better here.
            # If HR spikes (panic), it would reduce "intensity" or fail the hold in a real app.
            
            # Intensity tracks how close we are to the target retention time
            intensity = min(elapsed / self.retention_target, 1.0)
            
            if elapsed > self.retention_target:
                self._transition_to(ReactorState.IMPLOSION)

        elif self.current_state == ReactorState.IMPLOSION:
            # Phase 3: THE SPIKE
            # Goal: Squeeze energy to the head.
            target_rate = 0.0
            instruction_text = "INHALE & SQUEEZE. Push Energy to the Center."
            duration_remaining = max(0.0, self.implosion_duration - elapsed)
            intensity = 1.0 # MAX POWER
            
            if elapsed > self.implosion_duration:
                self._transition_to(ReactorState.RECOVERY)

        elif self.current_state == ReactorState.RECOVERY:
            # Phase 4: Integration
            target_rate = 6.0 # Slow coherent breathing
            instruction_text = "Release. Stabilize. Feel the Silence."
            duration_remaining = max(0.0, self.recovery_duration - elapsed)
            
            # Check coherence (phi alignment)
            # intensity = bio_state.phi_alignment.scalar_score # Visuals match coherence
            intensity = 0.5 # Placeholder until we have live bio_state in testing
            
            if elapsed > self.recovery_duration:
                # Next Cycle or Finish
                if self.current_cycle < 3:
                    self.current_cycle += 1
                    self.retention_target += 30.0 # Increase difficulty
                    self._transition_to(ReactorState.IGNITION)
                else:
                    self.stop_session()
                    instruction_text = "Sequence Complete."

        # Return the telemetry
        return ReactorInstruction(
            state=self.current_state,
            instruction_text=instruction_text,
            target_breathing_rate=target_rate,
            duration_remaining=duration_remaining,
            current_intensity=intensity
        )

    def _transition_to(self, new_state: ReactorState):
        logger.info(f"Transitioning: {self.current_state.name} -> {new_state.name}")
        self.current_state = new_state
        self.state_start_time = datetime.utcnow()
