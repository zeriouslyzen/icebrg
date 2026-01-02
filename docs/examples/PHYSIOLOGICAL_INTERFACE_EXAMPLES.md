# Physiological Interface Integration Examples

## Overview

This document provides practical examples for integrating ICEBURG's Physiological Interface System with real-world applications, including heart rate monitoring, stress detection, Earth connection, and consciousness amplification.

## Basic Setup

### 1. Initialize Physiological Interface

```python
from iceburg.physiological_interface import PhysiologicalAmplifier, EarthConnection, FrequencySynthesizer

# Initialize physiological amplifier
amplifier = PhysiologicalAmplifier({
    "monitoring": {
        "enabled": True,
        "interval": 1.0,
        "sensors": ["heart_rate", "breathing", "stress"]
    }
})

# Initialize Earth connection
earth_connection = EarthConnection(monitoring_interval=1.0)

# Initialize frequency synthesizer
synthesizer = FrequencySynthesizer(audio_device="Built-in Output")
```

### 2. Basic Heart Rate Monitoring

```python
import asyncio
from iceburg.physiological_interface import PhysiologicalAmplifier

async def monitor_heart_rate():
    """Basic heart rate monitoring example"""
    
    amplifier = PhysiologicalAmplifier()
    
    try:
        # Detect current heart rate
        heart_rate = await amplifier.detect_heart_rate()
        print(f"Current heart rate: {heart_rate} BPM")
        
        # Monitor for 30 seconds
        for i in range(30):
            heart_rate = await amplifier.detect_heart_rate()
            print(f"Heart rate reading {i+1}: {heart_rate} BPM")
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Error monitoring heart rate: {e}")

# Run the example
asyncio.run(monitor_heart_rate())
```

### 3. Stress Detection and Management

```python
import asyncio
from iceburg.physiological_interface import PhysiologicalAmplifier

async def stress_management():
    """Stress detection and management example"""
    
    amplifier = PhysiologicalAmplifier()
    
    try:
        # Detect stress level
        stress_level = await amplifier.detect_stress_level()
        print(f"Current stress level: {stress_level:.2f}")
        
        if stress_level > 0.7:
            print("High stress detected! Initiating relaxation protocol...")
            
            # Amplify consciousness to relaxed state
            success = await amplifier.amplify_consciousness("relaxed")
            if success:
                print("Consciousness amplified to relaxed state")
                
                # Generate relaxation frequency
                await synthesizer.generate_frequency(40.0, 10.0)  # 40Hz for 10 seconds
                
        elif stress_level > 0.4:
            print("Moderate stress detected. Monitoring...")
            
        else:
            print("Stress levels are normal")
            
    except Exception as e:
        print(f"Error in stress management: {e}")

# Run the example
asyncio.run(stress_management())
```

## Advanced Examples

### 4. Earth Connection Monitoring

```python
import asyncio
from iceburg.physiological_interface import EarthConnection

async def earth_connection_monitoring():
    """Earth connection monitoring example"""
    
    earth_connection = EarthConnection()
    
    try:
        # Monitor Schumann resonance
        resonance_data = await earth_connection.monitor_schumann_resonance()
        print(f"Schumann resonance data: {resonance_data}")
        
        # Detect Earth connection strength
        connection_strength = await earth_connection.detect_earth_connection()
        print(f"Earth connection strength: {connection_strength:.2f}")
        
        # Get Earth frequency profile
        frequency_profile = await earth_connection.get_earth_frequency_profile()
        print(f"Earth frequency profile: {frequency_profile}")
        
        # Monitor for 60 seconds
        for i in range(60):
            resonance_data = await earth_connection.monitor_schumann_resonance()
            connection_strength = await earth_connection.detect_earth_connection()
            
            print(f"Reading {i+1}: Resonance={resonance_data['frequency']:.2f}Hz, "
                  f"Connection={connection_strength:.2f}")
            
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Error in Earth connection monitoring: {e}")

# Run the example
asyncio.run(earth_connection_monitoring())
```

### 5. Consciousness Amplification

```python
import asyncio
from iceburg.physiological_interface import PhysiologicalAmplifier, FrequencySynthesizer

async def consciousness_amplification():
    """Consciousness amplification example"""
    
    amplifier = PhysiologicalAmplifier()
    synthesizer = FrequencySynthesizer()
    
    try:
        # Amplify consciousness to focused state
        print("Amplifying consciousness to focused state...")
        success = await amplifier.amplify_consciousness("focused")
        
        if success:
            print("Consciousness amplified successfully")
            
            # Generate focus frequency (40Hz gamma)
            await synthesizer.generate_frequency(40.0, 15.0)
            
            # Wait and then switch to relaxed state
            await asyncio.sleep(5)
            
            print("Switching to relaxed state...")
            success = await amplifier.amplify_consciousness("relaxed")
            
            if success:
                # Generate relaxation frequency (10Hz alpha)
                await synthesizer.generate_frequency(10.0, 15.0)
                
        else:
            print("Failed to amplify consciousness")
            
    except Exception as e:
        print(f"Error in consciousness amplification: {e}")

# Run the example
asyncio.run(consciousness_amplification())
```

### 6. Binaural Beats for Meditation

```python
import asyncio
from iceburg.physiological_interface import FrequencySynthesizer

async def meditation_binaural_beats():
    """Binaural beats for meditation example"""
    
    synthesizer = FrequencySynthesizer()
    
    try:
        # Generate binaural beats for meditation
        # Left ear: 200Hz, Right ear: 210Hz (10Hz difference for alpha state)
        print("Generating meditation binaural beats...")
        
        await synthesizer.generate_binaural_beats(200.0, 210.0)
        
        # Wait for meditation session
        await asyncio.sleep(20)  # 20-minute meditation
        
        print("Meditation session complete")
        
    except Exception as e:
        print(f"Error in meditation binaural beats: {e}")

# Run the example
asyncio.run(meditation_binaural_beats())
```

## ICEBURG Integration Examples

### 7. Physiological Data Integration with ICEBURG

```python
import asyncio
from iceburg.physiological_interface import ICEBURGPhysiologicalIntegration
from iceburg.agents import SurveyorAgent

async def iceburg_physiological_integration():
    """ICEBURG physiological integration example"""
    
    # Initialize ICEBURG physiological integration
    integration = ICEBURGPhysiologicalIntegration()
    
    # Initialize ICEBURG agent
    surveyor = SurveyorAgent()
    
    try:
        # Get agent output
        agent_output = await surveyor.analyze("Analyze market trends for renewable energy")
        
        # Integrate physiological data
        integrated_data = await integration.integrate_physiological_data(agent_output)
        print(f"Integrated physiological data: {integrated_data}")
        
        # Assess breakthrough probability
        physiological_state = {
            "heart_rate": 72,
            "stress_level": 0.3,
            "consciousness_state": "focused"
        }
        
        breakthrough_prob = await integration.assess_breakthrough_probability(physiological_state)
        print(f"Breakthrough probability: {breakthrough_prob:.2f}")
        
        # Get unified physiological state
        unified_state = await integration.get_unified_physiological_state()
        print(f"Unified physiological state: {unified_state}")
        
    except Exception as e:
        print(f"Error in ICEBURG physiological integration: {e}")

# Run the example
asyncio.run(iceburg_physiological_integration())
```

### 8. Real-time Physiological Monitoring

```python
import asyncio
import json
from iceburg.physiological_interface import PhysiologicalAmplifier, EarthConnection

async def real_time_monitoring():
    """Real-time physiological monitoring example"""
    
    amplifier = PhysiologicalAmplifier()
    earth_connection = EarthConnection()
    
    # Create monitoring data structure
    monitoring_data = {
        "heart_rate": [],
        "stress_level": [],
        "earth_connection": [],
        "timestamps": []
    }
    
    try:
        print("Starting real-time physiological monitoring...")
        
        # Monitor for 5 minutes
        for i in range(300):  # 5 minutes = 300 seconds
            # Get physiological data
            heart_rate = await amplifier.detect_heart_rate()
            stress_level = await amplifier.detect_stress_level()
            earth_connection_strength = await earth_connection.detect_earth_connection()
            
            # Store data
            monitoring_data["heart_rate"].append(heart_rate)
            monitoring_data["stress_level"].append(stress_level)
            monitoring_data["earth_connection"].append(earth_connection_strength)
            monitoring_data["timestamps"].append(i)
            
            # Print current status
            print(f"Time: {i}s | HR: {heart_rate:.1f} | Stress: {stress_level:.2f} | "
                  f"Earth: {earth_connection_strength:.2f}")
            
            # Check for anomalies
            if stress_level > 0.8:
                print("⚠️  High stress detected!")
            if heart_rate > 100:
                print("⚠️  Elevated heart rate detected!")
            if earth_connection_strength < 0.3:
                print("⚠️  Weak Earth connection detected!")
            
            await asyncio.sleep(1)
        
        # Save monitoring data
        with open("physiological_monitoring_data.json", "w") as f:
            json.dump(monitoring_data, f, indent=2)
        
        print("Monitoring complete. Data saved to physiological_monitoring_data.json")
        
    except Exception as e:
        print(f"Error in real-time monitoring: {e}")

# Run the example
asyncio.run(real_time_monitoring())
```

## Application-Specific Examples

### 9. Meditation App Integration

```python
import asyncio
from iceburg.physiological_interface import PhysiologicalAmplifier, FrequencySynthesizer

class MeditationApp:
    """Meditation app with physiological interface"""
    
    def __init__(self):
        self.amplifier = PhysiologicalAmplifier()
        self.synthesizer = FrequencySynthesizer()
        self.session_active = False
    
    async def start_meditation_session(self, duration_minutes: int = 10):
        """Start meditation session with physiological monitoring"""
        
        self.session_active = True
        print(f"Starting {duration_minutes}-minute meditation session...")
        
        try:
            # Generate meditation frequency
            await self.synthesizer.generate_frequency(10.0, duration_minutes * 60)
            
            # Monitor during session
            for i in range(duration_minutes * 60):
                if not self.session_active:
                    break
                
                # Check stress level
                stress_level = await self.amplifier.detect_stress_level()
                
                if stress_level > 0.6:
                    print("High stress detected. Adjusting meditation frequency...")
                    await self.synthesizer.generate_frequency(8.0, 30)  # Lower frequency
                
                await asyncio.sleep(1)
            
            print("Meditation session complete")
            
        except Exception as e:
            print(f"Error in meditation session: {e}")
    
    async def stop_meditation_session(self):
        """Stop meditation session"""
        self.session_active = False
        print("Meditation session stopped")

# Usage example
async def meditation_app_example():
    app = MeditationApp()
    await app.start_meditation_session(5)  # 5-minute session

# Run the example
asyncio.run(meditation_app_example())
```

### 10. Productivity Enhancement

```python
import asyncio
from iceburg.physiological_interface import PhysiologicalAmplifier, FrequencySynthesizer

class ProductivityEnhancer:
    """Productivity enhancement with physiological monitoring"""
    
    def __init__(self):
        self.amplifier = PhysiologicalAmplifier()
        self.synthesizer = FrequencySynthesizer()
        self.focus_sessions = 0
    
    async def start_focus_session(self, duration_minutes: int = 25):
        """Start focus session with physiological optimization"""
        
        print(f"Starting {duration_minutes}-minute focus session...")
        
        try:
            # Amplify consciousness to focused state
            await self.amplifier.amplify_consciousness("focused")
            
            # Generate focus frequency (40Hz gamma)
            await self.synthesizer.generate_frequency(40.0, duration_minutes * 60)
            
            # Monitor focus during session
            for i in range(duration_minutes * 60):
                # Check stress level
                stress_level = await self.amplifier.detect_stress_level()
                
                if stress_level > 0.5:
                    print("Stress detected. Taking break...")
                    await self.synthesizer.generate_frequency(10.0, 60)  # Relaxation
                    break
                
                await asyncio.sleep(1)
            
            self.focus_sessions += 1
            print(f"Focus session complete. Total sessions: {self.focus_sessions}")
            
        except Exception as e:
            print(f"Error in focus session: {e}")
    
    async def start_break_session(self, duration_minutes: int = 5):
        """Start break session for relaxation"""
        
        print(f"Starting {duration_minutes}-minute break session...")
        
        try:
            # Amplify consciousness to relaxed state
            await self.amplifier.amplify_consciousness("relaxed")
            
            # Generate relaxation frequency (10Hz alpha)
            await self.synthesizer.generate_frequency(10.0, duration_minutes * 60)
            
            await asyncio.sleep(duration_minutes * 60)
            print("Break session complete")
            
        except Exception as e:
            print(f"Error in break session: {e}")

# Usage example
async def productivity_enhancer_example():
    enhancer = ProductivityEnhancer()
    
    # Pomodoro technique: 25 minutes focus, 5 minutes break
    await enhancer.start_focus_session(25)
    await enhancer.start_break_session(5)
    await enhancer.start_focus_session(25)

# Run the example
asyncio.run(productivity_enhancer_example())
```

## Error Handling and Best Practices

### 11. Robust Error Handling

```python
import asyncio
import logging
from iceburg.physiological_interface import PhysiologicalAmplifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def robust_physiological_monitoring():
    """Robust physiological monitoring with error handling"""
    
    amplifier = PhysiologicalAmplifier()
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            # Attempt to get physiological data
            heart_rate = await amplifier.detect_heart_rate()
            stress_level = await amplifier.detect_stress_level()
            
            print(f"Heart rate: {heart_rate} BPM, Stress: {stress_level:.2f}")
            
            # Reset retry count on success
            retry_count = 0
            
            await asyncio.sleep(1)
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error in physiological monitoring (attempt {retry_count}): {e}")
            
            if retry_count >= max_retries:
                logger.error("Max retries reached. Stopping monitoring.")
                break
            
            # Wait before retry
            await asyncio.sleep(5)

# Run the example
asyncio.run(robust_physiological_monitoring())
```

### 12. Configuration Management

```python
import asyncio
import json
from iceburg.physiological_interface import PhysiologicalAmplifier

class PhysiologicalConfigManager:
    """Configuration manager for physiological interface"""
    
    def __init__(self, config_file: str = "physiological_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return {
                "monitoring": {
                    "enabled": True,
                    "interval": 1.0,
                    "sensors": ["heart_rate", "breathing", "stress"]
                },
                "earth_connection": {
                    "enabled": True,
                    "schumann_monitoring": True
                },
                "consciousness_amplification": {
                    "enabled": True,
                    "target_states": ["focused", "relaxed", "creative"]
                }
            }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
    
    async def run_with_config(self):
        """Run physiological monitoring with loaded configuration"""
        
        if not self.config["monitoring"]["enabled"]:
            print("Physiological monitoring is disabled in configuration")
            return
        
        amplifier = PhysiologicalAmplifier(self.config)
        
        try:
            # Run monitoring based on configuration
            interval = self.config["monitoring"]["interval"]
            
            for i in range(10):  # Run for 10 iterations
                heart_rate = await amplifier.detect_heart_rate()
                print(f"Reading {i+1}: Heart rate = {heart_rate} BPM")
                await asyncio.sleep(interval)
                
        except Exception as e:
            print(f"Error in configured monitoring: {e}")

# Usage example
async def config_manager_example():
    config_manager = PhysiologicalConfigManager()
    await config_manager.run_with_config()

# Run the example
asyncio.run(config_manager_example())
```

## Summary

These examples demonstrate how to integrate ICEBURG's Physiological Interface System with various applications, from basic monitoring to advanced consciousness amplification. The system provides:

- **Real-time physiological monitoring** (heart rate, stress, breathing)
- **Earth connection monitoring** (Schumann resonance, connection strength)
- **Consciousness amplification** (focused, relaxed, creative states)
- **Frequency synthesis** (audio frequencies, binaural beats)
- **ICEBURG integration** (physiological data with AI agents)

The examples show both basic usage and advanced integration patterns, with proper error handling and configuration management for production use.
