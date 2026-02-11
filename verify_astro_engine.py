"""
Verify Astro-Physiology Engine
==============================
Runs a live simulation of the Data Engine to verify:
1. Data Ingestion (Simulated)
2. Scalar Calculation (Phi Alignment)
3. Suppression Detection (Jamming)
4. Integration Logic (Unified Field)
"""

import asyncio
import sys
import os
import logging

# Configure Logging to see Ingestion Status
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from iceburg.physiological_interface.icberg_integration import ICEBURGPhysiologicalIntegration
from iceburg.config import load_config

async def run_verification():
    print("🔮 Initializing Astro-Physiology Engine...")
    cfg = load_config()
    engine = ICEBURGPhysiologicalIntegration(cfg)
    
    await engine.start_icberg_consciousness_integration()
    
    print("\n📡 Listening for Coherence Snapshots...")
    print("-" * 50)
    
    for i in range(5):
        try:
            await asyncio.sleep(1)
            status = engine.get_unified_consciousness_status()
            
            if status is None or status.get('status') == 'INITIALIZING':
                print(f"[{i}] Initializing...")
                continue
                
            if 'telemetry' not in status:
                print(f"[{i}] No telemetry available yet...")
                continue
                
            telemetry = status['telemetry']
            
            print(f"\n[{i}] SNAPSHOT TIMESTAMP: {status['timestamp']}")
            print(f"   > NET UNIFIED FIELD (COHERENCE): {status['unified_field_strength']:.4f}")
            
            # Celestial
            cel = telemetry.get('celestial', {})
            print(f"   [CELESTIAL SIGNAL]")
            print(f"     Solar Wind: {cel.get('solar_wind', 0):.1f} km/s")
            print(f"     Scalar Potential: {cel.get('scalar_potential', 0):.4f}")
            print(f"     Planetary Resonance: {cel.get('planetary_resonance', 0):.4f}")
            
            # Biological
            bio = telemetry.get('biological', {})
            print(f"   [BIO RECEIVER]")
            print(f"     Phi Alignment: {bio.get('phi_alignment', 0):.4f} (Negentropic: {bio.get('is_negentropic', False)})")
            print(f"     DNA Resonance: {bio.get('dna_resonance', 0):.4f}")
            
            # Suppression
            sup = telemetry.get('suppression', {})
            print(f"   [SUPPRESSION / JAMMING]")
            print(f"     HAARP Index: {sup.get('haarp_index', 0):.4f}")
            print(f"     Jamming Power: {sup.get('jamming_power', 0):.4f}")
            
            jamming = sup.get('jamming_power', 0)
            unified = status.get('unified_field_strength', 0)
            
            if jamming > 0.5:
                print("   ⚠️  HEAVY JAMMING DETECTED")
            
            if unified > 0.6:
                print("   ✨  HIGH COHERENCE - SYSTEM LOCKED")
                
        except Exception as e:
            print(f"Error in loop: {e}")
            
    await engine.stop_icberg_consciousness_integration()
    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    asyncio.run(run_verification())
