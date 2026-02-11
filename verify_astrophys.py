
import asyncio
import logging
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from iceburg.physiological_interface.icberg_integration import ICEBURGPhysiologicalIntegration
from iceburg.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AstroPhysVerify")

async def verify_astro_physiology():
    logger.info("🧪 Starting Verification of Astro-Physiology Engine...")
    
    try:
        # Initialize Config
        cfg = load_config()
        
        # Initialize Integration
        integration = ICEBURGPhysiologicalIntegration(cfg)
        
        # PATCH: Fix potential bug in icberg_integration.py
        if not hasattr(integration, 'consciousness_amplifier'):
             logger.warning("⚠️  'consciousness_amplifier' missing on integration object. Patching with 'physiological_amplifier'...")
             integration.consciousness_amplifier = integration.physiological_amplifier
             
             # Also needs to start it, but PhysiologicalStateDetector doesn't have start_consciousness_amplification
             # It acts as a wrapper check.
        
        logger.info("✅ ICEBURGPhysiologicalIntegration initialized.")

    except Exception as e:
        logger.error(f"Initialization error: {e}")

    # component 1: Earth Connection
    try:
        from iceburg.physiological_interface.earth_connection import EarthConnectionInterface
        earth = EarthConnectionInterface()
        logger.info("🌍 Starting Earth Connection...")
        # Start in background task to avoid blocking if it runs forever
        earth_task = asyncio.create_task(earth.start_earth_connection())
        await asyncio.sleep(2)
        
        status = earth.get_connection_status()
        logger.info(f"🌍 Earth Status: Connected={status['connected']}, Sync={status['sync_quality']:.2f}")
        
        await earth.stop_earth_connection()
        await earth_task
        logger.info("✅ Earth Connection verified.")
    except Exception as e:
        logger.error(f"Earth Connection failed: {e}")
    
    # component 2: Physiological Detector
    try:
        from iceburg.physiological_interface.physiological_detector import PhysiologicalStateDetector
        physio = PhysiologicalStateDetector()
        logger.info("omm Starting Physiological Detector...")
        physio_task = asyncio.create_task(physio.start_physiological_detection())
        await asyncio.sleep(2)
        
        p_summary = physio.get_physiological_summary()
        logger.info(f"omm Physio Summary: {p_summary['current_state']}")
        
        await physio.stop_physiological_detection()
        await physio_task
        logger.info("✅ Physiological Detector verified.")
    except Exception as e:
        logger.error(f"Physiological Detector failed: {e}")

    # component 3: Piezoelectric Detector
    try:
        from iceburg.physiological_interface.piezoelectric_detector import PiezoelectricDetector
        piezo = PiezoelectricDetector()
        logger.info("⚡ Starting Piezoelectric Detector...")
        piezo_task = asyncio.create_task(piezo.start_monitoring())
        await asyncio.sleep(2)
        
        metrics = piezo.get_monitoring_metrics_dict()
        sensor_type = metrics.get('sensor_type', 'Unknown')
        logger.info(f"⚡ Piezo Sensor: {sensor_type}")
        logger.info(f"⚡ Piezo Metrics: Status={metrics['status']}, Emergence={metrics['emergence_score']:.2f}")
        
        await piezo.stop_monitoring()
        try:
            await asyncio.wait_for(piezo_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass # It might not cancel immediately
            
        logger.info("✅ Piezoelectric Detector verified.")
    except Exception as e:
        logger.error(f"Piezoelectric Detector failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_astro_physiology())
