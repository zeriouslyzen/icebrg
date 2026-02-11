
import sys
import logging
from datetime import datetime
sys.path.append("/Users/jackdanger/Desktop/Projects/iceburg/src")

from iceburg.astro_physiology.ingestion import AstroPhysiologyIngestor

# Setup logging to see the planetary output
logging.basicConfig(level=logging.INFO)

def test_planetary_snapshot():
    print("🔭 TESTING PLANETARY HARMONICS INTEGRATION...")
    ingestor = AstroPhysiologyIngestor()
    
    # We test with a dummy user ID
    user_id = "test_user_planetary"
    
    try:
        snapshot = ingestor.fetch_current_snapshot(user_id)
        
        print("\n--- PLANETARY HARMONICS SNAPSHOT ---")
        print(f"Timestamp: {snapshot.timestamp}")
        print(f"Planetary Resonance Index: {snapshot.celestial_signal.planetary_resonance_index:.4f}")
        print(f"Solar Wind Speed: {snapshot.celestial_signal.solar_wind_speed} km/s")
        print(f"Net Coherence Score: {snapshot.net_coherence_score:.4f}")
        
        # Basic Assertions
        assert 0.0 <= snapshot.celestial_signal.planetary_resonance_index <= 1.0, "Resonance index should be between 0 and 1"
        assert snapshot.net_coherence_score >= 0.0, "Net Coherence should be non-negative"
        
        print("\n✅ Planetary Integration Verified!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_planetary_snapshot()
