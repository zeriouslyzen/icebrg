
import sys
import time
from datetime import datetime, timedelta
sys.path.append("/Users/jackdanger/Desktop/Projects/iceburg/src")

from iceburg.astro_physiology.models import PhysiologicalState, ScalarMetric
from iceburg.astro_physiology.bio_reactor import BioReactor, ReactorState

def mock_bio_state():
    return PhysiologicalState(
        timestamp=datetime.utcnow(),
        user_id="tester",
        hrv_rmssd=50.0,
        heart_rate=60.0,
        body_temperature=37.0,
        dominant_frequency=0.1 # coherent
    )

def test_reactor_cycle():
    print("🧪 STARTING ROBUST BIO-REACTOR TEST...")
    reactor = BioReactor()
    reactor.start_session()
    
    # 1. Verify IGNITION Start
    assert reactor.current_state == ReactorState.IGNITION, "Should start in IGNITION"
    print("✅ Cycle 1 Started: IGNITION")
    
    # 2. Fast Forward to End of Ignition (61s)
    # We cheat by moving start_time back
    reactor.state_start_time = datetime.utcnow() - timedelta(seconds=61)
    
    # Process Feedback to trigger transition
    instruction = reactor.process_bio_feedback(mock_bio_state())
    
    # 3. Verify Transition to RETENTION
    assert reactor.current_state == ReactorState.RETENTION, f"Should be RETENTION, got {reactor.current_state}"
    print("✅ Transitioned to RETENTION")
    
    # 4. Fast Forward to End of Retention (Target is 60s)
    # Note: Transition resets state_start_time to NOW.
    # So we move it back relative to NOW.
    reactor.state_start_time = datetime.utcnow() - timedelta(seconds=61)
    
    instruction = reactor.process_bio_feedback(mock_bio_state())
    
    # 5. Verify Transition to IMPLOSION
    assert reactor.current_state == ReactorState.IMPLOSION, f"Should be IMPLOSION, got {reactor.current_state}"
    print("✅ Transitioned to IMPLOSION")
    
    # 6. Fast Forward to End of Implosion (15s)
    reactor.state_start_time = datetime.utcnow() - timedelta(seconds=16)
    
    instruction = reactor.process_bio_feedback(mock_bio_state())
    
    # 7. Verify Transition to RECOVERY
    assert reactor.current_state == ReactorState.RECOVERY, f"Should be RECOVERY, got {reactor.current_state}"
    print("✅ Transitioned to RECOVERY")
    
    # 8. Fast Forward to End of Recovery (30s) -> Cycle 2
    reactor.state_start_time = datetime.utcnow() - timedelta(seconds=31)
    
    instruction = reactor.process_bio_feedback(mock_bio_state())
    
    # 9. Verify New Cycle (Back to IGNITION)
    assert reactor.current_state == ReactorState.IGNITION, f"Should be IGNITION (Cycle 2), got {reactor.current_state}"
    assert reactor.current_cycle == 2, f"Should be Cycle 2, got {reactor.current_cycle}"
    # Verify retention target increased (60 -> 90)
    assert reactor.retention_target == 90.0, f"Retention target should increase to 90.0, got {reactor.retention_target}"
    print("✅ Cycle 2 Started: Retention Target Increased")

    print("\n🎉 ALL TESTS PASSED!")

if __name__ == "__main__":
    try:
        test_reactor_cycle()
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        sys.exit(1)
