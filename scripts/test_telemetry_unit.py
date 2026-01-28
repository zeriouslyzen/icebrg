
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from iceburg.telemetry.advanced_telemetry import AdvancedTelemetry, PromptMetrics

def test_telemetry():
    print("🚀 Testing Telemetry...")
    
    # Initialize
    telemetry = AdvancedTelemetry()  # Use default path
    print(f"📂 Data dir: {telemetry.data_dir.absolute()}")
    
    # Create metric
    metric = PromptMetrics(
        prompt_id="test-123",
        prompt_text="Test prompt",
        response_time=0.5,
        token_count=100,
        model_used="test-model",
        success=True,
        quality_score=1.0
    )
    
    # Track
    telemetry.track_prompt(metric)
    print("✅ Tracked prompt.")
    
    # Check if file exists
    expected_file = telemetry.data_dir / "prompt_metrics.jsonl"
    if expected_file.exists():
        print(f"✅ File created: {expected_file}")
        print(f"📝 Content: {expected_file.read_text()}")
    else:
        print(f"❌ File NOT found: {expected_file}")

if __name__ == "__main__":
    test_telemetry()
