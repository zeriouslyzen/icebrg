#!/usr/bin/env python3
"""
Test specification generation with real performance data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iceburg.monitoring.unified_performance_tracker import UnifiedPerformanceTracker
from iceburg.evolution.specification_generator import SpecificationGenerator

def test_specification_generation():
    print("🔍 Testing specification generation...")
    
    # Create tracker and get real performance data
    tracker = UnifiedPerformanceTracker()
    performance_summary = tracker.get_performance_summary(hours=1)
    
    print(f"Performance summary keys: {list(performance_summary.keys())}")
    print(f"Performance summary: {performance_summary}")
    
    # Create specification generator
    spec_generator = SpecificationGenerator()
    
    # Test with real performance data
    print("\nTesting with real performance data...")
    specs = spec_generator.generate_improvement_specifications(performance_summary)
    print(f"Generated {len(specs)} specifications from real data")
    
    # Test with mock data that should work
    print("\nTesting with mock data...")
    mock_data = {
        "response_time": 8.5,
        "accuracy": 0.85,
        "memory_usage": 150.0,
        "cache_hit_rate": 0.4,
        "error_rate": 0.05,
        "success_rate": 0.95
    }
    specs2 = spec_generator.generate_improvement_specifications(mock_data)
    print(f"Generated {len(specs2)} specifications from mock data")
    
    # Test with averages data
    if "averages" in performance_summary:
        print("\nTesting with averages data...")
        specs3 = spec_generator.generate_improvement_specifications(performance_summary["averages"])
        print(f"Generated {len(specs3)} specifications from averages data")
        
        for i, spec in enumerate(specs3):
            print(f"  {i+1}. {spec.name}: {spec.description}")

if __name__ == "__main__":
    test_specification_generation()
