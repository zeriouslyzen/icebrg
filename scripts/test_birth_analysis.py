#!/usr/bin/env python3
"""
Test script for birth analysis with visualization.
Tests the complete physiology-celestial-chemistry system.
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from iceburg.config import load_config
from iceburg.agents.celestial_biological_framework import (
    analyze_birth_imprint,
    predict_behavioral_traits,
    get_tcm_health_predictions,
    get_current_celestial_conditions
)

# Dallas, Texas coordinates (Parkland Hospital)
DALLAS_COORDS = (32.8130, -96.8353)  # Latitude, Longitude

async def test_birth_analysis():
    """Test complete birth analysis system"""
    
    print("=" * 80)
    print("ICEBURG BIRTH ANALYSIS TEST")
    print("=" * 80)
    print()
    
    # Birth data
    birth_date = datetime(1991, 12, 26, 7, 20, 0, tzinfo=timezone.utc)
    location = DALLAS_COORDS
    
    print(f"📅 Birth Date: {birth_date.strftime('%B %d, %Y at %I:%M %p')}")
    print(f"📍 Location: Parkland Hospital, Dallas, TX ({location[0]}, {location[1]})")
    print()
    
    # Current date
    current_date = datetime(2025, 11, 26, tzinfo=timezone.utc)
    print(f"🗓️  Analysis Date: {current_date.strftime('%B %d, %Y')}")
    print()
    
    # Load config
    cfg = load_config()
    
    print("🔬 Step 1: Analyzing Birth Imprint...")
    print("-" * 80)
    
    try:
        # Analyze birth imprint
        molecular_imprint = await analyze_birth_imprint(birth_date, location)
        
        print("✅ Birth Imprint Analysis Complete")
        print()
        print("Molecular Imprint Data:")
        print(f"  - Birth DateTime: {molecular_imprint.birth_datetime}")
        print(f"  - Celestial Positions: {len(molecular_imprint.celestial_positions)} bodies")
        print(f"  - EM Environment: {len(molecular_imprint.electromagnetic_environment)} parameters")
        print(f"  - Molecular Configurations: {len(molecular_imprint.molecular_configurations)} types")
        print(f"  - Cellular Dependencies: {len(molecular_imprint.cellular_dependencies)} dependencies")
        print(f"  - Trait Amplifications: {len(molecular_imprint.trait_amplification_factors)} traits")
        print()
        
        # Get current celestial conditions
        print("🌌 Step 2: Getting Current Celestial Conditions...")
        print("-" * 80)
        
        current_conditions = await get_current_celestial_conditions(location)
        
        print("✅ Current Conditions Retrieved")
        print(f"  - Timestamp: {current_conditions.get('timestamp')}")
        print(f"  - Celestial Positions: {len(current_conditions.get('celestial_positions', {}))} bodies")
        print(f"  - EM Environment: {len(current_conditions.get('electromagnetic_environment', {}))} parameters")
        print(f"  - Earth Frequencies: {len(current_conditions.get('earth_frequencies', {}))} frequencies")
        print()
        
        # Predict behavioral traits
        print("🧠 Step 3: Predicting Behavioral Traits...")
        print("-" * 80)
        
        behavioral_predictions = await predict_behavioral_traits(
            molecular_imprint,
            current_conditions
        )
        
        print("✅ Behavioral Predictions Generated")
        print()
        print("Behavioral Traits:")
        for trait, value in behavioral_predictions.items():
            print(f"  - {trait.replace('_', ' ').title()}: {value:.3f}")
        print()
        
        # Get TCM predictions
        print("🏥 Step 4: Getting TCM Health Predictions...")
        print("-" * 80)
        
        tcm_predictions = await get_tcm_health_predictions(molecular_imprint)
        
        print("✅ TCM Predictions Generated")
        print()
        print("Organ System Correlations:")
        for planet, data in tcm_predictions.items():
            print(f"  - {planet.upper()}:")
            print(f"    Organ: {data.get('organ', 'N/A')}")
            print(f"    Element: {data.get('element', 'N/A')}")
            print(f"    Emotion: {data.get('emotion', 'N/A')}")
            print(f"    Strength: {data.get('strength', 0):.3f}")
            print()
        
        # Calculate age and time since birth
        age_years = (current_date - birth_date).days / 365.25
        print(f"⏰ Age: {age_years:.1f} years")
        print()
        
        # Prepare data for visualization
        visualization_data = {
            "birth_data": {
                "date": birth_date.isoformat(),
                "location": location,
                "age_years": age_years
            },
            "molecular_imprint": {
                "celestial_positions": {
                    k: {
                        "ra": v[0],
                        "dec": v[1],
                        "distance": v[2]
                    } for k, v in molecular_imprint.celestial_positions.items()
                },
                "electromagnetic_environment": molecular_imprint.electromagnetic_environment,
                "trait_amplification_factors": molecular_imprint.trait_amplification_factors
            },
            "current_conditions": current_conditions,
            "behavioral_predictions": behavioral_predictions,
            "tcm_predictions": tcm_predictions,
            "analysis_date": current_date.isoformat()
        }
        
        # Save data for visualization
        output_file = Path("data/birth_analysis_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(visualization_data, f, indent=2, default=str)
        
        print("💾 Data saved to: data/birth_analysis_results.json")
        print()
        
        return visualization_data
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_birth_analysis())
    if result:
        print("=" * 80)
        print("✅ TEST COMPLETE - Ready for visualization")
        print("=" * 80)

