#!/usr/bin/env python3
"""
Test Device Generation
Tests general-purpose device generation capabilities
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.generation.device_generator import DeviceGenerator
from iceburg.research.methodology_analyzer import MethodologyAnalyzer


async def test_device_generation():
    """Test device generation"""
    print("\n" + "="*70)
    print("DEVICE GENERATION TEST")
    print("="*70 + "\n")
    
    device_generator = DeviceGenerator()
    methodology_analyzer = MethodologyAnalyzer()
    
    # Test 1: Quantum Computer
    print("Test 1: Quantum Computer Generation")
    print("-" * 70)
    
    quantum_device = await device_generator.generate_device(
        device_type="quantum_computer",
        requirements={
            "qubits": 100,
            "fidelity": 0.99,
            "dimensions": {"width": 10, "height": 10, "depth": 10},
            "functions": ["quantum_computation", "error_correction"],
            "materials": ["superconducting_qubits", "cryogenic_system"]
        },
        domain="quantum_physics"
    )
    
    print(f"Device Type: {quantum_device.get('device_type')}")
    print(f"Specifications: {'✅' if quantum_device.get('specifications') else '❌'}")
    print(f"Schematics: {'✅' if quantum_device.get('schematics') else '❌'}")
    print(f"Code: {'✅' if quantum_device.get('code') else '❌'}")
    print(f"BOM: {'✅' if quantum_device.get('bom') else '❌'}")
    print(f"Assembly Instructions: {'✅' if quantum_device.get('assembly_instructions') else '❌'}")
    print(f"Validated: {'✅' if quantum_device.get('validated') else '❌'}")
    
    # Test 2: Energy Device (using past research)
    print("\n\nTest 2: Energy Device Generation (Using Past Research)")
    print("-" * 70)
    
    energy_device = await device_generator.generate_device(
        device_type="free_energy_device",
        requirements={
            "power_output": 1000,
            "efficiency": 0.95,
            "dimensions": {"width": 5, "height": 5, "depth": 5},
            "functions": ["energy_generation", "storage"],
            "materials": ["piezoelectric_materials", "capacitors"]
        },
        domain="energy_physics"
    )
    
    print(f"Device Type: {energy_device.get('device_type')}")
    print(f"Specifications: {'✅' if energy_device.get('specifications') else '❌'}")
    print(f"Schematics: {'✅' if energy_device.get('schematics') else '❌'}")
    print(f"Code: {'✅' if energy_device.get('code') else '❌'}")
    print(f"BOM: {'✅' if energy_device.get('bom') else '❌'}")
    
    # Test 3: Medical Device (pancreatic cancer research)
    print("\n\nTest 3: Medical Device Generation (Pancreatic Cancer Research)")
    print("-" * 70)
    
    medical_device = await device_generator.generate_device(
        device_type="bioelectric_therapy_device",
        requirements={
            "frequency_range": "1-100 Hz",
            "power": 10,
            "dimensions": {"width": 3, "height": 3, "depth": 2},
            "functions": ["bioelectric_stimulation", "monitoring"],
            "materials": ["electrodes", "microcontroller", "battery"]
        },
        domain="medical_devices"
    )
    
    print(f"Device Type: {medical_device.get('device_type')}")
    print(f"Specifications: {'✅' if medical_device.get('specifications') else '❌'}")
    print(f"Schematics: {'✅' if medical_device.get('schematics') else '❌'}")
    print(f"Code: {'✅' if medical_device.get('code') else '❌'}")
    
    # Show methodology application
    print("\n\nMethodology Applied:")
    methodology = methodology_analyzer.apply_methodology(
        "Generate bioelectric therapy device",
        domain="medical_devices"
    )
    print(f"Steps: {len(methodology.get('steps', []))}")
    for i, step in enumerate(methodology.get('steps', [])[:3], 1):
        print(f"  {i}. {step.get('name')}")
    
    print("\n" + "="*70)
    print("DEVICE GENERATION TEST COMPLETE")
    print("="*70 + "\n")
    
    return {
        "quantum_device": quantum_device.get('device_type') is not None,
        "energy_device": energy_device.get('device_type') is not None,
        "medical_device": medical_device.get('device_type') is not None,
        "total_devices": 3
    }


if __name__ == "__main__":
    asyncio.run(test_device_generation())

