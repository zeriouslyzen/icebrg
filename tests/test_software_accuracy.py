#!/usr/bin/env python3
"""
Software Accuracy Test
Tests software generation for accuracy and correctness
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.generation.device_generator import DeviceGenerator


async def test_software_accuracy():
    """Test software generation for accuracy"""
    print("\n" + "="*70)
    print("ICEBURG 2.0 - SOFTWARE ACCURACY TEST")
    print("="*70 + "\n")
    
    device_generator = DeviceGenerator()
    
    results = {
        "code_generation_tests": [],
        "specification_tests": [],
        "validation_tests": [],
        "accuracy_scores": {}
    }
    
    # Test 1: Code Generation Accuracy
    print("1. CODE GENERATION ACCURACY")
    print("-" * 70)
    
    code_tests = [
        {
            "type": "quantum_algorithm",
            "requirements": {"qubits": 10, "gates": 20},
            "expected_features": ["quantum_circuit", "gates", "measurement"]
        },
        {
            "type": "optimization_algorithm",
            "requirements": {"variables": 100, "constraints": 50},
            "expected_features": ["optimization", "constraints", "objective"]
        },
    ]
    
    for test in code_tests:
        try:
            device = await device_generator.generate_device(
                device_type=test["type"],
                requirements=test["requirements"],
                domain="software"
            )
            
            code = device.get("code", "")
            has_code = bool(code)
            
            # Check if code contains expected features
            expected_features = test.get("expected_features", [])
            features_found = sum(1 for feature in expected_features if feature.lower() in code.lower())
            feature_accuracy = features_found / len(expected_features) if expected_features else 0.0
            
            accuracy = (has_code + feature_accuracy) / 2.0
            
            results["code_generation_tests"].append({
                "type": test["type"],
                "accuracy": accuracy,
                "has_code": has_code,
                "code_length": len(code),
                "features_found": features_found,
                "expected_features": len(expected_features),
                "feature_accuracy": feature_accuracy
            })
            
            print(f"  ✅ {test['type']}")
            print(f"     Has Code: {has_code}")
            print(f"     Code Length: {len(code)} characters")
            print(f"     Features Found: {features_found}/{len(expected_features)}")
            print(f"     Accuracy: {accuracy*100:.1f}%")
        except Exception as e:
            print(f"  ❌ {test['type']} (FAILED: {e})")
            results["code_generation_tests"].append({
                "type": test["type"],
                "accuracy": 0.0,
                "error": str(e)
            })
    
    # Test 2: Specification Accuracy
    print("\n2. SPECIFICATION ACCURACY")
    print("-" * 70)
    
    spec_tests = [
        {
            "type": "quantum_computer",
            "requirements": {"qubits": 100, "fidelity": 0.99},
            "expected_specs": ["qubits", "fidelity", "error_rate"]
        },
    ]
    
    for test in spec_tests:
        try:
            device = await device_generator.generate_device(
                device_type=test["type"],
                requirements=test["requirements"],
                domain="quantum_physics"
            )
            
            specs = device.get("specifications", {})
            has_specs = bool(specs)
            
            # Check if specs contain expected fields
            expected_specs = test.get("expected_specs", [])
            specs_found = sum(1 for spec in expected_specs if spec in str(specs).lower())
            spec_accuracy = specs_found / len(expected_specs) if expected_specs else 0.0
            
            accuracy = (has_specs + spec_accuracy) / 2.0
            
            results["specification_tests"].append({
                "type": test["type"],
                "accuracy": accuracy,
                "has_specs": has_specs,
                "specs_found": specs_found,
                "expected_specs": len(expected_specs),
                "spec_accuracy": spec_accuracy
            })
            
            print(f"  ✅ {test['type']}")
            print(f"     Has Specifications: {has_specs}")
            print(f"     Specs Found: {specs_found}/{len(expected_specs)}")
            print(f"     Accuracy: {accuracy*100:.1f}%")
        except Exception as e:
            print(f"  ❌ {test['type']} (FAILED: {e})")
            results["specification_tests"].append({
                "type": test["type"],
                "accuracy": 0.0,
                "error": str(e)
            })
    
    # Test 3: Validation Accuracy
    print("\n3. VALIDATION ACCURACY")
    print("-" * 70)
    
    validation_tests = [
        {
            "type": "quantum_computer",
            "requirements": {"qubits": 100}
        },
    ]
    
    for test in validation_tests:
        try:
            device = await device_generator.generate_device(
                device_type=test["type"],
                requirements=test["requirements"],
                domain="quantum_physics"
            )
            
            validated = device.get("validated", False)
            has_validation = validated is not False
            
            accuracy = 1.0 if has_validation else 0.0
            
            results["validation_tests"].append({
                "type": test["type"],
                "accuracy": accuracy,
                "validated": validated
            })
            
            print(f"  ✅ {test['type']}")
            print(f"     Validated: {validated}")
            print(f"     Accuracy: {accuracy*100:.1f}%")
        except Exception as e:
            print(f"  ❌ {test['type']} (FAILED: {e})")
            results["validation_tests"].append({
                "type": test["type"],
                "accuracy": 0.0,
                "error": str(e)
            })
    
    # Calculate overall accuracy
    print("\n4. OVERALL SOFTWARE ACCURACY")
    print("-" * 70)
    
    code_accuracy = sum(t.get("accuracy", 0) for t in results["code_generation_tests"]) / len(results["code_generation_tests"]) if results["code_generation_tests"] else 0.0
    spec_accuracy = sum(t.get("accuracy", 0) for t in results["specification_tests"]) / len(results["specification_tests"]) if results["specification_tests"] else 0.0
    validation_accuracy = sum(t.get("accuracy", 0) for t in results["validation_tests"]) / len(results["validation_tests"]) if results["validation_tests"] else 0.0
    
    overall_accuracy = (code_accuracy + spec_accuracy + validation_accuracy) / 3.0
    
    results["accuracy_scores"] = {
        "code_accuracy": code_accuracy,
        "spec_accuracy": spec_accuracy,
        "validation_accuracy": validation_accuracy,
        "overall_accuracy": overall_accuracy
    }
    
    print(f"Code Generation Accuracy: {code_accuracy*100:.1f}%")
    print(f"Specification Accuracy: {spec_accuracy*100:.1f}%")
    print(f"Validation Accuracy: {validation_accuracy*100:.1f}%")
    print(f"Overall Software Accuracy: {overall_accuracy*100:.1f}%")
    
    print("\n" + "="*70)
    print("SOFTWARE ACCURACY TEST COMPLETE")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_software_accuracy())


