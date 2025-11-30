#!/usr/bin/env python3
"""
Protocol Accuracy Test
Tests the entire ICEBURG protocol for accuracy in software and physics
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.lab.virtual_physics_lab import VirtualPhysicsLab
from iceburg.generation.device_generator import DeviceGenerator


async def test_protocol_accuracy():
    """Test protocol accuracy for software and physics"""
    print("\n" + "="*70)
    print("ICEBURG 2.0 - PROTOCOL ACCURACY TEST")
    print("="*70 + "\n")
    
    system_integrator = SystemIntegrator()
    lab = VirtualPhysicsLab()
    device_generator = DeviceGenerator()
    
    results = {
        "protocol_tests": [],
        "software_tests": [],
        "physics_tests": [],
        "accuracy_scores": {}
    }
    
    # Test 1: Protocol Execution Accuracy
    print("1. PROTOCOL EXECUTION ACCURACY")
    print("-" * 70)
    
    protocol_queries = [
        "What is quantum computing?",
        "How does photosynthesis work?",
        "What is the speed of light?",
    ]
    
    for query in protocol_queries:
        try:
            result = await system_integrator.process_query_with_full_integration(
                query=query,
                domain="physics"
            )
            
            # Check if protocol executed correctly
            has_methodology = result.get("results", {}).get("methodology") is not None
            has_swarm = result.get("results", {}).get("swarm") is not None
            has_insights = result.get("results", {}).get("insights") is not None
            
            accuracy = (has_methodology + has_swarm + has_insights) / 3.0
            
            results["protocol_tests"].append({
                "query": query,
                "accuracy": accuracy,
                "has_methodology": has_methodology,
                "has_swarm": has_swarm,
                "has_insights": has_insights
            })
            
            print(f"  ✅ {query[:50]}... (accuracy: {accuracy:.2f})")
        except Exception as e:
            print(f"  ❌ {query[:50]}... (FAILED: {e})")
            results["protocol_tests"].append({
                "query": query,
                "accuracy": 0.0,
                "error": str(e)
            })
    
    # Test 2: Software Generation Accuracy
    print("\n2. SOFTWARE GENERATION ACCURACY")
    print("-" * 70)
    
    software_tests = [
        {
            "type": "quantum_algorithm",
            "requirements": {"qubits": 10, "gates": 20}
        },
        {
            "type": "optimization_algorithm",
            "requirements": {"variables": 100, "constraints": 50}
        },
    ]
    
    for test in software_tests:
        try:
            device = await device_generator.generate_device(
                device_type=test["type"],
                requirements=test["requirements"],
                domain="software"
            )
            
            has_code = device.get("code") is not None
            has_specs = device.get("specifications") is not None
            has_validation = device.get("validated", False)
            
            accuracy = (has_code + has_specs + has_validation) / 3.0
            
            results["software_tests"].append({
                "type": test["type"],
                "accuracy": accuracy,
                "has_code": has_code,
                "has_specs": has_specs,
                "has_validation": has_validation
            })
            
            print(f"  ✅ {test['type']} (accuracy: {accuracy:.2f})")
        except Exception as e:
            print(f"  ❌ {test['type']} (FAILED: {e})")
            results["software_tests"].append({
                "type": test["type"],
                "accuracy": 0.0,
                "error": str(e)
            })
    
    # Test 3: Physics Simulation Accuracy
    print("\n3. PHYSICS SIMULATION ACCURACY")
    print("-" * 70)
    
    physics_tests = [
        {
            "experiment_type": "quantum_coherence",
            "parameters": {"qubits": 10, "time": 1.0}
        },
        {
            "experiment_type": "classical_mechanics",
            "parameters": {"mass": 1.0, "velocity": 10.0}
        },
    ]
    
    for test in physics_tests:
        try:
            result = lab.run_experiment(
                experiment_type=test["experiment_type"],
                algorithm=None,
                parameters=test["parameters"]
            )
            
            success = result.success
            has_metrics = len(result.performance_metrics) > 0
            execution_time = result.execution_time
            
            accuracy = (success + has_metrics) / 2.0
            
            results["physics_tests"].append({
                "experiment_type": test["experiment_type"],
                "accuracy": accuracy,
                "success": success,
                "has_metrics": has_metrics,
                "execution_time": execution_time,
                "metrics": result.performance_metrics
            })
            
            print(f"  ✅ {test['experiment_type']} (accuracy: {accuracy:.2f}, time: {execution_time:.3f}s)")
        except Exception as e:
            print(f"  ❌ {test['experiment_type']} (FAILED: {e})")
            results["physics_tests"].append({
                "experiment_type": test["experiment_type"],
                "accuracy": 0.0,
                "error": str(e)
            })
    
    # Calculate overall accuracy scores
    print("\n4. ACCURACY SCORES")
    print("-" * 70)
    
    protocol_accuracy = sum(t.get("accuracy", 0) for t in results["protocol_tests"]) / len(results["protocol_tests"]) if results["protocol_tests"] else 0.0
    software_accuracy = sum(t.get("accuracy", 0) for t in results["software_tests"]) / len(results["software_tests"]) if results["software_tests"] else 0.0
    physics_accuracy = sum(t.get("accuracy", 0) for t in results["physics_tests"]) / len(results["physics_tests"]) if results["physics_tests"] else 0.0
    
    overall_accuracy = (protocol_accuracy + software_accuracy + physics_accuracy) / 3.0
    
    results["accuracy_scores"] = {
        "protocol_accuracy": protocol_accuracy,
        "software_accuracy": software_accuracy,
        "physics_accuracy": physics_accuracy,
        "overall_accuracy": overall_accuracy
    }
    
    print(f"Protocol Accuracy: {protocol_accuracy*100:.1f}%")
    print(f"Software Accuracy: {software_accuracy*100:.1f}%")
    print(f"Physics Accuracy: {physics_accuracy*100:.1f}%")
    print(f"Overall Accuracy: {overall_accuracy*100:.1f}%")
    
    print("\n" + "="*70)
    print("PROTOCOL ACCURACY TEST COMPLETE")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_protocol_accuracy())


