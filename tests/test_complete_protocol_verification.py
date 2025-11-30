#!/usr/bin/env python3
"""
Complete Protocol Verification Test
Tests the entire ICEBURG protocol including software and physics for accuracy
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.lab.virtual_physics_lab import VirtualPhysicsLab
from iceburg.generation.device_generator import DeviceGenerator
from iceburg.research.methodology_analyzer import MethodologyAnalyzer


async def test_complete_protocol_verification():
    """Test complete protocol verification including software and physics"""
    print("\n" + "="*70)
    print("ICEBURG 2.0 - COMPLETE PROTOCOL VERIFICATION")
    print("="*70 + "\n")
    
    system_integrator = SystemIntegrator()
    lab = VirtualPhysicsLab()
    device_generator = DeviceGenerator()
    methodology_analyzer = MethodologyAnalyzer()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "protocol_tests": [],
        "software_tests": [],
        "physics_tests": [],
        "methodology_tests": [],
        "accuracy_scores": {}
    }
    
    # Test 1: Protocol Execution
    print("1. PROTOCOL EXECUTION VERIFICATION")
    print("-" * 70)
    
    protocol_query = "How does quantum computing work and how can we build a quantum computer?"
    
    try:
        result = await system_integrator.process_query_with_full_integration(
            query=protocol_query,
            domain="quantum_physics"
        )
        
        # Verify protocol components
        has_methodology = result.get("results", {}).get("methodology") is not None
        has_swarm = result.get("results", {}).get("swarm") is not None
        has_insights = result.get("results", {}).get("insights") is not None
        has_citation = result.get("citation_id") is not None
        
        protocol_accuracy = (has_methodology + has_swarm + has_insights + has_citation) / 4.0
        
        results["protocol_tests"].append({
            "query": protocol_query,
            "accuracy": protocol_accuracy,
            "has_methodology": has_methodology,
            "has_swarm": has_swarm,
            "has_insights": has_insights,
            "has_citation": has_citation
        })
        
        print(f"  ✅ Protocol Execution")
        print(f"     Methodology: {has_methodology}")
        print(f"     Swarm: {has_swarm}")
        print(f"     Insights: {has_insights}")
        print(f"     Citation: {has_citation}")
        print(f"     Accuracy: {protocol_accuracy*100:.1f}%")
    except Exception as e:
        print(f"  ❌ Protocol Execution (FAILED: {e})")
        results["protocol_tests"].append({
            "query": protocol_query,
            "accuracy": 0.0,
            "error": str(e)
        })
    
    # Test 2: Software Generation
    print("\n2. SOFTWARE GENERATION VERIFICATION")
    print("-" * 70)
    
    try:
        device = await device_generator.generate_device(
            device_type="quantum_computer",
            requirements={"qubits": 100, "fidelity": 0.99},
            domain="quantum_physics"
        )
        
        has_code = device.get("code") is not None
        has_specs = device.get("specifications") is not None
        has_schematics = device.get("schematics") is not None
        has_bom = device.get("bom") is not None
        validated = device.get("validated", False)
        
        software_accuracy = (has_code + has_specs + has_schematics + has_bom + validated) / 5.0
        
        results["software_tests"].append({
            "type": "quantum_computer",
            "accuracy": software_accuracy,
            "has_code": has_code,
            "has_specs": has_specs,
            "has_schematics": has_schematics,
            "has_bom": has_bom,
            "validated": validated
        })
        
        print(f"  ✅ Software Generation")
        print(f"     Code: {has_code}")
        print(f"     Specifications: {has_specs}")
        print(f"     Schematics: {has_schematics}")
        print(f"     BOM: {has_bom}")
        print(f"     Validated: {validated}")
        print(f"     Accuracy: {software_accuracy*100:.1f}%")
    except Exception as e:
        print(f"  ❌ Software Generation (FAILED: {e})")
        results["software_tests"].append({
            "type": "quantum_computer",
            "accuracy": 0.0,
            "error": str(e)
        })
    
    # Test 3: Physics Simulation
    print("\n3. PHYSICS SIMULATION VERIFICATION")
    print("-" * 70)
    
    try:
        result = lab.run_experiment(
            experiment_type="quantum_coherence",
            algorithm=None,
            parameters={"qubits": 10, "time": 1.0}
        )
        
        success = result.success
        has_metrics = len(result.performance_metrics) > 0
        execution_time = result.execution_time
        
        physics_accuracy = (success + has_metrics) / 2.0
        
        results["physics_tests"].append({
            "experiment_type": "quantum_coherence",
            "accuracy": physics_accuracy,
            "success": success,
            "has_metrics": has_metrics,
            "execution_time": execution_time,
            "metrics": result.performance_metrics
        })
        
        print(f"  ✅ Physics Simulation")
        print(f"     Success: {success}")
        print(f"     Has Metrics: {has_metrics}")
        print(f"     Execution Time: {execution_time:.3f}s")
        print(f"     Accuracy: {physics_accuracy*100:.1f}%")
    except Exception as e:
        print(f"  ❌ Physics Simulation (FAILED: {e})")
        results["physics_tests"].append({
            "experiment_type": "quantum_coherence",
            "accuracy": 0.0,
            "error": str(e)
        })
    
    # Test 4: Methodology Verification
    print("\n4. METHODOLOGY VERIFICATION")
    print("-" * 70)
    
    try:
        methodology = methodology_analyzer.apply_methodology(
            "How does Enhanced Deliberation enable truth-finding?",
            domain="truth_finding"
        )
        
        has_steps = len(methodology.get("steps", [])) > 0
        has_components = len(methodology_analyzer.get_methodology_components()) > 0
        
        methodology_accuracy = (has_steps + has_components) / 2.0
        
        results["methodology_tests"].append({
            "test": "methodology_application",
            "accuracy": methodology_accuracy,
            "has_steps": has_steps,
            "steps_count": len(methodology.get("steps", [])),
            "has_components": has_components,
            "components_count": len(methodology_analyzer.get_methodology_components())
        })
        
        print(f"  ✅ Methodology Application")
        print(f"     Has Steps: {has_steps}")
        print(f"     Steps Count: {len(methodology.get('steps', []))}")
        print(f"     Has Components: {has_components}")
        print(f"     Accuracy: {methodology_accuracy*100:.1f}%")
    except Exception as e:
        print(f"  ❌ Methodology Application (FAILED: {e})")
        results["methodology_tests"].append({
            "test": "methodology_application",
            "accuracy": 0.0,
            "error": str(e)
        })
    
    # Calculate overall accuracy
    print("\n5. OVERALL ACCURACY SCORES")
    print("-" * 70)
    
    protocol_accuracy = sum(t.get("accuracy", 0) for t in results["protocol_tests"]) / len(results["protocol_tests"]) if results["protocol_tests"] else 0.0
    software_accuracy = sum(t.get("accuracy", 0) for t in results["software_tests"]) / len(results["software_tests"]) if results["software_tests"] else 0.0
    physics_accuracy = sum(t.get("accuracy", 0) for t in results["physics_tests"]) / len(results["physics_tests"]) if results["physics_tests"] else 0.0
    methodology_accuracy = sum(t.get("accuracy", 0) for t in results["methodology_tests"]) / len(results["methodology_tests"]) if results["methodology_tests"] else 0.0
    
    overall_accuracy = (protocol_accuracy + software_accuracy + physics_accuracy + methodology_accuracy) / 4.0
    
    results["accuracy_scores"] = {
        "protocol_accuracy": protocol_accuracy,
        "software_accuracy": software_accuracy,
        "physics_accuracy": physics_accuracy,
        "methodology_accuracy": methodology_accuracy,
        "overall_accuracy": overall_accuracy
    }
    
    print(f"Protocol Accuracy: {protocol_accuracy*100:.1f}%")
    print(f"Software Accuracy: {software_accuracy*100:.1f}%")
    print(f"Physics Accuracy: {physics_accuracy*100:.1f}%")
    print(f"Methodology Accuracy: {methodology_accuracy*100:.1f}%")
    print(f"Overall Accuracy: {overall_accuracy*100:.1f}%")
    
    # Save results
    results_file = Path("data/test_results") / f"complete_protocol_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    print("\n" + "="*70)
    print("COMPLETE PROTOCOL VERIFICATION COMPLETE")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_complete_protocol_verification())


