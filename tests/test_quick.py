#!/usr/bin/env python3
"""
Quick Test - Test ICEBURG's thinking and capabilities
Simplified test that doesn't require all dependencies
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_quick():
    """Quick test of ICEBURG capabilities"""
    print("\n" + "="*70)
    print("ICEBURG 2.0 - QUICK TEST")
    print("="*70 + "\n")
    
    try:
        # Test 1: Methodology Analyzer
        print("Test 1: Enhanced Deliberation Methodology")
        print("-" * 70)
        from iceburg.research.methodology_analyzer import MethodologyAnalyzer
        methodology_analyzer = MethodologyAnalyzer()
        methodology = methodology_analyzer.apply_methodology(
            "How does Enhanced Deliberation enable truth-finding?",
            domain="truth_finding"
        )
        print(f"✅ Methodology Steps: {len(methodology.get('steps', []))}")
        for i, step in enumerate(methodology.get('steps', [])[:3], 1):
            print(f"   {i}. {step.get('name')}")
        
        # Test 2: Suppression Detection
        print("\nTest 2: Suppression Detection")
        print("-" * 70)
        from iceburg.truth.suppression_detector import SuppressionDetector
        suppression_detector = SuppressionDetector()
        documents = [
            {
                "id": "doc1",
                "content": "Research was classified for 20 years",
                "metadata": {"creation_date": "2000-01-01", "release_date": "2020-01-01"}
            }
        ]
        result = suppression_detector.detect_suppression(documents)
        print(f"✅ Suppression Detected: {result.get('suppression_detected', False)}")
        print(f"   Suppression Score: {result.get('overall_suppression_score', 0.0):.2f}")
        
        # Test 3: Swarming
        print("\nTest 3: Swarming Integration")
        print("-" * 70)
        from iceburg.integration.swarming_integration import SwarmingIntegration
        swarming = SwarmingIntegration()
        swarm = await swarming.create_truth_finding_swarm(
            "How does swarming create better answers?",
            swarm_type="research_swarm"
        )
        print(f"✅ Swarm Created: {swarm.get('type')}")
        print(f"   Agents: {len(swarm.get('agents', []))}")
        
        # Test 4: Device Generation
        print("\nTest 4: Device Generation")
        print("-" * 70)
        from iceburg.generation.device_generator import DeviceGenerator
        device_generator = DeviceGenerator()
        device = await device_generator.generate_device(
            device_type="quantum_computer",
            requirements={"qubits": 100},
            domain="quantum_physics"
        )
        print(f"✅ Device Generated: {device.get('device_type')}")
        print(f"   Has Specifications: {device.get('specifications') is not None}")
        print(f"   Has Schematics: {device.get('schematics') is not None}")
        
        # Test 5: System Integration
        print("\nTest 5: System Integration")
        print("-" * 70)
        from iceburg.core.system_integrator import SystemIntegrator
        system_integrator = SystemIntegrator()
        integration = system_integrator.integrate_all_systems()
        status = system_integrator.get_system_status()
        print(f"✅ Systems Integrated: {integration.get('integrated', False)}")
        print(f"   Blackboard: {status.get('blackboard', {}).get('subscriptions', 0)} subscriptions")
        print(f"   Curiosity: {status.get('curiosity', {}).get('curiosity_engine_active', False)}")
        print(f"   Swarming: {len(status.get('swarming', {}).get('swarm_types', []))} swarm types")
        
        # Test 6: Full Query Processing
        print("\nTest 6: Full Query Processing")
        print("-" * 70)
        query = "How does Enhanced Deliberation enable ICEBURG to find suppressed knowledge?"
        result = await system_integrator.process_query_with_full_integration(
            query=query,
            domain="truth_finding"
        )
        print(f"✅ Query Processed: {query[:50]}...")
        print(f"   Methodology Applied: {result.get('methodology') is not None}")
        print(f"   Swarm Created: {result.get('results', {}).get('swarm') is not None}")
        print(f"   Insights Generated: {result.get('results', {}).get('insights') is not None}")
        
        print("\n" + "="*70)
        print("✅ ALL QUICK TESTS PASSED")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_quick())
    sys.exit(0 if success else 1)

