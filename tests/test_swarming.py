#!/usr/bin/env python3
"""
Test Swarming Capabilities
Tests how swarming creates better answers
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.integration.swarming_integration import SwarmingIntegration


async def test_swarming():
    """Test swarming capabilities"""
    print("\n" + "="*70)
    print("SWARMING TEST")
    print("="*70 + "\n")
    
    swarming = SwarmingIntegration()
    
    # Test 1: Research Swarm
    print("Test 1: Research Swarm")
    print("-" * 70)
    
    query1 = "How does swarming create better answers than single agents?"
    swarm1 = await swarming.create_truth_finding_swarm(
        query=query1,
        swarm_type="research_swarm"
    )
    
    result1 = await swarming.execute_swarm(swarm1, parallel=True)
    
    print(f"Query: {query1}")
    print(f"Swarm Type: {swarm1.get('type')}")
    print(f"Agents: {len(swarm1.get('agents', []))}")
    print(f"Agent Results: {len(result1.get('agent_results', []))}")
    print(f"Synthesized: {'✅' if result1.get('synthesized_result') else '❌'}")
    print(f"Diversity Score: {result1.get('synthesized_result', {}).get('diversity_score', 0):.2f}")
    print(f"Consensus Score: {result1.get('synthesized_result', {}).get('consensus_score', 0):.2f}")
    
    # Test 2: Archaeology Swarm
    print("\n\nTest 2: Archaeology Swarm (Suppressed Information)")
    print("-" * 70)
    
    query2 = "What suppressed knowledge exists about quantum consciousness?"
    swarm2 = await swarming.create_truth_finding_swarm(
        query=query2,
        swarm_type="archaeology_swarm"
    )
    
    result2 = await swarming.execute_swarm(swarm2, parallel=True)
    
    print(f"Query: {query2}")
    print(f"Swarm Type: {swarm2.get('type')}")
    print(f"Agents: {len(swarm2.get('agents', []))}")
    print(f"Agent Results: {len(result2.get('agent_results', []))}")
    print(f"Synthesized: {'✅' if result2.get('synthesized_result') else '❌'}")
    
    # Test 3: Contradiction Swarm
    print("\n\nTest 3: Contradiction Swarm")
    print("-" * 70)
    
    query3 = "What contradictions exist in narratives about AI development timelines?"
    swarm3 = await swarming.create_truth_finding_swarm(
        query=query3,
        swarm_type="contradiction_swarm"
    )
    
    result3 = await swarming.execute_swarm(swarm3, parallel=True)
    
    print(f"Query: {query3}")
    print(f"Swarm Type: {swarm3.get('type')}")
    print(f"Agents: {len(swarm3.get('agents', []))}")
    print(f"Agent Results: {len(result3.get('agent_results', []))}")
    print(f"Synthesized: {'✅' if result3.get('synthesized_result') else '❌'}")
    
    # Compare single vs swarm
    print("\n\nComparison: Single Agent vs Swarm")
    print("-" * 70)
    print("Single Agent:")
    print("  - Single perspective")
    print("  - Limited error correction")
    print("  - No cross-validation")
    print("\nSwarm:")
    print(f"  - Multiple perspectives: {len(swarm1.get('agents', []))} agents")
    print(f"  - Diversity score: {result1.get('synthesized_result', {}).get('diversity_score', 0):.2f}")
    print(f"  - Consensus score: {result1.get('synthesized_result', {}).get('consensus_score', 0):.2f}")
    print("  - Cross-validation enabled")
    print("  - Error correction through consensus")
    
    print("\n" + "="*70)
    print("SWARMING TEST COMPLETE")
    print("="*70 + "\n")
    
    return {
        "research_swarm": result1.get('success', False),
        "archaeology_swarm": result2.get('success', False),
        "contradiction_swarm": result3.get('success', False),
        "total_swarms": 3,
        "diversity_score": result1.get('synthesized_result', {}).get('diversity_score', 0)
    }


if __name__ == "__main__":
    asyncio.run(test_swarming())

