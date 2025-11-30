#!/usr/bin/env python3
"""
Test Full System Integration
Tests ICEBURG's thinking and capabilities with a comprehensive query
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.formatting.response_formatter import ResponseFormatter


async def test_full_system():
    """Test full system with comprehensive query"""
    print("\n" + "="*70)
    print("FULL SYSTEM INTEGRATION TEST")
    print("="*70 + "\n")
    
    system_integrator = SystemIntegrator()
    response_formatter = ResponseFormatter()
    
    # Comprehensive query that tests all features
    query = """
    Using Enhanced Deliberation methodology, how can ICEBURG:
    1. Detect suppressed knowledge about quantum consciousness research
    2. Use swarming to create better answers about this topic
    3. Generate a device that leverages quantum consciousness principles
    4. Apply past research on pancreatic cancer and free energy
    5. Demonstrate how Enhanced Deliberation enables truth-finding
    """
    
    print("Query:")
    print(query)
    print("\n" + "="*70)
    print("PROCESSING WITH FULL SYSTEM INTEGRATION...")
    print("="*70 + "\n")
    
    # Process query with full integration
    result = await system_integrator.process_query_with_full_integration(
        query=query,
        domain="quantum_consciousness"
    )
    
    # Format response
    formatted = response_formatter.format_from_analysis(result.get("results", {}))
    
    print("RESULTS:")
    print("-" * 70)
    
    # Show methodology
    methodology = result.get("results", {}).get("methodology", {})
    if methodology:
        print(f"\n📋 Enhanced Deliberation Methodology Applied")
        print(f"   Steps: {len(methodology.get('steps', []))}")
        for i, step in enumerate(methodology.get('steps', [])[:5], 1):
            print(f"   {i}. {step.get('name')} ({step.get('duration', 0)}s)")
    
    # Show swarm
    swarm = result.get("results", {}).get("swarm", {})
    if swarm:
        print(f"\n🐝 Swarm Created")
        print(f"   Type: {swarm.get('type')}")
        print(f"   Agents: {len(swarm.get('agents', []))}")
        for agent in swarm.get('agents', [])[:3]:
            print(f"   - {agent.get('name')} ({agent.get('role')})")
    
    # Show swarm results
    swarm_results = result.get("results", {}).get("swarm_results", {})
    if swarm_results:
        print(f"\n🔍 Swarm Results")
        print(f"   Agent Results: {len(swarm_results.get('agent_results', []))}")
        print(f"   Synthesized: {'✅' if swarm_results.get('synthesized_result') else '❌'}")
        synthesized = swarm_results.get('synthesized_result', {})
        if synthesized:
            print(f"   Diversity Score: {synthesized.get('diversity_score', 0):.2f}")
            print(f"   Consensus Score: {synthesized.get('consensus_score', 0):.2f}")
    
    # Show insights
    insights = result.get("results", {}).get("insights", {})
    if insights:
        print(f"\n💡 Insights Generated")
        print(f"   Total Insights: {len(insights.get('insights', []))}")
        print(f"   Breakthroughs: {len(insights.get('breakthroughs', []))}")
        print(f"   Suppression Detected: {'✅' if insights.get('suppression_detected') else '❌'}")
        print(f"   Cross-Domain Connections: {len(insights.get('cross_domain_connections', []))}")
        
        if insights.get('breakthroughs'):
            print(f"\n   Breakthroughs:")
            for i, breakthrough in enumerate(insights.get('breakthroughs', [])[:3], 1):
                print(f"   {i}. {breakthrough.get('insight', {}).get('description', '')[:80]}...")
    
    # Show formatted response
    print(f"\n📝 Formatted Response")
    print(f"   Content: {len(formatted.get('content', ''))} characters")
    print(f"   Thinking Items: {len(formatted.get('thinking', []))}")
    print(f"   Conclusions: {len(formatted.get('conclusions', []))}")
    
    if formatted.get('thinking'):
        print(f"\n   Thinking Process:")
        for i, thought in enumerate(formatted.get('thinking', [])[:5], 1):
            print(f"   {i}. {thought[:80]}...")
    
    if formatted.get('conclusions'):
        print(f"\n   Conclusions:")
        for i, conclusion in enumerate(formatted.get('conclusions', [])[:5], 1):
            print(f"   {i}. {conclusion[:80]}...")
    
    print("\n" + "="*70)
    print("FULL SYSTEM TEST COMPLETE")
    print("="*70 + "\n")
    
    return {
        "methodology_applied": methodology is not None,
        "swarm_created": swarm is not None,
        "swarm_results": swarm_results is not None,
        "insights_generated": insights is not None,
        "formatted_response": formatted is not None,
        "full_integration": True
    }


if __name__ == "__main__":
    asyncio.run(test_full_system())

