#!/usr/bin/env python3
"""
Test ICEBURG with actual prompts
Shows what ICEBURG thinks about and how it processes queries
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.formatting.response_formatter import ResponseFormatter


async def test_iceburg_prompt(query: str, domain: str = None):
    """Test ICEBURG with an actual prompt"""
    print("\n" + "="*70)
    print("ICEBURG 2.0 - PROMPT TEST")
    print("="*70 + "\n")
    
    print(f"Query: {query}")
    if domain:
        print(f"Domain: {domain}")
    print("\n" + "-"*70)
    print("ICEBURG Processing...")
    print("-"*70 + "\n")
    
    # Initialize system
    system_integrator = SystemIntegrator()
    response_formatter = ResponseFormatter()
    
    # Process query
    result = await system_integrator.process_query_with_full_integration(
        query=query,
        domain=domain
    )
    
    # Format response
    formatted = response_formatter.format_from_analysis(result.get("results", {}))
    
    # Display results
    print("="*70)
    print("ICEBURG'S RESPONSE")
    print("="*70 + "\n")
    
    # Show methodology
    methodology = result.get("results", {}).get("methodology", {})
    if methodology:
        print("📋 ENHANCED DELIBERATION METHODOLOGY")
        print("-" * 70)
        steps = methodology.get("steps", [])
        print(f"Steps Applied: {len(steps)}")
        for i, step in enumerate(steps[:5], 1):
            print(f"  {i}. {step.get('name', 'Unknown step')}")
        print()
    
    # Show curiosity queries
    curiosity_queries = result.get("results", {}).get("curiosity_queries", [])
    if curiosity_queries:
        print("🔍 CURIOSITY-DRIVEN QUERIES")
        print("-" * 70)
        for i, cq in enumerate(curiosity_queries[:3], 1):
            if isinstance(cq, str):
                print(f"  {i}. {cq}")
            elif hasattr(cq, 'query_text'):
                print(f"  {i}. {cq.query_text}")
        print()
    
    # Show swarm
    swarm = result.get("results", {}).get("swarm", {})
    if swarm:
        print("🐝 SWARM CREATED")
        print("-" * 70)
        print(f"Type: {swarm.get('type', 'Unknown')}")
        agents = swarm.get('agents', [])
        print(f"Agents: {len(agents)}")
        for i, agent in enumerate(agents[:3], 1):
            print(f"  {i}. {agent.get('name', 'Unknown')} - {agent.get('role', 'Unknown role')}")
        print()
    
    # Show swarm results
    swarm_results = result.get("results", {}).get("swarm_results", {})
    if swarm_results:
        print("🔬 SWARM RESULTS")
        print("-" * 70)
        agent_results = swarm_results.get('agent_results', [])
        print(f"Agent Results: {len(agent_results)}")
        synthesized = swarm_results.get('synthesized_result', {})
        if synthesized:
            print(f"Synthesized: ✅")
            if isinstance(synthesized, dict):
                print(f"  Diversity Score: {synthesized.get('diversity_score', 0):.2f}")
                print(f"  Consensus Score: {synthesized.get('consensus_score', 0):.2f}")
        print()
    
    # Show insights
    insights = result.get("results", {}).get("insights", {})
    if insights:
        print("💡 INSIGHTS GENERATED")
        print("-" * 70)
        insight_list = insights.get('insights', [])
        print(f"Total Insights: {len(insight_list)}")
        breakthroughs = insights.get('breakthroughs', [])
        print(f"Breakthroughs: {len(breakthroughs)}")
        print(f"Suppression Detected: {'✅' if insights.get('suppression_detected') else '❌'}")
        
        if insight_list:
            print("\nKey Insights:")
            for i, insight in enumerate(insight_list[:3], 1):
                if isinstance(insight, dict):
                    print(f"  {i}. {insight.get('type', 'insight')}: {insight.get('description', '')[:80]}...")
                else:
                    print(f"  {i}. {str(insight)[:80]}...")
        print()
    
    # Show formatted response
    print("📝 FORMATTED RESPONSE")
    print("-" * 70)
    
    # Thinking process
    thinking = formatted.get('thinking', [])
    if thinking:
        print("\n🧠 THINKING PROCESS:")
        for i, thought in enumerate(thinking[:5], 1):
            print(f"  {i}. {thought[:100]}...")
    
    # Informatics
    informatics = formatted.get('informatics', {})
    if informatics:
        print("\n📊 INFORMATICS:")
        for key, value in list(informatics.items())[:5]:
            print(f"  {key}: {value}")
    
    # Conclusions
    conclusions = formatted.get('conclusions', [])
    if conclusions:
        print("\n✅ CONCLUSIONS:")
        if isinstance(conclusions, list):
            for i, conclusion in enumerate(conclusions[:5], 1):
                print(f"  {i}. {str(conclusion)[:100]}...")
        else:
            print(f"  {str(conclusions)[:200]}...")
    
    # Main content
    content = formatted.get('content', '')
    if content:
        print("\n📄 MAIN CONTENT:")
        print(content[:500] + "..." if len(content) > 500 else content)
    
    print("\n" + "="*70)
    print("PROMPT TEST COMPLETE")
    print("="*70 + "\n")
    
    return result


async def main():
    """Run prompt tests"""
    queries = [
        ("What is Enhanced Deliberation methodology and how does it enable truth-finding?", "truth_finding"),
        ("How can swarming create better answers than single agents?", "swarming"),
        ("What suppressed knowledge exists about quantum consciousness?", "quantum_consciousness"),
        ("How does ICEBURG find suppressed knowledge and create devices?", "truth_finding"),
    ]
    
    for query, domain in queries:
        await test_iceburg_prompt(query, domain)
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

