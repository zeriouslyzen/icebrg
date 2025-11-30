#!/usr/bin/env python3
"""
Test ICEBURG's Thinking Process
Shows what ICEBURG actually thinks about when given a prompt
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.formatting.response_formatter import ResponseFormatter


async def test_iceburg_thinking():
    """Test what ICEBURG thinks about"""
    print("\n" + "="*70)
    print("ICEBURG 2.0 - THINKING TEST")
    print("="*70 + "\n")
    
    query = "What is Enhanced Deliberation methodology and how does it enable truth-finding?"
    
    print(f"📝 PROMPT:")
    print(f"   {query}\n")
    
    print("🧠 ICEBURG Processing...\n")
    
    # Initialize system
    system_integrator = SystemIntegrator()
    response_formatter = ResponseFormatter()
    
    # Process query
    result = await system_integrator.process_query_with_full_integration(
        query=query,
        domain="truth_finding"
    )
    
    # Format response
    formatted = response_formatter.format_from_analysis(result.get("results", {}))
    
    print("="*70)
    print("ICEBURG'S THINKING PROCESS")
    print("="*70 + "\n")
    
    # Show methodology steps (what ICEBURG is thinking about)
    methodology = result.get("results", {}).get("methodology", {})
    if methodology:
        steps = methodology.get("steps", [])
        print("📋 METHODOLOGY STEPS (ICEBURG's Thinking Process):")
        print("-" * 70)
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step.get('name', 'Unknown')}")
            if step.get('description'):
                print(f"     → {step.get('description')[:80]}...")
        print()
    
    # Show curiosity queries (what ICEBURG is curious about)
    curiosity_queries = result.get("results", {}).get("curiosity_queries", [])
    if curiosity_queries:
        print("🔍 CURIOSITY-DRIVEN QUERIES (What ICEBURG is Curious About):")
        print("-" * 70)
        for i, cq in enumerate(curiosity_queries[:5], 1):
            if isinstance(cq, str):
                print(f"  {i}. {cq}")
            elif hasattr(cq, 'query_text'):
                print(f"  {i}. {cq.query_text}")
        print()
    
    # Show swarm (how ICEBURG collaborates)
    swarm = result.get("results", {}).get("swarm", {})
    if swarm:
        print("🐝 SWARM CREATION (How ICEBURG Collaborates):")
        print("-" * 70)
        print(f"  Type: {swarm.get('type', 'Unknown')}")
        agents = swarm.get('agents', [])
        print(f"  Agents: {len(agents)}")
        for agent in agents[:3]:
            print(f"    - {agent.get('name', 'Unknown')}: {agent.get('role', 'Unknown role')}")
        print()
    
    # Show insights (what ICEBURG discovered)
    insights = result.get("results", {}).get("insights", {})
    if insights:
        print("💡 INSIGHTS (What ICEBURG Discovered):")
        print("-" * 70)
        insight_list = insights.get('insights', [])
        print(f"  Total Insights: {len(insight_list)}")
        
        if insight_list:
            print("\n  Key Insights:")
            for i, insight in enumerate(insight_list[:5], 1):
                if isinstance(insight, dict):
                    insight_type = insight.get('type', 'insight')
                    description = insight.get('description', '')
                    print(f"    {i}. [{insight_type}] {description[:100]}...")
                else:
                    print(f"    {i}. {str(insight)[:100]}...")
        print()
    
    # Show formatted thinking
    thinking = formatted.get('thinking', [])
    if thinking:
        print("🧠 THINKING PROCESS (ICEBURG's Internal Thoughts):")
        print("-" * 70)
        for i, thought in enumerate(thinking[:10], 1):
            print(f"  {i}. {thought[:150]}...")
        print()
    
    # Show informatics
    informatics = formatted.get('informatics', {})
    if informatics:
        print("📊 INFORMATICS (ICEBURG's Analysis Metrics):")
        print("-" * 70)
        for key, value in informatics.items():
            print(f"  {key}: {value}")
        print()
    
    # Show conclusions
    conclusions = formatted.get('conclusions', [])
    if conclusions:
        print("✅ CONCLUSIONS (ICEBURG's Final Thoughts):")
        print("-" * 70)
        if isinstance(conclusions, list):
            for i, conclusion in enumerate(conclusions[:5], 1):
                print(f"  {i}. {str(conclusion)[:150]}...")
        else:
            print(f"  {str(conclusions)[:200]}...")
        print()
    
    print("="*70)
    print("THINKING TEST COMPLETE")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_iceburg_thinking())

