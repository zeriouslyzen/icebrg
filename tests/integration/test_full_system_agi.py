#!/usr/bin/env python3
"""
Full System AGI Test for ICEBURG

Tests the complete ICEBURG system with a challenging real-world prompt
that requires multi-agent coordination, research, and creative problem-solving.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.agents.capability_registry import get_registry
from iceburg.config import load_config
from iceburg.vectorstore import VectorStore
from iceburg.graph_store import KnowledgeGraph

# Optional monitoring imports
try:
    from iceburg.monitoring.observability_dashboard import get_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    get_dashboard = None

try:
    from iceburg.monitoring.alerting_system import get_alerting_system
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False
    get_alerting_system = None


# Challenging prompts for AGI testing
CHALLENGING_PROMPTS = [
    {
        "name": "Novel Drug Discovery",
        "query": "How can we use AI and computational biology to discover a novel treatment for antibiotic-resistant bacterial infections? What are the most promising approaches combining machine learning, protein folding prediction, and molecular dynamics simulations?",
        "mode": "research",
        "agent": "auto"
    },
    {
        "name": "Climate Change Solution",
        "query": "What are the most innovative and scalable solutions to remove CO2 from the atmosphere at scale? Consider both technological approaches and biological systems. How can we combine multiple strategies for maximum impact?",
        "mode": "research",
        "agent": "auto"
    },
    {
        "name": "Quantum Computing Breakthrough",
        "query": "What are the fundamental barriers preventing practical quantum computing, and what novel approaches could overcome them? Consider error correction, coherence times, and new qubit architectures.",
        "mode": "research",
        "agent": "auto"
    },
    {
        "name": "Longevity Research",
        "query": "What are the most promising interventions to extend healthy human lifespan? Analyze recent research on cellular senescence, telomere extension, and metabolic pathways. What combination of approaches shows the most potential?",
        "mode": "research",
        "agent": "auto"
    }
]


class ProgressTracker:
    """Track progress of agent execution"""
    
    def __init__(self):
        self.thinking_messages = []
        self.actions = []
        self.engines = []
        self.algorithms = []
        self.agent_results = {}
    
    async def callback(self, update):
        """Progress callback for agent execution"""
        update_type = update.get("type")
        
        if update_type == "thinking":
            self.thinking_messages.append(update.get("content", ""))
            print(f"🧠 Thinking: {update.get('content', '')[:100]}...")
        
        elif update_type == "agent_thinking":
            agent = update.get("agent", "unknown")
            content = update.get("content", "")
            self.thinking_messages.append(f"{agent}: {content}")
            print(f"🤖 {agent.capitalize()}: {content[:100]}...")
        
        elif update_type == "action":
            action = update.get("action", "unknown")
            status = update.get("status", "unknown")
            description = update.get("description", "")
            self.actions.append({
                "action": action,
                "status": status,
                "description": description
            })
            print(f"⚡ Action: {action} - {status} - {description[:80]}...")
        
        elif update_type == "engines":
            engines = update.get("engines", [])
            self.engines.extend(engines)
            for engine in engines:
                print(f"🔧 Engine: {engine.get('engine', 'unknown')} - {engine.get('algorithm', 'unknown')}")
        
        elif update_type == "algorithms":
            algorithms = update.get("algorithms", [])
            self.algorithms.extend(algorithms)
            for algo in algorithms:
                print(f"📊 Algorithm: {algo.get('algorithm', 'unknown')}")


async def test_full_system(prompt_config):
    """Test the full ICEBURG system with a challenging prompt"""
    
    print("\n" + "="*80)
    print(f"TESTING: {prompt_config['name']}")
    print("="*80)
    print(f"Query: {prompt_config['query']}")
    print(f"Mode: {prompt_config['mode']}")
    print(f"Agent: {prompt_config['agent']}")
    print("="*80 + "\n")
    
    # Initialize components
    print("🔧 Initializing ICEBURG components...")
    
    # Load configuration
    cfg = load_config()
    print("✅ Configuration loaded")
    
    # Initialize vector store
    vs = VectorStore(cfg)
    print("✅ Vector store initialized")
    
    # Initialize knowledge graph
    kg = KnowledgeGraph(cfg)
    print("✅ Knowledge graph initialized")
    
    # Initialize system integrator (this includes all new features)
    integrator = SystemIntegrator()
    print("✅ System integrator initialized")
    
    # Verify new features are active
    print("\n🔍 Verifying new features are active...")
    
    # Check capability registry
    registry = get_registry()
    all_agents = registry.get_all_agents()
    print(f"✅ Capability Registry: {len(all_agents)} agents registered")
    
    # Check observability dashboard
    if DASHBOARD_AVAILABLE:
        dashboard = get_dashboard()
        print("✅ Observability Dashboard: Active")
    else:
        dashboard = None
        print("⚠️  Observability Dashboard: Not available (optional dependency)")
    
    # Check alerting system
    if ALERTING_AVAILABLE:
        alerting = get_alerting_system()
        print("✅ Alerting System: Active")
    else:
        alerting = None
        print("⚠️  Alerting System: Not available (optional dependency)")
    
    # Check resource allocator
    from iceburg.infrastructure.dynamic_resource_allocator import get_resource_allocator
    resource_allocator = get_resource_allocator()
    resource_status = resource_allocator.get_resource_status()
    print(f"✅ Resource Allocator: {resource_status['allocated']['active_agents']} active agents")
    
    # Check load balancer
    if hasattr(integrator, 'load_balancer'):
        lb_stats = integrator.load_balancer.get_load_balancer_stats()
        print(f"✅ Load Balancer: {lb_stats['total_workers']} workers registered")
    
    # Initialize progress tracker
    progress = ProgressTracker()
    
    # Execute query with full integration
    print("\n🚀 Executing query with full agent coordination...")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        result = await integrator.process_query_with_full_integration(
            query=prompt_config['query'],
            domain=prompt_config['mode'],
            custom_config=None,
            temperature=0.7,
            max_tokens=4000,
            progress_callback=progress.callback
        )
        
        execution_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETE")
        print("="*80)
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"🧠 Thinking messages: {len(progress.thinking_messages)}")
        print(f"⚡ Actions executed: {len(progress.actions)}")
        print(f"🔧 Engines used: {len(progress.engines)}")
        print(f"📊 Algorithms used: {len(progress.algorithms)}")
        
        # Display results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        agent_results = result.get("results", {}).get("agent_results", {})
        print(f"\n📋 Agent Results ({len(agent_results)} agents):")
        for agent, output in agent_results.items():
            if output and isinstance(output, str):
                output_preview = output[:200] + "..." if len(output) > 200 else output
                print(f"  • {agent.capitalize()}: {output_preview}")
        
        # Display final response
        final_response = result.get("results", {}).get("final_response", "")
        if final_response:
            print(f"\n💬 Final Response ({len(final_response)} chars):")
            print("-" * 80)
            print(final_response[:1000] + "..." if len(final_response) > 1000 else final_response)
            print("-" * 80)
        
        # Display engines and algorithms
        engines_used = result.get("results", {}).get("engines_used", [])
        if engines_used:
            print(f"\n🔧 Engines Used ({len(engines_used)}):")
            for engine in engines_used:
                print(f"  • {engine.get('engine', 'unknown')} - {engine.get('algorithm', 'unknown')}")
        
        algorithms_used = result.get("results", {}).get("algorithms_used", [])
        if algorithms_used:
            print(f"\n📊 Algorithms Used ({len(algorithms_used)}):")
            for algo in algorithms_used:
                print(f"  • {algo.get('algorithm', 'unknown')} - {algo.get('method', 'unknown')}")
        
        # Check observability metrics
        if dashboard:
            print("\n" + "="*80)
            print("OBSERVABILITY METRICS")
            print("="*80)
            
            dashboard_data = dashboard.get_dashboard_data()
            overview = dashboard_data.get("overview", {})
            print(f"📊 Total Executions: {overview.get('total_executions', 0)}")
            print(f"✅ Successful: {overview.get('successful_executions', 0)}")
            print(f"❌ Failed: {overview.get('failed_executions', 0)}")
            print(f"📈 Success Rate: {overview.get('success_rate', 0):.1%}")
            print(f"🔌 Active Agents: {overview.get('active_agents', 0)}")
            print(f"⚠️  Open Circuit Breakers: {overview.get('open_circuit_breakers', 0)}")
        
        # Check alerting
        if alerting:
            alert_summary = alerting.get_alert_summary()
            print(f"\n🚨 Alerts:")
            print(f"  • Active: {alert_summary.get('active_alerts', 0)}")
            print(f"  • Total: {alert_summary.get('total_alerts', 0)}")
        
        # Check resource utilization
        resource_status = resource_allocator.get_resource_status()
        allocated = resource_status.get("allocated", {})
        print(f"\n💾 Resource Utilization:")
        print(f"  • Memory: {allocated.get('used_memory_mb', 0):.1f} MB")
        print(f"  • CPU: {allocated.get('used_cpu_cores', 0):.1f} cores")
        print(f"  • Active Agents: {allocated.get('active_agents', 0)}")
        
        print("\n" + "="*80)
        print("✅ TEST COMPLETE")
        print("="*80 + "\n")
        
        return True, result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ ERROR after {execution_time:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def main():
    """Run full system tests"""
    
    print("\n" + "="*80)
    print("ICEBURG FULL SYSTEM AGI TEST")
    print("="*80)
    print("\nTesting complete ICEBURG system with challenging real-world prompts")
    print("All new features are active: capability registry, linguistic intelligence,")
    print("load balancing, resource allocation, observability, and alerting.\n")
    
    # Test with the first challenging prompt
    prompt_config = CHALLENGING_PROMPTS[0]  # Novel Drug Discovery
    
    success, result = await test_full_system(prompt_config)
    
    if success:
        print("✅ Full system test PASSED")
        return 0
    else:
        print("❌ Full system test FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

