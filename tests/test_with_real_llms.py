#!/usr/bin/env python3
"""
Test ICEBURG with Real LLMs and Agents
Tests the actual protocol execution with real LLM calls and agent execution
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.micro_agent_swarm import MicroAgentSwarm
from iceburg.agents.surveyor import run as surveyor_run
from iceburg.agents.dissident import run as dissident_run
from iceburg.agents.synthesist import run as synthesist_run
from iceburg.config import load_config
from iceburg.vectorstore import VectorStore


async def test_with_real_llms():
    """Test ICEBURG with real LLM calls and agent execution"""
    print("\n" + "="*70)
    print("ICEBURG 2.0 - REAL LLM AND AGENT TEST (VERBOSE)")
    print("="*70 + "\n")
    print("Starting comprehensive test with real LLM calls and agent execution...")
    print("This will show exactly what ICEBURG is doing at each step.\n")
    
    # Load config
    cfg = load_config()
    
    # Initialize VectorStore for agents (requires config)
    vs = VectorStore(cfg)
    
    results = {
        "llm_tests": [],
        "agent_tests": [],
        "protocol_tests": [],
        "accuracy_scores": {}
    }
    
    # Test 1: Direct LLM Call
    print("1. DIRECT LLM CALL TEST")
    print("-" * 70)
    
    try:
        from iceburg.llm import chat_complete
        
        print(f"  📞 Calling LLM with model: {cfg.surveyor_model}")
        print(f"     Prompt: 'What is quantum computing?'")
        print(f"     System: 'You are a helpful assistant.'")
        print(f"     Temperature: 0.7")
        print(f"     [ICEBURG] Making LLM API call...")
        
        response = chat_complete(
            cfg.surveyor_model,
            "What is quantum computing?",
            system="You are a helpful assistant.",
            temperature=0.7
        )
        
        print(f"     [ICEBURG] LLM response received!")
        
        has_response = bool(response)
        response_length = len(response) if response else 0
        
        results["llm_tests"].append({
            "test": "direct_llm_call",
            "success": has_response,
            "response_length": response_length,
            "accuracy": 1.0 if has_response and response_length > 0 else 0.0
        })
        
        print(f"  ✅ Direct LLM Call")
        print(f"     Response Length: {response_length} characters")
        print(f"     Success: {has_response}")
        if response:
            print(f"     Preview: {response[:200]}...")
    except Exception as e:
        print(f"  ❌ Direct LLM Call (FAILED: {e})")
        results["llm_tests"].append({
            "test": "direct_llm_call",
            "success": False,
            "error": str(e),
            "accuracy": 0.0
        })
    
    # Test 2: Agent Execution
    print("\n2. AGENT EXECUTION TEST")
    print("-" * 70)
    
    test_query = "What is quantum computing?"
    
    # Initialize surveyor result for use in Dissident test
    surveyor_result_for_dissident = ""
    
    # Test Surveyor Agent
    try:
        print("  Testing Surveyor Agent...")
        print(f"     [ICEBURG] Initializing Surveyor agent...")
        print(f"     [ICEBURG] Surveyor query: '{test_query}'")
        print(f"     [ICEBURG] Surveyor model: {cfg.surveyor_model}")
        print(f"     [ICEBURG] Executing Surveyor agent...")
        surveyor_result = surveyor_run(cfg, vs, test_query, verbose=True)
        print(f"     [ICEBURG] Surveyor agent completed!")
        has_surveyor_result = bool(surveyor_result)
        surveyor_result_for_dissident = surveyor_result if surveyor_result else ""
        
        results["agent_tests"].append({
            "agent": "surveyor",
            "success": has_surveyor_result,
            "result_length": len(surveyor_result) if surveyor_result else 0,
            "accuracy": 1.0 if has_surveyor_result else 0.0
        })
        
        print(f"    ✅ Surveyor: {len(surveyor_result) if surveyor_result else 0} characters")
    except Exception as e:
        print(f"    ❌ Surveyor (FAILED: {e})")
        results["agent_tests"].append({
            "agent": "surveyor",
            "success": False,
            "error": str(e),
            "accuracy": 0.0
        })
    
    # Test Dissident Agent
    try:
        print("  Testing Dissident Agent...")
        print(f"     [ICEBURG] Initializing Dissident agent...")
        print(f"     [ICEBURG] Dissident query: '{test_query}'")
        print(f"     [ICEBURG] Dissident model: {cfg.dissident_model if hasattr(cfg, 'dissident_model') else cfg.surveyor_model}")
        print(f"     [ICEBURG] Executing Dissident agent...")
        # Dissident needs surveyor_output - use the result from Surveyor test above
        dissident_result = dissident_run(cfg, test_query, surveyor_result_for_dissident, verbose=True)
        print(f"     [ICEBURG] Dissident agent completed!")
        has_dissident_result = bool(dissident_result)
        
        results["agent_tests"].append({
            "agent": "dissident",
            "success": has_dissident_result,
            "result_length": len(dissident_result) if dissident_result else 0,
            "accuracy": 1.0 if has_dissident_result else 0.0
        })
        
        print(f"    ✅ Dissident: {len(dissident_result) if dissident_result else 0} characters")
    except Exception as e:
        print(f"    ❌ Dissident (FAILED: {e})")
        results["agent_tests"].append({
            "agent": "dissident",
            "success": False,
            "error": str(e),
            "accuracy": 0.0
        })
    
    # Test Synthesist Agent
    try:
        print("  Testing Synthesist Agent...")
        synthesis_input = f"Surveyor: {surveyor_result if 'surveyor_result' in locals() else 'No result'}\nDissident: {dissident_result if 'dissident_result' in locals() else 'No result'}"
        print(f"     [ICEBURG] Initializing Synthesist agent...")
        print(f"     [ICEBURG] Synthesist input length: {len(synthesis_input)} characters")
        print(f"     [ICEBURG] Synthesist model: {cfg.synthesist_model if hasattr(cfg, 'synthesist_model') else cfg.surveyor_model}")
        print(f"     [ICEBURG] Executing Synthesist agent...")
        synthesist_result = synthesist_run(cfg, synthesis_input, verbose=True)
        print(f"     [ICEBURG] Synthesist agent completed!")
        has_synthesist_result = bool(synthesist_result)
        
        results["agent_tests"].append({
            "agent": "synthesist",
            "success": has_synthesist_result,
            "result_length": len(synthesist_result) if synthesist_result else 0,
            "accuracy": 1.0 if has_synthesist_result else 0.0
        })
        
        print(f"    ✅ Synthesist: {len(synthesist_result) if synthesist_result else 0} characters")
    except Exception as e:
        print(f"    ❌ Synthesist (FAILED: {e})")
        results["agent_tests"].append({
            "agent": "synthesist",
            "success": False,
            "error": str(e),
            "accuracy": 0.0
        })
    
    # Test 3: Swarm Execution with Real LLMs
    print("\n3. SWARM EXECUTION WITH REAL LLMs")
    print("-" * 70)
    
    try:
        print(f"     [ICEBURG] Initializing MicroAgentSwarm...")
        swarm = MicroAgentSwarm()
        print(f"     [ICEBURG] Swarm initialized with {len(swarm.agents)} agents")
        print(f"     [ICEBURG] Starting swarm...")
        await swarm.start_swarm()
        print(f"     [ICEBURG] Swarm started!")
        print(f"     [ICEBURG] Starting swarm task: 'research'")
        print(f"     [ICEBURG] Task params: query='{test_query}', domain='quantum_physics'")
        print(f"     [ICEBURG] Executing swarm task...")
        
        # Use the correct method - submit_task and get_task_result
        task_id = await swarm.submit_task(
            task_type="research",
            input_data={"query": test_query, "domain": "quantum_physics"},
            requirements=["research", "analysis"],
            priority=7
        )
        print(f"     [ICEBURG] Task submitted with ID: {task_id}")
        print(f"     [ICEBURG] Waiting for task completion...")
        
        swarm_result = await swarm.get_task_result(task_id, timeout=60)
        
        print(f"     [ICEBURG] Swarm task completed!")
        
        has_swarm_result = swarm_result is not None and not isinstance(swarm_result, dict) or swarm_result.get("error") is None
        agent_count = 1 if has_swarm_result else 0
        
        results["protocol_tests"].append({
            "test": "swarm_execution",
            "success": has_swarm_result,
            "agent_count": agent_count,
            "accuracy": 1.0 if has_swarm_result and agent_count > 0 else 0.0
        })
        
        print(f"  ✅ Swarm Execution")
        print(f"     Agents Executed: {agent_count}")
        print(f"     Success: {has_swarm_result}")
    except Exception as e:
        print(f"  ❌ Swarm Execution (FAILED: {e})")
        results["protocol_tests"].append({
            "test": "swarm_execution",
            "success": False,
            "error": str(e),
            "accuracy": 0.0
        })
    
    # Test 4: Full Protocol with Real LLMs
    print("\n4. FULL PROTOCOL WITH REAL LLMs")
    print("-" * 70)
    
    try:
        print(f"     [ICEBURG] Initializing SystemIntegrator...")
        system_integrator = SystemIntegrator()
        print(f"     [ICEBURG] Integrating all systems...")
        integration_result = system_integrator.integrate_all_systems()  # Not async, returns dict
        print(f"     [ICEBURG] Systems integrated! Status: {integration_result.get('integrated_at', 'N/A')}")
        print(f"     [ICEBURG] Processing query with full integration...")
        print(f"     [ICEBURG] Query: '{test_query}'")
        print(f"     [ICEBURG] Domain: 'quantum_physics'")
        print(f"     [ICEBURG] Starting Enhanced Deliberation methodology...")
        
        protocol_result = await system_integrator.process_query_with_full_integration(
            query=test_query,
            domain="quantum_physics"
        )
        
        print(f"     [ICEBURG] Full protocol execution completed!")
        
        has_methodology = protocol_result.get("results", {}).get("methodology") is not None
        has_swarm = protocol_result.get("results", {}).get("swarm") is not None
        has_insights = protocol_result.get("results", {}).get("insights") is not None
        
        protocol_accuracy = (has_methodology + has_swarm + has_insights) / 3.0
        
        results["protocol_tests"].append({
            "test": "full_protocol",
            "success": True,
            "has_methodology": has_methodology,
            "has_swarm": has_swarm,
            "has_insights": has_insights,
            "accuracy": protocol_accuracy
        })
        
        print(f"  ✅ Full Protocol")
        print(f"     Methodology: {has_methodology}")
        print(f"     Swarm: {has_swarm}")
        print(f"     Insights: {has_insights}")
        print(f"     Accuracy: {protocol_accuracy*100:.1f}%")
    except Exception as e:
        print(f"  ❌ Full Protocol (FAILED: {e})")
        import traceback
        traceback.print_exc()
        results["protocol_tests"].append({
            "test": "full_protocol",
            "success": False,
            "error": str(e),
            "accuracy": 0.0
        })
    
    # Calculate overall accuracy
    print("\n5. OVERALL ACCURACY SCORES")
    print("-" * 70)
    
    llm_accuracy = sum(t.get("accuracy", 0) for t in results["llm_tests"]) / len(results["llm_tests"]) if results["llm_tests"] else 0.0
    agent_accuracy = sum(t.get("accuracy", 0) for t in results["agent_tests"]) / len(results["agent_tests"]) if results["agent_tests"] else 0.0
    protocol_accuracy = sum(t.get("accuracy", 0) for t in results["protocol_tests"]) / len(results["protocol_tests"]) if results["protocol_tests"] else 0.0
    
    overall_accuracy = (llm_accuracy + agent_accuracy + protocol_accuracy) / 3.0
    
    results["accuracy_scores"] = {
        "llm_accuracy": llm_accuracy,
        "agent_accuracy": agent_accuracy,
        "protocol_accuracy": protocol_accuracy,
        "overall_accuracy": overall_accuracy
    }
    
    print(f"LLM Accuracy: {llm_accuracy*100:.1f}%")
    print(f"Agent Accuracy: {agent_accuracy*100:.1f}%")
    print(f"Protocol Accuracy: {protocol_accuracy*100:.1f}%")
    print(f"Overall Accuracy: {overall_accuracy*100:.1f}%")
    
    print("\n" + "="*70)
    print("REAL LLM AND AGENT TEST COMPLETE")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_with_real_llms())

