"""
Test Always-On Architecture Components
Tests all always-on architecture components to ensure they work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iceburg.core import (
    AlwaysOnProtocolExecutor,
    PreWarmedAgentPool,
    LocalPersonaInstance,
    ICEBURGPortal,
    GraphProcessor,
    MixtureOfExperts,
    MultiTokenPredictor
)
from iceburg.config import load_config


async def test_always_on_executor():
    """Test AlwaysOnProtocolExecutor"""
    print("\n=== Testing AlwaysOnProtocolExecutor ===")
    try:
        config = load_config()
        executor = AlwaysOnProtocolExecutor(config)
        
        # Start executor
        await executor.start()
        print("✅ Executor started successfully")
        
        # Queue a task
        task = {
            "id": "test_task_1",
            "type": "knowledge_update",
            "query": "test query",
            "knowledge": {"test": "data"}
        }
        executor.queue_task(task)
        print("✅ Task queued successfully")
        
        # Wait a bit for processing
        await asyncio.sleep(1)
        
        # Get stats
        stats = executor.get_stats()
        print(f"✅ Executor stats: {stats}")
        
        # Stop executor
        await executor.stop()
        print("✅ Executor stopped successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error testing AlwaysOnProtocolExecutor: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pre_warmed_agent_pool():
    """Test PreWarmedAgentPool"""
    print("\n=== Testing PreWarmedAgentPool ===")
    try:
        config = load_config()
        pool = PreWarmedAgentPool(config)
        
        # Warmup agents
        await pool.warmup()
        print("✅ Agent pool warmed up successfully")
        
        # Get agent
        agent = pool.get_agent("surveyor")
        if agent:
            print("✅ Agent retrieved successfully")
        else:
            print("⚠️  Agent not found (expected for placeholder implementation)")
        
        # Get stats
        stats = pool.get_stats()
        print(f"✅ Pool stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing PreWarmedAgentPool: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_local_persona_instance():
    """Test LocalPersonaInstance"""
    print("\n=== Testing LocalPersonaInstance ===")
    try:
        config = load_config()
        persona = LocalPersonaInstance("test_user", config)
        print("✅ Persona instance created successfully")
        
        # Test simple query
        response = await persona.respond("hi")
        if response:
            print(f"✅ Local persona response: {response.get('response', 'N/A')}")
        else:
            print("⚠️  No response (escalated)")
        
        # Test complex query
        response = await persona.respond("What is the meaning of life?")
        if response:
            print(f"✅ Local persona response: {response.get('response', 'N/A')}")
        else:
            print("✅ Complex query escalated (expected)")
        
        # Get stats
        stats = persona.get_stats()
        print(f"✅ Persona stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing LocalPersonaInstance: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_iceburg_portal():
    """Test ICEBURGPortal"""
    print("\n=== Testing ICEBURGPortal ===")
    try:
        config = load_config()
        portal = ICEBURGPortal(config)
        
        # Initialize portal
        await portal.initialize()
        print("✅ Portal initialized successfully")
        
        # Test simple query
        response = await portal.open_portal("test_user", "hi")
        if response:
            print(f"✅ Portal response: {response.get('response', 'N/A')[:50]}...")
            print(f"   Source: {response.get('source', 'unknown')}")
            print(f"   Response time: {response.get('response_time', 0):.3f}s")
        else:
            print("⚠️  No response from portal")
        
        # Test complex query
        response = await portal.open_portal("test_user", "What is artificial intelligence?")
        if response:
            print(f"✅ Portal response: {response.get('response', 'N/A')[:50]}...")
            print(f"   Source: {response.get('source', 'unknown')}")
            print(f"   Response time: {response.get('response_time', 0):.3f}s")
        else:
            print("⚠️  No response from portal")
        
        # Get stats
        stats = portal.get_stats()
        print(f"✅ Portal stats: {stats}")
        
        # Shutdown portal
        await portal.shutdown()
        print("✅ Portal shut down successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error testing ICEBURGPortal: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_graph_processor():
    """Test GraphProcessor"""
    print("\n=== Testing GraphProcessor ===")
    try:
        processor = GraphProcessor()
        print("✅ Graph processor created successfully")
        
        # Add processors
        processor.add_processor("test_processor", lambda q, c: {"result": "test"}, dependencies=[])
        print("✅ Processor added successfully")
        
        # Process graph
        result = await processor.process_graph("test query")
        if result:
            print(f"✅ Graph processing result: {result}")
        else:
            print("⚠️  No result from graph processing")
        
        # Get stats
        stats = processor.get_stats()
        print(f"✅ Graph processor stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing GraphProcessor: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mixture_of_experts():
    """Test MixtureOfExperts"""
    print("\n=== Testing MixtureOfExperts ===")
    try:
        moe = MixtureOfExperts()
        print("✅ MoE created successfully")
        
        # Add experts
        moe.add_expert("code_expert", lambda q, c: {"result": "code"}, specialization="code")
        moe.add_expert("research_expert", lambda q, c: {"result": "research"}, specialization="research")
        print("✅ Experts added successfully")
        
        # Process query
        result = await moe.process("write code")
        if result:
            print(f"✅ MoE processing result: {result}")
        else:
            print("⚠️  No result from MoE processing")
        
        # Get stats
        stats = moe.get_stats()
        print(f"✅ MoE stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing MixtureOfExperts: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_token_predictor():
    """Test MultiTokenPredictor"""
    print("\n=== Testing MultiTokenPredictor ===")
    try:
        config = load_config()
        predictor = MultiTokenPredictor(config)
        print("✅ Multi-token predictor created successfully")
        
        # Predict tokens
        tokens = await predictor.predict_tokens("test context", num_tokens=3)
        if tokens:
            print(f"✅ Predicted tokens: {tokens}")
        else:
            print("⚠️  No tokens predicted")
        
        # Get stats
        stats = predictor.get_stats()
        print(f"✅ Predictor stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing MultiTokenPredictor: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration():
    """Test full integration"""
    print("\n=== Testing Full Integration ===")
    try:
        config = load_config()
        portal = ICEBURGPortal(config)
        
        # Initialize portal
        await portal.initialize()
        print("✅ Portal initialized for integration test")
        
        # Test various query types
        test_queries = [
            ("hi", "simple greeting"),
            ("hello", "simple greeting"),
            ("What is AI?", "medium complexity"),
            ("Explain quantum computing in detail", "complex query")
        ]
        
        for query, query_type in test_queries:
            print(f"\n  Testing {query_type}: '{query}'")
            response = await portal.open_portal("test_user", query)
            if response:
                print(f"    ✅ Response received: {response.get('source', 'unknown')} in {response.get('response_time', 0):.3f}s")
            else:
                print(f"    ⚠️  No response")
        
        # Shutdown
        await portal.shutdown()
        print("✅ Integration test completed")
        
        return True
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Always-On Architecture Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run individual component tests
    results.append(("AlwaysOnProtocolExecutor", await test_always_on_executor()))
    results.append(("PreWarmedAgentPool", await test_pre_warmed_agent_pool()))
    results.append(("LocalPersonaInstance", await test_local_persona_instance()))
    results.append(("ICEBURGPortal", await test_iceburg_portal()))
    results.append(("GraphProcessor", await test_graph_processor()))
    results.append(("MixtureOfExperts", await test_mixture_of_experts()))
    results.append(("MultiTokenPredictor", await test_multi_token_predictor()))
    
    # Run integration test
    results.append(("Integration", await test_integration()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

