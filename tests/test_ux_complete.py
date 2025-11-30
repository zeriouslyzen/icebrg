#!/usr/bin/env python3
"""
Complete UX Test Suite
Tests all modes, agents, and features
"""
import asyncio
import json
import websockets
import sys
import time

async def test_query(mode, agent, query, degradation_mode=False):
    """Test a single query"""
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            message = {
                "query": query,
                "mode": mode,
                "agent": agent,
                "degradation_mode": degradation_mode,
                "settings": {
                    "primaryModel": "llama3.1:8b",
                    "temperature": 0.7,
                    "maxTokens": 2000
                }
            }
            
            await websocket.send(json.dumps(message))
            
            responses = []
            start_time = time.time()
            timeout = 60
            
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    data = json.loads(response)
                    responses.append(data)
                    
                    if data.get("type") == "done":
                        break
                    elif data.get("type") == "error":
                        return False, f"Error: {data.get('message')}", time.time() - start_time
                    
            except asyncio.TimeoutError:
                return False, f"Timeout after {timeout}s", time.time() - start_time
            
            # Extract content
            chunks = [r.get("content", "") for r in responses if r.get("type") == "chunk"]
            content = "".join(chunks)
            
            elapsed = time.time() - start_time
            
            return True, content, elapsed
            
    except Exception as e:
        return False, f"Connection error: {e}", 0

async def run_tests():
    """Run all UX tests"""
    print("🧪 ICEBURG UX Test Suite\n")
    print("=" * 60)
    
    tests = [
        ("chat", "auto", "hi", False, "Fast chat mode"),
        ("fast", "auto", "hi", False, "Fast mode explicit"),
        ("chat", "auto", "What is quantum computing?", False, "Complex query in chat mode"),
        ("research", "auto", "quantum computing", False, "Research mode"),
        ("chat", "surveyor", "hello", False, "Single agent (Surveyor)"),
    ]
    
    results = []
    
    for mode, agent, query, degradation, description in tests:
        print(f"\n📋 Test: {description}")
        print(f"   Mode: {mode}, Agent: {agent}, Query: '{query}'")
        
        success, result, elapsed = await test_query(mode, agent, query, degradation)
        
        if success:
            print(f"   ✅ PASSED ({elapsed:.2f}s)")
            print(f"   Response: {result[:100]}...")
            results.append(("PASS", description, elapsed))
        else:
            print(f"   ❌ FAILED: {result}")
            results.append(("FAIL", description, result))
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] == "FAIL")
    
    print(f"\n✅ Passed: {passed}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\n❌ Failed tests:")
        for status, desc, result in results:
            if status == "FAIL":
                print(f"   - {desc}: {result}")
    
    avg_time = sum(r[2] for r in results if r[0] == "PASS") / passed if passed > 0 else 0
    print(f"\n⏱️  Average response time: {avg_time:.2f}s")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)

