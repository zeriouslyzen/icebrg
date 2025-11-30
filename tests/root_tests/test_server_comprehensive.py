"""
Comprehensive Server Test Suite
Tests various query types, edge cases, and user inputs
"""

import asyncio
import json
import time
import websockets
from typing import Dict, Any, List

# Test cases
TEST_CASES = [
    # Simple queries (should use fast path)
    {"query": "hi", "mode": "chat", "expected": "fast"},
    {"query": "hello", "mode": "chat", "expected": "fast"},
    {"query": "hey", "mode": "chat", "expected": "fast"},
    {"query": "thanks", "mode": "chat", "expected": "fast"},
    {"query": "bye", "mode": "chat", "expected": "fast"},
    
    # Normal queries
    {"query": "What is quantum mechanics?", "mode": "chat", "expected": "normal"},
    {"query": "Explain AI", "mode": "chat", "expected": "normal"},
    {"query": "How does the brain work?", "mode": "chat", "expected": "normal"},
    
    # Edge cases
    {"query": "", "mode": "chat", "expected": "error"},  # Empty query
    {"query": "   ", "mode": "chat", "expected": "error"},  # Whitespace only
    {"query": "a" * 10000, "mode": "chat", "expected": "normal"},  # Very long query
    {"query": "!@#$%^&*()", "mode": "chat", "expected": "normal"},  # Special characters
    {"query": "测试中文", "mode": "chat", "expected": "normal"},  # Non-ASCII
    {"query": "مرحبا", "mode": "chat", "expected": "normal"},  # Arabic
    {"query": "こんにちは", "mode": "chat", "expected": "normal"},  # Japanese
    
    # Different modes
    {"query": "What is AI?", "mode": "fast", "expected": "normal"},
    {"query": "Research quantum computing", "mode": "research", "expected": "normal"},
    {"query": "Generate a calculator", "mode": "device", "expected": "normal"},
    
    # Different agents
    {"query": "What is physics?", "mode": "chat", "agent": "surveyor", "expected": "normal"},
    {"query": "Analyze this", "mode": "chat", "agent": "dissident", "expected": "normal"},
    
    # Complex queries
    {"query": "What is the relationship between quantum mechanics and general relativity, and how does this relate to consciousness?", "mode": "chat", "expected": "normal"},
    {"query": "Explain the history of AI from 1950 to present, including major breakthroughs and current state", "mode": "chat", "expected": "normal"},
]

async def test_query(ws, test_case: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    """Test a single query"""
    query = test_case["query"]
    mode = test_case.get("mode", "chat")
    agent = test_case.get("agent", "auto")
    expected = test_case.get("expected", "normal")
    
    print(f"\n{'='*60}")
    print(f"Testing: {query[:50]}...")
    print(f"Mode: {mode}, Agent: {agent}, Expected: {expected}")
    print(f"{'='*60}")
    
    start_time = time.time()
    response_received = False
    error_received = False
    chunks_received = []
    done_received = False
    
    try:
        # Send query
        message = {
            "query": query,
            "mode": mode,
            "agent": agent,
            "settings": {
                "primaryModel": "llama3.1:8b",
                "temperature": 0.7,
                "maxTokens": 2000
            }
        }
        
        await ws.send(json.dumps(message))
        print(f"✅ Query sent")
        
        # Wait for response with timeout
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                response = await asyncio.wait_for(ws.recv(), timeout=min(5.0, remaining))
                response_data = json.loads(response)
                
                response_type = response_data.get("type", "unknown")
                
                if response_type == "chunk":
                    chunks_received.append(response_data.get("content", ""))
                    print(f"  📝 Chunk received: {len(chunks_received)} chunks")
                elif response_type == "done":
                    done_received = True
                    print(f"  ✅ Done received")
                    break
                elif response_type == "error":
                    error_received = True
                    print(f"  ❌ Error: {response_data.get('message', 'Unknown error')}")
                    break
                elif response_type == "thinking":
                    print(f"  💭 Thinking: {response_data.get('content', '')[:50]}...")
                elif response_type == "action":
                    print(f"  ⚙️  Action: {response_data.get('action', '')} - {response_data.get('status', '')}")
                else:
                    print(f"  📨 {response_type}: {str(response_data)[:100]}...")
            except asyncio.TimeoutError:
                print(f"  ⏱️  Timeout waiting for response")
                continue
        
        if time.time() >= end_time:
            print(f"  ⏱️  Overall timeout after {timeout}s")
            
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {
            "test": query[:50],
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time
        }
    
    elapsed = time.time() - start_time
    response_received = done_received or len(chunks_received) > 0
    
    # Determine result
    if error_received:
        status = "error"
    elif response_received:
        if elapsed < 1.0 and expected == "fast":
            status = "fast_success"
        elif elapsed < 5.0:
            status = "success"
        else:
            status = "slow"
    else:
        status = "no_response"
    
    result = {
        "test": query[:50],
        "status": status,
        "time": elapsed,
        "chunks": len(chunks_received),
        "expected": expected,
        "response_received": response_received,
        "error_received": error_received
    }
    
    # Print result
    if status == "fast_success":
        print(f"  ✅ FAST SUCCESS ({elapsed:.2f}s)")
    elif status == "success":
        print(f"  ✅ SUCCESS ({elapsed:.2f}s)")
    elif status == "slow":
        print(f"  ⚠️  SLOW ({elapsed:.2f}s)")
    elif status == "error":
        print(f"  ❌ ERROR")
    else:
        print(f"  ⚠️  NO RESPONSE")
    
    return result

async def run_tests():
    """Run all test cases"""
    uri = "ws://localhost:8000/ws"
    
    print("="*60)
    print("ICEBURG Server Comprehensive Test Suite")
    print("="*60)
    
    results = []
    
    try:
        async with websockets.connect(uri) as ws:
            print(f"✅ Connected to {uri}")
            
            # Wait for connection confirmation
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                if data.get("type") == "connected":
                    print(f"✅ Connection confirmed: {data.get('message', '')}")
            except asyncio.TimeoutError:
                print("⚠️  No connection confirmation, continuing anyway...")
            
            # Run all test cases
            for i, test_case in enumerate(TEST_CASES, 1):
                print(f"\n[{i}/{len(TEST_CASES)}]")
                result = await test_query(ws, test_case)
                results.append(result)
                
                # Small delay between tests
                await asyncio.sleep(0.5)
            
    except ConnectionRefusedError:
        print("❌ Connection refused - is the server running?")
        print("   Start server: python3 -m uvicorn src.iceburg.api.server:app --host 0.0.0.0 --port 8000")
        return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    fast_success = sum(1 for r in results if r["status"] == "fast_success")
    success = sum(1 for r in results if r["status"] == "success")
    slow = sum(1 for r in results if r["status"] == "slow")
    errors = sum(1 for r in results if r["status"] == "error")
    no_response = sum(1 for r in results if r["status"] == "no_response")
    
    print(f"Total tests: {total}")
    print(f"✅ Fast success (<1s): {fast_success}")
    print(f"✅ Success (<5s): {success}")
    print(f"⚠️  Slow (>5s): {slow}")
    print(f"❌ Errors: {errors}")
    print(f"⚠️  No response: {no_response}")
    
    # Average times
    times = [r["time"] for r in results if r["status"] in ["fast_success", "success", "slow"]]
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage response time: {avg_time:.2f}s")
        print(f"Fastest: {min(times):.2f}s")
        print(f"Slowest: {max(times):.2f}s")
    
    # Failed tests
    failed = [r for r in results if r["status"] in ["error", "no_response"]]
    if failed:
        print(f"\n❌ Failed tests ({len(failed)}):")
        for r in failed:
            print(f"  - {r['test']}: {r['status']}")
    
    # Success rate
    success_rate = (fast_success + success) / total * 100 if total > 0 else 0
    print(f"\n✅ Success rate: {success_rate:.1f}%")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_tests())

