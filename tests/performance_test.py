#!/usr/bin/env python3
"""
ICEBURG Performance Testing Script
Tests response times, connection handling, and streaming performance
"""

import asyncio
import websockets
import json
import time
import statistics
from typing import List, Dict, Any
import sys

# Configuration
WS_URL = "ws://localhost:8000/ws"
TEST_QUERIES = [
    # Simple queries (should use fast path)
    "hi",
    "hello",
    "hey",
    
    # Medium queries (single agent chat)
    "What is quantum computing?",
    "Explain machine learning in simple terms.",
    
    # Complex queries (full protocol)
    "Research the latest developments in AGI and provide a comprehensive analysis.",
]

async def test_simple_query(query: str, api_key: str = None) -> Dict[str, Any]:
    """Test a single query and measure performance"""
    start_time = time.time()
    response_chunks = []
    thinking_received = False
    etymology_received = False
    first_chunk_time = None
    done_received = False
    
    try:
        # Connect with API key if provided
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        uri = WS_URL
        if api_key and "?" not in uri:
            uri = f"{uri}?api_key={api_key}"
        
        async with websockets.connect(uri, extra_headers=headers) as websocket:
            # Send query
            message = {
                "query": query,
                "mode": "chat",
                "agent": "auto",
                "degradation_mode": False
            }
            await websocket.send(json.dumps(message))
            
            # Receive messages
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "thinking":
                        thinking_received = True
                    elif data.get("type") == "word_breakdown":
                        etymology_received = True
                    elif data.get("type") == "chunk":
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - start_time
                        response_chunks.append(data.get("content", ""))
                    elif data.get("type") == "done":
                        done_received = True
                        break
                    elif data.get("type") == "error":
                        return {
                            "query": query,
                            "error": data.get("message", "Unknown error"),
                            "success": False
                        }
                except asyncio.TimeoutError:
                    return {
                        "query": query,
                        "error": "Timeout waiting for response",
                        "success": False
                    }
        
        total_time = time.time() - start_time
        response_text = "".join(response_chunks)
        
        return {
            "query": query,
            "success": True,
            "total_time": total_time,
            "first_chunk_time": first_chunk_time,
            "thinking_received": thinking_received,
            "etymology_received": etymology_received,
            "response_length": len(response_text),
            "chunk_count": len(response_chunks),
            "done_received": done_received
        }
    
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "success": False
        }

async def test_concurrent_connections(num_connections: int = 5, api_key: str = None) -> Dict[str, Any]:
    """Test multiple concurrent connections"""
    print(f"\n📊 Testing {num_connections} concurrent connections...")
    
    tasks = []
    for i in range(num_connections):
        query = f"Test query {i+1}: What is AI?"
        tasks.append(test_simple_query(query, api_key))
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if r.get("success", False))
    failed = num_connections - successful
    
    return {
        "total_connections": num_connections,
        "successful": successful,
        "failed": failed,
        "total_time": total_time,
        "avg_time_per_connection": total_time / num_connections if num_connections > 0 else 0,
        "results": results
    }

async def run_performance_tests(api_key: str = None):
    """Run comprehensive performance tests"""
    print("🚀 ICEBURG Performance Testing")
    print("=" * 60)
    
    # Test 1: Simple queries (fast path)
    print("\n📝 Test 1: Simple Queries (Fast Path)")
    print("-" * 60)
    simple_results = []
    for query in TEST_QUERIES[:3]:  # First 3 are simple
        print(f"  Testing: '{query}'...", end=" ", flush=True)
        result = await test_simple_query(query, api_key)
        simple_results.append(result)
        if result.get("success"):
            print(f"✅ {result['total_time']:.3f}s (first chunk: {result.get('first_chunk_time', 0):.3f}s)")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    # Test 2: Medium queries (single agent)
    print("\n📝 Test 2: Medium Queries (Single Agent Chat)")
    print("-" * 60)
    medium_results = []
    for query in TEST_QUERIES[3:5]:  # Medium queries
        print(f"  Testing: '{query[:50]}...'...", end=" ", flush=True)
        result = await test_simple_query(query, api_key)
        medium_results.append(result)
        if result.get("success"):
            print(f"✅ {result['total_time']:.3f}s (first chunk: {result.get('first_chunk_time', 0):.3f}s)")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    # Test 3: Concurrent connections
    concurrent_result = await test_concurrent_connections(5, api_key)
    print(f"\n✅ Successful: {concurrent_result['successful']}/{concurrent_result['total_connections']}")
    print(f"⏱️  Total time: {concurrent_result['total_time']:.3f}s")
    print(f"📊 Avg per connection: {concurrent_result['avg_time_per_connection']:.3f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Performance Summary")
    print("=" * 60)
    
    # Simple queries
    simple_times = [r['total_time'] for r in simple_results if r.get('success')]
    if simple_times:
        print(f"\n⚡ Simple Queries (Fast Path):")
        print(f"   Average: {statistics.mean(simple_times):.3f}s")
        print(f"   Min: {min(simple_times):.3f}s")
        print(f"   Max: {max(simple_times):.3f}s")
        print(f"   Target: <0.1s (✅ {'PASS' if statistics.mean(simple_times) < 0.1 else 'FAIL'})")
    
    # Medium queries
    medium_times = [r['total_time'] for r in medium_results if r.get('success')]
    if medium_times:
        print(f"\n💬 Medium Queries (Single Agent):")
        print(f"   Average: {statistics.mean(medium_times):.3f}s")
        print(f"   Min: {min(medium_times):.3f}s")
        print(f"   Max: {max(medium_times):.3f}s")
        print(f"   Target: <30s (✅ {'PASS' if statistics.mean(medium_times) < 30 else 'FAIL'})")
    
    # First chunk times (streaming responsiveness)
    first_chunk_times = [r.get('first_chunk_time', 0) for r in simple_results + medium_results if r.get('success') and r.get('first_chunk_time')]
    if first_chunk_times:
        print(f"\n📡 Streaming Responsiveness (First Chunk):")
        print(f"   Average: {statistics.mean(first_chunk_times):.3f}s")
        print(f"   Min: {min(first_chunk_times):.3f}s")
        print(f"   Max: {max(first_chunk_times):.3f}s")
        print(f"   Target: <1s (✅ {'PASS' if statistics.mean(first_chunk_times) < 1 else 'FAIL'})")
    
    # Etymology check
    etymology_count = sum(1 for r in simple_results + medium_results if r.get('etymology_received'))
    print(f"\n🔤 Etymology Breakdown:")
    print(f"   Received: {etymology_count}/{len(simple_results + medium_results)}")
    print(f"   Status: {'✅ Working' if etymology_count > 0 else '⚠️  Not received'}")
    
    # Thinking messages
    thinking_count = sum(1 for r in simple_results + medium_results if r.get('thinking_received'))
    print(f"\n🧠 Thinking Messages:")
    print(f"   Received: {thinking_count}/{len(simple_results + medium_results)}")
    print(f"   Status: {'✅ Working' if thinking_count > 0 else '⚠️  Not received'}")
    
    print("\n" + "=" * 60)
    print("✅ Performance testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Get API key from command line if provided
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    if api_key:
        print(f"🔑 Using API key: {api_key[:10]}...")
    else:
        print("⚠️  No API key provided (will fail if authentication required)")
    
    asyncio.run(run_performance_tests(api_key))

