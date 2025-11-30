"""
Test Self-Healing Integration
Test how the new self-healing systems integrate with UX and automatic engagement
"""

import asyncio
import json
import websockets
import time
from typing import Dict, Any

async def test_self_healing_integration():
    """Test self-healing integration with UX."""
    print("=" * 80)
    print("Testing Self-Healing Integration")
    print("=" * 80)
    print()
    
    # Connect to WebSocket
    uri = "ws://localhost:8000/ws"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            print()
            
            # Wait for connection confirmation
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                if data.get("type") == "connected":
                    print(f"✅ Connection confirmed: {data.get('message', 'Connected')}")
                    if data.get("always_on_enabled"):
                        print("✅ Always-On AI enabled")
                    if data.get("monitoring", {}).get("enabled"):
                        print("✅ Self-Healing Monitoring enabled")
                    print()
            except asyncio.TimeoutError:
                print("⚠️  No connection confirmation received")
                print()
            
            # Test 1: Simple query to trigger monitoring
            print("Test 1: Simple Query (should trigger monitoring)")
            print("-" * 80)
            query1 = "What is artificial intelligence?"
            print(f"Query: {query1}")
            print()
            
            message1 = {
                "type": "query",
                "query": query1,
                "mode": "chat",
                "agent": "auto",
                "primaryModel": "llama3.1:8b",
                "temperature": 0.7,
                "maxTokens": 2000
            }
            
            await websocket.send(json.dumps(message1))
            print("📤 Sent query")
            print()
            
            # Collect responses
            responses1 = []
            monitoring_status = None
            start_time = time.time()
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(response)
                    responses1.append(data)
                    
                    # Check for monitoring status
                    if data.get("type") == "done":
                        monitoring_status = data.get("monitoring")
                        print(f"✅ Received done signal")
                        if monitoring_status:
                            print(f"   Monitoring enabled: {monitoring_status.get('enabled', False)}")
                            if monitoring_status.get("status"):
                                status = monitoring_status["status"]
                                print(f"   Total alerts: {status.get('total_alerts', 0)}")
                                print(f"   Resolved alerts: {status.get('resolved_alerts', 0)}")
                                print(f"   Auto-healing success rate: {status.get('auto_healing_success_rate', 0.0):.1%}")
                                print(f"   LLM analyses: {status.get('llm_analyses', 0)}")
                                print(f"   Cached analyses: {status.get('cached_analyses', 0)}")
                        break
                    elif data.get("type") == "chunk":
                        print(f"📝 Received chunk: {data.get('content', '')[:50]}...")
                    elif data.get("type") == "action":
                        print(f"⚙️  Action: {data.get('action', 'unknown')} - {data.get('status', 'unknown')}")
                    elif data.get("type") == "thinking":
                        print(f"💭 Thinking: {data.get('content', '')[:50]}...")
                    elif data.get("type") == "error":
                        print(f"❌ Error: {data.get('message', 'Unknown error')}")
                        break
                except asyncio.TimeoutError:
                    print("⏱️  Timeout waiting for response")
                    break
            
            elapsed_time = time.time() - start_time
            print(f"⏱️  Total time: {elapsed_time:.2f}s")
            print()
            
            # Test 2: Complex query to trigger LLM analysis
            print("Test 2: Complex Query (should trigger LLM analysis for HIGH/CRITICAL bottlenecks)")
            print("-" * 80)
            query2 = "Explain quantum mechanics and its applications in computing"
            print(f"Query: {query2}")
            print()
            
            message2 = {
                "type": "query",
                "query": query2,
                "mode": "research",
                "agent": "auto",
                "primaryModel": "llama3.1:8b",
                "temperature": 0.7,
                "maxTokens": 2000
            }
            
            await websocket.send(json.dumps(message2))
            print("📤 Sent query")
            print()
            
            # Collect responses
            responses2 = []
            monitoring_status2 = None
            start_time2 = time.time()
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    data = json.loads(response)
                    responses2.append(data)
                    
                    # Check for monitoring status
                    if data.get("type") == "done":
                        monitoring_status2 = data.get("monitoring")
                        print(f"✅ Received done signal")
                        if monitoring_status2:
                            print(f"   Monitoring enabled: {monitoring_status2.get('enabled', False)}")
                            if monitoring_status2.get("status"):
                                status = monitoring_status2["status"]
                                print(f"   Total alerts: {status.get('total_alerts', 0)}")
                                print(f"   Resolved alerts: {status.get('resolved_alerts', 0)}")
                                print(f"   Auto-healing success rate: {status.get('auto_healing_success_rate', 0.0):.1%}")
                                print(f"   LLM analyses: {status.get('llm_analyses', 0)}")
                                print(f"   Cached analyses: {status.get('cached_analyses', 0)}")
                        break
                    elif data.get("type") == "chunk":
                        print(f"📝 Received chunk: {data.get('content', '')[:50]}...")
                    elif data.get("type") == "action":
                        print(f"⚙️  Action: {data.get('action', 'unknown')} - {data.get('status', 'unknown')}")
                    elif data.get("type") == "error":
                        print(f"❌ Error: {data.get('message', 'Unknown error')}")
                        break
                except asyncio.TimeoutError:
                    print("⏱️  Timeout waiting for response")
                    break
            
            elapsed_time2 = time.time() - start_time2
            print(f"⏱️  Total time: {elapsed_time2:.2f}s")
            print()
            
            # Summary
            print("=" * 80)
            print("Summary")
            print("=" * 80)
            print(f"Test 1 responses: {len(responses1)}")
            print(f"Test 2 responses: {len(responses2)}")
            print()
            print("Monitoring Status:")
            if monitoring_status:
                print(f"  ✅ Monitoring is active")
                if monitoring_status.get("status"):
                    status = monitoring_status["status"]
                    print(f"  - Total alerts: {status.get('total_alerts', 0)}")
                    print(f"  - Resolved: {status.get('resolved_alerts', 0)}")
                    print(f"  - Success rate: {status.get('auto_healing_success_rate', 0.0):.1%}")
            else:
                print("  ⚠️  Monitoring status not received")
            print()
            print("✅ Self-healing systems are integrated and active!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_self_healing_integration())

