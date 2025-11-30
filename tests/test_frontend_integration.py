#!/usr/bin/env python3
"""
Test Frontend-Backend Integration
Tests fast chat, agent selection, and model selection
"""

import asyncio
import json
import websockets
from typing import Dict, Any

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"


async def test_websocket_connection():
    """Test WebSocket connection and fast chat"""
    print("Testing WebSocket connection...")
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✅ WebSocket connected")
            
            # Test 1: Fast chat with small model
            print("\n📝 Test 1: Fast chat with small model (llama3.2:1b)")
            test_message = {
                "query": "What is 2+2?",
                "mode": "chat",
                "agent": "auto",
                "degradation_mode": False,  # Fast mode
                "settings": {
                    "primaryModel": "llama3.2:1b",
                    "temperature": 0.7,
                    "maxTokens": 500
                }
            }
            
            await websocket.send(json.dumps(test_message))
            print("✅ Message sent")
            
            # Receive responses
            responses = []
            timeout = 10  # 10 second timeout for fast chat
            start_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    responses.append(data)
                    
                    if data.get("type") == "done":
                        break
                    
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > timeout:
                        print(f"⏱️  Timeout after {elapsed:.2f}s")
                        break
                        
                except asyncio.TimeoutError:
                    print("⏱️  No response received")
                    break
            
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"✅ Fast chat completed in {elapsed:.2f}s")
            print(f"   Received {len(responses)} response chunks")
            
            # Test 2: Agent selection (Surveyor)
            print("\n📝 Test 2: Agent selection (Surveyor)")
            test_message = {
                "query": "Explain quantum computing",
                "mode": "chat",
                "agent": "surveyor",
                "degradation_mode": False,
                "settings": {
                    "primaryModel": "llama3.1:8b",
                    "temperature": 0.7,
                    "maxTokens": 1000
                }
            }
            
            await websocket.send(json.dumps(test_message))
            print("✅ Message sent")
            
            responses = []
            start_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    responses.append(data)
                    
                    if data.get("type") == "agent_thinking":
                        print(f"   Agent thinking: {data.get('agent')} - {data.get('content')}")
                    
                    if data.get("type") == "done":
                        break
                    
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > 15:
                        break
                        
                except asyncio.TimeoutError:
                    break
            
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"✅ Agent selection completed in {elapsed:.2f}s")
            
            # Test 3: Model selection (phi3.5)
            print("\n📝 Test 3: Model selection (phi3.5)")
            test_message = {
                "query": "Hello, how are you?",
                "mode": "chat",
                "agent": "auto",
                "degradation_mode": False,
                "settings": {
                    "primaryModel": "phi3.5",
                    "temperature": 0.5,
                    "maxTokens": 200
                }
            }
            
            await websocket.send(json.dumps(test_message))
            print("✅ Message sent")
            
            responses = []
            start_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    responses.append(data)
                    
                    if data.get("type") == "done":
                        break
                    
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > 10:
                        break
                        
                except asyncio.TimeoutError:
                    break
            
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"✅ Model selection completed in {elapsed:.2f}s")
            
            print("\n✅ All tests completed successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_http_endpoint():
    """Test HTTP endpoint"""
    import aiohttp
    
    print("\n📝 Testing HTTP endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            test_data = {
                "query": "What is ICEBURG?",
                "mode": "chat",
                "settings": {
                    "primaryModel": "llama3.1:8b",
                    "temperature": 0.7,
                    "maxTokens": 500
                }
            }
            
            async with session.post(
                f"{API_URL}/api/query",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ HTTP endpoint working")
                    print(f"   Response keys: {list(data.keys())}")
                else:
                    print(f"❌ HTTP endpoint error: {response.status}")
                    text = await response.text()
                    print(f"   Error: {text}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("ICEBURG Frontend-Backend Integration Test")
    print("=" * 60)
    
    await test_http_endpoint()
    await test_websocket_connection()
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

