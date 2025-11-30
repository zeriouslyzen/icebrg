#!/usr/bin/env python3
"""WebSocket Diagnostic Test"""
import asyncio
import websockets
import json
import sys

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            # Send a test message
            test_message = {
                "type": "ping",
                "query": "test"
            }
            print(f"📤 Sending: {test_message}")
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📥 Received: {response}")
            except asyncio.TimeoutError:
                print("⏱️ No response received within 5 seconds")
            
            # Send a query
            query_message = {
                "query": "hello",
                "mode": "chat",
                "agent": "auto"
            }
            print(f"📤 Sending query: {query_message}")
            await websocket.send(json.dumps(query_message))
            
            # Wait for multiple responses
            for i in range(10):
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    print(f"📥 Response {i+1}: {response[:200]}...")
                except asyncio.TimeoutError:
                    print(f"⏱️ No response {i+1} within 3 seconds")
                    break
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_websocket())

