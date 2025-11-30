#!/usr/bin/env python3
"""
Test WebSocket connection with "hi" query
"""
import asyncio
import json
import websockets
import sys

async def test_websocket():
    """Test WebSocket connection with 'hi' query"""
    uri = "ws://localhost:8000/ws"
    
    try:
        print("Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Send "hi" query
            message = {
                "query": "hi",
                "mode": "chat",
                "agent": "auto",
                "degradation_mode": False,
                "settings": {
                    "primaryModel": "llama3.1:8b",
                    "temperature": 0.7,
                    "maxTokens": 2000
                }
            }
            
            print(f"\n📤 Sending message: {json.dumps(message, indent=2)}")
            await websocket.send(json.dumps(message))
            
            # Wait for responses
            print("\n📥 Waiting for responses...")
            responses = []
            timeout = 30  # 30 second timeout
            
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    data = json.loads(response)
                    responses.append(data)
                    
                    msg_type = data.get("type", "unknown")
                    content = data.get("content", data.get("message", ""))
                    
                    if msg_type == "thinking":
                        print(f"💭 Thinking: {content}")
                    elif msg_type == "chunk":
                        print(f"📝 Chunk: {content[:50]}...")
                    elif msg_type == "error":
                        print(f"❌ Error: {content}")
                        break
                    elif msg_type == "done":
                        print("✅ Done signal received")
                        break
                    else:
                        print(f"📦 {msg_type}: {str(data)[:100]}...")
                    
            except asyncio.TimeoutError:
                print(f"\n⏱️  Timeout after {timeout} seconds")
                print(f"Received {len(responses)} responses")
            
            # Print summary
            print(f"\n📊 Summary:")
            print(f"  Total responses: {len(responses)}")
            print(f"  Response types: {[r.get('type') for r in responses]}")
            
            # Check for errors
            errors = [r for r in responses if r.get("type") == "error"]
            if errors:
                print(f"\n❌ Errors found:")
                for error in errors:
                    print(f"  - {error.get('message', error)}")
                return False
            
            # Check for content
            chunks = [r.get("content", "") for r in responses if r.get("type") == "chunk"]
            if chunks:
                full_content = "".join(chunks)
                print(f"\n✅ Received content ({len(full_content)} chars):")
                print(f"  {full_content[:200]}...")
                return True
            else:
                print("\n⚠️  No content chunks received")
                return False
                
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Is the server running?")
        print("   Start server with: python -m iceburg.api.server")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing WebSocket connection with 'hi' query\n")
    success = asyncio.run(test_websocket())
    sys.exit(0 if success else 1)

