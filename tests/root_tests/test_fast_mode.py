#!/usr/bin/env python3
"""
Test fast mode optimization
"""
import asyncio
import json
import time
import websockets
from typing import Dict, Any

async def test_fast_mode():
    """Test fast mode with a simple query"""
    uri = "ws://localhost:8000/ws"
    
    print("🔌 Connecting to WebSocket...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Wait for connection confirmation
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                if data.get("type") == "connected":
                    print(f"✅ Connection confirmed: {data.get('message')}")
            except asyncio.TimeoutError:
                print("⚠️ No connection confirmation received")
            
            # Test query
            test_query = "What is quantum mechanics?"
            print(f"\n📤 Sending test query: '{test_query}'")
            print("   Mode: fast")
            print("   Agent: surveyor")
            
            message = {
                "query": test_query,
                "mode": "fast",
                "agent": "surveyor",
                "settings": {
                    "primaryModel": "llama3.1:8b",
                    "temperature": 0.7,
                    "maxTokens": 2000
                }
            }
            
            start_time = time.time()
            await websocket.send(json.dumps(message))
            print("✅ Query sent, waiting for response...\n")
            
            # Track response times
            first_chunk_time = None
            response_times = []
            chunks_received = 0
            status_updates = []
            
            # Receive responses
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    elapsed = time.time() - start_time
                    
                    try:
                        data = json.loads(response)
                        msg_type = data.get("type", "unknown")
                        
                        if msg_type == "thinking":
                            print(f"💭 [{elapsed:.2f}s] Thinking: {data.get('content', '')[:50]}...")
                            status_updates.append(("thinking", elapsed))
                        
                        elif msg_type == "action":
                            action = data.get("action", "unknown")
                            status = data.get("status", "unknown")
                            print(f"🔍 [{elapsed:.2f}s] Action: {action} - {status}")
                            status_updates.append((f"action_{action}", elapsed))
                        
                        elif msg_type == "chunk":
                            if first_chunk_time is None:
                                first_chunk_time = elapsed
                                print(f"📝 [{elapsed:.2f}s] First chunk received!")
                            chunks_received += 1
                            chunk_text = data.get("content", "")
                            if chunks_received <= 3:
                                print(f"   Chunk {chunks_received}: {chunk_text[:50]}...")
                        
                        elif msg_type == "done":
                            total_time = time.time() - start_time
                            print(f"\n✅ [{total_time:.2f}s] Response complete!")
                            print(f"   First chunk: {first_chunk_time:.2f}s" if first_chunk_time else "   No chunks received")
                            print(f"   Total chunks: {chunks_received}")
                            print(f"   Status updates: {len(status_updates)}")
                            
                            # Performance analysis
                            print(f"\n📊 Performance Analysis:")
                            print(f"   Total time: {total_time:.2f}s")
                            if first_chunk_time:
                                print(f"   Time to first chunk: {first_chunk_time:.2f}s")
                                print(f"   Streaming time: {total_time - first_chunk_time:.2f}s")
                            
                            if total_time < 15:
                                print(f"   ✅ FAST MODE: Response under 15s (target: 5-15s)")
                            elif total_time < 30:
                                print(f"   ⚠️ MODERATE: Response 15-30s (should be faster)")
                            else:
                                print(f"   ❌ SLOW: Response over 30s (needs optimization)")
                            
                            break
                        
                        elif msg_type == "error":
                            print(f"❌ [{elapsed:.2f}s] Error: {data.get('message', 'Unknown error')}")
                            break
                        
                        else:
                            print(f"📨 [{elapsed:.2f}s] {msg_type}: {str(data)[:50]}...")
                    
                    except json.JSONDecodeError:
                        print(f"⚠️ [{elapsed:.2f}s] Non-JSON response: {response[:50]}...")
            
            except asyncio.TimeoutError:
                total_time = time.time() - start_time
                print(f"\n⏱️ Timeout after {total_time:.2f}s")
                print("   Response may still be processing...")
    
    except ConnectionRefusedError:
        print("❌ Connection refused - is the server running?")
        print("   Start server with: python -m uvicorn src.iceburg.api.server:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Fast Mode Optimization\n")
    asyncio.run(test_fast_mode())

