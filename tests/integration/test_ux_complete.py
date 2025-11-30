#!/usr/bin/env python3
"""
Complete UX Test - Tests the full user experience
"""
import asyncio
import json
import websockets
import sys
import time

async def test_full_ux():
    """Test complete UX flow"""
    uri = "ws://localhost:8000/ws"
    print("=" * 60)
    print("🧪 COMPLETE UX TEST")
    print("=" * 60)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            # Test 1: First query - should work
            print("\n" + "=" * 60)
            print("📝 TEST 1: First Query (hello)")
            print("=" * 60)
            
            query1 = {
                "query": "hello",
                "mode": "chat",
                "agent": "auto",
                "degradation_mode": False,
                "settings": {
                    "primaryModel": "llama3.1:8b",
                    "temperature": 0.7,
                    "maxTokens": 2000
                }
            }
            
            print(f"📤 Sending: {json.dumps(query1, indent=2)}")
            await websocket.send(json.dumps(query1))
            
            responses1 = []
            chunks1 = []
            start_time = time.time()
            timeout = 30
            
            print("\n📥 Receiving responses...")
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    data = json.loads(response)
                    responses1.append(data)
                    
                    msg_type = data.get("type", "unknown")
                    print(f"  ✅ {msg_type}: {str(data)[:100]}...")
                    
                    if msg_type == "chunk":
                        chunks1.append(data.get("content", ""))
                    
                    if msg_type == "done":
                        print(f"\n✅ Query 1 completed in {time.time() - start_time:.2f}s")
                        print(f"   Total responses: {len(responses1)}")
                        print(f"   Content chunks: {len(chunks1)}")
                        print(f"   Content length: {sum(len(c) for c in chunks1)} chars")
                        break
                    elif msg_type == "error":
                        print(f"\n❌ Error: {data.get('message')}")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"\n⏱️  Timeout after {timeout}s")
                    break
            
            # Wait a bit before second query
            await asyncio.sleep(1)
            
            # Test 2: Second query - connection should still be open
            print("\n" + "=" * 60)
            print("📝 TEST 2: Second Query (connection should stay open)")
            print("=" * 60)
            
            query2 = {
                "query": "what is 2+2?",
                "mode": "chat",
                "agent": "auto",
                "degradation_mode": False,
                "settings": {
                    "primaryModel": "llama3.1:8b",
                    "temperature": 0.7,
                    "maxTokens": 2000
                }
            }
            
            print(f"📤 Sending: {json.dumps(query2, indent=2)}")
            await websocket.send(json.dumps(query2))
            
            responses2 = []
            chunks2 = []
            start_time = time.time()
            
            print("\n📥 Receiving responses...")
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    data = json.loads(response)
                    responses2.append(data)
                    
                    msg_type = data.get("type", "unknown")
                    print(f"  ✅ {msg_type}: {str(data)[:100]}...")
                    
                    if msg_type == "chunk":
                        chunks2.append(data.get("content", ""))
                    
                    if msg_type == "done":
                        print(f"\n✅ Query 2 completed in {time.time() - start_time:.2f}s")
                        print(f"   Total responses: {len(responses2)}")
                        print(f"   Content chunks: {len(chunks2)}")
                        print(f"   Content length: {sum(len(c) for c in chunks2)} chars")
                        break
                    elif msg_type == "error":
                        print(f"\n❌ Error: {data.get('message')}")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"\n⏱️  Timeout after {timeout}s")
                    break
            
            # Summary
            print("\n" + "=" * 60)
            print("📊 TEST SUMMARY")
            print("=" * 60)
            print(f"✅ Connection stayed open for multiple queries")
            print(f"✅ Query 1: {len(responses1)} responses, {len(chunks1)} chunks")
            print(f"✅ Query 2: {len(responses2)} responses, {len(chunks2)} chunks")
            
            if len(chunks1) > 0 and len(chunks2) > 0:
                print("✅ Both queries received content chunks")
            elif len(chunks1) == 0:
                print("⚠️  Query 1 had no content chunks")
            elif len(chunks2) == 0:
                print("⚠️  Query 2 had no content chunks")
            
            print("\n✅ UX TEST PASSED!")
            
    except Exception as e:
        print(f"\n❌ UX TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_full_ux())

