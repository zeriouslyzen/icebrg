#!/usr/bin/env python3
"""
Reliability Test - Tests the system with multiple queries to ensure 100% reliability
"""
import asyncio
import json
import websockets
import sys
import time

async def test_reliability():
    """Test system reliability with multiple queries"""
    uri = "ws://localhost:8000/ws"
    print("=" * 60)
    print("🧪 RELIABILITY TEST - 100% Success Rate")
    print("=" * 60)
    
    queries = [
        "hello",
        "what is 2+2?",
        "test",
        "hi",
        "explain quantum computing",
        "what is the meaning of life?",
        "how does a computer work?",
        "what is AI?",
        "tell me a joke",
        "what is the weather?"
    ]
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            success_count = 0
            total_queries = len(queries)
            
            for i, query in enumerate(queries, 1):
                print(f"\n{'='*60}")
                print(f"📝 Query {i}/{total_queries}: {query}")
                print(f"{'='*60}")
                
                message = {
                    "query": query,
                    "mode": "chat",
                    "agent": "auto",
                    "degradation_mode": False,
                    "settings": {
                        "primaryModel": "llama3.1:8b",
                        "temperature": 0.7,
                        "maxTokens": 2000
                    }
                }
                
                print(f"📤 Sending: {json.dumps(message, indent=2)}")
                await websocket.send(json.dumps(message))
                
                responses = []
                chunks = []
                start_time = time.time()
                timeout = 30
                done_received = False
                error_received = False
                
                print("\n📥 Receiving responses...")
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                        data = json.loads(response)
                        responses.append(data)
                        
                        msg_type = data.get("type", "unknown")
                        
                        if msg_type == "chunk":
                            chunks.append(data.get("content", ""))
                        elif msg_type == "done":
                            done_received = True
                            print(f"  ✅ done received")
                            break
                        elif msg_type == "error":
                            error_received = True
                            print(f"  ⚠️  error: {data.get('message', 'Unknown error')}")
                            # Continue to wait for done signal
                        else:
                            print(f"  ✅ {msg_type}")
                        
                        elapsed = time.time() - start_time
                        if elapsed > timeout:
                            print(f"  ⏱️  Timeout after {timeout}s")
                            break
                            
                    except asyncio.TimeoutError:
                        print(f"  ⏱️  No response within {timeout}s")
                        break
                
                elapsed = time.time() - start_time
                content_length = sum(len(c) for c in chunks)
                
                # Check if query was successful
                if done_received and (len(chunks) > 0 or not error_received):
                    success_count += 1
                    status = "✅ SUCCESS"
                else:
                    status = "❌ FAILED"
                
                print(f"\n{status}")
                print(f"   Time: {elapsed:.2f}s")
                print(f"   Responses: {len(responses)}")
                print(f"   Chunks: {len(chunks)}")
                print(f"   Content length: {content_length} chars")
                print(f"   Done received: {done_received}")
                print(f"   Error received: {error_received}")
                
                # Wait a bit before next query
                await asyncio.sleep(0.5)
            
            # Summary
            print("\n" + "=" * 60)
            print("📊 RELIABILITY TEST SUMMARY")
            print("=" * 60)
            print(f"Total queries: {total_queries}")
            print(f"Successful: {success_count}")
            print(f"Failed: {total_queries - success_count}")
            print(f"Success rate: {(success_count / total_queries) * 100:.1f}%")
            
            if success_count == total_queries:
                print("\n✅ ALL TESTS PASSED - 100% RELIABILITY!")
                return True
            else:
                print(f"\n❌ {total_queries - success_count} TEST(S) FAILED")
                return False
            
    except Exception as e:
        print(f"\n❌ RELIABILITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_reliability())
    sys.exit(0 if success else 1)

