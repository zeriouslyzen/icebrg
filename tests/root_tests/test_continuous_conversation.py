#!/usr/bin/env python3
"""
Test continuous conversation flow and reasoning quality
"""
import asyncio
import websockets
import json
import time

async def test_continuous_conversation():
    """Test a continuous conversation with multiple queries"""
    uri = "ws://localhost:8000/ws?conversation_id=test_conv_continuous"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Wait for connection confirmation
            try:
                confirmation = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(confirmation)
                if data.get("type") == "connected":
                    print("✅ Connection confirmed")
            except asyncio.TimeoutError:
                print("⚠️ No confirmation received, continuing...")
            
            # Test queries in sequence
            queries = [
                "What is 2+2?",
                "What about 3+3?",
                "Can you explain why those answers are correct?"
            ]
            
            for i, query in enumerate(queries, 1):
                print(f"\n{'='*60}")
                print(f"Query {i}: {query}")
                print(f"{'='*60}")
                
                # Send query
                message = {
                    "query": query,
                    "mode": "chat",
                    "agent": "auto",
                    "settings": {
                        "primaryModel": "llama3.1:8b",
                        "temperature": 0.7,
                        "maxTokens": 2000
                    }
                }
                await websocket.send(json.dumps(message))
                print(f"📤 Sent: {query}")
                
                # Collect response
                response_chunks = []
                thinking_messages = []
                word_breakdowns = []
                done_received = False
                
                start_time = time.time()
                timeout = 60.0
                
                while not done_received and (time.time() - start_time) < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(response)
                        
                        msg_type = data.get("type", "")
                        
                        if msg_type == "thinking":
                            thinking_messages.append(data.get("content", ""))
                            print(f"💭 Thinking: {data.get('content', '')[:50]}...")
                        
                        elif msg_type == "word_breakdown":
                            word_breakdowns.append(data)
                            print(f"📚 Word breakdown: {data.get('word', '')}")
                        
                        elif msg_type == "chunk":
                            response_chunks.append(data.get("content", ""))
                            # Print first few chunks to show streaming
                            if len(response_chunks) <= 5:
                                print(f"📝 Chunk: {data.get('content', '')}")
                        
                        elif msg_type == "done":
                            done_received = True
                            print(f"✅ Done signal received")
                            break
                        
                        elif msg_type == "error":
                            print(f"❌ Error: {data.get('message', '')}")
                            break
                    
                    except asyncio.TimeoutError:
                        print("⏱️ Waiting for response...")
                        continue
                    except Exception as e:
                        print(f"❌ Error receiving: {e}")
                        break
                
                # Display full response
                full_response = "".join(response_chunks)
                response_time = time.time() - start_time
                
                print(f"\n📊 Response Summary:")
                print(f"   Time: {response_time:.2f}s")
                print(f"   Chunks: {len(response_chunks)}")
                print(f"   Length: {len(full_response)} chars")
                print(f"   Thinking messages: {len(thinking_messages)}")
                print(f"   Word breakdowns: {len(word_breakdowns)}")
                
                if full_response:
                    print(f"\n💬 Full Response:")
                    print(f"   {full_response[:200]}...")
                    if len(full_response) > 200:
                        print(f"   ... ({len(full_response) - 200} more chars)")
                else:
                    print("⚠️ No response content received")
                
                # Wait a bit between queries
                if i < len(queries):
                    await asyncio.sleep(2)
            
            print(f"\n{'='*60}")
            print("✅ Continuous conversation test complete!")
            print(f"{'='*60}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_continuous_conversation())

