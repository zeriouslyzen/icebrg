#!/usr/bin/env python3
"""
Test Fast -> Research -> Fast Chat pipeline via WebSocket.
Simulates the full UX flow and prints results.
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

WS_URL = "ws://localhost:8000/ws"
CONV_ID = f"pipeline_test_{int(time.time())}"
TIMEOUT = 300  # 5 minutes for long-running research


async def recv_with_timeout(websocket, timeout=TIMEOUT):
    """Receive message with timeout."""
    try:
        return await asyncio.wait_for(websocket.recv(), timeout=timeout)
    except asyncio.TimeoutError:
        raise Exception(f"WebSocket receive timeout after {timeout}s")


async def test_pipeline():
    """Test full pipeline: Fast -> Research -> Fast Chat with coherence."""
    print("=" * 70)
    print("ICEBURG Pipeline Test: Fast -> Research -> Fast Chat")
    print("=" * 70)
    print()
    
    uri = f"{WS_URL}?conversation_id={CONV_ID}"
    
    # Use longer ping_interval and ping_timeout for long-running operations
    async with websockets.connect(
        uri,
        ping_interval=30,
        ping_timeout=10,
        close_timeout=10
    ) as websocket:
        # Wait for connection confirmation
        init_msg = await websocket.recv()
        print(f"✅ Connected: {json.loads(init_msg).get('type', 'unknown')}")
        print()
        
        # === STEP 1: Fast mode research question ===
        print("📝 STEP 1: Sending research question in Fast mode...")
        query1 = "What do we know about connections between astrology and organs?"
        await websocket.send(json.dumps({
            "query": query1,
            "mode": "fast",
            "agent": "secretary",
            "settings": {"primaryModel": "llama3.1:8b", "temperature": 0.7, "maxTokens": 2000}
        }))
        
        fast_response = ""
        print(f"   Query: {query1}")
        print("   Waiting for Fast mode response...")
        while True:
            try:
                raw_msg = await recv_with_timeout(websocket, timeout=120)
                msg = json.loads(raw_msg)
            except Exception as e:
                print(f"\n⚠️  Receive error: {e}")
                if fast_response:
                    print(f"   Partial response received ({len(fast_response)} chars)")
                    break
                raise
            msg_type = msg.get("type")
            
            if msg_type == "chunk":
                fast_response += msg.get("content", "")
                print(".", end="", flush=True)
            elif msg_type == "done":
                print()
                print(f"✅ Fast mode response received ({len(fast_response)} chars)")
                print(f"   Preview: {fast_response[:150]}...")
                break
            elif msg_type == "error":
                print(f"\n❌ Error: {msg.get('message', 'Unknown error')}")
                return
        
        print()
        await asyncio.sleep(1)
        
        # === STEP 2: Run Research mode ===
        print("🔬 STEP 2: Running Research mode (same query)...")
        await websocket.send(json.dumps({
            "query": query1,  # Same query
            "mode": "research",
            "agent": "auto",
            "settings": {"primaryModel": "llama3.1:8b", "temperature": 0.7, "maxTokens": 2000}
        }))
        
        research_content = ""
        research_stages = []
        research_start = time.time()
        print("   Waiting for research to complete...")
        
        while True:
            try:
                raw_msg = await recv_with_timeout(websocket, timeout=TIMEOUT)
                msg = json.loads(raw_msg)
            except Exception as e:
                print(f"\n⚠️  Receive error during research: {e}")
                if research_stages:
                    print(f"   Research stages completed: {[s[0] for s in research_stages]}")
                if research_content:
                    print(f"   Partial research content received ({len(research_content)} chars)")
                    break
                raise
            msg_type = msg.get("type")
            
            if msg_type == "research_status":
                stage = msg.get("stage", "unknown")
                elapsed = msg.get("elapsed_seconds", 0)
                research_stages.append((stage, elapsed))
                if stage == "complete":
                    print(f"\n✅ Research complete! Total time: {elapsed}s")
                    print(f"   Stages: {', '.join([s[0] for s in research_stages])}")
                    break
                else:
                    print(f"   [{elapsed}s] {stage}...", end="\r", flush=True)
            elif msg_type == "chunk":
                research_content += msg.get("content", "")
            elif msg_type == "thinking_stream":
                # Progress updates
                pass
            elif msg_type == "done":
                if not research_content:
                    research_content = "Research completed but no content streamed"
                break
            elif msg_type == "error":
                print(f"\n❌ Research error: {msg.get('message', 'Unknown')}")
                return
        
        print()
        print(f"📊 Research report length: {len(research_content)} chars")
        print(f"   Preview: {research_content[:200]}...")
        print()
        await asyncio.sleep(1)
        
        # === STEP 3: Fast Chat follow-up (should see last_research) ===
        print("💬 STEP 3: Fast Chat follow-up (should reference research)...")
        query2 = "Summarize the main finding from the research and what we should do next."
        await websocket.send(json.dumps({
            "query": query2,
            "mode": "fast",  # Fast Chat
            "agent": "secretary",
            "settings": {"primaryModel": "llama3.1:8b", "temperature": 0.7, "maxTokens": 2000}
        }))
        
        chat_response = ""
        print(f"   Query: {query2}")
        print("   Waiting for Fast Chat response...")
        
        while True:
            try:
                raw_msg = await recv_with_timeout(websocket, timeout=120)
                msg = json.loads(raw_msg)
            except Exception as e:
                print(f"\n⚠️  Receive error: {e}")
                if chat_response:
                    print(f"   Partial response received ({len(chat_response)} chars)")
                    break
                raise
            msg_type = msg.get("type")
            
            if msg_type == "chunk":
                chat_response += msg.get("content", "")
                print(".", end="", flush=True)
            elif msg_type == "done":
                print()
                print(f"✅ Fast Chat response received ({len(chat_response)} chars)")
                break
            elif msg_type == "error":
                print(f"\n❌ Error: {msg.get('message', 'Unknown')}")
                return
        
        print()
        
        # === STEP 4: Check coherence ===
        print("🔍 STEP 4: Checking coherence...")
        query3 = "What did the Dissident say about that finding?"
        await websocket.send(json.dumps({
            "query": query3,
            "mode": "fast",
            "agent": "secretary",
            "settings": {"primaryModel": "llama3.1:8b", "temperature": 0.7, "maxTokens": 2000}
        }))
        
        coherence_response = ""
        print(f"   Query: {query3}")
        print("   Waiting for response...")
        
        while True:
            try:
                raw_msg = await recv_with_timeout(websocket, timeout=120)
                msg = json.loads(raw_msg)
            except Exception as e:
                print(f"\n⚠️  Receive error: {e}")
                if coherence_response:
                    print(f"   Partial response received ({len(coherence_response)} chars)")
                    break
                raise
            msg_type = msg.get("type")
            
            if msg_type == "chunk":
                coherence_response += msg.get("content", "")
                print(".", end="", flush=True)
            elif msg_type == "done":
                print()
                print(f"✅ Coherence check response received ({len(coherence_response)} chars)")
                break
            elif msg_type == "error":
                print(f"\n❌ Error: {msg.get('message', 'Unknown')}")
                return
        
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print()
        print("1️⃣ Fast mode response:")
        print(f"   {fast_response[:300]}...")
        print()
        print("2️⃣ Research report:")
        print(f"   {research_content[:300]}...")
        print()
        print("3️⃣ Fast Chat follow-up (should reference research):")
        print(f"   {chat_response[:300]}...")
        print()
        print("4️⃣ Coherence check (should remember research):")
        print(f"   {coherence_response[:300]}...")
        print()
        
        # Check if responses reference research
        research_keywords = ["research", "surveyor", "dissident", "synthesist", "oracle", "finding", "report"]
        chat_has_research = any(kw in chat_response.lower() for kw in research_keywords)
        coherence_has_research = any(kw in coherence_response.lower() for kw in research_keywords)
        
        print("=" * 70)
        print("VALIDATION")
        print("=" * 70)
        print(f"✅ Fast Chat references research: {chat_has_research}")
        print(f"✅ Coherence check references research: {coherence_has_research}")
        print()
        
        if chat_has_research and coherence_has_research:
            print("🎉 SUCCESS: Pipeline working! Fast Chat sees and builds on research.")
        else:
            print("⚠️  WARNING: Fast Chat may not be seeing last_research. Check backend logs.")


if __name__ == "__main__":
    try:
        asyncio.run(test_pipeline())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
