"""
WebSocket Diagnostics Tool
Tests WebSocket connection, ping/pong, keepalive, and message handling
"""

import asyncio
import json
import time
import websockets
from typing import Dict, Any, List
import sys

class WebSocketDiagnostics:
    def __init__(self, uri: str = "ws://localhost:8000/ws"):
        self.uri = uri
        self.results = []
        self.errors = []
        
    async def test_connection(self):
        """Test basic WebSocket connection"""
        print("\n" + "="*60)
        print("TEST 1: Basic Connection")
        print("="*60)
        
        try:
            # Increase timeout for initial connection (15s instead of 5s)
            async with websockets.connect(self.uri, ping_interval=None, ping_timeout=None) as ws:
                print("✅ WebSocket connected")
                
                # Wait for connection confirmation (longer timeout)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    data = json.loads(response)
                    if data.get("type") == "connected":
                        print(f"✅ Connection confirmed: {data.get('message', '')}")
                        return True
                    else:
                        print(f"⚠️  Unexpected response: {data}")
                        return False
                except asyncio.TimeoutError:
                    print("⚠️  No connection confirmation received (15s timeout)")
                    return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.errors.append(f"Connection: {e}")
            return False
    
    async def test_ping_pong(self):
        """Test ping/pong keepalive"""
        print("\n" + "="*60)
        print("TEST 2: Ping/Pong Keepalive")
        print("="*60)
        
        try:
            # Increase timeout for initial connection
            async with websockets.connect(self.uri, ping_interval=None, ping_timeout=None) as ws:
                print("✅ WebSocket connected")
                
                # Wait for connection confirmation (longer timeout)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    data = json.loads(response)
                    if data.get("type") == "connected":
                        print("✅ Connection confirmed")
                except asyncio.TimeoutError:
                    print("⚠️  No connection confirmation (15s timeout)")
                
                # Test ping/pong
                ping_count = 0
                pong_count = 0
                
                for i in range(5):
                    try:
                        # Send ping
                        await ws.send(json.dumps({"type": "ping"}))
                        ping_count += 1
                        print(f"  📤 Sent ping #{i+1}")
                        
                        # Wait for pong
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                            data = json.loads(response)
                            if data.get("type") == "pong":
                                pong_count += 1
                                print(f"  📥 Received pong #{i+1}")
                            else:
                                print(f"  ⚠️  Unexpected response: {data.get('type')}")
                        except asyncio.TimeoutError:
                            print(f"  ⏱️  No pong received for ping #{i+1}")
                        
                        await asyncio.sleep(1)
                    except Exception as e:
                        print(f"  ❌ Error during ping/pong test: {e}")
                        self.errors.append(f"Ping/Pong: {e}")
                        break
                
                print(f"\n  Results: {ping_count} pings sent, {pong_count} pongs received")
                if pong_count == ping_count:
                    print("  ✅ Ping/pong working correctly")
                    return True
                else:
                    print(f"  ⚠️  Ping/pong mismatch: {ping_count} pings, {pong_count} pongs")
                    return False
                    
        except Exception as e:
            print(f"❌ Ping/pong test failed: {e}")
            self.errors.append(f"Ping/Pong: {e}")
            return False
    
    async def test_query_sequence(self):
        """Test sequence of queries to check for connection drops"""
        print("\n" + "="*60)
        print("TEST 3: Query Sequence (Check for Connection Drops)")
        print("="*60)
        
        queries = [
            "hi",
            "hello",
            "What is AI?",
            "Explain quantum mechanics",
            "How does the brain work?"
        ]
        
        try:
            # Increase timeout for initial connection
            async with websockets.connect(self.uri, ping_interval=None, ping_timeout=None) as ws:
                print("✅ WebSocket connected")
                
                # Wait for connection confirmation (longer timeout)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    data = json.loads(response)
                    if data.get("type") == "connected":
                        print("✅ Connection confirmed")
                except asyncio.TimeoutError:
                    print("⚠️  No connection confirmation (15s timeout)")
                
                for i, query in enumerate(queries, 1):
                    print(f"\n  Query {i}/{len(queries)}: {query[:50]}...")
                    
                    try:
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
                        
                        await ws.send(json.dumps(message))
                        print(f"    ✅ Query sent")
                        
                        # Wait for response
                        response_received = False
                        chunks_received = 0
                        start_time = time.time()
                        timeout = 30.0
                        
                        while time.time() - start_time < timeout:
                            try:
                                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                                data = json.loads(response)
                                
                                response_type = data.get("type", "unknown")
                                
                                if response_type == "chunk":
                                    chunks_received += 1
                                    print(f"    📝 Chunk {chunks_received} received")
                                elif response_type == "done":
                                    response_received = True
                                    print(f"    ✅ Done received")
                                    break
                                elif response_type == "error":
                                    print(f"    ❌ Error: {data.get('message', 'Unknown error')}")
                                    break
                                elif response_type == "thinking":
                                    print(f"    💭 Thinking: {data.get('content', '')[:50]}...")
                                elif response_type == "action":
                                    print(f"    ⚙️  Action: {data.get('action', '')} - {data.get('status', '')}")
                                else:
                                    print(f"    📨 {response_type}")
                                    
                            except asyncio.TimeoutError:
                                print(f"    ⏱️  Timeout waiting for response")
                                continue
                        
                        if response_received:
                            elapsed = time.time() - start_time
                            print(f"    ✅ Response received in {elapsed:.2f}s")
                        else:
                            print(f"    ⚠️  No response received")
                            break
                        
                        # Small delay between queries
                        await asyncio.sleep(1)
                        
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"    ❌ Connection closed: {e}")
                        self.errors.append(f"Query {i}: Connection closed - {e}")
                        return False
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                        self.errors.append(f"Query {i}: {e}")
                        return False
                
                print(f"\n  ✅ All {len(queries)} queries completed successfully")
                return True
                
        except Exception as e:
            print(f"❌ Query sequence test failed: {e}")
            self.errors.append(f"Query Sequence: {e}")
            return False
    
    async def test_keepalive_timeout(self):
        """Test if connection stays alive during idle period"""
        print("\n" + "="*60)
        print("TEST 4: Keepalive Timeout (Idle Connection)")
        print("="*60)
        
        try:
            # Increase timeout for initial connection
            async with websockets.connect(self.uri, ping_interval=None, ping_timeout=None) as ws:
                print("✅ WebSocket connected")
                
                # Wait for connection confirmation (longer timeout)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    data = json.loads(response)
                    if data.get("type") == "connected":
                        print("✅ Connection confirmed")
                except asyncio.TimeoutError:
                    print("⚠️  No connection confirmation (15s timeout)")
                
                # Wait idle for 30 seconds
                print("  ⏳ Waiting idle for 30 seconds...")
                for i in range(30):
                    await asyncio.sleep(1)
                    if i % 5 == 0:
                        print(f"    {i}s elapsed...")
                    
                    # Check if connection is still alive
                    try:
                        # Try to send a ping
                        await ws.send(json.dumps({"type": "ping"}))
                        # Wait for pong
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            data = json.loads(response)
                            if data.get("type") == "pong":
                                print(f"    ✅ Connection alive at {i}s")
                        except asyncio.TimeoutError:
                            print(f"    ⚠️  No pong at {i}s")
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"    ❌ Connection closed at {i}s: {e}")
                        self.errors.append(f"Keepalive: Connection closed at {i}s - {e}")
                        return False
                    except Exception as e:
                        print(f"    ❌ Error at {i}s: {e}")
                        self.errors.append(f"Keepalive: Error at {i}s - {e}")
                        return False
                
                print("  ✅ Connection stayed alive for 30 seconds")
                return True
                
        except Exception as e:
            print(f"❌ Keepalive test failed: {e}")
            self.errors.append(f"Keepalive: {e}")
            return False
    
    async def test_concurrent_connections(self):
        """Test multiple concurrent connections"""
        print("\n" + "="*60)
        print("TEST 5: Concurrent Connections")
        print("="*60)
        
        async def test_single_connection(conn_id: int):
            try:
                # Increase timeout for initial connection
                async with websockets.connect(self.uri, ping_interval=None, ping_timeout=None) as ws:
                    # Wait for connection confirmation (longer timeout)
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                        data = json.loads(response)
                        if data.get("type") == "connected":
                            print(f"  ✅ Connection {conn_id} confirmed")
                    except asyncio.TimeoutError:
                        print(f"  ⚠️  Connection {conn_id} - no confirmation (15s timeout)")
                    
                    # Send a query
                    message = {
                        "query": f"Test query from connection {conn_id}",
                        "mode": "chat",
                        "agent": "auto",
                        "settings": {
                            "primaryModel": "llama3.1:8b",
                            "temperature": 0.7,
                            "maxTokens": 2000
                        }
                    }
                    
                    await ws.send(json.dumps(message))
                    print(f"  📤 Connection {conn_id} sent query")
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        data = json.loads(response)
                        print(f"  📥 Connection {conn_id} received: {data.get('type', 'unknown')}")
                        return True
                    except asyncio.TimeoutError:
                        print(f"  ⏱️  Connection {conn_id} - timeout")
                        return False
                        
            except Exception as e:
                print(f"  ❌ Connection {conn_id} failed: {e}")
                self.errors.append(f"Concurrent {conn_id}: {e}")
                return False
        
        # Test 3 concurrent connections
        tasks = [test_single_connection(i) for i in range(1, 4)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        print(f"\n  Results: {success_count}/3 connections successful")
        
        if success_count == 3:
            print("  ✅ All concurrent connections worked")
            return True
        else:
            print(f"  ⚠️  Only {success_count}/3 connections successful")
            return False
    
    async def run_all_tests(self):
        """Run all diagnostic tests"""
        print("="*60)
        print("WebSocket Diagnostics Tool")
        print("="*60)
        print(f"Testing: {self.uri}")
        
        tests = [
            ("Basic Connection", self.test_connection),
            ("Ping/Pong Keepalive", self.test_ping_pong),
            ("Query Sequence", self.test_query_sequence),
            ("Keepalive Timeout", self.test_keepalive_timeout),
            ("Concurrent Connections", self.test_concurrent_connections),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = result
            except Exception as e:
                print(f"\n❌ Test '{test_name}' crashed: {e}")
                import traceback
                traceback.print_exc()
                results[test_name] = False
                self.errors.append(f"{test_name}: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        print(f"\nResults: {passed}/{total} tests passed")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        return results

async def main():
    uri = "ws://localhost:8000/ws"
    if len(sys.argv) > 1:
        uri = sys.argv[1]
    
    diagnostics = WebSocketDiagnostics(uri)
    results = await diagnostics.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

