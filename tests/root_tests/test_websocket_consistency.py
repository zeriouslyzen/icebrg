#!/usr/bin/env python3
"""
Comprehensive WebSocket Connection Consistency Test
Tests WebSocket connection multiple times to ensure it works reliably
"""
import asyncio
import websockets
import json
import time
from datetime import datetime

WS_URL = "ws://localhost:8000/ws"
NUM_TESTS = 20
TIMEOUT = 10.0

async def test_websocket_connection(test_num):
    """Test a single WebSocket connection"""
    try:
        print(f"\n[TEST {test_num}/{NUM_TESTS}] Starting connection test...")
        start_time = time.time()
        
        # Connect with timeout
        async with websockets.connect(WS_URL, ping_interval=None, ping_timeout=None) as websocket:
            connect_time = time.time() - start_time
            print(f"[TEST {test_num}] Connected in {connect_time:.3f}s")
            
            # Wait for "connected" message
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "connected":
                    print(f"[TEST {test_num}] ✅ Received connection confirmation")
                    
                    # Send a test query
                    test_query = {
                        "type": "query",
                        "query": "hi",
                        "mode": "chat",
                        "agent": "surveyor"
                    }
                    await websocket.send(json.dumps(test_query))
                    print(f"[TEST {test_num}] ✅ Sent test query")
                    
                    # Wait for response (should get chunks or done)
                    response_received = False
                    timeout = time.time() + 10.0
                    while time.time() < timeout:
                        try:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            msg_data = json.loads(msg)
                            
                            if msg_data.get("type") in ["chunk", "done", "error"]:
                                response_received = True
                                print(f"[TEST {test_num}] ✅ Received response: {msg_data.get('type')}")
                                break
                        except asyncio.TimeoutError:
                            continue
                    
                    if response_received:
                        print(f"[TEST {test_num}] ✅ SUCCESS - Connection works end-to-end")
                        return True
                    else:
                        print(f"[TEST {test_num}] ❌ FAILED - No response received")
                        return False
                else:
                    print(f"[TEST {test_num}] ❌ FAILED - Unexpected message type: {response_data.get('type')}")
                    return False
                    
            except asyncio.TimeoutError:
                print(f"[TEST {test_num}] ❌ FAILED - Timeout waiting for connection confirmation")
                return False
            except json.JSONDecodeError as e:
                print(f"[TEST {test_num}] ❌ FAILED - Invalid JSON: {e}")
                return False
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"[TEST {test_num}] ❌ FAILED - Connection closed: {e.code} - {e.reason}")
        return False
    except websockets.exceptions.InvalidURI as e:
        print(f"[TEST {test_num}] ❌ FAILED - Invalid URI: {e}")
        return False
    except Exception as e:
        print(f"[TEST {test_num}] ❌ FAILED - Error: {type(e).__name__}: {e}")
        return False

async def run_consistency_tests():
    """Run multiple WebSocket connection tests"""
    print(f"Starting WebSocket consistency tests ({NUM_TESTS} tests)...")
    print(f"WebSocket URL: {WS_URL}")
    print(f"Timeout: {TIMEOUT}s per test")
    print("=" * 60)
    
    results = []
    for i in range(1, NUM_TESTS + 1):
        result = await test_websocket_connection(i)
        results.append(result)
        
        # Small delay between tests
        if i < NUM_TESTS:
            await asyncio.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    failed = NUM_TESTS - passed
    success_rate = (passed / NUM_TESTS) * 100
    
    print(f"Total Tests: {NUM_TESTS}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("\n✅ CONSISTENCY TEST PASSED - Connection is reliable")
        return 0
    else:
        print("\n❌ CONSISTENCY TEST FAILED - Connection is unreliable")
        print(f"Need at least 95% success rate, got {success_rate:.1f}%")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_consistency_tests())
    exit(exit_code)

