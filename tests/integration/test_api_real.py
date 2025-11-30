#!/usr/bin/env python3
"""
Test parallel execution through the API server with a real query.
"""

import requests
import json
import time
import sys

def test_api_query():
    """Test a real query through the API."""
    
    # Practical, real-world query
    test_query = "How do I build a REST API with authentication using Python and Flask? Include best practices for security, error handling, and testing."
    
    print("=" * 80)
    print("ICEBURG API Parallel Execution Test")
    print("=" * 80)
    print(f"\nQuery: {test_query}\n")
    
    # Wait for server to be ready
    print("[0] Checking server status...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("   ✅ Server is ready")
                break
        except:
            if i < max_retries - 1:
                print(f"   ⏳ Waiting for server... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("   ❌ Server not responding")
                return {"success": False, "error": "Server not responding"}
    
    # Send query
    print("\n[1] Sending query to API...")
    start_time = time.time()
    
    try:
        # Use query endpoint
        response = requests.post(
            "http://localhost:8000/api/query",
            json={
                "query": test_query,
                "mode": "smart",
                "verbose": True
            },
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout
        )
        
        exec_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n[2] Query processed in {exec_time:.2f}s")
            print(f"   Status: {result.get('status', 'unknown')}")
            
            # Extract results
            if result.get("status") == "completed":
                result_data = result.get("result", {})
                
                if isinstance(result_data, dict):
                    content = result_data.get("content", "")
                    agent_results = result_data.get("agent_results", {})
                    metadata = result_data.get("metadata", {})
                    
                    print(f"\n[3] Results:")
                    print(f"   Content length: {len(content)} characters")
                    print(f"   Agents executed: {len(agent_results)}")
                    
                    # Check for parallel execution
                    if metadata.get("parallel_execution"):
                        print(f"   ✅ Parallel execution: YES")
                        print(f"   Groups executed: {metadata.get('groups_executed', 'N/A')}")
                    else:
                        print(f"   ⚠️  Parallel execution: NO (sequential)")
                    
                    # Calculate quality score
                    print(f"\n[4] Quality Metrics:")
                    quality_score = result_data.get("quality_score", 0.0)
                    if quality_score:
                        print(f"   Quality score: {quality_score:.2f}")
                    else:
                        print(f"   Quality score: Not calculated")
                    
                    # Summary
                    print("\n" + "=" * 80)
                    print("Test Summary")
                    print("=" * 80)
                    print(f"Query: {test_query[:60]}...")
                    print(f"Execution time: {exec_time:.2f}s")
                    print(f"Parallel execution: {metadata.get('parallel_execution', False)}")
                    print(f"Quality score: {quality_score:.2f if quality_score else 'N/A'}")
                    print(f"Content length: {len(content)} characters")
                    print("=" * 80)
                    
                    return {
                        "success": True,
                        "execution_time": exec_time,
                        "quality_score": quality_score,
                        "parallel_execution": metadata.get("parallel_execution", False),
                        "content_length": len(content)
                    }
                else:
                    print(f"   Result: {str(result_data)[:200]}...")
                    return {"success": True, "execution_time": exec_time}
            else:
                error = result.get("error", "Unknown error")
                print(f"   ❌ Query failed: {error}")
                return {"success": False, "error": error}
        else:
            print(f"   ❌ API error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        exec_time = time.time() - start_time
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "execution_time": exec_time}


if __name__ == "__main__":
    print("\n🚀 Starting ICEBURG API Test\n")
    
    result = test_api_query()
    
    if result.get("success"):
        print("\n✅ Test completed successfully!")
        if result.get("parallel_execution"):
            print(f"✅ Parallel execution is working")
        print(f"✅ Execution time: {result.get('execution_time', 0.0):.2f}s")
        if result.get("quality_score"):
            print(f"✅ Quality score: {result.get('quality_score', 0.0):.2f}")
    else:
        print(f"\n❌ Test failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

