#!/usr/bin/env python3
"""
Test parallel execution through the API server with a real query.
"""

import asyncio
import time
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_api_query():
    """Test a real query through the API."""
    
    # Practical, real-world query
    test_query = "How do I build a REST API with authentication using Python and Flask? Include best practices for security, error handling, and testing."
    
    print("=" * 80)
    print("ICEBURG API Parallel Execution Test")
    print("=" * 80)
    print(f"\nQuery: {test_query}\n")
    
    try:
        # Import the protocol
        from iceburg.protocol.protocol import process_query
        
        print("[1] Processing query through ICEBURG protocol...")
        start_time = time.time()
        
        # Process query
        result = await process_query(
            query=test_query,
            mode="smart",
            verbose=True
        )
        
        exec_time = time.time() - start_time
        
        print(f"\n[2] Query processed in {exec_time:.2f}s")
        
        # Extract results
        if result and result.get("results"):
            content = result.get("results", {}).get("content", "")
            agent_results = result.get("results", {}).get("agent_results", {})
            metadata = result.get("results", {}).get("metadata", {})
            
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
            from iceburg.utils.quality_calculator import calculate_quality_score
            
            quality_metadata = {}
            quality_score = calculate_quality_score(
                response_text=content,
                query_text=test_query,
                agent_results=agent_results,
                response_time=exec_time,
                metadata=quality_metadata
            )
            
            print(f"\n[4] Quality Metrics:")
            print(f"   Quality score: {quality_score:.2f}")
            
            if "quality_factors" in quality_metadata:
                factors = quality_metadata.get("quality_factors", {})
                print(f"   Quality factors:")
                for factor, score in factors.items():
                    print(f"     {factor}: {score:.2f}")
            
            # Summary
            print("\n" + "=" * 80)
            print("Test Summary")
            print("=" * 80)
            print(f"Query: {test_query[:60]}...")
            print(f"Execution time: {exec_time:.2f}s")
            print(f"Parallel execution: {metadata.get('parallel_execution', False)}")
            print(f"Quality score: {quality_score:.2f}")
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
            print("❌ No results returned")
            return {"success": False, "error": "No results"}
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("\n🚀 Starting ICEBURG API Test\n")
    
    result = asyncio.run(test_api_query())
    
    if result.get("success"):
        print("\n✅ Test completed successfully!")
        if result.get("parallel_execution"):
            print(f"✅ Parallel execution is working")
        print(f"✅ Quality score: {result.get('quality_score', 0.0):.2f}")
        print(f"✅ Execution time: {result.get('execution_time', 0.0):.2f}s")
    else:
        print(f"\n❌ Test failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

