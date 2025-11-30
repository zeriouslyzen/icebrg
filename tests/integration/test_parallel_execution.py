#!/usr/bin/env python3
"""
Test script for parallel execution and quality score improvements.
Tests with a practical, real-world query.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iceburg.protocol.config import ProtocolConfig
from iceburg.protocol.models import Query, Mode
from iceburg.protocol.execution.runner import run_agent_tasks
from iceburg.protocol.planner import plan
from iceburg.utils.quality_calculator import calculate_quality_score


async def test_parallel_execution():
    """Test parallel execution with a practical query."""
    
    # Practical, real-world query
    test_query = "How do I build a REST API with authentication using Python and Flask? Include best practices for security, error handling, and testing."
    
    print("=" * 80)
    print("ICEBURG Parallel Execution Test")
    print("=" * 80)
    print(f"\nQuery: {test_query}\n")
    
    # Create query object
    query = Query(
        text=test_query,
        metadata={}
    )
    
    # Create config
    cfg = ProtocolConfig(
        verbose=True,
        fast=False,
        enable_multimodal_processing=False,
        enable_visual_generation=False,
        enable_blockchain_verification=False
    )
    
    # Create mode
    from iceburg.protocol.models import Mode
    mode = Mode(name="smart", reason="Complex query requiring full analysis", confidence=0.9)
    
    # Plan tasks
    print("\n[1] Planning agent tasks...")
    start_plan = time.time()
    tasks = plan(query, mode, cfg)
    plan_time = time.time() - start_plan
    
    print(f"   Planned {len(tasks)} tasks in {plan_time:.2f}s")
    print(f"   Agents: {[t.agent for t in tasks]}")
    
    # Check for parallelizable groups
    from iceburg.protocol.planner import get_parallelizable_groups
    groups = get_parallelizable_groups(tasks, cfg)
    print(f"   Parallel groups: {len(groups)}")
    for i, group in enumerate(groups):
        print(f"     Group {i+1}: {[t.agent for t in group]}")
    
    # Execute tasks
    print("\n[2] Executing tasks (parallel by default)...")
    start_exec = time.time()
    
    try:
        results = await run_agent_tasks(tasks, query, cfg)
        exec_time = time.time() - start_exec
        
        print(f"\n   Execution completed in {exec_time:.2f}s")
        
        # Analyze results
        print("\n[3] Analyzing results...")
        
        successful = sum(1 for r in results if r.payload and r.metadata.get("success", False))
        failed = len(results) - successful
        
        print(f"   Successful agents: {successful}/{len(results)}")
        print(f"   Failed agents: {failed}")
        
        # Check parallel execution
        parallel_count = sum(1 for r in results if r.metadata.get("parallel_execution", False))
        sequential_count = len(results) - parallel_count
        
        print(f"\n   Parallel execution: {parallel_count} agents")
        print(f"   Sequential execution: {sequential_count} agents")
        
        # Calculate total latency
        total_latency = sum(r.latency_ms for r in results) / 1000  # Convert to seconds
        max_latency = max((r.latency_ms for r in results), default=0) / 1000
        
        print(f"\n   Total agent latency: {total_latency:.2f}s")
        print(f"   Max agent latency: {max_latency:.2f}s")
        
        if parallel_count > 0:
            speedup_ratio = total_latency / max_latency if max_latency > 0 else 1.0
            print(f"   Estimated speedup: {speedup_ratio:.2f}x")
        
        # Extract content for quality score
        content = ""
        agent_results = {}
        
        for result in results:
            if result.payload:
                agent_results[result.agent] = result.payload
                if isinstance(result.payload, str):
                    content += result.payload + "\n\n"
                elif isinstance(result.payload, dict):
                    content += str(result.payload.get("content", result.payload)) + "\n\n"
        
        # Calculate quality score
        print("\n[4] Calculating quality score...")
        quality_metadata = {}
        quality_score = calculate_quality_score(
            response_text=content,
            query_text=test_query,
            agent_results=agent_results,
            response_time=exec_time,
            metadata=quality_metadata
        )
        
        print(f"   Quality score: {quality_score:.2f}")
        if "quality_factors" in quality_metadata:
            factors = quality_metadata.get("quality_factors", {})
            print(f"   Factors:")
            for factor, score in factors.items():
                print(f"     {factor}: {score:.2f}")
        
        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        print(f"Query: {test_query[:60]}...")
        print(f"Total execution time: {exec_time:.2f}s")
        print(f"Parallel execution: {parallel_count > 0}")
        print(f"Quality score: {quality_score:.2f}")
        print(f"Successful agents: {successful}/{len(results)}")
        
        if parallel_count > 0:
            print(f"Estimated speedup: {speedup_ratio:.2f}x")
        
        print("=" * 80)
        
        return {
            "success": True,
            "execution_time": exec_time,
            "quality_score": quality_score,
            "parallel_execution": parallel_count > 0,
            "successful_agents": successful,
            "total_agents": len(results),
            "speedup_ratio": speedup_ratio if parallel_count > 0 else 1.0
        }
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("\n🚀 Starting ICEBURG Parallel Execution Test\n")
    
    result = asyncio.run(test_parallel_execution())
    
    if result.get("success"):
        print("\n✅ Test completed successfully!")
        if result.get("parallel_execution"):
            print(f"✅ Parallel execution is working")
            print(f"✅ Speedup: {result.get('speedup_ratio', 1.0):.2f}x")
        print(f"✅ Quality score: {result.get('quality_score', 0.0):.2f}")
    else:
        print(f"\n❌ Test failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

