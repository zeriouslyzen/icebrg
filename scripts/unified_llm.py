#!/usr/bin/env python3
"""
Simple CLI wrapper for Unified LLM Interface
"""
import sys
import asyncio
from iceburg.unified_llm_interface import UnifiedLLMInterface

async def main():
    if len(sys.argv) < 2:
        print("Usage: python unified_llm.py 'Your query here'")
        print()
        print("Examples:")
        print("  python unified_llm.py 'What is ICEBURG?'")
        print("  python unified_llm.py 'Analyze quantum consciousness comprehensively'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print("=" * 70)
    print("UNIFIED LLM INTERFACE - Multi-ASI System")
    print("=" * 70)
    print()
    print(f"Query: {query}")
    print()
    print("Processing...")
    print()
    
    llm = UnifiedLLMInterface()
    
    # Query with automatic routing
    response = await llm.query_sync(query)
    
    print("=" * 70)
    print(f"Mode: {response.mode.value}")
    print(f"Processing Time: {response.processing_time:.2f}s")
    print(f"Complexity Score: {response.complexity_score:.2f}")
    print(f"Confidence: {response.confidence:.2f}")
    print("=" * 70)
    print()
    print("Response:")
    print("-" * 70)
    print(response.content)
    print("-" * 70)
    print()
    
    # Show stats
    stats = llm.get_stats()
    print("Statistics:")
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Average Response Time: {stats['average_response_time']:.2f}s")
    print(f"  Mode Distribution:")
    for mode, percentage in stats['mode_distribution'].items():
        print(f"    {mode}: {percentage:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
