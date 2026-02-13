#!/usr/bin/env python3
"""
Test ICEBURG core protocol with a simple query to verify it still works
"""
import sys
sys.path.insert(0, 'src')

from iceburg.config import load_config
from iceburg.protocol_fixed import run as run_protocol
import asyncio

async def test_simple_query():
    cfg = load_config()
    
    # Simple test query that should work quickly
    test_query = "What is quantum entanglement?"
    
    print(f"[TEST] Testing ICEBURG protocol with simple query...")
    print(f"[TEST] Query: {test_query}")
    print()
    
    try:
        result = await run_protocol(test_query, verbose=True)
        print()
        print("=" * 80)
        print("[RESULT]")
        print("=" * 80)
        print(result)
        print()
        print("[PASS] ✅ ICEBURG protocol completed successfully")
        return True
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_simple_query())
