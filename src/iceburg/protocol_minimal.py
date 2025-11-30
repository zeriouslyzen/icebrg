"""
Minimal working protocol for ICEBURG
This provides the essential functions needed by the unified interface
"""

import asyncio
import time
from typing import Dict, Any, Optional, List

def run_iceberg_protocol(
    initial_query: str,
    cfg: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    **kwargs
) -> str:
    """
    Minimal working protocol implementation
    """
    if verbose:
        print(f"[ICEBURG] Processing query: {initial_query[:100]}...")
    
    # Simple response for now
    response = f"ICEBURG Response: {initial_query}"
    
    if verbose:
        print(f"[ICEBURG] Response generated: {len(response)} characters")
    
    return response

def iceberg_protocol(
    initial_query: str,
    cfg: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    **kwargs
) -> str:
    """
    Synchronous wrapper for the protocol
    """
    return run_iceberg_protocol(initial_query, cfg, verbose, **kwargs)

async def _iceberg_protocol_async(
    initial_query: str,
    cfg: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    **kwargs
) -> str:
    """
    Async version of the protocol
    """
    return run_iceberg_protocol(initial_query, cfg, verbose, **kwargs)
