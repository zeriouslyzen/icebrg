#!/usr/bin/env python3
"""
Emergent Architect Agent
Integrates the emergent software architecture algorithm with ICEBURG's existing system
"""

from typing import Dict, Any, Optional
from ..config import IceburgConfig
import asyncio

# Optional import - stub if missing
try:
    from ..emergent_software_architect import EmergentSoftwareArchitect
except ImportError:
    # Stub for missing module
    class EmergentSoftwareArchitect:
        def __init__(self, config):
            pass
        async def generate_software(self, principle, verbose=False):
            return "EmergentSoftwareArchitect module not available"


class EmergentArchitect:
    """
    ICEBURG agent that uses emergent software architecture instead of rigid frameworks
    """
    
    def __init__(self, config: IceburgConfig):
        self.config = config
        self.architect = EmergentSoftwareArchitect(config)
    
    async def run(self, cfg: IceburgConfig, principle: str, verbose: bool = False) -> str:
        """
        Generate software using emergent architecture algorithm
        """
        try:
            if verbose:
                print("[EMERGENT_ARCHITECT] Starting emergent architecture generation...")
            # Generate the software architecture
            software_arch = await self.architect.generate_software(principle, verbose=verbose)
            
            if verbose:
                print("[EMERGENT_ARCHITECT] Architecture generation complete, formatting report...")
            # Format the complete report
            report = self.architect.format_architecture_report(software_arch)
            
            if verbose:
                print("[EMERGENT_ARCHITECT] Report ready")
            return report
            
        except Exception as e:
            error_msg = f"[EMERGENT_ARCHITECT] ❌ Error: {e}"
            if verbose:
                print(f"[EMERGENT_ARCHITECT] Error: {e}")
            return error_msg


# For backward compatibility with existing ICEBURG system
def run(cfg: IceburgConfig, principle: str, verbose: bool = False) -> str:
    """
    Synchronous wrapper for the emergent architect
    """
    architect = EmergentArchitect(cfg)
    return asyncio.run(architect.run(cfg, principle, verbose))
