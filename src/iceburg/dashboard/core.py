"""Dashboard Core - Minimal stub for ICEBURG integration."""

from __future__ import annotations

from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class DashboardCore:
    """Minimal stub for dashboard core."""
    
    def __init__(self, *args, **kwargs) -> None:
        self.cfg = kwargs.get('cfg', None)
    
    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render dashboard."""
        return {"rendered": True, "status": "active"}
    
    async def start(self) -> None:
        """Start dashboard."""
        pass
    
    async def stop(self) -> None:
        """Stop dashboard."""
        pass