"""Physics Prior Gate - Minimal stub for ICEBURG integration."""

from __future__ import annotations

from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class PhysicsPriorGate:
    """Minimal stub for physics prior gate."""
    
    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg
    
    def gate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply physics prior gating."""
        return {"gated": True, "confidence": 1.0}
