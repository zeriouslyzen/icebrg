"""
Telemetry: structured event logging + workspace broadcast.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .memory.unified_memory import UnifiedMemory
from .global_workspace import GlobalWorkspace


class Telemetry:
    _memory = UnifiedMemory()
    _workspace = GlobalWorkspace(verbose=False)

    @staticmethod
    def write(event_name: str, data: Optional[Dict[str, Any]] = None, *, run_id: str = "runtime", agent_id: str = "app", task_id: str = "event") -> None:
        payload = data or {}
        Telemetry._memory.log_event({
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "agent_id": agent_id,
            "task_id": task_id,
            "event_type": event_name,
            "payload": payload,
        })
        Telemetry._workspace.publish("telemetry/runtime", {"event": event_name, **payload})
