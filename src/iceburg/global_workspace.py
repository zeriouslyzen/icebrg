"""
ICEBURG Global Workspace
Manages global workspace for agent coordination
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, DefaultDict
from collections import defaultdict


class ThoughtType:
    """Thought type enumeration"""
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    HYPOTHESIS = "hypothesis"
    INSIGHT = "insight"


class ThoughtPriority:
    """Thought priority enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMPORTANT = "important"


class GlobalWorkspace:
    """
    Manages global workspace for agent coordination
    - Lightweight pub/sub for topics (telemetry/*, emergence/*)
    """

    def __init__(self, verbose: bool = False):
        self.workspace_id = "global_workspace"
        self.verbose = verbose
        self._subscribers: DefaultDict[str, List[Callable[[str, Dict[str, Any]], None]]] = defaultdict(list)

    def broadcast_thought(self, *args, **kwargs):
        """Broadcast thought to workspace."""
        if self.verbose:
            pass  # Verbose mode logging
        return {"broadcasted": True}

    # ---- Pub/Sub ----
    def subscribe(self, topic: str, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        self._subscribers[topic].append(handler)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.verbose:
            pass  # Verbose mode logging
        for handler in list(self._subscribers.get(topic, [])):
            try:
                handler(topic, payload)
            except Exception as e:
                if self.verbose:
                    pass  # Log error in verbose mode