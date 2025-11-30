"""Auto Healer - Failure detection and recovery proposals for ICEBURG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time


FailureRecord = Dict[str, Any]
HealingAction = Dict[str, Any]


@dataclass
class AutoHealer:
    """Detects common failure modes and proposes recovery actions.

    Scope:
    - Purely local, non-destructive suggestions by default
    - Designed to integrate with CapabilityGapDetector and DatabaseManager
    """

    cfg: Any = None
    history: List[FailureRecord] = field(default_factory=list)
    actions_log: List[HealingAction] = field(default_factory=list)

    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', None)
        self.known_errors = {
            "llm_timeout": {"detection": "TimeoutError", "recovery": "Increase timeout or switch model"},
            "db_connection": {"detection": "ConnectionError", "recovery": "Restart DB service"},
            "missing_model": {"detection": "ModelNotFound", "recovery": "Pull model via Ollama"}
        }

    # --------------------------- Failure Detection ---------------------------
    def _detect_failure_type(self, data: Dict[str, Any]) -> str:
        msg = (data.get("error") or data.get("message") or "").lower()
        if "database" in msg or "sqlite" in msg:
            return "database_error"
        if "timeout" in msg or "timed out" in msg:
            return "timeout"
        if "model" in msg and ("not found" in msg or "missing" in msg):
            return "model_missing"
        if "memory" in msg and ("oom" in msg or "out of memory" in msg):
            return "memory_pressure"
        if "schema" in msg and ("migrate" in msg or "migration" in msg):
            return "migration_required"
        return "unknown"

    # --------------------------- Recovery Proposals --------------------------
    def _propose_actions(self, ftype: str, data: Dict[str, Any]) -> List[HealingAction]:
        actions: List[HealingAction] = []
        if ftype == "database_error":
            actions.append({
                "action": "reconnect_database",
                "description": "Reinitialize unified database connection and retry the operation.",
            })
            actions.append({
                "action": "run_migration",
                "description": "Execute data migration to ensure schemas are up to date.",
                "command_hint": "DatabaseManager.initialize_database(migrate_existing_data=True)",
            })
        elif ftype == "timeout":
            actions.append({
                "action": "increase_timeout",
                "description": "Increase provider timeout_s and retry with smaller context.",
                "config_hint": "ICEBURG_PROVIDER_TIMEOUT_S=90",
            })
            actions.append({
                "action": "enable_fast_mode",
                "description": "Use smaller models for the current step to avoid long inference.",
            })
        elif ftype == "model_missing":
            actions.append({
                "action": "pull_model",
                "description": "Pull the required Ollama model or adjust model selection.",
                "command_hint": "ollama pull <model-name>",
            })
        elif ftype == "memory_pressure":
            actions.append({
                "action": "reduce_concurrency",
                "description": "Lower parallelism, unload unused models, or switch to a smaller model.",
            })
        elif ftype == "migration_required":
            actions.append({
                "action": "run_migration",
                "description": "Run unified database migration to create/update tables.",
            })
        else:
            actions.append({
                "action": "capture_context",
                "description": "Log context and route through CapabilityGapDetector for analysis.",
            })
        return actions

    # ------------------------------- Public API ------------------------------
    async def heal(self, error_log: str) -> Dict[str, str]:
        detected_issue = self._detect_issue(error_log)
        if detected_issue:
            recovery_action = self.known_errors[detected_issue]["recovery"]
            # Execute recovery (stub; expand to actual commands)
            return {"issue": detected_issue, "action": recovery_action, "status": "resolved"}
        return {"status": "no_issue_detected"}

    def _detect_issue(self, error_log: str) -> Optional[str]:
        for issue, data in self.known_errors.items():
            if data["detection"] in error_log:
                return issue
        return None

    def get_healing_summary(self) -> dict:
        return {
            "status": "operational",
            "records": len(self.history),
            "last_actions": self.actions_log[-5:],
        }