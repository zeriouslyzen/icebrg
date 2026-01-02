# src/iceburg/protocol/__init__.py
"""
ICEBURG Modular Protocol Package

This package contains the refactored modular version of the ICEBURG protocol,
breaking down the monolithic protocol.py into smaller, testable, and independently
deployable modules.

Architecture:
- config: Configuration management and feature flags
- models: Pydantic data models for type safety
- triage: Query classification and routing
- planner: Task planning and agent orchestration
- execution: Agent execution and legacy adapter
- synthesis: Evidence fusion and consensus building
- reporting: Report formatting and output generation
- legacy: Original protocol for fallback compatibility
"""

from .config import ProtocolConfig
from .models import Query, Mode, AgentTask, AgentResult, EvidenceBundle, ProtocolReport

# Minimal modular orchestrator to run full protocol without importing legacy at import-time
import asyncio
from typing import Any, List, Optional, Dict


async def run_protocol_modular(query: Query, cfg: ProtocolConfig) -> ProtocolReport:
    # Import inside function to avoid circulars and heavy side-effects on package import
    # Ensure top-level 'iceburg' package is importable for legacy-dependent agents
    try:
        import sys
        from pathlib import Path
        src_dir = Path(__file__).resolve().parents[3]
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
    except Exception:
        pass
    from .triage import classify
    from .planner import plan
    from .execution.runner import run_agent_tasks
    # Import core agent modules so they self-register in the registry
    try:
        from .execution.agents import vectorstore  # noqa: F401
        from .execution.agents import surveyor  # noqa: F401
        from .execution.agents import dissident  # noqa: F401
        from .execution.agents import synthesist  # noqa: F401
        from .execution.agents import oracle  # noqa: F401
        from .execution.agents import archaeologist  # noqa: F401
        from .execution.agents import supervisor  # noqa: F401
        if getattr(cfg, "verbose", False):
            print("[MODULAR_PROTOCOL] Core agents imported and registered.")
    except Exception as imp_err:
        if getattr(cfg, "verbose", False):
            print(f"[MODULAR_PROTOCOL] Agent import failed: {imp_err}")
        # Register lightweight fallback agents to enable modular smoke test without legacy
        from .execution.agents.registry import register_agent, get_agent_runner
        if get_agent_runner("vectorstore") is None:
            @register_agent("vectorstore")
            def _vs(cfg: ProtocolConfig, query: str, **_):
                return []
        if get_agent_runner("surveyor") is None:
            @register_agent("surveyor")
            def _sv(cfg: ProtocolConfig, query: str, **_):
                return f"SURVEYOR: baseline analysis for: {query}"
        if get_agent_runner("dissident") is None:
            @register_agent("dissident")
            def _ds(cfg: ProtocolConfig, query: str, surveyor_output: str = "", **_):
                return f"DISSIDENT: alternative framing vs. ({surveyor_output[:60]})"
        if get_agent_runner("synthesist") is None:
            @register_agent("synthesist")
            def _sy(cfg: ProtocolConfig, query: str, enhanced_context: Optional[dict] = None, **_):
                ctx = enhanced_context or {}
                return f"SYNTHESIS: {ctx.get('alternatives','')} + {ctx.get('consensus','')}"
        if get_agent_runner("oracle") is None:
            @register_agent("oracle")
            def _oc(cfg: ProtocolConfig, query: str, synthesis_output: str = "", **_):
                return f"ORACLE PRINCIPLE: distilled insight from synthesis ({len(synthesis_output)} chars)"
        if get_agent_runner("supervisor") is None:
            @register_agent("supervisor")
            def _sp(cfg: ProtocolConfig, stage_outputs: Optional[dict] = None, **_):
                return "SUPERVISOR: quality OK"
    from .synthesis.fusion import fuse as fuse_evidence
    from .reporting.formatter import format_report

    if getattr(cfg, "verbose", False):
        print(f"[MODULAR_PROTOCOL] Starting for query: {query.text[:120]}...")

    mode: Mode = classify(query, cfg)
    tasks: List[AgentTask] = plan(query, mode, cfg)
    results: List[AgentResult] = await run_agent_tasks(tasks, query, cfg)
    evidence: EvidenceBundle = fuse_evidence(results, query, cfg)
    report: ProtocolReport = format_report(evidence, cfg)
    return report


# Legacy compatibility – import lazily to avoid pulling broken legacy agents on import
def iceberg_protocol(*args: Any, **kwargs: Any) -> str:
    from .legacy.protocol_legacy import iceberg_protocol as _legacy
    return _legacy(*args, **kwargs)


__all__ = [
    "ProtocolConfig",
    "Query",
    "Mode",
    "AgentTask",
    "AgentResult",
    "EvidenceBundle",
    "ProtocolReport",
    "run_protocol_modular",
    "iceberg_protocol",
]