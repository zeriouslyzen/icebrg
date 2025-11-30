from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


QueryMode = Literal["fast", "hybrid", "smart", "legacy"]


@dataclass
class Query:
    """User query payload passed into the protocol."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    multimodal_input: Optional[Any] = None
    documents: Optional[List[str]] = None
    multimodal_evidence: Optional[List[Any]] = None


@dataclass
class Mode:
    """Execution mode selected during triage."""

    name: QueryMode
    reason: str = ""
    confidence: float = 1.0


@dataclass
class Capability:
    """Individual capability toggled for execution planning."""

    name: str
    required: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Planned task to be executed by a registered agent."""

    agent: str
    payload: Dict[str, Any]
    capability: Optional[Capability] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result returned by agent execution."""

    agent: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None


@dataclass
class EvidenceBundle:
    """Aggregated evidence after synthesis stages."""

    results: List[AgentResult] = field(default_factory=list)
    consensus: Optional[str] = None
    confidence: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolReport:
    """Structured report returned by the protocol façade."""

    sections: Dict[str, Any]
    audit: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
