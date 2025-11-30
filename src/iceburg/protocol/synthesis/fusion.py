from __future__ import annotations

from ..config import ProtocolConfig
from ..models import EvidenceBundle, AgentResult
from .evidence import synthesize


def fuse(results: list[AgentResult], config: ProtocolConfig) -> EvidenceBundle:
    bundle = synthesize(results, config)
    bundle.diagnostics.update({
        "total_results": len(results),
        "agents": [result.agent for result in results],
    })
    return bundle
