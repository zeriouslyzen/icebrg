from __future__ import annotations

from typing import Any, Dict

from .config import ProtocolConfig
from .models import Mode, Query


_SIMPLE_KEYWORDS = {"test", "ping", "hello", "hi", "help", "status", "check"}
_EXPERIMENTAL_KEYWORDS = {
    "research",
    "breakthrough",
    "investigate",
    "analyze",
    "simulate",
    "virtual",
    "ecosystem",
    "nutrition",
    "chronic",
    "disease",
    "prevent",
    "medicine",
    "health",
    "cancer",
    "diabetes",
    "heart",
    "therapy",
    "treatment",
    "cure",
    "healing",
    "biochemical",
    "molecular",
    "enzyme",
    "pathway",
    "signal",
    "gene",
    "genomic",
    "metabolic",
    "quantum",
    "field",
    "plasma",
    "resonance",
    "synthetic biology",
    "bioelectric",
}


def _score_field_conditions(metadata: Dict[str, Any]) -> float:
    conditions = metadata.get("field_conditions")
    if not conditions or conditions.get("error"):
        return 0.0

    earth_sync = conditions.get("earth_sync", 0.0)
    human_state = conditions.get("human_state", "unknown")
    boost = 0.0
    if earth_sync > 0.7 or human_state in {"creative_flow", "insight"}:
        boost += 0.3
    if earth_sync > 0.9:
        boost += 0.2
    return boost


def classify(query: Query, config: ProtocolConfig, **kwargs: Any) -> Mode:
    text = query.text.strip()
    query_lower = text.lower()

    field_boost = _score_field_conditions(query.metadata)

    if (
        query_lower in _SIMPLE_KEYWORDS
        or len(text.split()) < 3
        or query_lower.startswith("test")
    ):
        return Mode(name="fast", reason="Simple keyword detection")

    experimental_score = field_boost
    if any(keyword in query_lower for keyword in _EXPERIMENTAL_KEYWORDS):
        experimental_score += 0.5

    if experimental_score >= 0.5 and config.hybrid_mode_enabled:
        return Mode(name="hybrid", reason="Experimental routing")

    if config.smart_mode_enabled and len(text) > 280:
        return Mode(name="smart", reason="Long-form smart routing")

    if config.fast_mode_enabled:
        return Mode(name="fast", reason="Default fast path")

    return Mode(name="legacy", reason="Fallback")
