"""Evidence schema and store for the dossier pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class Evidence:
    """Single piece of evidence supporting an investigation.

    This is the common currency between Gatherer, Decoder, Mapper and
    Colossus. Everything we reason about should be traceable back to one
    or more Evidence items.
    """

    id: str
    query: str
    source_url: str
    source_title: str
    source_type: str  # surface | alternative | academic | historical | deep | corpus
    snippet: str
    collected_at: datetime
    credibility: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    entity_mentions: List[str] = field(default_factory=list)
    relationship_hints: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def from_source(query: str, source: Any) -> "Evidence":
        """Create Evidence from an IntelligenceSource-like object.

        We keep this loose to avoid tight coupling; anything with
        url/title/content/source_type/timestamp/credibility_score/metadata
        fields can be converted.
        """
        ts = getattr(source, "timestamp", None) or datetime.utcnow()
        meta = dict(getattr(source, "metadata", {}) or {})
        return Evidence(
            id=str(uuid.uuid4()),
            query=query,
            source_url=getattr(source, "url", ""),
            source_title=getattr(source, "title", ""),
            source_type=getattr(source, "source_type", "unknown"),
            snippet=getattr(source, "content", ""),
            collected_at=ts,
            credibility=getattr(source, "credibility_score", 0.5),
            metadata=meta,
        )


class EvidenceStore:
    """In-memory evidence store with hooks for vector/graph backends.

    For now this is a light abstraction that lets the pipeline carry
    around Evidence objects instead of just raw strings. Later it can be
    extended to materialize into Chroma/FAISS, SQLite, or Neo4j.
    """

    def __init__(self) -> None:
        self._items: List[Evidence] = []

    def add(self, ev: Evidence) -> None:
        self._items.append(ev)

    def extend(self, items: List[Evidence]) -> None:
        self._items.extend(items)

    @property
    def items(self) -> List[Evidence]:
        return list(self._items)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Return evidence as plain dicts (JSON-serializable)."""
        out: List[Dict[str, Any]] = []
        for ev in self._items:
            d: Dict[str, Any] = {
                "id": ev.id,
                "query": ev.query,
                "source_url": ev.source_url,
                "source_title": ev.source_title,
                "source_type": ev.source_type,
                "snippet": ev.snippet,
                "collected_at": ev.collected_at.isoformat(),
                "credibility": ev.credibility,
                "metadata": ev.metadata,
                "entity_mentions": ev.entity_mentions,
                "relationship_hints": ev.relationship_hints,
            }
            out.append(d)
        return out
