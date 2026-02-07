"""
Silence / mention tracker.

Tracks which entities are mentioned vs silent across a corpus (e.g. documents, emails).
Scans corpus text for entity IDs and optional names; returns mention counts and
sources where each entity appears.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_text(item: Dict[str, Any]) -> str:
    """Extract searchable text from a corpus item."""
    return (item.get("text") or item.get("content") or item.get("body") or "").strip()


def _count_mentions(text: str, pattern: str, case_sensitive: bool = False) -> int:
    """Count non-overlapping occurrences of pattern in text. Uses word-boundary for alphabetic patterns."""
    if not text or not pattern:
        return 0
    flags = 0 if case_sensitive else re.IGNORECASE
    if pattern.isalpha() or " " in pattern:
        try:
            regex = re.compile(r"\b" + re.escape(pattern) + r"\b", flags)
        except re.error:
            regex = re.compile(re.escape(pattern), flags)
    else:
        regex = re.compile(re.escape(pattern), flags)
    return len(regex.findall(text))


def track_silence_mentions(
    corpus_sources: List[Dict[str, Any]],
    entity_ids: List[str],
    *,
    entity_id_to_name: Optional[Dict[str, str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Scan corpus for entity mentions; return per-entity mention counts and sources.

    Args:
        corpus_sources: List of corpus items with "text" or "content" key.
        entity_ids: Entity IDs to check for mention/silence.
        entity_id_to_name: Optional mapping entity_id -> display name for text matching.
        limit: Max entities to return.

    Returns:
        List of {"entity_id", "mentioned", "mention_count", "sources_where_mentioned"}.
    """
    if not corpus_sources or not entity_ids:
        return []
    name_map = entity_id_to_name or {}
    results = []
    for eid in entity_ids[:limit]:
        name = name_map.get(eid) or eid
        mention_count = 0
        sources_where_mentioned = []
        for item in corpus_sources:
            text = _get_text(item)
            if not text:
                continue
            source_id = item.get("id") or item.get("source") or ""
            n = _count_mentions(text, name)
            if n > 0:
                mention_count += n
                if source_id and source_id not in sources_where_mentioned:
                    sources_where_mentioned.append(source_id)
            if name != eid:
                n_id = _count_mentions(text, eid)
                if n_id > 0:
                    mention_count += n_id
                    if source_id and source_id not in sources_where_mentioned:
                        sources_where_mentioned.append(source_id)
        results.append({
            "entity_id": eid,
            "mentioned": mention_count > 0,
            "mention_count": mention_count,
            "sources_where_mentioned": sources_where_mentioned,
        })
    return results


def entities_silent_in_corpus(
    corpus_sources: List[Dict[str, Any]],
    entity_ids: List[str],
    *,
    entity_id_to_name: Optional[Dict[str, str]] = None,
    limit: int = 100,
) -> List[str]:
    """
    Return entity IDs that are not mentioned in the corpus.

    Args:
        corpus_sources: List of corpus items with text content.
        entity_ids: Entity IDs to check.
        entity_id_to_name: Optional mapping entity_id -> name for text matching.
        limit: Max IDs to return.

    Returns:
        Subset of entity_ids that appear to be "silent" (unmentioned).
    """
    results = track_silence_mentions(
        corpus_sources, entity_ids,
        entity_id_to_name=entity_id_to_name,
        limit=limit,
    )
    return [r["entity_id"] for r in results if not r.get("mentioned", True)]
