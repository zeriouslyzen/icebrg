"""
Linguistic marker detector for intel-style analysis.

Detects gatekeeper, reciprocity, euphemism, and compartmentation phrases
in text (e.g. dossier narrative, email body). Used by Decoder and Mapper
to tag entities/relationships with linguistic_flags and infer relationship types.
"""

import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Epstein-derived phrase lists (gatekeeper, reciprocity, euphemism, compartmentation)
GATEKEEPER_PHRASES = [
    "craft an answer",
    "craft a response",
    "i can connect you with",
    "connect you with",
    "introductions to",
    "introduction to",
    "prominent global figures",
    "prominent figures",
    "wing man",
    "wingman",
    "broker access",
    "brokering access",
]

RECIPROCITY_PHRASES = [
    "generating a debt",
    "generate a debt",
    "save him",
    "save her",
    "valuable currency",
    "political currency",
    "pr and political currency",
    "best shot",
    "best shot is",
    "positive benefit for you",
    "hang him",
    "hang her",
]

EUPHEMISM_PHRASES = [
    "dog that hasn't barked",
    "hasn't barked",
    "the girls",
    "the girl",
    "forced holding pattern",
    "holding pattern",
    "wing man",
    "wingman",
]

COMPARTMENTATION_PHRASES = [
    "off the record",
    "on background",
    "not for attribution",
    "between us",
    "confidentially",
]


def detect_linguistic_markers(text: str) -> List[Dict[str, Any]]:
    """
    Detect linguistic markers (gatekeeper, reciprocity, euphemism, compartmentation) in text.

    Args:
        text: Plain text to analyze (e.g. dossier narrative, email body).

    Returns:
        List of {"phrase": str, "type": str, "span": (start, end)} where type is one of
        gatekeeper, reciprocity, euphemism, compartmentation.
    """
    if not text or not text.strip():
        return []

    results: List[Dict[str, Any]] = []
    text_lower = text.lower()

    phrase_lists = [
        ("gatekeeper", GATEKEEPER_PHRASES),
        ("reciprocity", RECIPROCITY_PHRASES),
        ("euphemism", EUPHEMISM_PHRASES),
        ("compartmentation", COMPARTMENTATION_PHRASES),
    ]

    for marker_type, phrases in phrase_lists:
        for phrase in phrases:
            start = 0
            while True:
                pos = text_lower.find(phrase, start)
                if pos == -1:
                    break
                results.append({
                    "phrase": phrase,
                    "type": marker_type,
                    "span": (pos, pos + len(phrase)),
                })
                start = pos + 1

    logger.debug(f"Detected {len(results)} linguistic markers in text length {len(text)}")
    return results
