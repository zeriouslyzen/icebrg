"""
Corpus ingest pipeline for dossier protocol.

Reads a directory or file of documents (.txt, .md, .json, .jsonl, .eml) and produces
corpus_sources for downstream use (silence/mention tracker, dossier synthesis).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".eml"}
TEXT_KEYS = ("text", "content", "body", "snippet")


def _read_text(path: Path, encoding: str = "utf-8", errors: str = "replace") -> str:
    """Read file as text."""
    try:
        return path.read_text(encoding=encoding, errors=errors)
    except Exception as e:
        logger.debug("Could not read %s: %s", path, e)
        return ""


def _extract_text_from_json(data: Any) -> str:
    """Extract a single text field from JSON object."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in TEXT_KEYS:
            if key in data and data[key]:
                val = data[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, list):
                    return " ".join(str(x) for x in val)
    return ""


def _parse_eml(content: str) -> str:
    """Extract subject + body from EML-style content (simple)."""
    subject = ""
    body = content
    if "Subject:" in content:
        try:
            idx = content.index("Subject:")
            end = content.index("\n", idx) if "\n" in content[idx:] else len(content)
            subject = content[idx:end].replace("Subject:", "").strip()
        except ValueError:
            pass
    if "Content-Type: text/plain" in content or "\n\n" in content:
        parts = content.split("\n\n", 1)
        if len(parts) > 1:
            body = parts[1]
    return f"{subject}\n{body}".strip()


def load_document(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a single document into a corpus source dict.

    Returns:
        {"id", "text", "source", "path", "ext"} or None if unreadable/unsupported.
    """
    if not path.is_file():
        return None
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None
    raw = _read_text(path)
    if not raw.strip():
        return None
    text = raw
    if ext == ".json":
        try:
            data = json.loads(raw)
            text = _extract_text_from_json(data) or raw
        except json.JSONDecodeError:
            pass
    elif ext == ".jsonl":
        lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
        texts = [_extract_text_from_json(obj) for obj in lines if _extract_text_from_json(obj)]
        text = "\n".join(texts) if texts else raw
    elif ext == ".eml":
        text = _parse_eml(raw)
    return {
        "id": path.stem,
        "text": text,
        "source": str(path),
        "path": path,
        "ext": ext,
    }


def load_corpus_from_path(
    path: Path,
    *,
    extensions: Optional[List[str]] = None,
    max_files: int = 10_000,
) -> List[Dict[str, Any]]:
    """
    Load a corpus from a directory or single file.

    Args:
        path: Directory to scan or path to a single file.
        extensions: Allowed extensions (default: .txt, .md, .json, .jsonl, .eml).
        max_files: Maximum number of files to load.

    Returns:
        List of {"id", "text", "source", "path", "ext"} for downstream use.
    """
    exts = set(extensions) if extensions else SUPPORTED_EXTENSIONS
    sources = []
    if path.is_file():
        doc = load_document(path)
        if doc:
            sources.append(doc)
        return sources
    if not path.is_dir():
        logger.warning("Corpus path is not a file or directory: %s", path)
        return sources
    for f in path.rglob("*"):
        if len(sources) >= max_files:
            break
        if f.suffix.lower() not in exts:
            continue
        doc = load_document(f)
        if doc:
            sources.append(doc)
    logger.info("Corpus ingest: loaded %s documents from %s", len(sources), path)
    return sources


def ingest_corpus_for_dossier(
    path: Path,
    *,
    max_files: int = 10_000,
) -> Dict[str, Any]:
    """
    Ingest a corpus and return a result suitable for dossier/Gatherer hook.

    Returns:
        {"status": "ok", "path": str, "sources_count": int, "corpus_sources": [...]}
        or {"status": "error", "path": str, "error": str}.
    """
    try:
        corpus_sources = load_corpus_from_path(path, max_files=max_files)
        return {
            "status": "ok",
            "path": str(path),
            "sources_count": len(corpus_sources),
            "corpus_sources": corpus_sources,
        }
    except Exception as e:
        logger.exception("Corpus ingest failed: %s", e)
        return {"status": "error", "path": str(path), "error": str(e)}
