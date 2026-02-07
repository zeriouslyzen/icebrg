"""
Robust parsing of JSON from LLM responses.

LLMs often return markdown-wrapped JSON, leading text, or malformed output.
This module provides a single, well-tested path for parsing so the pipeline
does not fail with "Expecting value: line 1 column 1 (char 0)".
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    s = raw.strip()
    if s.startswith("```"):
        # Take content after first ```
        parts = s.split("```", 2)
        if len(parts) >= 2:
            s = parts[1]
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()
    return s


def _find_json_array_or_object(s: str) -> Optional[str]:
    """Find first complete JSON array [...] or object {...} by bracket matching."""
    s = s.strip()
    open_char = None
    close_char = None
    for i, c in enumerate(s):
        if c == "[":
            open_char = "["
            close_char = "]"
            break
        if c == "{":
            open_char = "{"
            close_char = "}"
            break
    if not open_char:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_char:
            depth += 1
        elif s[j] == close_char:
            depth -= 1
            if depth == 0:
                return s[i : j + 1]
    return None


def parse_llm_json(
    raw: str,
    default: Any = None,
    *,
    expect_list: bool = True,
    log_context: str = "",
) -> Any:
    """
    Parse JSON from an LLM response with fallbacks.

    Steps:
    1. Strip whitespace and remove markdown code fences (```json ... ```).
    2. Try json.loads(cleaned).
    3. If that fails, find first complete [...] or {...} by bracket matching and parse.
    4. On any failure, return default and log a warning.

    Args:
        raw: Raw LLM response string.
        default: Value to return on parse failure (e.g. [] or {}).
        expect_list: If True, default is [] when not provided; else {}.
        log_context: Short label for logs (e.g. "claim extraction").

    Returns:
        Parsed JSON (list or dict) or default.
    """
    if default is None:
        default = [] if expect_list else {}
    if not raw or not isinstance(raw, str):
        return default
    s = _strip_markdown_fences(raw)
    if not s:
        if log_context:
            logger.warning("%s: empty response after strip", log_context)
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try to extract a single JSON value from the string
    extracted = _find_json_array_or_object(s)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    if log_context:
        logger.warning(
            "%s: could not parse JSON (first 200 chars): %s",
            log_context,
            repr(s[:200]),
        )
    return default
