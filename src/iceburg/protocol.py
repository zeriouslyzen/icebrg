"""Compatibility wrapper for the modular ICEBURG protocol package."""

from __future__ import annotations

from typing import Any

from .protocol import run_protocol
from .protocol.models import ProtocolReport


def _report_to_legacy_output(report: ProtocolReport) -> str:
    sections = report.sections
    if "legacy_output" in sections:
        return str(sections["legacy_output"])
    if "summary" in sections and isinstance(sections["summary"], str):
        return sections["summary"]
    if sections.get("results"):
        return str(sections["results"][0])
    return ""


def iceberg_protocol(initial_query: str, *args: Any, **kwargs: Any) -> str:
    if args:
        raise TypeError("iceberg_protocol only accepts keyword arguments in this refactor phase")

    report = run_protocol(initial_query, **kwargs)
    return _report_to_legacy_output(report)
