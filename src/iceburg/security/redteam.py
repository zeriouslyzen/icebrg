"""
RedTeamAnalyzer: static policy checks for generated apps before build/package.

Checks (Swift/macOS focus for now):
- Risky APIs: Process(), system(), popen, NSTask usage
- Network risks: http:// URLs (non-TLS)
- WebKit remote scripts
- Mic/Speech/Camera usage without Info.plist privacy keys
- File access assumptions (home dir) without sandbox entitlements (informational)

Returns a report with failures, warnings, and required Info.plist privacy keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import re

from ..memory.unified_memory import UnifiedMemory


RISKY_API_PATTERNS = [
    re.compile(r"\bProcess\("),          # Swift Process
    re.compile(r"(?<!\.)\bsystem\("),    # C system(), but NOT .system( from SwiftUI fonts
    re.compile(r"\bpopen\("),            # popen()
    re.compile(r"\bNSTask\b"),           # ObjC NSTask
]

NETWORK_RISK_PATTERNS = [
    "http://",             # non-TLS
]

WEBKIT_REMOTE_SCRIPT = [
    "<script src=\"http://",  # non-TLS remote scripts
]

PRIVACY_API_TO_KEYS = {
    # API token → required privacy usage keys
    "AVFoundation": ["NSCameraUsageDescription", "NSMicrophoneUsageDescription"],
    "AVAudioEngine": ["NSMicrophoneUsageDescription"],
    "SFSpeechRecognizer": ["NSSpeechRecognitionUsageDescription"],
}


@dataclass
class RedTeamReport:
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_privacy_keys: Dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0


class RedTeamAnalyzer:
    def __init__(self):
        self._memory = UnifiedMemory()

    def analyze_swift_project(self, sources_dir: Path, run_id: str = "redteam", 
                            app_metadata: Dict = None) -> RedTeamReport:
        report = RedTeamReport()

        if not sources_dir.exists():
            report.failures.append(f"Sources directory not found: {sources_dir}")
            return report

        swift_files = list(sources_dir.rglob("*.swift"))
        for file_path in swift_files:
            try:
                text = file_path.read_text(errors="ignore")
            except Exception:
                report.warnings.append(f"Could not read {file_path}")
                continue

            # Risky APIs
            for pattern in RISKY_API_PATTERNS:
                if pattern.search(text):
                    name = pattern.pattern
                    msg = f"Risky API matched '{name}' in {file_path}"
                    report.failures.append(msg)
                    self._memory.log_and_index(run_id, "redteam", str(file_path), "policy_hit", msg)

            # Network risks
            for token in NETWORK_RISK_PATTERNS:
                if token in text:
                    msg = f"Non-TLS URL found in {file_path}"
                    report.warnings.append(msg)
                    self._memory.log_and_index(run_id, "redteam", str(file_path), "policy_warn", msg)

            # WebKit remote scripts (non-TLS)
            for token in WEBKIT_REMOTE_SCRIPT:
                if token in text:
                    msg = f"Insecure remote script over HTTP in {file_path}"
                    report.failures.append(msg)
                    self._memory.log_and_index(run_id, "redteam", str(file_path), "policy_hit", msg)

            # Privacy API usage → enforce Info.plist keys
            for api, keys in PRIVACY_API_TO_KEYS.items():
                if api in text:
                    for key in keys:
                        if key not in report.required_privacy_keys:
                            # Provide default value. User should customize later.
                            report.required_privacy_keys[key] = "This app requires access for core functionality."

        return report


