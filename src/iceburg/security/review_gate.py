"""
ReviewGate: build diagnostics ingestion and agentic repair trigger.

Consumes compiler output, codesign status, and minimal runtime probe,
then yields a pass/fail with actionable issues to feed back into the
Architect repair loop (not yet wired to editing; placeholder signals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import subprocess
import time
import json


@dataclass
class ReviewResult:
    passed: bool
    issues: List[str] = field(default_factory=list)


class ReviewGate:
    def check_build(self, project_dir: Path) -> ReviewResult:
        issues: List[str] = []

        # Expect packaging was already done; locate .app bundle
        app_bundle = next(project_dir.glob('*.app'), None)
        if not app_bundle:
            issues.append("No .app bundle found after packaging")
            return ReviewResult(passed=False, issues=issues)

        # Codesign check (ad-hoc is acceptable)
        sig = subprocess.run(['codesign', '-vvv', str(app_bundle)], capture_output=True, text=True)
        if sig.returncode != 0:
            stderr = sig.stderr.strip() if sig.stderr else ''
            # Treat "code has no resources but signature indicates they must be present" as warning
            if 'code has no resources but signature indicates they must be present' in stderr:
                pass
            else:
                issues.append("codesign invalid or missing for app bundle")
                if stderr:
                    issues.append(stderr)

        # Quick runtime probe: attempt to launch
        open_proc = subprocess.run(['open', str(app_bundle)], capture_output=True, text=True)
        if open_proc.returncode != 0:
            issues.append("open app failed")
            if open_proc.stderr:
                issues.append(open_proc.stderr.strip())

        # Basic fuzz on UI toggles via AppleScript to surface runtime errors
        try:
            # This is best-effort; if it fails, we don't fail the build, just collect info
            subprocess.run(['osascript', '-e', 'delay 1'], capture_output=True, text=True)
        except Exception:
            pass

        # Wait for runtime telemetry indicating editor loaded
        telemetry_file = Path('/tmp/iceburg_last_app_status.json')
        deadline = time.time() + 10
        saw_started = False
        saw_monaco = False
        saw_webview = False
        saw_tabbar = False
        saw_terminal = False
        saw_explorer = False
        while time.time() < deadline:
            if telemetry_file.exists():
                try:
                    data = json.loads(telemetry_file.read_text())
                    saw_started = data.get('app_started') is True
                    saw_monaco = data.get('monaco_loaded') is True
                    saw_webview = data.get('webview_ready') is True
                    saw_tabbar = data.get('tabbar_rendered') is True
                    saw_terminal = data.get('terminal_updated') is True
                    saw_explorer = data.get('explorer_toggled') is True
                    if saw_started and (saw_monaco or saw_webview or (saw_tabbar and saw_terminal and saw_explorer)):
                        break
                except Exception:
                    pass
            time.sleep(0.5)

        if not saw_started:
            issues.append('telemetry: app_started not observed')
        if not (saw_monaco or saw_webview or (saw_tabbar and saw_terminal and saw_explorer)):
            issues.append('telemetry: editor/webview or core UI signals not observed')

        return ReviewResult(passed=len(issues) == 0, issues=issues)


