"""
Backend mode prompt tests against Ollama via /v2/query.

This hits the real FastAPI server (http://localhost:8000) and exercises
all user-facing modes end-to-end, without touching the UI.

Usage (from repo root):

    python -m tests.run_mode_smoke

Make sure:
- ./scripts/start_iceburg.sh is running
- Ollama is running with the configured models
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib import request, error


API_URL = "http://localhost:8000/v2/query"

MODES: List[str] = [
    "fast",
    "chat",
    "research",
    "civilization",
    "astrophysiology",
    "dossier",
    "deep_research",
    "unbounded",
    "truth",
    "web_research",
    "local_rag",
    "hybrid",
    "protocol",
    "code",
    "finance",
]

TEST_QUERIES: Dict[str, str] = {
    "fast": "Quick sanity check: 2+2?",
    "chat": "Say hello in one short sentence.",
    "research": "Summarize one scientific topic in two sentences.",
    "civilization": "Describe a simple agent-based simulation in one paragraph.",
    "astrophysiology": "Give a one-paragraph high-level description of astro-physiology.",
    "dossier": "Summarize a public figure in two sentences.",
    "deep_research": "List three open questions in AI safety.",
    "unbounded": "Describe an ambitious but safe AI system in one paragraph.",
    "truth": "Give one short factual statement about the Moon.",
    "web_research": "Summarize recent AI news in two sentences (if web tools available).",
    "local_rag": "Explain how this project handles conversation memory.",
    "hybrid": "Compare two AI model families briefly.",
    "protocol": "Describe the multi-agent research protocol.",
    "code": "Write a one-line Python function that returns 42.",
    "finance": "Give one high-level observation about stock markets.",
}


@dataclass
class ModeResult:
    mode: str
    ok: bool
    status_code: Optional[int]
    error: Optional[str]
    latency_s: float
    snippet: str


def _post_json(url: str, payload: Dict) -> (int, str):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as resp:
            status = resp.getcode()
            text = resp.read().decode("utf-8", errors="replace")
            return status, text
    except error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        return e.code, text
    except Exception as e:  # noqa: BLE001
        # Network / connection / timeout errors
        raise RuntimeError(f"Request failed: {e!r}") from e


def test_mode(mode: str) -> ModeResult:
    query = TEST_QUERIES.get(mode, f"Test query for mode {mode}")

    payload = {
        "query": query,
        "mode": mode,
        "conversation_id": f"smoke_{mode}_{int(time.time())}",
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 256,
        "data": None,
    }

    start = time.time()
    try:
        status, text = _post_json(API_URL, payload)
        latency = time.time() - start
    except Exception as e:  # noqa: BLE001
        latency = time.time() - start
        return ModeResult(
            mode=mode,
            ok=False,
            status_code=None,
            error=str(e),
            latency_s=latency,
            snippet="",
        )

    ok = 200 <= status < 300
    snippet = text[:260].replace("\n", " ")

    # Best-effort extra check: try to see if JSON with an "answer" or "error"
    extra_error = None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if data.get("error"):
                ok = False
                extra_error = f"API error: {data.get('error')}"
            elif data.get("detail"):
                # FastAPI-style error
                ok = False
                extra_error = f"API detail: {data.get('detail')}"
    except Exception:
        # Non-JSON body is fine; we rely on HTTP status only
        pass

    return ModeResult(
        mode=mode,
        ok=ok,
        status_code=status,
        error=extra_error,
        latency_s=latency,
        snippet=snippet,
    )


def main() -> int:
    print("ICEBURG backend mode prompt test against Ollama (/v2/query)")
    print(f"Target: {API_URL}\n")

    results: List[ModeResult] = []
    for mode in MODES:
        print(f"=== Mode: {mode} ===")
        res = test_mode(mode)
        results.append(res)

        status_str = (
            f"HTTP {res.status_code}" if res.status_code is not None else "NO RESPONSE"
        )
        outcome = "OK" if res.ok else "FAIL"
        print(f"Result: {outcome} ({status_str}), latency={res.latency_s:.2f}s")
        if res.error:
            print(f"  Error: {res.error}")
        if res.snippet:
            print(f"  Body snippet: {res.snippet}")
        print()

    total = len(results)
    passed = sum(1 for r in results if r.ok)
    failed = total - passed

    print("=" * 72)
    print(f"Total modes: {total}, Passed: {passed}, Failed: {failed}")
    if failed:
        print("Failed modes:")
        for r in results:
            if not r.ok:
                print(
                    f"  - {r.mode}: status={r.status_code}, error={r.error or 'HTTP error'}"
                )
    print("=" * 72)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

