from __future__ import annotations
import json
from typing import Any, Optional, Dict, List
import urllib.request
import os


class LlamaCppProvider:
    def __init__(self, base_url: str = None, timeout_s: int = 60):
        if base_url is None:
            host = os.getenv("HOST", "localhost")
            base_url = f"http://{host}:8080"
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s

    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        # llama.cpp server ignores model name; bound at server start
        pl: dict[str, Any] = {"prompt": prompt, "temperature": temperature}
        if system:
            pl["system_prompt"] = system
        if options:
            pl.update(options)
        req = urllib.request.Request(
            f"{self._base}/completion", data=json.dumps(pl).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("content") or data.get("generation", "")

    def embed_texts(self, model: str, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            pl = {"content": t}
            req = urllib.request.Request(
                f"{self._base}/embedding", data=json.dumps(pl).encode("utf-8"), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out.append(data.get("embedding", []))
        return out


