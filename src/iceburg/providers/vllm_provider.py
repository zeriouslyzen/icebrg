from __future__ import annotations
import json
from typing import Any, Optional, Dict, List
import urllib.request
import os


class VLLMProvider:
    def __init__(self, base_url: str = None, timeout_s: int = 60):
        if base_url is None:
            host = os.getenv("HOST", "localhost")
            base_url = f"http://{host}:8000/v1"
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
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if options:
            payload.update(options)
        req = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def embed_texts(self, model: str, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            payload = {"model": model, "input": t}
            req = urllib.request.Request(
                f"{self._base}/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out.append(data.get("data", [{}])[0].get("embedding", []))
        return out


