from __future__ import annotations
import json
from typing import Any
import urllib.request


class LlamaCppProvider:
    def __init__(self, base_url: str = "http://os.getenv("HOST", "localhost"):8080", timeout_s: int = 60):
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s

    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        options: dict[str, Any] | None = None,
        images: list[str] | None = None,
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


