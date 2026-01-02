from __future__ import annotations
import json
import os
from typing import Any, Optional, Dict, List
import urllib.request


class OllamaProvider:
    def __init__(self, base_url: str = None, timeout_s: int = 60):
        if base_url is None:
            base_url = f"http://{os.getenv('HOST', 'localhost')}:11434"
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": [],
            "options": {"temperature": temperature},
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        if images:
            content: list[Any] = [{"type": "text", "text": prompt}]
            for p in images:
                content.append({"type": "image_url", "image_url": {"url": f"file://{p}"}})
            payload["messages"].append({"role": "user", "content": content})
        else:
            payload["messages"].append({"role": "user", "content": prompt})
        if options:
            payload["options"].update(options)

        # Disable streaming for simpler parsing
        payload["stream"] = False
        
        # Phase 3.3: Hardware acceleration for local models
        # Ollama automatically uses Metal on macOS and CUDA on Linux if available
        # No additional configuration needed - Ollama handles this automatically
        
        req = urllib.request.Request(
            f"{self._base}/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            import logging
            logger = logging.getLogger(__name__)
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except:
                pass
            logger.error(f"❌ Ollama HTTP Error {e.code}: {e.reason}")
            logger.error(f"❌ Request model: {model}, payload size: {len(json.dumps(payload))} bytes")
            logger.error(f"❌ Error body: {error_body[:500]}")
            # Re-raise with more context
            raise urllib.error.HTTPError(
                req.full_url, e.code, f"Ollama error: {e.reason}. Model: {model}. Details: {error_body[:200]}", e.headers, None
            )
        
        # Handle single response object
        # Ollama can return response in different formats
        if "message" in data:
            msg = data.get("message") or {}
            content = msg.get("content", "")
        elif "response" in data:
            # Some Ollama versions return "response" directly
            content = data.get("response", "")
        else:
            # Fallback: try to extract from any nested structure
            content = str(data.get("content", data.get("text", "")))
        
        # Log if response is suspiciously short
        if content and len(content.strip()) < 5:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ Ollama returned very short response ({len(content)} chars): '{content}'")
            logger.debug(f"Full Ollama response: {json.dumps(data, indent=2)}")
        
        return content if content else ""

    def embed_texts(self, model: str, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            payload = {"model": model, "prompt": t}
            req = urllib.request.Request(
                f"{self._base}/api/embeddings", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out.append(data.get("embedding", []))
        return out


