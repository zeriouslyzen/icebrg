from __future__ import annotations
from typing import Any, Protocol, Optional, Dict, List


class LLMProvider(Protocol):
    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        ...

    def embed_texts(self, model: str, texts: list[str]) -> list[list[float]]:
        ...


