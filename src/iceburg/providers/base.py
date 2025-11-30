from __future__ import annotations
from typing import Any, Protocol


class LLMProvider(Protocol):
    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        options: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> str:
        ...

    def embed_texts(self, model: str, texts: list[str]) -> list[list[float]]:
        ...


