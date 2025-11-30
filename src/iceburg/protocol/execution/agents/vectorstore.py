from __future__ import annotations

from typing import Dict

from iceburg.vectorstore import VectorStore

from ..registry import Agent, register
from ...config import load_config
from ...models import AgentResult


@register("vectorstore")
class VectorStoreAgent:
    def __call__(self) -> Agent:
        return self

    def run(self, payload: Dict, **kwargs) -> AgentResult:
        cfg = load_config()
        store = VectorStore(cfg)
        query = payload.get("query", "")
        result = store.search(query)
        return AgentResult(agent="vectorstore", payload=result)
