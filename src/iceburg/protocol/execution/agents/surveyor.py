from __future__ import annotations

from typing import Dict

from iceburg.vectorstore import VectorStore
from iceburg.agents import surveyor as surveyor_module
from iceburg.config import load_config

from ..registry import register
from ...models import AgentResult


@register("surveyor")
class SurveyorAgent:
    def __call__(self) -> "SurveyorAgent":
        return self

    def run(self, payload: Dict, **kwargs) -> AgentResult:
        cfg = load_config()
        vs = VectorStore(cfg)
        result = surveyor_module.run(cfg, vs, payload.get("query", ""), **payload)
        return AgentResult(agent="surveyor", payload=result)
