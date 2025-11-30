from __future__ import annotations

from typing import Dict

from iceburg.config import load_config
from iceburg.agents import dissident as dissident_module

from ..registry import register
from ...models import AgentResult


@register("dissident")
class DissidentAgent:
    def __call__(self) -> "DissidentAgent":
        return self

    def run(self, payload: Dict, **kwargs) -> AgentResult:
        cfg = load_config()
        surveyor_output = payload.get("surveyor_output", "")
        result = dissident_module.run(cfg, payload.get("query", ""), surveyor_output)
        return AgentResult(agent="dissident", payload=result)
