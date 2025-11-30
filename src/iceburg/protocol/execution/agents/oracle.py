from __future__ import annotations

from typing import Dict

from iceburg.config import load_config
from iceburg.agents import oracle as oracle_module

from ..registry import register
from ...models import AgentResult


@register("oracle")
class OracleAgent:
    def __call__(self) -> "OracleAgent":
        return self

    def run(self, payload: Dict, **kwargs) -> AgentResult:
        cfg = load_config()
        oracle_context = {
            "synthesis": payload.get("synthesist_output"),
            "initial_query": payload.get("query"),
        }
        result = oracle_module.run(cfg, oracle_context, verbose=False)
        return AgentResult(agent="oracle", payload=result)
