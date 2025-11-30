from __future__ import annotations

from typing import Dict

from iceburg.config import load_config
from iceburg.agents import synthesist as synthesist_module

from ..registry import register
from ...models import AgentResult


@register("synthesist")
class SynthesistAgent:
    def __call__(self) -> "SynthesistAgent":
        return self

    def run(self, payload: Dict, **kwargs) -> AgentResult:
        cfg = load_config()
        enhanced_context = {
            "surveyor": payload.get("surveyor_output"),
            "dissident": payload.get("dissident_output"),
            "initial_query": payload.get("query"),
        }
        result = synthesist_module.run(
            cfg,
            enhanced_context,
            multimodal_evidence=payload.get("multimodal_evidence"),
        )
        return AgentResult(agent="synthesist", payload=result)
