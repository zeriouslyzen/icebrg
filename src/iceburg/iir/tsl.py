from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class IOType:
    name: str
    type: str  # e.g., "tensor<float32>[N]" or "float32"


@dataclass
class Budget:
    latency_ms: Optional[int] = None
    memory_mb: Optional[int] = None


@dataclass
class TaskSpec:
    name: str
    inputs: List[IOType]
    outputs: List[IOType]
    pre: List[str] = field(default_factory=list)
    post: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    budgets: Budget = field(default_factory=Budget)


def tsl_to_json(spec: TaskSpec) -> str:
    d: Dict[str, Any] = asdict(spec)
    return json.dumps(d, indent=2)


def tsl_from_json(data: str | Dict[str, Any]) -> TaskSpec:
    obj: Dict[str, Any] = json.loads(data) if isinstance(data, str) else data
    budgets = obj.get("budgets", {})
    spec = TaskSpec(
        name=obj["name"],
        inputs=[IOType(**io) for io in obj.get("inputs", [])],
        outputs=[IOType(**io) for io in obj.get("outputs", [])],
        pre=list(obj.get("pre", [])),
        post=list(obj.get("post", [])),
        invariants=list(obj.get("invariants", [])),
        effects=list(obj.get("effects", [])),
        budgets=Budget(**budgets) if isinstance(budgets, dict) else Budget(),
    )
    return spec


