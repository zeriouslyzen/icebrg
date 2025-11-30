"""
AblationRunner: orchestrates controlled experiments (model size, swarm size, prompt compression)
Outputs JSONL metrics and leverages UnifiedMemory for indexing summaries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory.unified_memory import UnifiedMemory


@dataclass
class AblationConfig:
    run_id: str
    grid: Dict[str, List[Any]]  # e.g., {"model_size": ["small","base"], "swarm_size":[1,4], "prompt_compression":[0.0,0.3]}
    task: str
    repetitions: int = 1


@dataclass
class AblationResult:
    config: Dict[str, Any]
    quality_score: float
    tokens_in: int
    tokens_out: int
    duration_ms: int
    notes: str = ""


class AblationRunner:
    def __init__(self, memory: Optional[UnifiedMemory] = None, out_dir: Optional[Path] = None):
        self.memory = memory or UnifiedMemory()
        self.out_dir = out_dir or Path("experiments")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, cfg: AblationConfig) -> Path:
        results_path = self.out_dir / f"ablation_{cfg.run_id}.jsonl"
        with open(results_path, "a", encoding="utf-8") as f:
            for combo in self._product(cfg.grid):
                for rep in range(cfg.repetitions):
                    # Placeholder: call into agents/system with combo
                    metrics = self._simulate_run(cfg.task, combo)
                    rec = AblationResult(config=combo, **metrics)
                    f.write(json.dumps({
                        "timestamp": datetime.utcnow().isoformat(),
                        "run_id": cfg.run_id,
                        "task": cfg.task,
                        "config": rec.config,
                        "quality_score": rec.quality_score,
                        "tokens_in": rec.tokens_in,
                        "tokens_out": rec.tokens_out,
                        "duration_ms": rec.duration_ms,
                        "notes": rec.notes,
                    }) + "\n")
            
        # Index summary
        self.memory.log_and_index(
            run_id=cfg.run_id,
            agent_id="ablation_runner",
            task_id="ablation",
            event_type="ablation_summary",
            text=f"Completed ablation for task {cfg.task} with grid {list(cfg.grid.keys())}",
            meta={"results_path": str(results_path)}
        )
        return results_path

    def _product(self, grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        # Cartesian product over config grid
        keys = list(grid.keys())
        if not keys:
            return [{}]
        combos: List[Dict[str, Any]] = [{}]
        for k in keys:
            new_combos: List[Dict[str, Any]] = []
            for base in combos:
                for v in grid[k]:
                    c = dict(base)
                    c[k] = v
                    new_combos.append(c)
            combos = new_combos
        return combos

    def _simulate_run(self, task: str, combo: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder simulation; integrate with real pipeline later
        size = combo.get("model_size", "base")
        swarm = combo.get("swarm_size", 1)
        comp = combo.get("prompt_compression", 0.0)
        # Heuristic scoring
        quality = 0.6 + (0.1 if swarm > 1 else 0.0) + (0.05 if size != "small" else -0.05) - (comp * 0.1)
        quality = max(min(quality, 1.0), 0.0)
        return {
            "quality_score": quality,
            "tokens_in": int(1000 * (1.0 - comp)),
            "tokens_out": int(600 * (1.0 - comp/2)),
            "duration_ms": int(2000 + 300 * swarm),
            "notes": "simulated"
        }
