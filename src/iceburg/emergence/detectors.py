"""
Emergence detectors for ICEBURG
- Novelty: embedding distance to nearest historical items
- Surprise: proxy via quality/loss deltas provided by callers
- Compression gain: summary ratio and minhash-like ngram Jaccard heuristic
- CoT diversity: diversity across traces
- Episode detection: windowed thresholds + hysteresis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean

@dataclass
class EmergenceMetrics:
    novelty_score: float
    surprise_score: float
    compression_gain: float
    cot_diversity: float
    consensus_delta: float


class NoveltyDetector:
    def score(self, distances: List[float]) -> float:
        if not distances:
            return 0.0
        # Higher distance → higher novelty; use top-k average for stability
        topk = sorted(distances, reverse=True)[:3]
        return min(mean(topk), 1.0)


class SurpriseDetector:
    def score(self, baseline_loss: Optional[float], observed_loss: Optional[float]) -> float:
        if baseline_loss is None or observed_loss is None:
            return 0.0
        # Positive delta (worse than baseline) is not surprising in a good way; we clamp at 0
        delta = max(baseline_loss - observed_loss, 0.0)
        # Normalize with a heuristic scale
        return max(min(delta / 2.0, 1.0), 0.0)


class CompressionGainDetector:
    def score(self, original_len: int, summary_len: int, jaccard: float) -> float:
        if original_len <= 0 or summary_len <= 0:
            return 0.0
        ratio = original_len / max(summary_len, 1)
        # Favor higher compression with decent content overlap
        combined = 0.7 * min(ratio / 10.0, 1.0) + 0.3 * (1.0 - jaccard)
        return max(min(combined, 1.0), 0.0)


class CoTDiversityDetector:
    def score(self, trace_embeddings: List[List[float]]) -> float:
        # Simple proxy: average pairwise cosine distance across traces
        if len(trace_embeddings) < 2:
            return 0.0
        import math
        def cos(a, b):
            num = sum(x*y for x, y in zip(a, b))
            da = math.sqrt(sum(x*x for x in a))
            db = math.sqrt(sum(y*y for y in b))
            if da == 0 or db == 0:
                return 0.0
            return num / (da * db)
        dists = []
        for i in range(len(trace_embeddings)):
            for j in range(i+1, len(trace_embeddings)):
                d = 1.0 - cos(trace_embeddings[i], trace_embeddings[j])
                dists.append(max(min(d, 1.0), 0.0))
        return mean(dists) if dists else 0.0


class EpisodeDetector:
    def __init__(self, novelty_thr: float = 0.6, surprise_thr: float = 0.5, diversity_thr: float = 0.4, hysteresis: float = 0.1):
        self.n_thr = novelty_thr
        self.s_thr = surprise_thr
        self.d_thr = diversity_thr
        self.h = hysteresis
        self.active = False

    def step(self, metrics: EmergenceMetrics) -> Optional[str]:
        start = (metrics.novelty_score > self.n_thr) and (metrics.surprise_score > self.s_thr or metrics.cot_diversity > self.d_thr)
        end = (metrics.novelty_score < (self.n_thr - self.h)) and (metrics.surprise_score < (self.s_thr - self.h)) and (metrics.cot_diversity < (self.d_thr - self.h))
        if not self.active and start:
            self.active = True
            return "emergence_episode_start"
        if self.active and end:
            self.active = False
            return "emergence_episode_end"
        return None
