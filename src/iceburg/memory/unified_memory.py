"""
UnifiedMemory: unified event logging + vector indexing for ICEBURG.
- JSONL append-only logs for events and metrics
- Chroma-backed vector stores per namespace for retrieval/learning
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, Embeddings, IDs, Metadatas
from chromadb.utils import embedding_functions

from ..config import IceburgConfig, load_config
from ..providers.factory import provider_factory


class SafeEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Embedding function with a safe local fallback (no network)."""
    def __init__(self, cfg: IceburgConfig, dim: int = 384):
        self._dim = dim
        self._provider = None
        try:
            self._provider = provider_factory(cfg)
            self._model = cfg.embed_model
        except Exception:
            self._provider = None
            self._model = ""

    def __call__(self, texts: Documents) -> Embeddings:  # type: ignore[override]
        # Try provider first
        if self._provider is not None and self._model:
            try:
                return self._provider.embed_texts(self._model, list(texts))
            except Exception:
                # fall back to local
                pass
        # Deterministic local fallback embedding
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> List[float]:
        # Simple hashing-based embedding (not semantic, deterministic and offline)
        vec = [0.0] * self._dim
        if not text:
            return vec
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self._dim] += (float(ch) / 255.0)
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@dataclass
class MemoryEvent:
    timestamp: str
    run_id: str
    agent_id: str
    task_id: str
    event_type: str
    payload: Dict[str, Any]
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    duration_ms: Optional[int] = None


class UnifiedMemory:
    def __init__(self, cfg: Optional[IceburgConfig] = None):
        self._cfg = cfg or load_config()
        self._root: Path = self._cfg.data_dir / "memory"
        self._root.mkdir(parents=True, exist_ok=True)
        self._events_dir: Path = self._root / "events"
        self._events_dir.mkdir(parents=True, exist_ok=True)
        # Singleton PersistentClient with telemetry disabled
        global _UNIFIED_MEMORY_CHROMA_CLIENT
        try:
            _UNIFIED_MEMORY_CHROMA_CLIENT
        except NameError:
            _UNIFIED_MEMORY_CHROMA_CLIENT = None  # type: ignore
        if _UNIFIED_MEMORY_CHROMA_CLIENT is None:
            _UNIFIED_MEMORY_CHROMA_CLIENT = chromadb.PersistentClient(
                path=str(self._cfg.data_dir / "chroma"),
                settings=Settings(anonymized_telemetry=False),
            )
        self._client = _UNIFIED_MEMORY_CHROMA_CLIENT
        self._embed_fn = SafeEmbeddingFunction(self._cfg)
        self._collections: Dict[str, Any] = {}

    # ---------- JSONL logging ----------
    def _events_file_for_run(self, run_id: str) -> Path:
        date = datetime.utcnow().strftime("%Y%m%d")
        return self._events_dir / f"{date}-{run_id}.jsonl"

    def log_event(self, event: Dict[str, Any]) -> None:
        run_id = event.get("run_id", "default")
        fpath = self._events_file_for_run(run_id)
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ---------- Vector indexing & search ----------
    def _get_collection(self, namespace: str):
        if namespace not in self._collections:
            self._collections[namespace] = self._client.get_or_create_collection(
                name=f"iceburg_{namespace}",
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[namespace]

    def index_texts(
        self,
        namespace: str,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        col = self._get_collection(namespace)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            # Use time-based ids to avoid collisions
            base = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            ids = [f"{base}_{i}" for i in range(len(texts))]
        col.add(ids=list(ids), documents=list(texts), metadatas=list(metadatas))
        return list(ids)

    def search(self, namespace: str, query: str, k: int = 8) -> List[Dict[str, Any]]:
        col = self._get_collection(namespace)
        try:
            res = col.query(query_texts=[query], n_results=k)
        except Exception:
            return []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])
        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            out.append(
                {
                    "id": ids[i],
                    "document": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distances[0][i] if distances and distances[0] and i < len(distances[0]) else None,
                }
            )
        return out

    # ---------- Convenience helpers ----------
    def log_and_index(self, run_id: str, agent_id: str, task_id: str, event_type: str, text: str, meta: Optional[Dict[str, Any]] = None):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "agent_id": agent_id,
            "task_id": task_id,
            "event_type": event_type,
            "payload": {"text": text, **(meta or {})},
        }
        self.log_event(event)
        self.index_texts("events", [text], metadatas=[{**(meta or {}), "event_type": event_type, "agent_id": agent_id, "task_id": task_id, "run_id": run_id}])
