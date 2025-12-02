from __future__ import annotations
from typing import List, Sequence, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import uuid

# Try to import chromadb, but make it optional
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import Documents, Embeddings, IDs, Metadatas
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Create dummy types for type checking
    class Settings:
        pass
    class Documents:
        pass
    class Embeddings:
        pass
    class IDs:
        pass
    class Metadatas:
        pass
    class embedding_functions:
        class EmbeddingFunction:
            pass

from .config import IceburgConfig
from .providers.factory import provider_factory


class ProviderEmbeddingFunction(embedding_functions.EmbeddingFunction if CHROMADB_AVAILABLE else object):
    def __init__(self, cfg: IceburgConfig):
        self._provider = provider_factory(cfg)
        self._model = cfg.embed_model

    def __call__(self, texts: Documents) -> Embeddings:  # type: ignore[override]
        return self._provider.embed_texts(self._model, list(texts))


@dataclass
class VectorHit:
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: Optional[float]


class VectorStore:
    def __init__(self, cfg: IceburgConfig):
        self._cfg = cfg
        self._persist_dir: Path = cfg.data_dir / "chroma"
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: If ChromaDB fails, just work without it
        if not CHROMADB_AVAILABLE:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("ChromaDB not available - VectorStore will work in mock mode")
            self._client = None
            self._collection = None
            return
        
        # Singleton PersistentClient with telemetry disabled
        # CRITICAL: Handle all ChromaDB errors gracefully - never raise exceptions
        global _VECTORSTORE_CHROMA_CLIENT
        try:
            _VECTORSTORE_CHROMA_CLIENT
        except NameError:
            _VECTORSTORE_CHROMA_CLIENT = None  # type: ignore
        
        # Only try to create client if we don't have one and ChromaDB is available
        if _VECTORSTORE_CHROMA_CLIENT is None and CHROMADB_AVAILABLE:
            try:
                _VECTORSTORE_CHROMA_CLIENT = chromadb.PersistentClient(
                    path=str(self._persist_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
                import logging
                logger = logging.getLogger(__name__)
                logger.info("✅ ChromaDB PersistentClient created successfully")
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️ ChromaDB initialization failed: {e}. VectorStore will work in mock mode (no persistence)")
                # Set to a sentinel value to prevent retrying
                _VECTORSTORE_CHROMA_CLIENT = "FAILED"  # type: ignore
        
        # If client creation failed, use mock mode
        if _VECTORSTORE_CHROMA_CLIENT == "FAILED" or _VECTORSTORE_CHROMA_CLIENT is None:
            self._client = None
            self._collection = None
            import logging
            logger = logging.getLogger(__name__)
            logger.info("✅ VectorStore operating in mock mode (no ChromaDB)")
            return
        
        self._client = _VECTORSTORE_CHROMA_CLIENT
        
        # Try to create collection, but don't fail if it doesn't work
        try:
            self._collection = self._client.get_or_create_collection(
                name="iceburg",
                embedding_function=ProviderEmbeddingFunction(cfg),
                metadata={"hnsw:space": "cosine"},
            )
            import logging
            logger = logging.getLogger(__name__)
            logger.info("✅ ChromaDB collection created successfully")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ ChromaDB collection creation failed: {e}. VectorStore will work in mock mode")
            self._collection = None
            # Don't raise - just work without collection

    def add(self, texts: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None, ids: Optional[Sequence[str]] = None) -> List[str]:
        if self._collection is None:
            # Mock mode - just return fake IDs
            return [str(uuid.uuid4()) for _ in texts]
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        self._collection.add(
            ids=cast(IDs, list(ids)),
            documents=cast(Documents, list(texts)),
            metadatas=cast(Metadatas, list(metadatas)),
        )
        return list(ids)

    def semantic_search(self, query: str, k: int = 8, where: Optional[Dict[str, Any]] = None) -> List[VectorHit]:
        # If ChromaDB is not available or collection is None, return empty results
        if self._collection is None:
            return []
        
        try:
            res = self._collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
            )
        except Exception:
            return []
        hits: List[VectorHit] = []
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])
        for i, doc in enumerate(docs):
            hits.append(VectorHit(
                id=ids[i],
                document=doc,
                metadata=metadatas[i] if i < len(metadatas) else {},
                distance=distances[0][i] if distances and distances[0] and i < len(distances[0]) else None,
            ))
        return hits


# typing helpers without importing typing.cast at top-level repeatedly
from typing import cast  # noqa: E402
