"""
RAG Memory Integration for ICEBURG - October 2025
==============================================

Implements retrieval-augmented generation across all 7 layers
for enhanced cross-agent knowledge sharing and context retention.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import chromadb
from chromadb.config import Settings
from ..config import load_config
from ..providers.factory import provider_factory

@dataclass
class MemoryEntry:
    """Memory entry for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    layer: str
    agent: str
    embedding: List[float] = field(default_factory=list)

@dataclass
class RAGQuery:
    """Query for RAG system"""
    query_text: str
    layer: str
    agent: str
    context_limit: int = 5
    similarity_threshold: float = 0.7

class RAGMemoryIntegration:
    """
    Retrieval-Augmented Generation memory system for ICEBURG layers.
    Provides cross-layer knowledge sharing and context retention.
    """

    def __init__(self, persist_directory: str = "data/rag_memory"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the RAG memory system"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create or get collection for ICEBURG memories
            self.collection = self.client.get_or_create_collection(
                name="iceburg_memories",
                metadata={"description": "ICEBURG multi-layer memory system"}
            )

            self.is_initialized = True
            return True

        except Exception as e:
            return False

    async def store_memory(self, layer: str, agent: str, content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory entry in the RAG system"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Generate unique ID
            memory_id = f"{layer}_{agent}_{int(time.time() * 1000)}"

            # Prepare metadata
            entry_metadata = metadata or {}
            entry_metadata.update({
                "layer": layer,
                "agent": agent,
                "timestamp": datetime.now().isoformat(),
                "content_length": len(content)
            })

            # Create memory entry
            memory_entry = MemoryEntry(
                id=memory_id,
                content=content,
                metadata=entry_metadata,
                timestamp=datetime.now(),
                layer=layer,
                agent=agent
            )

            # Produce real embeddings via provider
            cfg = load_config()
            provider = provider_factory(cfg)
            vecs = provider.embed_texts(cfg.embed_model, [content])
            embedding = vecs[0] if vecs else []

            # Add to ChromaDB
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[entry_metadata],
                documents=[content]
            )

            return memory_id

        except Exception as e:
            return ""

    async def query_memory(self, query: RAGQuery) -> List[MemoryEntry]:
        """Query the RAG system for relevant memories"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Create embedding for query via provider
            cfg = load_config()
            provider = provider_factory(cfg)
            qv = provider.embed_texts(cfg.embed_model, [query.query_text])
            query_embedding = qv[0] if qv else []

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=query.context_limit,
                where={"layer": query.layer} if query.layer != "all" else None,
                include=["documents", "metadatas", "distances"]
            )

            # Convert results to MemoryEntry objects
            memories = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    if results["distances"] and results["distances"][0][i] <= (1 - query.similarity_threshold):
                        metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}

                        memory = MemoryEntry(
                            id=f"query_result_{i}",
                            content=doc,
                            metadata=metadata,
                            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                            layer=metadata.get("layer", "unknown"),
                            agent=metadata.get("agent", "unknown")
                        )
                        memories.append(memory)

            return memories

        except Exception as e:
            return []

    async def get_layer_context(self, layer: str, agent: str, limit: int = 3) -> str:
        """Get context from other layers for a specific agent"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Query for memories from all layers except current
            query = RAGQuery(
                query_text=f"context for {layer} layer",
                layer="all",  # Get from all layers
                agent=agent,
                context_limit=limit
            )

            memories = await self.query_memory(query)

            # Filter out memories from the same layer
            relevant_memories = [m for m in memories if m.layer != layer]

            if not relevant_memories:
                return ""

            # Build context string
            context_parts = []
            for memory in relevant_memories[:limit]:
                context_parts.append(f"[{memory.layer}/{memory.agent}]: {memory.content[:200]}...")

            context = "\n".join(context_parts)

            return context

        except Exception as e:
            return ""

    async def update_agent_knowledge(self, layer: str, agent: str, new_insights: str):
        """Update agent knowledge based on new insights"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Store new insights
            await self.store_memory(layer, agent, new_insights)

            # Query for related memories to build enhanced context
            query = RAGQuery(
                query_text=new_insights[:100],  # Use first 100 chars as query
                layer=layer,
                agent=agent,
                context_limit=5
            )

            related_memories = await self.query_memory(query)

            # Generate enhanced knowledge by combining insights with related memories
            enhanced_knowledge = new_insights
            if related_memories:
                related_context = "\n".join([m.content for m in related_memories[:3]])
                enhanced_knowledge = f"{new_insights}\n\nRelated Context:\n{related_context}"

            return enhanced_knowledge

        except Exception as e:
            return new_insights

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG memory system"""
        if not self.is_initialized or not self.collection:
            return {"error": "Not initialized"}

        try:
            count = self.collection.count()

            # Get metadata statistics
            all_metadata = []
            try:
                # This is a simplified approach - in production you'd query more efficiently
                results = self.collection.get(include=["metadatas"])
                if results["metadatas"]:
                    all_metadata = results["metadatas"]
            except:
                pass

            layer_counts = {}
            agent_counts = {}

            for metadata in all_metadata:
                layer = metadata.get("layer", "unknown")
                agent = metadata.get("agent", "unknown")

                layer_counts[layer] = layer_counts.get(layer, 0) + 1
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

            return {
                "total_memories": count,
                "layers_distribution": layer_counts,
                "agents_distribution": agent_counts,
                "initialization_status": "ready"
            }

        except Exception as e:
            return {"error": f"Failed to get stats: {e}"}

# Global RAG instance
_rag_memory: Optional[RAGMemoryIntegration] = None

async def get_rag_memory() -> RAGMemoryIntegration:
    """Get or create the global RAG memory instance"""
    global _rag_memory
    if _rag_memory is None:
        _rag_memory = RAGMemoryIntegration()
        await _rag_memory.initialize()
    return _rag_memory

async def store_cross_layer_memory(layer: str, agent: str, content: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
    """Store memory across layers for enhanced knowledge sharing"""
    rag = await get_rag_memory()
    return await rag.store_memory(layer, agent, content, metadata)

async def query_cross_layer_context(query: RAGQuery) -> List[MemoryEntry]:
    """Query for cross-layer context and knowledge sharing"""
    rag = await get_rag_memory()
    return await rag.query_memory(query)

async def get_enhanced_agent_context(layer: str, agent: str, limit: int = 3) -> str:
    """Get enhanced context for an agent including cross-layer knowledge"""
    rag = await get_rag_memory()
    return await rag.get_layer_context(layer, agent, limit)
