"""
Investigation Storage - Persistence layer for ICEBURG dossiers.
Saves investigations to disk with metadata, sources, and network graphs.
"""

import json
import os
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

logger = logging.getLogger(__name__)

# Default investigations directory
DEFAULT_INVESTIGATIONS_DIR = Path.home() / "Documents" / "iceburg_data" / "investigations"


@dataclass
class InvestigationMetadata:
    """Metadata for an investigation."""
    investigation_id: str
    title: str
    query: str
    created_at: str
    updated_at: str
    status: str = "active"  # active, archived, flagged
    tags: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    sources_count: int = 0
    entities_count: int = 0
    depth: str = "standard"
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationMetadata":
        return cls(**data)


@dataclass
class Investigation:
    """Complete investigation with dossier, sources, and network data."""
    metadata: InvestigationMetadata
    dossier_markdown: str = ""
    executive_summary: str = ""
    official_narrative: str = ""
    alternative_narratives: List[Dict[str, str]] = field(default_factory=list)
    key_players: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, str]] = field(default_factory=list)
    historical_parallels: List[Dict[str, str]] = field(default_factory=list)
    network_graph: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    symbol_analysis: Dict[str, Any] = field(default_factory=dict)
    follow_up_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "dossier_markdown": self.dossier_markdown,
            "executive_summary": self.executive_summary,
            "official_narrative": self.official_narrative,
            "alternative_narratives": self.alternative_narratives,
            "key_players": self.key_players,
            "contradictions": self.contradictions,
            "historical_parallels": self.historical_parallels,
            "network_graph": self.network_graph,
            "sources": self.sources,
            "symbol_analysis": self.symbol_analysis,
            "follow_up_suggestions": self.follow_up_suggestions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Investigation":
        metadata = InvestigationMetadata.from_dict(data.get("metadata", {}))
        return cls(
            metadata=metadata,
            dossier_markdown=data.get("dossier_markdown", ""),
            executive_summary=data.get("executive_summary", ""),
            official_narrative=data.get("official_narrative", ""),
            alternative_narratives=data.get("alternative_narratives", []),
            key_players=data.get("key_players", []),
            contradictions=data.get("contradictions", []),
            historical_parallels=data.get("historical_parallels", []),
            network_graph=data.get("network_graph", {}),
            sources=data.get("sources", []),
            symbol_analysis=data.get("symbol_analysis", {}),
            follow_up_suggestions=data.get("follow_up_suggestions", [])
        )
    
    @classmethod
    def from_dossier(cls, dossier: Any, query: str, depth: str = "standard") -> "Investigation":
        """Create Investigation from an IcebergDossier object."""
        # Generate investigation ID from query
        slug = cls._slugify(query)
        date_str = datetime.now().strftime("%Y-%m-%d")
        investigation_id = f"{date_str}_{slug}"
        
        # Extract data from dossier
        metadata = InvestigationMetadata(
            investigation_id=investigation_id,
            title=query[:100],
            query=query,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status="active",
            tags=cls._extract_tags(query),
            confidence_score=dossier.metadata.get("confidence_score", 0.7),
            sources_count=dossier.metadata.get("total_sources", 0),
            entities_count=dossier.metadata.get("entities_found", 0),
            depth=depth,
            version=1
        )
        
        return cls(
            metadata=metadata,
            dossier_markdown=dossier.to_markdown(),
            executive_summary=dossier.executive_summary,
            official_narrative=dossier.official_narrative,
            alternative_narratives=dossier.alternative_narratives,
            key_players=dossier.key_players,
            contradictions=dossier.contradictions,
            historical_parallels=dossier.historical_parallels,
            network_graph=getattr(dossier, 'network_map', {}),  # IcebergDossier uses network_map
            sources=dossier.sources,
            symbol_analysis=dossier.symbol_analysis,
            follow_up_suggestions=getattr(dossier, 'follow_up_research', [])  # IcebergDossier uses follow_up_research
        )
    
    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to URL-safe slug."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text[:50]
    
    @staticmethod
    def _extract_tags(query: str) -> List[str]:
        """Extract potential tags from query."""
        # Common topic keywords
        keywords = [
            "politics", "finance", "tech", "crypto", "war", "conspiracy",
            "government", "corporation", "secret", "intelligence", "military",
            "geopolitics", "economy", "resources", "energy", "climate"
        ]
        query_lower = query.lower()
        tags = [kw for kw in keywords if kw in query_lower]
        return tags[:5]  # Limit to 5 tags


class InvestigationStore:
    """
    Persistent storage for investigations.
    Saves to ~/Documents/iceburg_data/investigations/
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or DEFAULT_INVESTIGATIONS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / "index.json"
        self._ensure_index()
        logger.info(f"📁 Investigation store initialized at: {self.base_dir}")
    
    def _ensure_index(self):
        """Ensure index file exists."""
        if not self.index_file.exists():
            self._save_index({"investigations": [], "version": 1})
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the master index."""
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return {"investigations": [], "version": 1}
    
    def _save_index(self, index: Dict[str, Any]):
        """Save the master index."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
    
    def save(self, investigation: Investigation) -> str:
        """
        Save an investigation to disk.
        Returns the investigation_id.
        """
        inv_id = investigation.metadata.investigation_id
        inv_dir = self.base_dir / inv_id
        inv_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dossier markdown
        dossier_file = inv_dir / "dossier.md"
        with open(dossier_file, "w", encoding="utf-8") as f:
            f.write(investigation.dossier_markdown)
        
        # Save metadata
        metadata_file = inv_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(investigation.metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save network graph
        if investigation.network_graph:
            graph_file = inv_dir / "network_graph.json"
            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(investigation.network_graph, f, indent=2, ensure_ascii=False)
        
        # Save sources
        if investigation.sources:
            sources_file = inv_dir / "sources.json"
            with open(sources_file, "w", encoding="utf-8") as f:
                json.dump(investigation.sources, f, indent=2, ensure_ascii=False)
        
        # Save full investigation data
        data_file = inv_dir / "investigation.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(investigation.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Create exports directory
        (inv_dir / "exports").mkdir(exist_ok=True)
        
        # Update index
        index = self._load_index()
        # Remove existing entry if updating
        index["investigations"] = [
            inv for inv in index["investigations"] 
            if inv.get("investigation_id") != inv_id
        ]
        # Add new entry
        index["investigations"].append({
            "investigation_id": inv_id,
            "title": investigation.metadata.title,
            "query": investigation.metadata.query,
            "created_at": investigation.metadata.created_at,
            "updated_at": investigation.metadata.updated_at,
            "status": investigation.metadata.status,
            "tags": investigation.metadata.tags,
            "confidence_score": investigation.metadata.confidence_score,
            "sources_count": investigation.metadata.sources_count
        })
        self._save_index(index)
        
        logger.info(f"💾 Saved investigation: {inv_id}")
        return inv_id
    
    def load(self, investigation_id: str) -> Optional[Investigation]:
        """Load an investigation by ID."""
        inv_dir = self.base_dir / investigation_id
        data_file = inv_dir / "investigation.json"
        
        if not data_file.exists():
            logger.warning(f"Investigation not found: {investigation_id}")
            return None
        
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Investigation.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load investigation {investigation_id}: {e}")
            return None
    
    def list_all(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List all investigations, optionally filtered by status."""
        index = self._load_index()
        investigations = index.get("investigations", [])
        
        # Filter by status if specified
        if status:
            investigations = [inv for inv in investigations if inv.get("status") == status]
        
        # Sort by updated_at descending
        investigations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return investigations[:limit]
    
    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search investigations by query text."""
        query_lower = query.lower()
        index = self._load_index()
        investigations = index.get("investigations", [])
        
        # Simple text matching
        matches = []
        for inv in investigations:
            title = inv.get("title", "").lower()
            inv_query = inv.get("query", "").lower()
            tags = " ".join(inv.get("tags", [])).lower()
            
            if query_lower in title or query_lower in inv_query or query_lower in tags:
                matches.append(inv)
        
        return matches[:limit]
    
    def update_status(self, investigation_id: str, status: str) -> bool:
        """Update investigation status (active, archived, flagged)."""
        investigation = self.load(investigation_id)
        if not investigation:
            return False
        
        investigation.metadata.status = status
        investigation.metadata.updated_at = datetime.now().isoformat()
        self.save(investigation)
        return True
    
    def delete(self, investigation_id: str) -> bool:
        """Delete an investigation (moves to archived)."""
        return self.update_status(investigation_id, "archived")
    
    def get_investigation_dir(self, investigation_id: str) -> Path:
        """Get the directory path for an investigation."""
        return self.base_dir / investigation_id


# Singleton instance
_investigation_store: Optional[InvestigationStore] = None


def get_investigation_store() -> InvestigationStore:
    """Get the singleton investigation store instance."""
    global _investigation_store
    if _investigation_store is None:
        _investigation_store = InvestigationStore()
    return _investigation_store
