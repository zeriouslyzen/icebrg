"""
Tool Inventory
Persistent storage for generated tools with semantic search
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from ..vectorstore import VectorStore
    from ..config import IceburgConfig, load_config
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False


class ToolInventory:
    """
    Persistent tool inventory with JSON storage and semantic search
    """
    
    def __init__(self, storage_path: Optional[str] = None, cfg: Optional[IceburgConfig] = None):
        """
        Initialize tool inventory
        
        Args:
            storage_path: Path to JSON storage file (default: data/tool_inventory.json)
            cfg: ICEBURG config (for VectorStore integration)
        """
        if storage_path is None:
            if cfg is None:
                cfg = load_config()
            storage_path = str(Path(cfg.data_dir) / "tool_inventory.json")
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.cfg = cfg
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.vectorstore: Optional[VectorStore] = None
        
        # Load existing tools
        self._load_tools()
        
        # Initialize VectorStore for semantic search if available
        if VECTORSTORE_AVAILABLE and cfg:
            try:
                self.vectorstore = VectorStore(cfg)
                # Index existing tools
                self._index_tools()
            except Exception:
                pass  # VectorStore not available or failed to initialize
    
    def _load_tools(self) -> None:
        """Load tools from JSON file"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tools = data.get("tools", {})
            except Exception:
                self.tools = {}
        else:
            self.tools = {}
    
    def _save_tools(self) -> None:
        """Save tools to JSON file"""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "tools": self.tools
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving tools: {e}")
    
    def _index_tools(self) -> None:
        """Index tools in VectorStore for semantic search"""
        if not self.vectorstore:
            return
        
        try:
            # Get all tool descriptions for indexing
            texts = []
            metadatas = []
            ids = []
            
            for tool_id, tool in self.tools.items():
                # Create searchable text from tool metadata
                description_parts = [
                    tool.get("type", ""),
                    tool.get("vulnerability_type", ""),
                    tool.get("scan_type", ""),
                    tool.get("payload_type", ""),
                    tool.get("target", ""),
                    tool.get("language", ""),
                ]
                description = " ".join(filter(None, description_parts))
                
                if description:
                    texts.append(description)
                    metadatas.append({
                        "tool_id": tool_id,
                        "type": tool.get("type", ""),
                        "created_at": tool.get("generated_at", ""),
                        "source": "tool_inventory"
                    })
                    ids.append(f"tool_{tool_id}")
            
            if texts:
                self.vectorstore.add(texts, metadatas=metadatas, ids=ids)
        except Exception:
            pass  # Indexing failed, continue without semantic search
    
    def add_tool(self, tool: Dict[str, Any]) -> str:
        """
        Add tool to inventory
        
        Args:
            tool: Tool dictionary with type, script, etc.
            
        Returns:
            Tool ID
        """
        tool_id = tool.get("id") or str(uuid.uuid4())
        tool["id"] = tool_id
        tool["generated_at"] = tool.get("generated_at", datetime.now().isoformat())
        tool["updated_at"] = datetime.now().isoformat()
        
        self.tools[tool_id] = tool
        self._save_tools()
        
        # Index in VectorStore if available
        if self.vectorstore:
            try:
                description_parts = [
                    tool.get("type", ""),
                    tool.get("vulnerability_type", ""),
                    tool.get("scan_type", ""),
                    tool.get("payload_type", ""),
                    tool.get("target", ""),
                    tool.get("language", ""),
                ]
                description = " ".join(filter(None, description_parts))
                
                if description:
                    self.vectorstore.add(
                        [description],
                        metadatas=[{
                            "tool_id": tool_id,
                            "type": tool.get("type", ""),
                            "created_at": tool.get("generated_at", ""),
                            "source": "tool_inventory"
                        }],
                        ids=[f"tool_{tool_id}"]
                    )
            except Exception:
                pass
        
        return tool_id
    
    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific tool by ID
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Tool dictionary or None
        """
        return self.tools.get(tool_id)
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools
        
        Returns:
            List of all tools
        """
        return list(self.tools.values())
    
    def search_tools(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search tools by query using semantic search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of matching tools
        """
        if not self.vectorstore:
            # Fallback to keyword search
            return self._keyword_search(query)
        
        try:
            # Semantic search in VectorStore
            hits = self.vectorstore.semantic_search(query, k=k, where={"source": "tool_inventory"})
            
            # Retrieve full tool data
            results = []
            for hit in hits:
                tool_id = hit.metadata.get("tool_id")
                if tool_id and tool_id in self.tools:
                    tool = self.tools[tool_id].copy()
                    tool["search_score"] = 1.0 - (hit.distance or 0.0)  # Convert distance to score
                    results.append(tool)
            
            return results
        except Exception:
            # Fallback to keyword search
            return self._keyword_search(query)
    
    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Fallback keyword search"""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            score = 0.0
            tool_text = " ".join([
                str(tool.get("type", "")),
                str(tool.get("vulnerability_type", "")),
                str(tool.get("scan_type", "")),
                str(tool.get("payload_type", "")),
                str(tool.get("target", "")),
            ]).lower()
            
            # Simple keyword matching
            if query_lower in tool_text:
                score = 1.0
            else:
                query_words = query_lower.split()
                tool_words = tool_text.split()
                matches = sum(1 for word in query_words if word in tool_words)
                score = matches / len(query_words) if query_words else 0.0
            
            if score > 0:
                tool_copy = tool.copy()
                tool_copy["search_score"] = score
                results.append(tool_copy)
        
        # Sort by score
        results.sort(key=lambda x: x.get("search_score", 0.0), reverse=True)
        return results
    
    def remove_tool(self, tool_id: str) -> bool:
        """
        Remove tool from inventory
        
        Args:
            tool_id: Tool ID
            
        Returns:
            True if removed, False if not found
        """
        if tool_id in self.tools:
            del self.tools[tool_id]
            self._save_tools()
            return True
        return False
    
    def update_tool(self, tool_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update tool in inventory
        
        Args:
            tool_id: Tool ID
            updates: Dictionary of updates
            
        Returns:
            True if updated, False if not found
        """
        if tool_id in self.tools:
            self.tools[tool_id].update(updates)
            self.tools[tool_id]["updated_at"] = datetime.now().isoformat()
            self._save_tools()
            return True
        return False

