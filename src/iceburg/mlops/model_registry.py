"""
Model Registry
Manages model registration and retrieval
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime


class ModelRegistry:
    """Registry for model management"""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path or "data/models/registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_registry(self) -> bool:
        """Save registry to file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            return True
        except Exception:
            return False
    
    def register_model(
        self,
        name: str,
        version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a model"""
        model_key = f"{name}:{version}"
        self.registry[model_key] = {
            "name": name,
            "version": version,
            "model_path": model_path,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }
        return self._save_registry()
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get model by name and optional version"""
        if version:
            model_key = f"{name}:{version}"
            return self.registry.get(model_key)
        else:
            # Get latest version
            versions = [k for k in self.registry.keys() if k.startswith(f"{name}:")]
            if not versions:
                return None
            # Sort by version and get latest
            latest = sorted(versions)[-1]
            return self.registry.get(latest)
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(set([k.split(':')[0] for k in self.registry.keys()]))
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model"""
        versions = [k.split(':')[1] for k in self.registry.keys() if k.startswith(f"{name}:")]
        return sorted(versions)
    
    def update_model_status(self, name: str, version: str, status: str) -> bool:
        """Update model status"""
        model_key = f"{name}:{version}"
        if model_key in self.registry:
            self.registry[model_key]["status"] = status
            return self._save_registry()
        return False
    
    def delete_model(self, name: str, version: str) -> bool:
        """Delete a model from registry"""
        model_key = f"{name}:{version}"
        if model_key in self.registry:
            del self.registry[model_key]
            return self._save_registry()
        return False

