"""
Model Versioning
Manages model versioning and lifecycle
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import hashlib
import json


class ModelVersioning:
    """Manages model versioning"""
    
    def __init__(self):
        self.versions: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_version(
        self,
        name: str,
        model_data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new model version"""
        # Generate version hash
        model_str = json.dumps(model_data, sort_keys=True) if isinstance(model_data, dict) else str(model_data)
        version_hash = hashlib.sha256(model_str.encode()).hexdigest()[:8]
        
        version = {
            "version": version_hash,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "model_data": model_data
        }
        
        if name not in self.versions:
            self.versions[name] = []
        
        self.versions[name].append(version)
        return version_hash
    
    def get_version(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get a specific version"""
        if name not in self.versions:
            return None
        
        for v in self.versions[name]:
            if v["version"] == version:
                return v
        return None
    
    def get_latest_version(self, name: str) -> Optional[Dict[str, Any]]:
        """Get latest version"""
        if name not in self.versions or not self.versions[name]:
            return None
        
        return sorted(self.versions[name], key=lambda x: x["created_at"])[-1]
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model"""
        if name not in self.versions:
            return []
        
        return [v["version"] for v in self.versions[name]]
    
    def compare_versions(self, name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions"""
        v1 = self.get_version(name, version1)
        v2 = self.get_version(name, version2)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        return {
            "version1": v1["version"],
            "version2": v2["version"],
            "created_at_diff": (datetime.fromisoformat(v2["created_at"]) - datetime.fromisoformat(v1["created_at"])).total_seconds(),
            "metadata_diff": self._diff_dicts(v1["metadata"], v2["metadata"])
        }
    
    def _diff_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two dictionaries"""
        diff = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        for key in all_keys:
            if key not in dict1:
                diff[key] = {"added": dict2[key]}
            elif key not in dict2:
                diff[key] = {"removed": dict1[key]}
            elif dict1[key] != dict2[key]:
                diff[key] = {"changed": {"from": dict1[key], "to": dict2[key]}}
        return diff

