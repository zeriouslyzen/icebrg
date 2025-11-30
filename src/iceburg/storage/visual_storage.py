"""
Visual Artifact Storage
Stores and manages generated visual artifacts
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict
import json
import uuid
import shutil


class VisualArtifactStorage:
    """Manages storage and retrieval of visual artifacts"""
    
    def __init__(self, storage_dir: Path = None):
        if storage_dir is None:
            storage_dir = Path("data/visual_artifacts")
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.storage_dir / "artifact_index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load artifact index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading index: {e}")
                return {}
        return {}
    
    def _save_index(self) -> None:
        """Save artifact index"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def _generate_artifact_id(self) -> str:
        """Generate unique artifact ID"""
        return str(uuid.uuid4())
    
    def store_visual_generation(
        self,
        result: Any,  # VisualGenerationResult
        project_id: str = "default"
    ) -> str:
        """
        Store visual generation result
        
        Args:
            result: VisualGenerationResult from VisualArchitect
            project_id: Project identifier
            
        Returns:
            Artifact ID
        """
        artifact_id = self._generate_artifact_id()
        artifact_dir = self.storage_dir / project_id / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Store spec
        spec_file = artifact_dir / "spec.json"
        try:
            spec_dict = result.spec.to_dict()
            spec_file.write_text(json.dumps(spec_dict, indent=2))
        except Exception as e:
            print(f"Error storing spec: {e}")
        
        # Store IR
        ir_file = artifact_dir / "visual_ir.json"
        try:
            ir_dict = result.ir.to_dict()
            ir_file.write_text(json.dumps(ir_dict, indent=2))
        except Exception as e:
            print(f"Error storing IR: {e}")
        
        # Store artifacts by backend
        for backend, artifacts in result.artifacts.items():
            backend_dir = artifact_dir / backend.value
            backend_dir.mkdir(exist_ok=True)
            try:
                artifacts.save_to_directory(backend_dir)
            except Exception as e:
                print(f"Error storing {backend.value} artifacts: {e}")
        
        # Store validation results
        validation_file = artifact_dir / "validation.json"
        try:
            validation_dict = {
                "passed": result.validation.passed,
                "errors": [asdict(v) for v in result.validation.errors],
                "warnings": [asdict(v) for v in result.validation.warnings]
            }
            validation_file.write_text(json.dumps(validation_dict, indent=2))
        except Exception as e:
            print(f"Error storing validation: {e}")
        
        # Store optimization results
        optimization_file = artifact_dir / "optimization.json"
        try:
            optimization_dict = {
                "applied_rules": result.optimization.applied_rules,
                "improvements": result.optimization.improvements
            }
            optimization_file.write_text(json.dumps(optimization_dict, indent=2))
        except Exception as e:
            print(f"Error storing optimization: {e}")
        
        # Store metadata
        metadata_file = artifact_dir / "metadata.json"
        metadata_file.write_text(json.dumps(result.metadata, indent=2))
        
        # Update index
        self._update_index(artifact_id, project_id, result)
        
        return artifact_id
    
    def _update_index(self, artifact_id: str, project_id: str, result: Any) -> None:
        """Update artifact index"""
        if project_id not in self.index:
            self.index[project_id] = {}
        
        self.index[project_id][artifact_id] = {
            "created_at": result.metadata.get("generated_at"),
            "user_intent": result.metadata.get("user_intent"),
            "components_count": result.metadata.get("components_generated"),
            "backends": result.metadata.get("backends_compiled"),
            "validation_passed": result.metadata.get("validation_passed")
        }
        
        self._save_index()
    
    def load_artifact(self, artifact_id: str, project_id: str = "default") -> Optional[Dict[str, Any]]:
        """Load artifact by ID"""
        artifact_dir = self.storage_dir / project_id / artifact_id
        
        if not artifact_dir.exists():
            return None
        
        artifact = {}
        
        # Load spec
        spec_file = artifact_dir / "spec.json"
        if spec_file.exists():
            artifact["spec"] = json.loads(spec_file.read_text())
        
        # Load IR
        ir_file = artifact_dir / "visual_ir.json"
        if ir_file.exists():
            artifact["ir"] = json.loads(ir_file.read_text())
        
        # Load validation
        validation_file = artifact_dir / "validation.json"
        if validation_file.exists():
            artifact["validation"] = json.loads(validation_file.read_text())
        
        # Load metadata
        metadata_file = artifact_dir / "metadata.json"
        if metadata_file.exists():
            artifact["metadata"] = json.loads(metadata_file.read_text())
        
        # List available backends
        artifact["backends"] = {}
        for backend_dir in artifact_dir.iterdir():
            if backend_dir.is_dir() and backend_dir.name in ["html5", "react", "swiftui"]:
                artifact["backends"][backend_dir.name] = str(backend_dir)
        
        return artifact
    
    def load_artifacts(self, artifact_id: str, backend: Any) -> Optional[Any]:
        """Load specific backend artifacts"""
        # This would load the actual artifacts for preview
        # For now, return the path
        artifact_dir = self.storage_dir / "default" / artifact_id / backend.value
        if artifact_dir.exists():
            return artifact_dir
        return None
    
    def list_artifacts(self, project_id: str = "default") -> List[Dict[str, Any]]:
        """List all artifacts for a project"""
        if project_id not in self.index:
            return []
        
        artifacts = []
        for artifact_id, metadata in self.index[project_id].items():
            artifacts.append({
                "artifact_id": artifact_id,
                **metadata
            })
        
        return artifacts
    
    def delete_artifact(self, artifact_id: str, project_id: str = "default") -> bool:
        """Delete an artifact"""
        artifact_dir = self.storage_dir / project_id / artifact_id
        
        if not artifact_dir.exists():
            return False
        
        try:
            shutil.rmtree(artifact_dir)
            
            # Update index
            if project_id in self.index and artifact_id in self.index[project_id]:
                del self.index[project_id][artifact_id]
                self._save_index()
            
            return True
        except Exception as e:
            print(f"Error deleting artifact: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_artifacts = sum(len(artifacts) for artifacts in self.index.values())
        projects = list(self.index.keys())
        
        # Calculate total size
        total_size = 0
        for root, dirs, files in self.storage_dir.walk():
            total_size += sum((root / file).stat().st_size for file in files)
        
        return {
            "total_artifacts": total_artifacts,
            "projects": len(projects),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_path": str(self.storage_dir)
        }


if __name__ == "__main__":
    # Example usage
    storage = VisualArtifactStorage()
    
    print("Visual Artifact Storage initialized")
    print(f"Storage directory: {storage.storage_dir}")
    
    stats = storage.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total artifacts: {stats['total_artifacts']}")
    print(f"  Projects: {stats['projects']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")

