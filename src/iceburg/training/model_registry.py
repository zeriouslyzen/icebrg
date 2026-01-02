"""
ICEBURG Model Registry
======================

Registry for ICEBURG fine-tuned models.
Provides model discovery, loading, and integration with ICEBURG agents.
"""

from __future__ import annotations
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a registered model."""
    AVAILABLE = "available"         # Model files exist
    DEPLOYED = "deployed"           # Model is loaded in Ollama
    UNAVAILABLE = "unavailable"     # Model files missing
    TRAINING = "training"           # Currently training


@dataclass
class RegisteredModel:
    """Metadata for a registered ICEBURG model."""
    name: str
    model_type: str  # surveyor, dissident, synthesist, oracle, base
    base_model: str
    path: Path
    created_at: str
    training_samples: int
    final_loss: float
    status: ModelStatus = ModelStatus.AVAILABLE
    ollama_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "path": str(self.path),
            "created_at": self.created_at,
            "training_samples": self.training_samples,
            "final_loss": self.final_loss,
            "status": self.status.value,
            "ollama_name": self.ollama_name,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredModel":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            model_type=data["model_type"],
            base_model=data["base_model"],
            path=Path(data["path"]),
            created_at=data["created_at"],
            training_samples=data.get("training_samples", 0),
            final_loss=data.get("final_loss", 0.0),
            status=ModelStatus(data.get("status", "available")),
            ollama_name=data.get("ollama_name"),
            metadata=data.get("metadata", {})
        )


class ICEBURGModelRegistry:
    """
    Registry for ICEBURG fine-tuned models.
    
    Provides:
    - Model discovery from filesystem
    - Model registration and metadata
    - Ollama deployment
    - Model loading for inference
    """
    
    DEFAULT_MODELS_DIR = Path("models/iceburg")
    REGISTRY_FILE = "model_registry.json"
    
    # Model type to agent mapping
    AGENT_MODEL_MAP = {
        "surveyor": "iceburg-surveyor",
        "dissident": "iceburg-dissident",
        "synthesist": "iceburg-synthesist",
        "oracle": "iceburg-oracle",
        "base": "iceburg-base"
    }
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model registry.
        
        Args:
            models_dir: Directory containing ICEBURG models
        """
        self.models_dir = models_dir or self.DEFAULT_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self._registry: Dict[str, RegisteredModel] = {}
        self._load_registry()
        self._discover_models()
        
    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_path = self.models_dir / self.REGISTRY_FILE
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    data = json.load(f)
                for name, model_data in data.items():
                    self._registry[name] = RegisteredModel.from_dict(model_data)
                logger.info(f"Loaded {len(self._registry)} models from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                
    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_path = self.models_dir / self.REGISTRY_FILE
        try:
            data = {name: model.to_dict() for name, model in self._registry.items()}
            with open(registry_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            
    def _discover_models(self) -> None:
        """Discover models from filesystem."""
        if not self.models_dir.exists():
            return
            
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            # Check for training_result.json
            result_file = model_dir / "training_result.json"
            if not result_file.exists():
                continue
                
            model_name = model_dir.name
            if model_name in self._registry:
                # Update status
                self._registry[model_name].status = ModelStatus.AVAILABLE
                continue
                
            # Load training result
            try:
                with open(result_file, "r") as f:
                    result = json.load(f)
                    
                # Determine model type from name
                model_type = "base"
                for agent_type in ["surveyor", "dissident", "synthesist", "oracle"]:
                    if agent_type in model_name.lower():
                        model_type = agent_type
                        break
                        
                # Register model
                model = RegisteredModel(
                    name=model_name,
                    model_type=model_type,
                    base_model=result.get("base_model", "unknown"),
                    path=model_dir,
                    created_at=model_dir.stat().st_mtime,
                    training_samples=result.get("filtered_samples", 0),
                    final_loss=result.get("final_loss", 0.0),
                    status=ModelStatus.AVAILABLE,
                    metadata=result.get("evaluation_metrics", {})
                )
                
                self._registry[model_name] = model
                logger.info(f"Discovered model: {model_name} ({model_type})")
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                
        self._save_registry()
        
    def get_model(self, name: str) -> Optional[RegisteredModel]:
        """Get a registered model by name."""
        return self._registry.get(name)
        
    def get_models_by_type(self, model_type: str) -> List[RegisteredModel]:
        """Get all models of a specific type."""
        return [m for m in self._registry.values() if m.model_type == model_type]
        
    def get_best_model(self, model_type: str) -> Optional[RegisteredModel]:
        """Get the best (lowest loss) model of a specific type."""
        models = self.get_models_by_type(model_type)
        if not models:
            return None
        return min(models, key=lambda m: m.final_loss)
        
    def get_latest_model(self, model_type: str) -> Optional[RegisteredModel]:
        """Get the most recent model of a specific type."""
        models = self.get_models_by_type(model_type)
        if not models:
            return None
        return max(models, key=lambda m: m.created_at)
        
    def list_models(self) -> List[RegisteredModel]:
        """List all registered models."""
        return list(self._registry.values())
        
    def deploy_to_ollama(self, model_name: str) -> Tuple[bool, str]:
        """
        Deploy a model to Ollama.
        
        Args:
            model_name: Name of the model to deploy
            
        Returns:
            Tuple of (success, message)
        """
        model = self.get_model(model_name)
        if not model:
            return False, f"Model not found: {model_name}"
            
        modelfile_path = model.path / "Modelfile"
        if not modelfile_path.exists():
            return False, f"Modelfile not found at {modelfile_path}"
            
        # Create Ollama model
        ollama_name = f"iceburg-{model.model_type}"
        
        try:
            result = subprocess.run(
                ["ollama", "create", ollama_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                cwd=str(model.path)
            )
            
            if result.returncode == 0:
                model.status = ModelStatus.DEPLOYED
                model.ollama_name = ollama_name
                self._save_registry()
                return True, f"Model deployed as '{ollama_name}'"
            else:
                return False, f"Ollama error: {result.stderr}"
                
        except FileNotFoundError:
            return False, "Ollama not installed or not in PATH"
        except Exception as e:
            return False, f"Deployment failed: {e}"
            
    def get_ollama_model_name(self, agent_type: str) -> Optional[str]:
        """
        Get the Ollama model name for an agent type.
        
        Args:
            agent_type: Type of agent (surveyor, dissident, etc.)
            
        Returns:
            Ollama model name if deployed, None otherwise
        """
        # Find deployed model of this type
        models = self.get_models_by_type(agent_type)
        for model in models:
            if model.status == ModelStatus.DEPLOYED and model.ollama_name:
                return model.ollama_name
                
        # Return expected name even if not deployed
        return self.AGENT_MODEL_MAP.get(agent_type)
        
    def register_model(
        self,
        name: str,
        model_type: str,
        base_model: str,
        path: Path,
        training_samples: int = 0,
        final_loss: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> RegisteredModel:
        """
        Register a new model.
        
        Args:
            name: Model name
            model_type: Type (surveyor, dissident, etc.)
            base_model: Base model used
            path: Path to model directory
            training_samples: Number of training samples
            final_loss: Final training loss
            metadata: Additional metadata
            
        Returns:
            RegisteredModel instance
        """
        model = RegisteredModel(
            name=name,
            model_type=model_type,
            base_model=base_model,
            path=path,
            created_at=datetime.now().isoformat(),
            training_samples=training_samples,
            final_loss=final_loss,
            metadata=metadata or {}
        )
        
        self._registry[name] = model
        self._save_registry()
        
        logger.info(f"Registered model: {name} ({model_type})")
        return model
        
    def unregister_model(self, name: str) -> bool:
        """Unregister a model."""
        if name in self._registry:
            del self._registry[name]
            self._save_registry()
            return True
        return False
        

# Global registry instance
_registry: Optional[ICEBURGModelRegistry] = None


def get_model_registry() -> ICEBURGModelRegistry:
    """Get or create global model registry."""
    global _registry
    if _registry is None:
        _registry = ICEBURGModelRegistry()
    return _registry


def get_iceburg_model_for_agent(agent_type: str) -> Optional[str]:
    """
    Get the ICEBURG model name for an agent type.
    
    Args:
        agent_type: Type of agent
        
    Returns:
        Model name for Ollama or None
    """
    registry = get_model_registry()
    return registry.get_ollama_model_name(agent_type)

