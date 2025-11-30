"""
Persistent Model Manager for ICEBURG
Handles model saving, loading, and cross-session persistence
"""

import torch
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for saved models"""
    model_name: str
    agent_type: str
    session_id: str
    timestamp: float
    training_data_size: int
    performance_metrics: Dict[str, float]
    model_architecture: Dict[str, Any]
    training_parameters: Dict[str, Any]

class PersistentModelManager:
    """Manages persistent model storage and retrieval"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.models_dir / "model_registry.json"
        self.model_registry = self._load_model_registry()

    def _load_model_registry(self) -> Dict[str, ModelMetadata]:
        """Load model registry from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {k: ModelMetadata(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")
        return {}

    def _save_model_registry(self):
        """Save model registry to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.model_registry.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def save_model(self, model, agent_name: str, session_id: str,
                   training_data_size: int, performance_metrics: Dict[str, float],
                   model_architecture: Dict[str, Any], training_parameters: Dict[str, Any]) -> str:
        """Save a trained model with metadata"""

        model_id = f"{agent_name}_{session_id}_{int(time.time())}"
        model_path = self.models_dir / f"{model_id}.pth"

        # Create metadata
        metadata = ModelMetadata(
            model_name=model_id,
            agent_type=agent_name,
            session_id=session_id,
            timestamp=time.time(),
            training_data_size=training_data_size,
            performance_metrics=performance_metrics,
            model_architecture=model_architecture,
            training_parameters=training_parameters
        )

        # Save model state
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': asdict(metadata),
            'model_architecture': model_architecture
        }

        torch.save(checkpoint, model_path)

        # Update registry
        self.model_registry[model_id] = metadata
        self._save_model_registry()

        logger.info(f"Saved model {model_id} for agent {agent_name}")
        return model_id

    def load_model(self, model_id: str) -> Optional[Tuple[Any, ModelMetadata]]:
        """Load a saved model and its metadata"""

        if model_id not in self.model_registry:
            # Try to find most recent model for agent
            agent_models = {k: v for k, v in self.model_registry.items()
                          if model_id in k}
            if agent_models:
                model_id = max(agent_models.keys(), key=lambda x: self.model_registry[x].timestamp)
            else:
                return None

        model_path = self.models_dir / f"{model_id}.pth"
        if not model_path.exists():
            return None

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            metadata = ModelMetadata(**checkpoint['metadata'])

            # Recreate model architecture (simplified - would need full architecture recreation)
            # For now, return the state dict and metadata
            return checkpoint['model_state_dict'], metadata

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None

    def get_models_for_agent(self, agent_name: str) -> Dict[str, ModelMetadata]:
        """Get all saved models for a specific agent"""
        return {k: v for k, v in self.model_registry.items()
                if v.agent_type == agent_name}

    def cleanup_old_models(self, keep_recent: int = 10):
        """Clean up old model files, keeping only the most recent ones"""
        agent_groups = {}
        for model_id, metadata in self.model_registry.items():
            agent = metadata.agent_type
            if agent not in agent_groups:
                agent_groups[agent] = []
            agent_groups[agent].append((model_id, metadata.timestamp))

        for agent, models in agent_groups.items():
            # Sort by timestamp, keep only recent ones
            sorted_models = sorted(models, key=lambda x: x[1], reverse=True)
            to_remove = sorted_models[keep_recent:]

            for model_id, _ in to_remove:
                # Remove file and registry entry
                model_path = self.models_dir / f"{model_id}.pth"
                if model_path.exists():
                    model_path.unlink()
                del self.model_registry[model_id]

        self._save_model_registry()
        logger.info(f"Cleaned up old models, keeping {keep_recent} recent models per agent")
