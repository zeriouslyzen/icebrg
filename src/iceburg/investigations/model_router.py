"""
Model Router - Intelligent model selection with fallback and staged processing.
Prevents Ollama crashes by routing tasks to appropriate model sizes.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for model routing."""
    SIMPLE = "simple"      # Quick answers, summaries, chat
    MEDIUM = "medium"      # Analysis, entity extraction
    COMPLEX = "complex"    # Deep research, multi-step reasoning
    CRITICAL = "critical"  # Full dossier synthesis


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    size_gb: float
    context_window: int
    tokens_per_second: float  # Approximate
    suitable_for: List[TaskComplexity]


# Model registry
MODELS = {
    "llama3.1:8b": ModelConfig(
        name="llama3.1:8b",
        size_gb=4.7,
        context_window=131072,
        tokens_per_second=50,
        suitable_for=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]
    ),
    "llama3.1:70b": ModelConfig(
        name="llama3.1:70b",
        size_gb=40,
        context_window=131072,
        tokens_per_second=8,
        suitable_for=[TaskComplexity.MEDIUM, TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]
    ),
    "gemini-2.0-flash-exp": ModelConfig(
        name="gemini-2.0-flash-exp",
        size_gb=0,  # Cloud model
        context_window=1000000,
        tokens_per_second=100,
        suitable_for=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM, TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]
    ),
}


class OllamaHealthChecker:
    """Check Ollama server health and model availability."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def is_healthy(self) -> bool:
        """Check if Ollama server is responding."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def is_model_loaded(self, model: str) -> bool:
        """Check if a specific model is currently loaded."""
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return any(m.get("name", "").startswith(model.split(":")[0]) for m in models)
            return False
        except Exception:
            return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
            return []
        except Exception:
            return []
    
    def warm_model(self, model: str) -> bool:
        """Warm up a model by loading it."""
        try:
            logger.info(f"🔥 Warming model: {model}")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "Hello", "stream": False},
                timeout=60
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to warm model {model}: {e}")
            return False


class ModelRouter:
    """
    Routes tasks to appropriate models based on complexity and availability.
    Implements automatic fallback on failure.
    """
    
    def __init__(self, cfg: Optional[Any] = None):
        self.cfg = cfg
        self.health_checker = OllamaHealthChecker()
        self.fallback_chain = ["llama3.1:70b", "llama3.1:8b", "gemini-2.0-flash-exp"]
        self._last_failure: Dict[str, float] = {}
        self._failure_cooldown = 60  # seconds
    
    def select_model(
        self,
        task_type: str,
        context_length: int = 0,
        prefer_local: bool = True
    ) -> str:
        """
        Select the best model for a task.
        
        Args:
            task_type: Type of task (chat, follow_up, analysis, synthesis, full_dossier)
            context_length: Approximate context size in characters
            prefer_local: Prefer local Ollama models over cloud
            
        Returns:
            Model name to use
        """
        complexity = self._task_to_complexity(task_type)
        
        # For simple tasks, always use 8b - fast and reliable
        if complexity == TaskComplexity.SIMPLE:
            return "llama3.1:8b"
        
        # For medium tasks, prefer 8b but allow 70b
        if complexity == TaskComplexity.MEDIUM:
            if self._is_model_available("llama3.1:8b"):
                return "llama3.1:8b"
            return "llama3.1:70b"
        
        # For complex/critical, try 70b first
        if self._is_model_available("llama3.1:70b"):
            return "llama3.1:70b"
        
        # Fallback to 8b
        return "llama3.1:8b"
    
    def _task_to_complexity(self, task_type: str) -> TaskComplexity:
        """Map task type to complexity level."""
        mapping = {
            "chat": TaskComplexity.SIMPLE,
            "follow_up": TaskComplexity.SIMPLE,
            "quick_answer": TaskComplexity.SIMPLE,
            "summarize": TaskComplexity.SIMPLE,
            "entity_extraction": TaskComplexity.MEDIUM,
            "analysis": TaskComplexity.MEDIUM,
            "network_mapping": TaskComplexity.MEDIUM,
            "narrative_synthesis": TaskComplexity.COMPLEX,
            "full_dossier": TaskComplexity.CRITICAL,
            "deep_research": TaskComplexity.CRITICAL,
        }
        return mapping.get(task_type, TaskComplexity.MEDIUM)
    
    def _is_model_available(self, model: str) -> bool:
        """Check if model is available and not in cooldown."""
        # Check cooldown from recent failures
        if model in self._last_failure:
            elapsed = time.time() - self._last_failure[model]
            if elapsed < self._failure_cooldown:
                logger.debug(f"Model {model} in cooldown for {self._failure_cooldown - elapsed:.0f}s")
                return False
        
        # For local models, check Ollama health
        if model.startswith("llama"):
            return self.health_checker.is_healthy()
        
        return True
    
    def record_failure(self, model: str):
        """Record a model failure for cooldown."""
        self._last_failure[model] = time.time()
        logger.warning(f"⚠️ Recorded failure for {model}, cooldown {self._failure_cooldown}s")
    
    def clear_failure(self, model: str):
        """Clear failure record for a model."""
        if model in self._last_failure:
            del self._last_failure[model]
    
    def get_fallback(self, failed_model: str) -> Optional[str]:
        """Get fallback model after a failure."""
        try:
            idx = self.fallback_chain.index(failed_model)
            if idx + 1 < len(self.fallback_chain):
                fallback = self.fallback_chain[idx + 1]
                logger.info(f"🔄 Falling back from {failed_model} to {fallback}")
                return fallback
        except ValueError:
            pass
        
        # Return first available in chain
        for model in self.fallback_chain:
            if model != failed_model and self._is_model_available(model):
                return model
        
        return None


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    model_used: str = ""
    duration_seconds: float = 0
    

class StagedPipeline:
    """
    Execute dossier generation in stages with checkpointing.
    If one stage fails, previous stages are preserved.
    """
    
    def __init__(self, router: ModelRouter):
        self.router = router
        self.stages: List[str] = [
            "gather_sources",
            "extract_entities", 
            "map_network",
            "decode_symbols",
            "analyze_narratives",
            "synthesize_dossier"
        ]
        self.checkpoints: Dict[str, StageResult] = {}
        self.current_stage: int = 0
    
    def get_stage_model(self, stage: str) -> str:
        """Get the appropriate model for each stage."""
        stage_tasks = {
            "gather_sources": "quick_answer",  # Just web queries
            "extract_entities": "entity_extraction",
            "map_network": "network_mapping",
            "decode_symbols": "analysis",
            "analyze_narratives": "narrative_synthesis",
            "synthesize_dossier": "full_dossier"
        }
        task_type = stage_tasks.get(stage, "analysis")
        return self.router.select_model(task_type)
    
    def save_checkpoint(self, stage: str, result: StageResult):
        """Save stage result as checkpoint."""
        self.checkpoints[stage] = result
        logger.info(f"💾 Checkpoint saved: {stage} (success={result.success})")
    
    def get_checkpoint(self, stage: str) -> Optional[StageResult]:
        """Get a previously saved checkpoint."""
        return self.checkpoints.get(stage)
    
    def can_resume_from(self, stage: str) -> bool:
        """Check if we can resume from a stage (all previous stages complete)."""
        stage_idx = self.stages.index(stage)
        for i in range(stage_idx):
            prev_stage = self.stages[i]
            checkpoint = self.checkpoints.get(prev_stage)
            if not checkpoint or not checkpoint.success:
                return False
        return True
    
    def get_resume_point(self) -> str:
        """Get the stage to resume from after a failure."""
        for stage in self.stages:
            checkpoint = self.checkpoints.get(stage)
            if not checkpoint or not checkpoint.success:
                return stage
        return self.stages[-1]  # All complete
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current pipeline progress."""
        completed = sum(1 for s in self.stages if self.checkpoints.get(s, StageResult(s, False)).success)
        return {
            "total_stages": len(self.stages),
            "completed_stages": completed,
            "current_stage": self.get_resume_point(),
            "progress_percent": (completed / len(self.stages)) * 100,
            "checkpoints": {s: self.checkpoints.get(s, StageResult(s, False)).success for s in self.stages}
        }


def execute_with_fallback(
    func: Callable,
    router: ModelRouter,
    task_type: str,
    max_retries: int = 2,
    **kwargs
) -> Tuple[Any, str]:
    """
    Execute a function with automatic model fallback on failure.
    
    Args:
        func: Function to execute (should accept 'model' kwarg)
        router: ModelRouter instance
        task_type: Task type for model selection
        max_retries: Maximum retry attempts
        **kwargs: Additional arguments to pass to func
        
    Returns:
        Tuple of (result, model_used)
    """
    model = router.select_model(task_type)
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"🚀 Executing {task_type} with {model} (attempt {attempt + 1})")
            result = func(model=model, **kwargs)
            router.clear_failure(model)  # Success!
            return result, model
            
        except Exception as e:
            last_error = e
            logger.warning(f"❌ {task_type} failed with {model}: {e}")
            router.record_failure(model)
            
            # Get fallback model
            fallback = router.get_fallback(model)
            if fallback:
                model = fallback
                logger.info(f"🔄 Retrying with fallback: {model}")
            else:
                break
    
    raise last_error or Exception("All models failed")


# Singleton router instance
_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """Get the singleton model router instance."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
