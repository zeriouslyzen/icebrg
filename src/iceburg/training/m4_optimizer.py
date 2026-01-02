"""
ICEBURG M4 Optimizer
====================

Optimizes training for Apple M4 Mac with:
- PyTorch MPS (Metal Performance Shaders) detection and configuration
- MLX integration (optional, for native Apple Silicon acceleration)
- Memory management for M4's unified memory architecture
- Adaptive batch sizes based on available memory
"""

from __future__ import annotations
import os
import logging
import platform
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types for training."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    MLX = "mlx"  # Apple MLX framework


@dataclass
class M4Config:
    """Configuration for M4 optimization."""
    device: DeviceType
    device_name: str
    max_memory_gb: float
    recommended_batch_size: int
    gradient_checkpointing: bool
    mixed_precision: bool
    mlx_available: bool
    mps_available: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device": self.device.value,
            "device_name": self.device_name,
            "max_memory_gb": self.max_memory_gb,
            "recommended_batch_size": self.recommended_batch_size,
            "gradient_checkpointing": self.gradient_checkpointing,
            "mixed_precision": self.mixed_precision,
            "mlx_available": self.mlx_available,
            "mps_available": self.mps_available
        }


class M4Optimizer:
    """
    Optimizer for Apple M4 Mac training.
    
    Detects available hardware and configures training for optimal performance.
    Supports PyTorch MPS as primary backend with MLX as optional optimization.
    """
    
    # Model size to memory requirements (approximate)
    MODEL_MEMORY_GB = {
        "1b": 2.0,
        "3b": 6.0,
        "7b": 14.0,
        "8b": 16.0,
        "14b": 28.0,
        "32b": 64.0,
    }
    
    # Recommended batch sizes based on model size and available memory
    BATCH_SIZE_RECOMMENDATIONS = {
        "1b": {"16gb": 8, "32gb": 16, "64gb": 32},
        "3b": {"16gb": 4, "32gb": 8, "64gb": 16},
        "7b": {"16gb": 2, "32gb": 4, "64gb": 8},
        "8b": {"16gb": 1, "32gb": 2, "64gb": 4},
        "14b": {"16gb": 1, "32gb": 2, "64gb": 4},
        "32b": {"16gb": 1, "32gb": 1, "64gb": 2},
    }
    
    def __init__(self, prefer_mlx: bool = False):
        """
        Initialize M4 optimizer.
        
        Args:
            prefer_mlx: If True, prefer MLX over MPS when both available
        """
        self.prefer_mlx = prefer_mlx
        self._config: Optional[M4Config] = None
        self._torch_device = None
        
    def detect_hardware(self) -> M4Config:
        """
        Detect available hardware and configure for optimal training.
        
        Returns:
            M4Config with detected hardware and recommended settings
        """
        if self._config is not None:
            return self._config
            
        # Check platform
        is_mac = platform.system() == "Darwin"
        is_apple_silicon = is_mac and platform.machine() == "arm64"
        
        # Check MPS availability
        mps_available = False
        try:
            import torch
            mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except (ImportError, AttributeError):
            pass
            
        # Check MLX availability
        mlx_available = False
        try:
            import mlx.core as mx
            mlx_available = True
        except ImportError:
            pass
            
        # Check CUDA availability
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass
            
        # Determine device
        if self.prefer_mlx and mlx_available:
            device = DeviceType.MLX
            device_name = "Apple MLX"
        elif mps_available:
            device = DeviceType.MPS
            device_name = "Apple MPS (Metal)"
        elif cuda_available:
            device = DeviceType.CUDA
            device_name = "NVIDIA CUDA"
        else:
            device = DeviceType.CPU
            device_name = "CPU"
            
        # Get memory info
        max_memory_gb = self._get_available_memory()
        
        # Determine recommended settings
        recommended_batch_size = self._get_recommended_batch_size("3b", max_memory_gb)
        # Disable gradient checkpointing for now (causes issues with some models)
        gradient_checkpointing = False
        mixed_precision = device in [DeviceType.MPS, DeviceType.CUDA, DeviceType.MLX]
        
        self._config = M4Config(
            device=device,
            device_name=device_name,
            max_memory_gb=max_memory_gb,
            recommended_batch_size=recommended_batch_size,
            gradient_checkpointing=gradient_checkpointing,
            mixed_precision=mixed_precision,
            mlx_available=mlx_available,
            mps_available=mps_available
        )
        
        logger.info(f"M4 Optimizer detected: {device_name}")
        logger.info(f"  Available memory: {max_memory_gb:.1f} GB")
        logger.info(f"  MPS available: {mps_available}")
        logger.info(f"  MLX available: {mlx_available}")
        logger.info(f"  Recommended batch size (3B model): {recommended_batch_size}")
        
        return self._config
        
    def _get_available_memory(self) -> float:
        """Get available memory in GB."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.total / (1024 ** 3)
        except ImportError:
            # Fallback: assume 16GB for Mac
            if platform.system() == "Darwin":
                return 16.0
            return 8.0
            
    def _get_recommended_batch_size(self, model_size: str, memory_gb: float) -> int:
        """
        Get recommended batch size for model size and available memory.
        
        Args:
            model_size: Model size string (e.g., "3b", "7b")
            memory_gb: Available memory in GB
            
        Returns:
            Recommended batch size
        """
        model_size = model_size.lower().replace("b", "b")
        
        if model_size not in self.BATCH_SIZE_RECOMMENDATIONS:
            model_size = "3b"  # Default
            
        recommendations = self.BATCH_SIZE_RECOMMENDATIONS[model_size]
        
        if memory_gb >= 64:
            return recommendations.get("64gb", 4)
        elif memory_gb >= 32:
            return recommendations.get("32gb", 2)
        else:
            return recommendations.get("16gb", 1)
            
    def get_torch_device(self) -> "torch.device":
        """
        Get PyTorch device for training.
        
        Returns:
            torch.device configured for optimal training
        """
        if self._torch_device is not None:
            return self._torch_device
            
        import torch
        
        config = self.detect_hardware()
        
        if config.device == DeviceType.MPS:
            self._torch_device = torch.device("mps")
            # Set MPS environment variables for optimization
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif config.device == DeviceType.CUDA:
            self._torch_device = torch.device("cuda")
        else:
            self._torch_device = torch.device("cpu")
            
        return self._torch_device
        
    def configure_training_args(
        self,
        model_size: str = "3b",
        base_batch_size: Optional[int] = None,
        base_learning_rate: float = 2e-4,
        epochs: int = 3
    ) -> Dict[str, Any]:
        """
        Configure training arguments optimized for detected hardware.
        
        Args:
            model_size: Model size string (e.g., "3b", "7b")
            base_batch_size: Base batch size (auto-detected if None)
            base_learning_rate: Base learning rate
            epochs: Number of training epochs
            
        Returns:
            Dictionary of training arguments
        """
        config = self.detect_hardware()
        
        # Get batch size
        if base_batch_size is None:
            batch_size = self._get_recommended_batch_size(model_size, config.max_memory_gb)
        else:
            batch_size = base_batch_size
            
        # Adjust learning rate based on batch size
        # Linear scaling rule: lr = base_lr * (batch_size / base_batch_size)
        effective_lr = base_learning_rate * (batch_size / 4)
        
        # Gradient accumulation to simulate larger batch sizes
        gradient_accumulation_steps = max(1, 16 // batch_size)
        
        args = {
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": effective_lr,
            "num_train_epochs": epochs,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "eval_strategy": "no",  # Disable eval (no separate eval dataset)
            "gradient_checkpointing": config.gradient_checkpointing,
            "optim": "adamw_torch",
        }
        
        # Add mixed precision settings
        if config.mixed_precision:
            if config.device == DeviceType.MPS:
                args["fp16"] = False  # MPS doesn't support fp16 well
                args["bf16"] = False  # MPS doesn't support bf16
            elif config.device == DeviceType.CUDA:
                args["fp16"] = True
                
        # Add dataloader settings
        args["dataloader_pin_memory"] = config.device != DeviceType.MPS
        args["dataloader_num_workers"] = 0 if config.device == DeviceType.MPS else 4
        
        return args
        
    def get_lora_config(self, model_size: str = "3b") -> Dict[str, Any]:
        """
        Get LoRA configuration optimized for model size.
        
        Args:
            model_size: Model size string
            
        Returns:
            LoRA configuration dictionary
        """
        config = self.detect_hardware()
        
        # Base LoRA config
        lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        
        # Adjust based on model size and memory
        model_size_lower = model_size.lower()
        
        if "7b" in model_size_lower or "8b" in model_size_lower:
            lora_config["r"] = 32
            lora_config["lora_alpha"] = 64
        elif "14b" in model_size_lower:
            lora_config["r"] = 64
            lora_config["lora_alpha"] = 128
        elif "32b" in model_size_lower:
            lora_config["r"] = 64
            lora_config["lora_alpha"] = 128
            # Add more target modules for larger models
            lora_config["target_modules"] = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
            
        return lora_config
        

# Global optimizer instance
_m4_optimizer: Optional[M4Optimizer] = None


def get_m4_optimizer(prefer_mlx: bool = False) -> M4Optimizer:
    """
    Get or create global M4 optimizer instance.
    
    Args:
        prefer_mlx: If True, prefer MLX over MPS
        
    Returns:
        M4Optimizer instance
    """
    global _m4_optimizer
    if _m4_optimizer is None:
        _m4_optimizer = M4Optimizer(prefer_mlx=prefer_mlx)
    return _m4_optimizer

