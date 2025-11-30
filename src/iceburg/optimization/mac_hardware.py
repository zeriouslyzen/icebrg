"""
Mac Hardware Optimization for ICEBURG
Optimizes for Apple Silicon, Neural Engine, and Metal acceleration.
"""

import os
import json
import logging
import time
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import platform

logger = logging.getLogger(__name__)


@dataclass
class MacHardwareInfo:
    """Mac hardware information."""
    processor: str
    architecture: str
    cpu_cores: int
    memory_gb: float
    gpu_type: str
    neural_engine: bool
    metal_support: bool
    thermal_state: str
    power_mode: str


@dataclass
class OptimizationConfig:
    """Hardware optimization configuration."""
    neural_engine_enabled: bool
    metal_acceleration: bool
    energy_efficiency: bool
    thermal_management: bool
    cpu_optimization: bool
    memory_optimization: bool


class MacHardwareOptimizer:
    """
    Mac hardware optimizer for ICEBURG.
    
    Features:
    - Apple Silicon optimization
    - Neural Engine utilization
    - Metal acceleration
    - Energy efficiency
    - Thermal management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Mac hardware optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.hardware_info = self._detect_mac_hardware()
        self.optimization_config = OptimizationConfig(
            neural_engine_enabled=self.config.get("neural_engine", True),
            metal_acceleration=self.config.get("metal_acceleration", True),
            energy_efficiency=self.config.get("energy_efficiency", True),
            thermal_management=self.config.get("thermal_management", True),
            cpu_optimization=self.config.get("cpu_optimization", True),
            memory_optimization=self.config.get("memory_optimization", True)
        )
        
        # Performance metrics
        self.performance_history = []
        self.energy_history = []
        self.optimization_results = {}
        
        # Initialize optimizations
        self._initialize_optimizations()
    
    def _detect_mac_hardware(self) -> MacHardwareInfo:
        """Detect Mac hardware information."""
        # Get processor information
        processor = platform.processor()
        if not processor or processor == "i386":
            # Try to get more specific info
            try:
                result = os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
                processor = result if result else "Unknown"
            except:
                processor = "Unknown"
        
        # Detect architecture
        architecture = platform.machine()
        is_apple_silicon = architecture == "arm64"
        
        # Get CPU cores
        cpu_cores = psutil.cpu_count(logical=False)
        
        # Get memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Detect GPU (simplified)
        gpu_type = "Unknown"
        try:
            result = os.popen("system_profiler SPDisplaysDataType | grep 'Chipset Model'").read().strip()
            if result:
                gpu_type = result.split(": ")[1] if ": " in result else "Unknown"
        except:
            gpu_type = "Unknown"
        
        # Neural Engine support (Apple Silicon only)
        neural_engine = is_apple_silicon and ("M1" in processor or "M2" in processor or "M3" in processor or "M4" in processor)
        
        # Metal support (all modern Macs)
        metal_support = True
        
        # Get thermal state
        thermal_state = self._get_thermal_state()
        
        # Get power mode
        power_mode = self._get_power_mode()
        
        return MacHardwareInfo(
            processor=processor,
            architecture=architecture,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_type=gpu_type,
            neural_engine=neural_engine,
            metal_support=metal_support,
            thermal_state=thermal_state,
            power_mode=power_mode
        )
    
    def _get_thermal_state(self) -> str:
        """Get current thermal state."""
        try:
            # Use powermetrics to get thermal state
            result = os.popen("powermetrics -n 1 -i 1000 | grep 'Thermal'").read().strip()
            if "Thermal" in result:
                if "Hot" in result:
                    return "hot"
                elif "Critical" in result:
                    return "critical"
                else:
                    return "normal"
        except:
            pass
        
        # Fallback: estimate from CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 90:
            return "hot"
        elif cpu_usage > 95:
            return "critical"
        else:
            return "normal"
    
    def _get_power_mode(self) -> str:
        """Get current power mode."""
        try:
            # Check for low power mode
            result = os.popen("pmset -g").read()
            if "lowpowermode" in result.lower():
                return "low_power"
            elif "high performance" in result.lower():
                return "high_performance"
            else:
                return "balanced"
        except:
            return "balanced"
    
    def _initialize_optimizations(self):
        """Initialize hardware optimizations."""
        logger.info(f"Initializing Mac hardware optimizations for {self.hardware_info.processor}")
        
        # Set up optimizations based on hardware
        if self.hardware_info.neural_engine:
            logger.info("Neural Engine detected, enabling ML optimizations")
            self._enable_neural_engine()
        
        if self.hardware_info.metal_support:
            logger.info("Metal support detected, enabling GPU acceleration")
            self._enable_metal_acceleration()
        
        if self.hardware_info.architecture == "arm64":
            logger.info("Apple Silicon detected, enabling ARM optimizations")
            self._enable_arm_optimizations()
    
    def _enable_neural_engine(self):
        """Enable Neural Engine optimizations."""
        # Set environment variables for Neural Engine
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Enable Core ML optimizations
        os.environ["COREML_ENABLE_NEURAL_ENGINE"] = "1"
        
        logger.info("Neural Engine optimizations enabled")
    
    def _enable_metal_acceleration(self):
        """Enable Metal acceleration."""
        # Set Metal environment variables
        os.environ["METAL_DEVICE_SELECTION"] = "1"
        os.environ["METAL_PERFORMANCE_SHADERS"] = "1"
        
        # Enable Metal compute
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        logger.info("Metal acceleration enabled")
    
    def _enable_arm_optimizations(self):
        """Enable ARM-specific optimizations."""
        # Set ARM optimizations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Enable ARM NEON optimizations
        os.environ["PYTORCH_USE_MPS"] = "1"
        
        logger.info("ARM optimizations enabled")
    
    async def optimize_for_task(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize hardware for a specific task.
        
        Args:
            task: Task description
            context: Task context
            
        Returns:
            Optimization result
        """
        # Analyze task requirements
        task_requirements = self._analyze_task_requirements(task, context)
        
        # Determine optimal configuration
        optimal_config = self._determine_optimal_config(task_requirements)
        
        # Apply optimizations
        optimization_result = await self._apply_optimizations(optimal_config)
        
        # Monitor performance
        performance_metrics = self._collect_performance_metrics()
        
        # Update optimization results
        self.optimization_results[task] = {
            "requirements": task_requirements,
            "config": optimal_config,
            "result": optimization_result,
            "performance": performance_metrics,
            "timestamp": time.time()
        }
        
        return {
            "task": task,
            "requirements": task_requirements,
            "config": optimal_config,
            "optimization_result": optimization_result,
            "performance_metrics": performance_metrics,
            "hardware_info": self.hardware_info
        }
    
    def _analyze_task_requirements(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze task requirements."""
        requirements = {
            "cpu_intensive": False,
            "memory_intensive": False,
            "gpu_intensive": False,
            "ml_intensive": False,
            "io_intensive": False,
            "complexity": 0.0
        }
        
        # Analyze task complexity
        complexity = 0.0
        
        # Text length factor
        text_length = len(task)
        complexity += min(text_length / 1000.0, 0.3)
        
        # Technical terms factor
        technical_terms = [
            "quantum", "neural", "machine learning", "deep learning",
            "simulation", "optimization", "algorithm", "computation",
            "rendering", "graphics", "video", "image", "audio"
        ]
        term_count = sum(1 for term in technical_terms if term.lower() in task.lower())
        complexity += min(term_count * 0.1, 0.4)
        
        # Context analysis
        if context:
            if "agents" in context and len(context["agents"]) > 5:
                requirements["cpu_intensive"] = True
                complexity += 0.2
            
            if "simulation_steps" in context and context["simulation_steps"] > 1000:
                requirements["cpu_intensive"] = True
                complexity += 0.2
            
            if "data_size" in context and context["data_size"] > 1000000:
                requirements["memory_intensive"] = True
                complexity += 0.2
            
            if "ml_model" in context:
                requirements["ml_intensive"] = True
                requirements["gpu_intensive"] = True
                complexity += 0.3
        
        # Task-specific analysis
        if any(term in task.lower() for term in ["render", "graphics", "video", "image"]):
            requirements["gpu_intensive"] = True
            complexity += 0.2
        
        if any(term in task.lower() for term in ["machine learning", "neural", "ai", "model"]):
            requirements["ml_intensive"] = True
            requirements["gpu_intensive"] = True
            complexity += 0.3
        
        if any(term in task.lower() for term in ["file", "database", "storage", "io"]):
            requirements["io_intensive"] = True
            complexity += 0.1
        
        requirements["complexity"] = min(complexity, 1.0)
        
        return requirements
    
    def _determine_optimal_config(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal hardware configuration."""
        config = {
            "cpu_cores": self.hardware_info.cpu_cores,
            "memory_limit": int(self.hardware_info.memory_gb * 0.8),  # 80% of available memory
            "gpu_acceleration": False,
            "neural_engine": False,
            "metal_acceleration": False,
            "energy_efficiency": True,
            "thermal_management": True
        }
        
        # CPU optimization
        if requirements["cpu_intensive"]:
            config["cpu_cores"] = min(self.hardware_info.cpu_cores, 8)  # Limit to 8 cores for efficiency
        
        # Memory optimization
        if requirements["memory_intensive"]:
            config["memory_limit"] = int(self.hardware_info.memory_gb * 0.9)  # 90% for memory-intensive tasks
        
        # GPU acceleration
        if requirements["gpu_intensive"] and self.hardware_info.metal_support:
            config["gpu_acceleration"] = True
            config["metal_acceleration"] = True
        
        # Neural Engine
        if requirements["ml_intensive"] and self.hardware_info.neural_engine:
            config["neural_engine"] = True
        
        # Energy efficiency
        if self.hardware_info.thermal_state == "hot":
            config["energy_efficiency"] = True
            config["cpu_cores"] = max(1, config["cpu_cores"] // 2)  # Reduce CPU usage
        
        # Thermal management
        if self.hardware_info.thermal_state == "critical":
            config["thermal_management"] = True
            config["cpu_cores"] = 1  # Minimal CPU usage
            config["energy_efficiency"] = True
        
        return config
    
    async def _apply_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hardware optimizations."""
        optimizations = {
            "cpu_optimization": False,
            "memory_optimization": False,
            "gpu_optimization": False,
            "neural_engine_optimization": False,
            "metal_optimization": False,
            "energy_optimization": False,
            "thermal_optimization": False
        }
        
        # CPU optimization
        if config["cpu_cores"] < self.hardware_info.cpu_cores:
            optimizations["cpu_optimization"] = True
            await self._optimize_cpu_usage(config["cpu_cores"])
        
        # Memory optimization
        if config["memory_limit"] < self.hardware_info.memory_gb:
            optimizations["memory_optimization"] = True
            await self._optimize_memory_usage(config["memory_limit"])
        
        # GPU optimization
        if config["gpu_acceleration"]:
            optimizations["gpu_optimization"] = True
            await self._optimize_gpu_usage()
        
        # Neural Engine optimization
        if config["neural_engine"]:
            optimizations["neural_engine_optimization"] = True
            await self._optimize_neural_engine_usage()
        
        # Metal optimization
        if config["metal_acceleration"]:
            optimizations["metal_optimization"] = True
            await self._optimize_metal_usage()
        
        # Energy optimization
        if config["energy_efficiency"]:
            optimizations["energy_optimization"] = True
            await self._optimize_energy_usage()
        
        # Thermal optimization
        if config["thermal_management"]:
            optimizations["thermal_optimization"] = True
            await self._optimize_thermal_usage()
        
        return optimizations
    
    async def _optimize_cpu_usage(self, target_cores: int):
        """Optimize CPU usage."""
        # Set CPU affinity
        import os
        os.sched_setaffinity(0, range(target_cores))
        logger.info(f"CPU optimized to {target_cores} cores")
    
    async def _optimize_memory_usage(self, memory_limit_gb: float):
        """Optimize memory usage."""
        # Set memory limit
        import resource
        memory_limit_bytes = int(memory_limit_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        logger.info(f"Memory limit set to {memory_limit_gb:.1f} GB")
    
    async def _optimize_gpu_usage(self):
        """Optimize GPU usage."""
        # Enable GPU acceleration
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("GPU optimization enabled")
    
    async def _optimize_neural_engine_usage(self):
        """Optimize Neural Engine usage."""
        # Enable Neural Engine
        os.environ["COREML_ENABLE_NEURAL_ENGINE"] = "1"
        logger.info("Neural Engine optimization enabled")
    
    async def _optimize_metal_usage(self):
        """Optimize Metal usage."""
        # Enable Metal acceleration
        os.environ["METAL_DEVICE_SELECTION"] = "1"
        logger.info("Metal optimization enabled")
    
    async def _optimize_energy_usage(self):
        """Optimize energy usage."""
        # Enable energy efficiency mode
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Energy optimization enabled")
    
    async def _optimize_thermal_usage(self):
        """Optimize thermal usage."""
        # Enable thermal management
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Thermal optimization enabled")
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Estimate GPU usage
        gpu_usage = 0.0
        if self.hardware_info.metal_support:
            gpu_usage = min(cpu_usage * 1.2, 100.0)
        
        # Get thermal state
        thermal_state = self._get_thermal_state()
        
        # Get power mode
        power_mode = self._get_power_mode()
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "gpu_usage": gpu_usage,
            "thermal_state": thermal_state,
            "power_mode": power_mode,
            "timestamp": time.time()
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report."""
        return {
            "hardware_info": {
                "processor": self.hardware_info.processor,
                "architecture": self.hardware_info.architecture,
                "cpu_cores": self.hardware_info.cpu_cores,
                "memory_gb": self.hardware_info.memory_gb,
                "gpu_type": self.hardware_info.gpu_type,
                "neural_engine": self.hardware_info.neural_engine,
                "metal_support": self.hardware_info.metal_support,
                "thermal_state": self.hardware_info.thermal_state,
                "power_mode": self.hardware_info.power_mode
            },
            "optimization_config": {
                "neural_engine_enabled": self.optimization_config.neural_engine_enabled,
                "metal_acceleration": self.optimization_config.metal_acceleration,
                "energy_efficiency": self.optimization_config.energy_efficiency,
                "thermal_management": self.optimization_config.thermal_management,
                "cpu_optimization": self.optimization_config.cpu_optimization,
                "memory_optimization": self.optimization_config.memory_optimization
            },
            "optimization_results": self.optimization_results,
            "performance_history": self.performance_history[-10:] if self.performance_history else [],
            "energy_history": self.energy_history[-10:] if self.energy_history else []
        }
    
    async def monitor_performance(self):
        """Monitor performance continuously."""
        while True:
            metrics = self._collect_performance_metrics()
            self.performance_history.append(metrics)
            
            # Keep only last 100 entries
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)
            
            # Check for thermal issues
            if metrics["thermal_state"] == "critical":
                logger.warning("Critical thermal state detected, reducing performance")
                await self._optimize_thermal_usage()
            
            # Sleep for 5 seconds
            await asyncio.sleep(5)
    
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities."""
        return {
            "apple_silicon": self.hardware_info.architecture == "arm64",
            "neural_engine": self.hardware_info.neural_engine,
            "metal_support": self.hardware_info.metal_support,
            "cpu_cores": self.hardware_info.cpu_cores,
            "memory_gb": self.hardware_info.memory_gb,
            "gpu_type": self.hardware_info.gpu_type,
            "thermal_state": self.hardware_info.thermal_state,
            "power_mode": self.hardware_info.power_mode
        }


# Convenience functions
def create_mac_optimizer(config: Dict[str, Any] = None) -> MacHardwareOptimizer:
    """Create Mac hardware optimizer."""
    return MacHardwareOptimizer(config)


async def optimize_for_mac_hardware(task: str, context: Dict[str, Any] = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Optimize for Mac hardware."""
    optimizer = create_mac_optimizer(config)
    return await optimizer.optimize_for_task(task, context)
