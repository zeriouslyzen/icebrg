"""
ICEBURG Training Module
=======================

Internal fine-tuning framework for creating ICEBURG-specialized LLMs.

Components:
- ICEBURGFineTuner: Main orchestrator for fine-tuning
- M4Optimizer: Apple M4 Mac optimization
- TruthFilter: Quality filtering using InstantTruthSystem
- EmergenceProcessor: Emergence-aware curriculum learning
- ModelExporter: Multi-format model export (Ollama, HuggingFace, GGUF)
"""

# Existing imports
try:
    from .elite_model_tuner import EliteModelTuner, TuningResult
    ELITE_TUNER_AVAILABLE = True
except ImportError:
    ELITE_TUNER_AVAILABLE = False
    EliteModelTuner = None
    TuningResult = None

from .specialized_model_tuner import SpecializedModelTuner, AgentModelConfig, FineTuningResult

try:
    from .elite_training_collector import EliteTrainingCollector
    ELITE_COLLECTOR_AVAILABLE = True
except ImportError:
    ELITE_COLLECTOR_AVAILABLE = False
    EliteTrainingCollector = None

# New ICEBURG Internal Fine-Tuning Framework imports
try:
    from .m4_optimizer import (
        M4Optimizer,
        M4Config,
        DeviceType,
        get_m4_optimizer
    )
    M4_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    M4_OPTIMIZER_AVAILABLE = False
    M4Optimizer = None
    M4Config = None
    DeviceType = None
    get_m4_optimizer = None

try:
    from .truth_filter import (
        TruthFilter,
        FilteredDatapoint,
        FilterStats,
        TruthCategory,
        filter_training_data
    )
    TRUTH_FILTER_AVAILABLE = True
except ImportError:
    TRUTH_FILTER_AVAILABLE = False
    TruthFilter = None
    FilteredDatapoint = None
    FilterStats = None
    TruthCategory = None
    filter_training_data = None

try:
    from .emergence_processor import (
        EmergenceProcessor,
        EmergenceDatapoint,
        EmergenceStats,
        EmergenceCategory,
        process_for_emergence
    )
    EMERGENCE_PROCESSOR_AVAILABLE = True
except ImportError:
    EMERGENCE_PROCESSOR_AVAILABLE = False
    EmergenceProcessor = None
    EmergenceDatapoint = None
    EmergenceStats = None
    EmergenceCategory = None
    process_for_emergence = None

try:
    from .model_exporter import (
        ModelExporter,
        ExportFormat,
        ExportResult,
        QuantizationType,
        export_model
    )
    MODEL_EXPORTER_AVAILABLE = True
except ImportError:
    MODEL_EXPORTER_AVAILABLE = False
    ModelExporter = None
    ExportFormat = None
    ExportResult = None
    QuantizationType = None
    export_model = None

try:
    from .iceburg_fine_tuner import (
        ICEBURGFineTuner,
        TrainingConfig,
        TrainingResult,
        TrainingStatus,
        ModelType,
        train_iceburg_model,
        get_recommended_config
    )
    ICEBURG_FINETUNER_AVAILABLE = True
except ImportError:
    ICEBURG_FINETUNER_AVAILABLE = False
    ICEBURGFineTuner = None
    TrainingConfig = None
    TrainingResult = None
    TrainingStatus = None
    ModelType = None
    train_iceburg_model = None
    get_recommended_config = None

# Build __all__
__all__ = [
    # Existing exports
    "SpecializedModelTuner",
    "AgentModelConfig",
    "FineTuningResult",
]

if ELITE_TUNER_AVAILABLE:
    __all__.extend(["EliteModelTuner", "TuningResult"])

if ELITE_COLLECTOR_AVAILABLE:
    __all__.append("EliteTrainingCollector")

# New ICEBURG Fine-Tuning Framework exports
if M4_OPTIMIZER_AVAILABLE:
    __all__.extend([
        "M4Optimizer",
        "M4Config",
        "DeviceType",
        "get_m4_optimizer"
    ])

if TRUTH_FILTER_AVAILABLE:
    __all__.extend([
        "TruthFilter",
        "FilteredDatapoint",
        "FilterStats",
        "TruthCategory",
        "filter_training_data"
    ])

if EMERGENCE_PROCESSOR_AVAILABLE:
    __all__.extend([
        "EmergenceProcessor",
        "EmergenceDatapoint",
        "EmergenceStats",
        "EmergenceCategory",
        "process_for_emergence"
    ])

if MODEL_EXPORTER_AVAILABLE:
    __all__.extend([
        "ModelExporter",
        "ExportFormat",
        "ExportResult",
        "QuantizationType",
        "export_model"
    ])

if ICEBURG_FINETUNER_AVAILABLE:
    __all__.extend([
        "ICEBURGFineTuner",
        "TrainingConfig",
        "TrainingResult",
        "TrainingStatus",
        "ModelType",
        "train_iceburg_model",
        "get_recommended_config"
    ])

# Model Registry
try:
    from .model_registry import (
        ICEBURGModelRegistry,
        RegisteredModel,
        ModelStatus,
        get_model_registry,
        get_iceburg_model_for_agent
    )
    MODEL_REGISTRY_AVAILABLE = True
    __all__.extend([
        "ICEBURGModelRegistry",
        "RegisteredModel",
        "ModelStatus",
        "get_model_registry",
        "get_iceburg_model_for_agent"
    ])
except ImportError:
    MODEL_REGISTRY_AVAILABLE = False
    ICEBURGModelRegistry = None
    RegisteredModel = None
    ModelStatus = None
    get_model_registry = None
    get_iceburg_model_for_agent = None


def get_available_features() -> dict:
    """
    Get dictionary of available training features.
    
    Returns:
        Dictionary with feature names and availability status
    """
    return {
        "elite_tuner": ELITE_TUNER_AVAILABLE,
        "elite_collector": ELITE_COLLECTOR_AVAILABLE,
        "m4_optimizer": M4_OPTIMIZER_AVAILABLE,
        "truth_filter": TRUTH_FILTER_AVAILABLE,
        "emergence_processor": EMERGENCE_PROCESSOR_AVAILABLE,
        "model_exporter": MODEL_EXPORTER_AVAILABLE,
        "iceburg_finetuner": ICEBURG_FINETUNER_AVAILABLE,
        "model_registry": MODEL_REGISTRY_AVAILABLE,
    }
