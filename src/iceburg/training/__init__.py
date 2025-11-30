"""
ICEBURG Training Module
"""

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

__all__ = [
    "SpecializedModelTuner",
    "AgentModelConfig",
    "FineTuningResult",
]

if ELITE_TUNER_AVAILABLE:
    __all__.extend(["EliteModelTuner", "TuningResult"])

if ELITE_COLLECTOR_AVAILABLE:
    __all__.append("EliteTrainingCollector")

