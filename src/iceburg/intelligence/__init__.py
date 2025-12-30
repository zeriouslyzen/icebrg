"""Intelligence module for V2 Advanced Prediction Market System"""

from .multi_source_aggregator import (
    IntelligenceSource,
    SignalPriority,
    IntelligenceSignal,
    CorrelatedIntelligence,
    MultiSourceIntelligenceAggregator,
    get_intelligence_aggregator
)
from .quant_signal_processor import (
    SignalType,
    AlphaSignal,
    StatArbPair,
    QuantitativeSignalProcessor,
    get_quant_processor
)
from .backtest_engine import (
    Trade,
    BacktestResults,
    BacktestEngine
)

__all__ = [
    # Intelligence aggregation
    'IntelligenceSource',
    'SignalPriority',
    'IntelligenceSignal',
    'CorrelatedIntelligence',
    'MultiSourceIntelligenceAggregator',
    'get_intelligence_aggregator',
    # Quant signals
    'SignalType',
    'AlphaSignal',
    'StatArbPair',
    'QuantitativeSignalProcessor',
    'get_quant_processor',
    # Backtesting
    'Trade',
    'BacktestResults',
    'BacktestEngine'
]
