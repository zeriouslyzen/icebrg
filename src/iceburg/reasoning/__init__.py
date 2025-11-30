"""
ICEBURG Reasoning Module
Hybrid COCONUT + ICEBURG Reasoning System

This module provides the hybrid reasoning capabilities that combine:
- COCONUT-style latent reasoning for silent processing
- ICEBURG's emergence detection and deliberation pauses
- Enhanced dual-layer emergence detection
- Seamless integration with existing ICEBURG agents

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

from .coconut_latent_reasoning import (
    COCONUTLatentReasoning,
    LatentReasoningStep,
    LatentReasoningResult
)

from .hybrid_reasoning_engine import (
    HybridReasoningEngine,
    HybridReasoningStep,
    HybridReasoningResult
)

__all__ = [
    'COCONUTLatentReasoning',
    'LatentReasoningStep', 
    'LatentReasoningResult',
    'HybridReasoningEngine',
    'HybridReasoningStep',
    'HybridReasoningResult'
]
