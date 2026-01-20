"""
ICEBURG Dossier Protocol
Deep investigative research and intelligence dossier generation.
"""

from .gatherer import GathererAgent, gather_intelligence
from .decoder import DecoderAgent, decode_symbols
from .mapper import MapperAgent, map_network
from .synthesizer import DossierSynthesizer, generate_dossier
from .recursive_pipeline import RecursiveDossierPipeline, generate_deep_dossier, DeepDossier

__all__ = [
    "GathererAgent",
    "gather_intelligence",
    "DecoderAgent", 
    "decode_symbols",
    "MapperAgent",
    "map_network",
    "DossierSynthesizer",
    "generate_dossier",
    "RecursiveDossierPipeline",
    "generate_deep_dossier",
    "DeepDossier",
]

