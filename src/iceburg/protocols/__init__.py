"""
ICEBURG Protocols Package
Contains specialized research protocols beyond basic chat/research.
"""

# Dossier Protocol - Deep investigative research
from .dossier import (
    GathererAgent,
    gather_intelligence,
    DecoderAgent,
    decode_symbols,
    MapperAgent,
    map_network,
    DossierSynthesizer,
    generate_dossier,
)

__all__ = [
    # Dossier Protocol
    "GathererAgent",
    "gather_intelligence",
    "DecoderAgent",
    "decode_symbols",
    "MapperAgent", 
    "map_network",
    "DossierSynthesizer",
    "generate_dossier",
]
