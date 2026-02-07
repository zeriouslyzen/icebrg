"""
ICEBURG Dossier Protocol
Deep investigative research and intelligence dossier generation.
"""

from .gatherer import GathererAgent, gather_intelligence
from .decoder import DecoderAgent, decode_symbols
from .mapper import MapperAgent, map_network
from .synthesizer import DossierSynthesizer, generate_dossier
from .recursive_pipeline import RecursiveDossierPipeline, generate_deep_dossier, DeepDossier
from .silence_mention_tracker import track_silence_mentions, entities_silent_in_corpus
from .corpus_ingest import load_corpus_from_path, ingest_corpus_for_dossier, load_document

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
    "track_silence_mentions",
    "entities_silent_in_corpus",
    "load_corpus_from_path",
    "ingest_corpus_for_dossier",
    "load_document",
]

