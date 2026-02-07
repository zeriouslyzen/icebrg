"""
OSINT Tools Package
Open-source intelligence gathering tools for the Dossier Protocol.
"""

from .entity_extractor import EntityExtractor, extract_entities
from .network_graph import NetworkGraphBuilder, build_network_graph
from .source_scorer import SourceScorer, score_source
from .apis import OpenCorporatesClient, OpenSecretsClient, WikidataClient
from .deep_entity import gather_entity_osint

__all__ = [
    "EntityExtractor",
    "extract_entities",
    "NetworkGraphBuilder",
    "build_network_graph",
    "SourceScorer",
    "score_source",
    "OpenCorporatesClient",
    "OpenSecretsClient",
    "WikidataClient",
    "gather_entity_osint",
]
