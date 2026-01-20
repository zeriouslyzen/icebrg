"""
Matrix Crawler Extension - Self-sovereign data gathering for ICEBURG.
Autonomously crawls free public data sources to build a comprehensive
knowledge graph of entities, relationships, and power structures.
"""

from .crawler_engine import CrawlerEngine, CrawlerStatus
from .scheduler import CrawlerScheduler
from .graph_storage import MatrixGraph
from .entity_extractor import EntityExtractor
from .entity_resolver import EntityResolver

__all__ = [
    "CrawlerEngine",
    "CrawlerStatus", 
    "CrawlerScheduler",
    "MatrixGraph",
    "EntityExtractor",
    "EntityResolver",
]
