"""
COLOSSUS - Enterprise Intelligence Platform

Internal extension of ICEBURG for high-value intelligence operations.
Palantir-class capabilities with self-sovereign, local-first architecture.

Capabilities:
- 50M+ entity knowledge graph
- Real-time sanction monitoring
- AI-powered entity extraction & resolution
- Graph-based network analysis
- Natural language investigation

Target Users:
- Enterprise due diligence
- M&A research
- Political & regulatory mapping
- Threat actor analysis
"""

from .core.graph import ColossusGraph
from .core.search import ColossusSearch
from .core.storage import ColossusStorage
from .intelligence.extraction import EntityExtractor
from .intelligence.resolution import EntityResolver
from .intelligence.risk import RiskScorer

__version__ = "0.1.0"
__codename__ = "COLOSSUS"

__all__ = [
    "ColossusGraph",
    "ColossusSearch", 
    "ColossusStorage",
    "EntityExtractor",
    "EntityResolver",
    "RiskScorer",
]
