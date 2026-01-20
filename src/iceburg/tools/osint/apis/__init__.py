"""
OSINT API Clients
External API integrations for open-source intelligence gathering.
"""

from .opencorporates import OpenCorporatesClient
from .opensecrets import OpenSecretsClient
from .wikidata import WikidataClient

__all__ = [
    "OpenCorporatesClient",
    "OpenSecretsClient",
    "WikidataClient",
]
