"""
Deep Entity Flow - OSINT gathering for a single entity (company or person).
Used when depth=deep and query is entity-like; can be called from dossier route
or Gatherer to pre-populate entity/relationship data for Colossus/Matrix.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def gather_entity_osint(entity_name: str, limit_companies: int = 10) -> Dict[str, Any]:
    """
    Gather OSINT for a single entity name (company or person).
    Calls OpenCorporates (companies search); optional OpenSecrets when candidate ID available.
    Returns structured data that can be merged into dossier context or Colossus/Matrix.

    Args:
        entity_name: Company or person name to search.
        limit_companies: Max companies to return from OpenCorporates.

    Returns:
        Dict with keys:
          - companies: list of company dicts (name, jurisdiction, status, incorporation_date, registry_url, etc.)
          - raw_sources: list of dicts suitable for IntelligenceSource or mapper (title, content, url, source_type)
          - entities_found: list of entity dicts for mapper (name, type, context)
    """
    result: Dict[str, Any] = {
        "companies": [],
        "raw_sources": [],
        "entities_found": [],
    }

    try:
        from .apis.opencorporates import OpenCorporatesClient
        client = OpenCorporatesClient()
        companies = client.search_companies(entity_name.strip(), limit=limit_companies)

        for co in companies:
            co_dict = {
                "name": co.name,
                "company_number": co.company_number,
                "jurisdiction_code": co.jurisdiction_code,
                "incorporation_date": co.incorporation_date,
                "company_type": getattr(co, "company_type", None),
                "registry_url": getattr(co, "registry_url", None),
                "status": co.status,
            }
            result["companies"].append(co_dict)

            content = f"Company: {co.name}. Jurisdiction: {co.jurisdiction_code}. Status: {co.status or 'unknown'}. Incorporation: {co.incorporation_date or 'unknown'}."
            url = getattr(co, "registry_url", None) or (getattr(co, "metadata", None) or {}).get("opencorporates_url") or "#"
            result["raw_sources"].append({
                "url": url,
                "title": co.name,
                "content": content,
                "source_type": "deep",
                "credibility_score": 0.8,
                "metadata": {"source_api": "opencorporates", "company_number": co.company_number},
            })
            result["entities_found"].append({
                "name": co.name,
                "type": "organization",
                "mentions": 1,
                "context": f"Jurisdiction: {co.jurisdiction_code}, Status: {co.status}",
            })

        if companies:
            logger.info(f"Deep entity OSINT: found {len(companies)} companies for '{entity_name}'")
    except Exception as e:
        logger.debug(f"Deep entity OSINT (OpenCorporates) failed: {e}")

    return result
