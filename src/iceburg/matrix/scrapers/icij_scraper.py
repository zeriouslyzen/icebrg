"""
ICIJ Offshore Leaks Scraper - Panama Papers, Paradise Papers, Pandora Papers.
Downloads and imports the ICIJ Offshore Leaks Database.
"""

import asyncio
import csv
import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_scraper import BaseScraper, ScraperResult

logger = logging.getLogger(__name__)


class ICIJScraper(BaseScraper):
    """
    Scraper for ICIJ Offshore Leaks Database.
    
    Downloads:
    - Entities (officers, intermediaries, addresses)
    - Relationships between offshore entities
    - Data from Panama, Paradise, Pandora, and Offshore Leaks investigations
    """
    
    # ICIJ download URLs
    BASE_URL = "https://offshoreleaks.icij.org"
    DOWNLOAD_URL = "https://offshoreleaks-data.icij.org/offshoreleaks/csv"
    
    # File names in the download
    FILES = {
        "nodes_entity": "nodes-entities.csv",
        "nodes_officer": "nodes-officers.csv",
        "nodes_intermediary": "nodes-intermediaries.csv",
        "nodes_address": "nodes-addresses.csv",
        "relationships": "relationships.csv",
    }
    
    @property
    def source_name(self) -> str:
        return "ICIJ Offshore Leaks"
    
    @property
    def source_url(self) -> str:
        return "https://offshoreleaks.icij.org/"
    
    async def _scrape(self, options: Dict[str, Any]) -> ScraperResult:
        """
        Scrape ICIJ Offshore Leaks data.
        
        Options:
            limit: Maximum records per file (for testing)
            datasets: List of datasets to import ("panama", "paradise", "pandora", "offshore")
        """
        limit = options.get("limit", None)
        
        result = ScraperResult(success=True)
        
        # Download the main CSV package
        logger.info("📥 Downloading ICIJ Offshore Leaks database...")
        
        try:
            # Try direct download of CSV files
            zip_path = await self._download_file(
                f"{self.DOWNLOAD_URL}/full-oldb.zip",
                "icij_offshore_leaks.zip"
            )
            
            # Process the downloaded data
            entities, relationships = await self._process_zip(zip_path, limit)
            
            result.entities_count = entities
            result.relationships_count = relationships
            result.documents_count = len(self.FILES)
            
        except Exception as e:
            logger.error(f"ICIJ download failed: {e}")
            result.errors.append(str(e))
            
            # Try alternative: scrape the search API
            logger.info("Trying alternative API scrape...")
            entities, relationships = await self._scrape_api(limit)
            result.entities_count = entities
            result.relationships_count = relationships
        
        return result
    
    async def _process_zip(
        self,
        zip_path: Path,
        limit: Optional[int]
    ) -> tuple[int, int]:
        """Process the ICIJ zip file."""
        total_entities = 0
        total_relationships = 0
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List what's in the zip
            logger.info(f"ZIP contents: {zf.namelist()}")
            
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    continue
                
                logger.info(f"Processing: {name}")
                
                with zf.open(name) as f:
                    data = f.read().decode("utf-8", errors="replace")
                    
                    if "relationships" in name.lower():
                        relationships = await self._process_relationships(data, limit)
                        total_relationships += relationships
                    else:
                        entities = await self._process_entities(data, name, limit)
                        total_entities += entities
        
        return total_entities, total_relationships
    
    async def _process_entities(
        self,
        data: str,
        filename: str,
        limit: Optional[int]
    ) -> int:
        """Process entity CSV data."""
        entities = 0
        
        # Determine entity type from filename
        if "officer" in filename.lower():
            entity_type = "person"
        elif "intermediary" in filename.lower():
            entity_type = "organization"
        elif "address" in filename.lower():
            entity_type = "location"
        else:
            entity_type = "company"
        
        reader = csv.DictReader(io.StringIO(data))
        
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            
            try:
                name = row.get("name", row.get("node_id", ""))
                if not name:
                    continue
                
                # Extract properties
                properties = {
                    "icij_id": row.get("node_id", ""),
                    "jurisdiction": row.get("jurisdiction", row.get("country_codes", "")),
                    "source": row.get("sourceID", row.get("source", "")),
                    "countries": row.get("countries", row.get("country_codes", "")),
                    "incorporation_date": row.get("incorporation_date", ""),
                    "inactivation_date": row.get("inactivation_date", ""),
                    "status": row.get("status", ""),
                }
                
                await self._save_entity(entity_type, name, properties)
                entities += 1
                
            except Exception as e:
                logger.debug(f"Parse error: {e}")
            
            if i > 0 and i % 10000 == 0:
                logger.info(f"Processed {i} entities from {filename}")
        
        return entities
    
    async def _process_relationships(
        self,
        data: str,
        limit: Optional[int]
    ) -> int:
        """Process relationship CSV data."""
        relationships = 0
        
        reader = csv.DictReader(io.StringIO(data))
        
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            
            try:
                source_id = row.get("node_id_start", row.get("START_ID", ""))
                target_id = row.get("node_id_end", row.get("END_ID", ""))
                rel_type = row.get("rel_type", row.get("TYPE", "connected_to"))
                
                if not source_id or not target_id:
                    continue
                
                # Normalize relationship type
                rel_type = rel_type.lower().replace(" ", "_")
                if rel_type in ("officer_of", "shareholder_of", "director_of"):
                    rel_type = "officer_of"
                elif rel_type in ("registered_address", "same_address_as"):
                    rel_type = "located_at"
                elif rel_type in ("intermediary_of", "similar_name_and_address_as"):
                    rel_type = "connected_to"
                
                properties = {
                    "source": row.get("sourceID", "icij"),
                    "link_type": row.get("link", ""),
                }
                
                await self._save_relationship(source_id, target_id, rel_type, properties)
                relationships += 1
                
            except Exception as e:
                logger.debug(f"Parse error: {e}")
        
        return relationships
    
    async def _scrape_api(self, limit: Optional[int]) -> tuple[int, int]:
        """
        Alternative: Scrape ICIJ search API.
        Used when bulk download is unavailable.
        """
        entities = 0
        relationships = 0
        
        # Search API endpoint
        api_url = f"{self.BASE_URL}/api/search"
        
        # Sample searches to seed the database
        search_terms = [
            "offshore", "trust", "foundation", "holdings",
            "investment", "management", "group", "capital",
        ]
        
        for term in search_terms:
            try:
                response = await self._fetch(
                    api_url,
                    params={"q": term, "size": min(limit or 100, 100)}
                )
                data = response.json()
                
                for hit in data.get("hits", {}).get("hits", []):
                    source = hit.get("_source", {})
                    name = source.get("name", "")
                    node_type = source.get("node_type", "entity")
                    
                    if not name:
                        continue
                    
                    entity_type = "company" if node_type == "Entity" else "person"
                    
                    properties = {
                        "icij_id": source.get("node_id", ""),
                        "jurisdiction": source.get("jurisdiction", ""),
                        "source": source.get("sourceID", ""),
                    }
                    
                    await self._save_entity(entity_type, name, properties)
                    entities += 1
                    
                    if limit and entities >= limit:
                        break
                        
            except Exception as e:
                logger.warning(f"API search failed for '{term}': {e}")
        
        return entities, relationships
    
    async def _save_entity(
        self,
        entity_type: str,
        name: str,
        data: Dict[str, Any]
    ):
        """Save an extracted entity."""
        try:
            from ..graph_storage import MatrixGraph, Entity
            
            graph = MatrixGraph(self.data_dir.parent)
            entity = Entity(
                entity_id=f"icij_{data.get('icij_id', name)}",
                name=name,
                entity_type=entity_type,
                properties=data,
                sources=["icij_offshore_leaks"],
            )
            graph.add_entity(entity)
            
        except Exception as e:
            logger.debug(f"Could not save entity: {e}")
    
    async def _save_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        data: Dict[str, Any]
    ):
        """Save an extracted relationship."""
        try:
            from ..graph_storage import MatrixGraph, Relationship
            import hashlib
            
            graph = MatrixGraph(self.data_dir.parent)
            rel = Relationship(
                relationship_id=hashlib.md5(f"{source_id}_{target_id}_{rel_type}".encode()).hexdigest()[:12],
                source_id=f"icij_{source_id}",
                target_id=f"icij_{target_id}",
                relationship_type=rel_type,
                properties=data,
                sources=["icij_offshore_leaks"],
            )
            graph.add_relationship(rel)
            
        except Exception as e:
            logger.debug(f"Could not save relationship: {e}")
