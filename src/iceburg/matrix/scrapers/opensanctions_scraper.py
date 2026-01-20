"""
OpenSanctions Scraper - Sanctioned individuals, PEPs, and wanted persons.
Downloads bulk data from OpenSanctions.org.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_scraper import BaseScraper, ScraperResult

logger = logging.getLogger(__name__)


class OpenSanctionsScraper(BaseScraper):
    """
    Scraper for OpenSanctions data.
    
    Downloads:
    - Sanctioned individuals and entities
    - Politically Exposed Persons (PEPs)
    - Wanted persons (Interpol, FBI, etc.)
    """
    
    # OpenSanctions API
    BASE_URL = "https://data.opensanctions.org"
    
    # Available datasets - using latest format
    DATASETS = {
        "sanctions": {
            "url": f"{BASE_URL}/datasets/latest/default/entities.ftm.json",
            "description": "Global sanctions consolidated",
        },
        "peps": {
            "url": f"{BASE_URL}/datasets/latest/peps/entities.ftm.json",
            "description": "Politically Exposed Persons",
        },
        "crime": {
            "url": f"{BASE_URL}/datasets/latest/crime/entities.ftm.json",
            "description": "Wanted criminals",
        },
    }
    
    @property
    def source_name(self) -> str:
        return "OpenSanctions"
    
    @property
    def source_url(self) -> str:
        return "https://opensanctions.org/"
    
    async def _scrape(self, options: Dict[str, Any]) -> ScraperResult:
        """
        Scrape OpenSanctions data.
        
        Options:
            datasets: List of datasets to download
            limit: Maximum records to process
        """
        datasets = options.get("datasets", ["sanctions"])
        limit = options.get("limit", None)
        
        result = ScraperResult(success=True)
        result.metadata["datasets"] = datasets
        
        total_entities = 0
        
        for dataset in datasets:
            if dataset not in self.DATASETS:
                logger.warning(f"Unknown dataset: {dataset}")
                continue
            
            try:
                entities = await self._download_and_process(dataset, limit)
                total_entities += entities
                result.documents_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {dataset}: {e}")
                result.errors.append(f"{dataset}: {str(e)}")
        
        result.entities_count = total_entities
        
        return result
    
    async def _download_and_process(
        self,
        dataset: str,
        limit: Optional[int]
    ) -> int:
        """Download and process a dataset."""
        info = self.DATASETS[dataset]
        url = info["url"]
        
        logger.info(f"📥 Downloading OpenSanctions {dataset}...")
        
        # Download JSON file
        filename = f"opensanctions_{dataset}.json"
        file_path = await self._download_file(url, filename)
        
        # Process entities
        entities = await self._process_entities(file_path, dataset, limit)
        
        return entities
    
    async def _process_entities(
        self,
        file_path: Path,
        dataset: str,
        limit: Optional[int]
    ) -> int:
        """Process entities from JSON file."""
        entities = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # Extract entity info
                    name = None
                    entity_type = data.get("schema", "unknown").lower()
                    
                    # Get name from properties
                    props = data.get("properties", {})
                    names = props.get("name", [])
                    if names:
                        name = names[0] if isinstance(names, list) else names
                    
                    if not name:
                        continue
                    
                    # Normalize entity type
                    if entity_type in ("person", "legalentity", "company", "organization"):
                        pass
                    else:
                        entity_type = "organization"
                    
                    # Build properties
                    properties = {
                        "opensanctions_id": data.get("id", ""),
                        "dataset": dataset,
                        "schema": data.get("schema", ""),
                        "countries": props.get("country", []),
                        "birth_date": props.get("birthDate", [""])[0] if props.get("birthDate") else "",
                        "nationality": props.get("nationality", []),
                        "position": props.get("position", []),
                        "topics": data.get("topics", []),
                        "sanctions": props.get("sanctions", []),
                        "aliases": props.get("alias", []),
                    }
                    
                    # Save entity
                    await self._save_entity(entity_type, name, properties)
                    entities += 1
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Parse error: {e}")
                
                if i > 0 and i % 10000 == 0:
                    logger.info(f"Processed {i} entities from {dataset}")
                    self._report_progress(i / (limit or 100000), i, limit or 100000)
        
        return entities
    
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
            
            # Add aliases from data
            aliases = data.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            
            entity = Entity(
                entity_id=f"osanc_{data.get('opensanctions_id', name)}",
                name=name,
                entity_type=entity_type,
                aliases=aliases,
                properties=data,
                sources=["opensanctions"],
            )
            graph.add_entity(entity)
            
        except Exception as e:
            logger.debug(f"Could not save entity: {e}")
