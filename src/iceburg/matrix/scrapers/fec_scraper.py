"""
FEC Scraper - Federal Election Commission campaign finance data.
Downloads bulk CSV files from FEC.gov with contribution and expenditure data.
"""

import asyncio
import csv
import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_scraper import BaseScraper, ScraperResult

logger = logging.getLogger(__name__)


class FECScraper(BaseScraper):
    """
    Scraper for FEC (Federal Election Commission) data.
    
    Downloads:
    - Individual contributions
    - Committee contributions
    - Candidate data
    - Committee data
    """
    
    # FEC bulk data URLs
    BASE_URL = "https://www.fec.gov/files/bulk-downloads"
    
    # Data file types
    DATA_TYPES = {
        "contributions": {
            "pattern": "{cycle}/indiv{cycle}.zip",
            "description": "Individual contributions to committees",
        },
        "committee_contributions": {
            "pattern": "{cycle}/pas2{cycle}.zip",
            "description": "Contributions from committees to candidates",
        },
        "candidates": {
            "pattern": "{cycle}/cn{cycle}.zip",
            "description": "Candidate master file",
        },
        "committees": {
            "pattern": "{cycle}/cm{cycle}.zip",
            "description": "Committee master file",
        },
    }
    
    # Current election cycle (2-year period ending in election year)
    DEFAULT_CYCLE = "2024"
    
    @property
    def source_name(self) -> str:
        return "FEC Campaign Finance"
    
    @property
    def source_url(self) -> str:
        return "https://www.fec.gov/data/"
    
    async def _scrape(self, options: Dict[str, Any]) -> ScraperResult:
        """
        Scrape FEC data.
        
        Options:
            cycle: Election cycle (e.g., "2024", "2022")
            data_types: List of data types to download
            limit: Maximum records to process (for testing)
        """
        cycle = options.get("cycle", self.DEFAULT_CYCLE)
        data_types = options.get("data_types", ["contributions", "candidates", "committees"])
        limit = options.get("limit", None)
        
        result = ScraperResult(success=True)
        result.metadata["cycle"] = cycle
        result.metadata["data_types"] = data_types
        
        total_entities = 0
        total_relationships = 0
        
        for data_type in data_types:
            if data_type not in self.DATA_TYPES:
                logger.warning(f"Unknown data type: {data_type}")
                continue
            
            try:
                entities, relationships = await self._download_and_process(
                    data_type, cycle, limit
                )
                total_entities += entities
                total_relationships += relationships
                result.documents_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {data_type}: {e}")
                result.errors.append(f"{data_type}: {str(e)}")
        
        result.entities_count = total_entities
        result.relationships_count = total_relationships
        
        return result
    
    async def _download_and_process(
        self,
        data_type: str,
        cycle: str,
        limit: Optional[int]
    ) -> tuple[int, int]:
        """Download and process a data file."""
        info = self.DATA_TYPES[data_type]
        pattern = info["pattern"].format(cycle=cycle)
        url = f"{self.BASE_URL}/{pattern}"
        filename = f"fec_{data_type}_{cycle}.zip"
        
        logger.info(f"📥 Downloading FEC {data_type} for cycle {cycle}")
        
        # Download zip file
        zip_path = await self._download_file(url, filename)
        
        # Extract and process
        entities = 0
        relationships = 0
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".txt"):
                    with zf.open(name) as f:
                        data = f.read().decode("utf-8", errors="replace")
                        e, r = await self._process_csv(data, data_type, limit)
                        entities += e
                        relationships += r
        
        return entities, relationships
    
    async def _process_csv(
        self,
        data: str,
        data_type: str,
        limit: Optional[int]
    ) -> tuple[int, int]:
        """Process CSV data and extract entities."""
        entities = 0
        relationships = 0
        
        # Parse based on data type
        if data_type == "contributions":
            entities, relationships = await self._process_contributions(data, limit)
        elif data_type == "candidates":
            entities = await self._process_candidates(data, limit)
        elif data_type == "committees":
            entities = await self._process_committees(data, limit)
        elif data_type == "committee_contributions":
            entities, relationships = await self._process_committee_contributions(data, limit)
        
        return entities, relationships
    
    async def _process_contributions(
        self,
        data: str,
        limit: Optional[int]
    ) -> tuple[int, int]:
        """Process individual contribution records."""
        # FEC contribution format (pipe-delimited)
        # CMTE_ID|AMNDT_IND|RPT_TP|TRANSACTION_PGI|...|NAME|CITY|STATE|ZIP_CODE|EMPLOYER|OCCUPATION|...
        
        entities = 0
        relationships = 0
        
        lines = data.strip().split("\n")
        for i, line in enumerate(lines):
            if limit and i >= limit:
                break
            
            fields = line.split("|")
            if len(fields) < 15:
                continue
            
            try:
                cmte_id = fields[0]
                name = fields[7] if len(fields) > 7 else ""
                employer = fields[11] if len(fields) > 11 else ""
                occupation = fields[12] if len(fields) > 12 else ""
                amount = fields[14] if len(fields) > 14 else "0"
                
                if name:
                    # Save contribution record
                    record = {
                        "committee_id": cmte_id,
                        "contributor_name": name,
                        "employer": employer,
                        "occupation": occupation,
                        "amount": amount,
                    }
                    await self._save_entity("contributor", name, record)
                    entities += 1
                    relationships += 1  # Contribution = relationship
                    
            except Exception as e:
                logger.debug(f"Parse error on line {i}: {e}")
            
            if i > 0 and i % 10000 == 0:
                self._report_progress(i / len(lines), i, len(lines))
        
        return entities, relationships
    
    async def _process_candidates(self, data: str, limit: Optional[int]) -> int:
        """Process candidate master records."""
        # CAND_ID|CAND_NAME|CAND_PTY_AFFILIATION|CAND_ELECTION_YR|CAND_OFFICE_ST|...
        
        entities = 0
        lines = data.strip().split("\n")
        
        for i, line in enumerate(lines):
            if limit and i >= limit:
                break
            
            fields = line.split("|")
            if len(fields) < 6:
                continue
            
            try:
                cand_id = fields[0]
                name = fields[1]
                party = fields[2] if len(fields) > 2 else ""
                year = fields[3] if len(fields) > 3 else ""
                state = fields[4] if len(fields) > 4 else ""
                office = fields[5] if len(fields) > 5 else ""
                
                record = {
                    "candidate_id": cand_id,
                    "name": name,
                    "party": party,
                    "election_year": year,
                    "state": state,
                    "office": office,
                }
                await self._save_entity("candidate", name, record)
                entities += 1
                
            except Exception as e:
                logger.debug(f"Parse error: {e}")
        
        return entities
    
    async def _process_committees(self, data: str, limit: Optional[int]) -> int:
        """Process committee master records."""
        # CMTE_ID|CMTE_NM|TRES_NM|CMTE_ST1|CMTE_ST2|CMTE_CITY|CMTE_ST|CMTE_ZIP|...
        
        entities = 0
        lines = data.strip().split("\n")
        
        for i, line in enumerate(lines):
            if limit and i >= limit:
                break
            
            fields = line.split("|")
            if len(fields) < 4:
                continue
            
            try:
                cmte_id = fields[0]
                name = fields[1]
                treasurer = fields[2] if len(fields) > 2 else ""
                
                record = {
                    "committee_id": cmte_id,
                    "name": name,
                    "treasurer": treasurer,
                }
                await self._save_entity("committee", name, record)
                entities += 1
                
            except Exception as e:
                logger.debug(f"Parse error: {e}")
        
        return entities
    
    async def _process_committee_contributions(
        self,
        data: str,
        limit: Optional[int]
    ) -> tuple[int, int]:
        """Process committee-to-candidate contributions."""
        # CMTE_ID|AMNDT_IND|RPT_TP|TRANSACTION_PGI|...|CAND_ID|TRANSACTION_AMT|...
        
        entities = 0
        relationships = 0
        lines = data.strip().split("\n")
        
        for i, line in enumerate(lines):
            if limit and i >= limit:
                break
            
            fields = line.split("|")
            if len(fields) < 16:
                continue
            
            try:
                cmte_id = fields[0]
                cand_id = fields[16] if len(fields) > 16 else ""
                amount = fields[14] if len(fields) > 14 else "0"
                
                if cmte_id and cand_id:
                    record = {
                        "committee_id": cmte_id,
                        "candidate_id": cand_id,
                        "amount": amount,
                    }
                    await self._save_relationship(cmte_id, cand_id, "contributed_to", record)
                    relationships += 1
                    
            except Exception as e:
                logger.debug(f"Parse error: {e}")
        
        return entities, relationships
    
    async def _save_entity(
        self,
        entity_type: str,
        name: str,
        data: Dict[str, Any]
    ):
        """Save an extracted entity."""
        # Import graph storage and add entity
        try:
            from ..graph_storage import MatrixGraph, Entity
            
            graph = MatrixGraph(self.data_dir.parent)
            entity = Entity(
                entity_id=f"fec_{entity_type}_{data.get('candidate_id', data.get('committee_id', name))}",
                name=name,
                entity_type=entity_type,
                properties=data,
                sources=["fec"],
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
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                properties=data,
                sources=["fec"],
            )
            graph.add_relationship(rel)
            
        except Exception as e:
            logger.debug(f"Could not save relationship: {e}")
