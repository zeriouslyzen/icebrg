"""
Wikidata API Client
Access structured knowledge from Wikidata (the data behind Wikipedia).
https://www.wikidata.org/w/api.php
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import requests

logger = logging.getLogger(__name__)


@dataclass
class WikidataEntity:
    """Entity from Wikidata."""
    qid: str  # Wikidata entity ID (e.g., Q76 for Barack Obama)
    label: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    entity_type: str = "item"  # 'item' or 'property'
    claims: Dict[str, Any] = field(default_factory=dict)
    sitelinks: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "label": self.label,
            "description": self.description,
            "aliases": self.aliases,
            "entity_type": self.entity_type,
            "claims": self.claims,
            "sitelinks": self.sitelinks
        }


class WikidataClient:
    """
    Wikidata API client for structured knowledge queries.
    
    No API key required.
    Rate limit: Be respectful, add delays for bulk queries.
    
    Docs: https://www.wikidata.org/w/api.php
    SPARQL: https://query.wikidata.org/
    """
    
    API_URL = "https://www.wikidata.org/w/api.php"
    SPARQL_URL = "https://query.wikidata.org/sparql"
    
    # Common property IDs
    PROPERTIES = {
        "instance_of": "P31",
        "occupation": "P106",
        "employer": "P108",
        "member_of": "P463",
        "educated_at": "P69",
        "position_held": "P39",
        "political_party": "P102",
        "country_of_citizenship": "P27",
        "spouse": "P26",
        "child": "P40",
        "parent": "P22",
        "sibling": "P3373",
        "headquarters": "P159",
        "founder": "P112",
        "ceo": "P169",
        "owner": "P127",
        "subsidiary": "P355",
        "industry": "P452",
        "date_of_birth": "P569",
        "date_of_death": "P570",
        "inception": "P571",
        "dissolved": "P576",
        "image": "P18",
        "official_website": "P856",
    }
    
    def __init__(self):
        """Initialize Wikidata client."""
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ICEBURG/1.0 (Research AI; contact@iceburg.ai)"
    
    def _api_request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request."""
        params["format"] = "json"
        
        try:
            response = self.session.get(self.API_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Wikidata API error: {e}")
            return None
    
    def _sparql_request(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute SPARQL query."""
        try:
            response = self.session.get(
                self.SPARQL_URL,
                params={"query": query, "format": "json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Wikidata SPARQL error: {e}")
            return None
    
    def search(self, query: str, language: str = "en", limit: int = 10) -> List[WikidataEntity]:
        """
        Search for entities.
        
        Args:
            query: Search query
            language: Language code
            limit: Maximum results
            
        Returns:
            List of WikidataEntity objects
        """
        data = self._api_request({
            "action": "wbsearchentities",
            "search": query,
            "language": language,
            "limit": limit,
            "type": "item"
        })
        
        if not data:
            return []
        
        entities = []
        for item in data.get("search", []):
            entities.append(WikidataEntity(
                qid=item.get("id", ""),
                label=item.get("label", ""),
                description=item.get("description", ""),
                aliases=item.get("aliases", [])
            ))
        
        logger.info(f"Wikidata: Found {len(entities)} entities for '{query}'")
        return entities
    
    def get_entity(self, qid: str, language: str = "en") -> Optional[WikidataEntity]:
        """
        Get detailed entity information.
        
        Args:
            qid: Wikidata entity ID (e.g., 'Q76')
            language: Language code
            
        Returns:
            WikidataEntity with full details or None
        """
        data = self._api_request({
            "action": "wbgetentities",
            "ids": qid,
            "languages": language,
            "props": "labels|descriptions|aliases|claims|sitelinks"
        })
        
        if not data or "entities" not in data:
            return None
        
        entity_data = data["entities"].get(qid, {})
        
        if not entity_data or entity_data.get("missing"):
            return None
        
        # Extract labels
        labels = entity_data.get("labels", {})
        label = labels.get(language, {}).get("value", qid)
        
        # Extract description
        descs = entity_data.get("descriptions", {})
        description = descs.get(language, {}).get("value", "")
        
        # Extract aliases
        alias_data = entity_data.get("aliases", {}).get(language, [])
        aliases = [a.get("value") for a in alias_data]
        
        # Extract sitelinks
        sitelinks = {}
        for site, link_data in entity_data.get("sitelinks", {}).items():
            sitelinks[site] = link_data.get("title", "")
        
        # Process claims
        claims = self._process_claims(entity_data.get("claims", {}), language)
        
        return WikidataEntity(
            qid=qid,
            label=label,
            description=description,
            aliases=aliases,
            claims=claims,
            sitelinks=sitelinks
        )
    
    def _process_claims(self, raw_claims: Dict, language: str) -> Dict[str, Any]:
        """Process claims into readable format."""
        claims = {}
        
        # Map property IDs to names
        prop_names = {v: k for k, v in self.PROPERTIES.items()}
        
        for prop_id, claim_list in raw_claims.items():
            prop_name = prop_names.get(prop_id, prop_id)
            
            values = []
            for claim in claim_list:
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                
                value = self._extract_value(datavalue)
                if value:
                    values.append(value)
            
            if values:
                claims[prop_name] = values
        
        return claims
    
    def _extract_value(self, datavalue: Dict) -> Optional[str]:
        """Extract value from Wikidata datavalue."""
        if not datavalue:
            return None
        
        value_type = datavalue.get("type")
        value = datavalue.get("value")
        
        if value_type == "wikibase-entityid":
            # Return entity ID for now (could resolve labels)
            return value.get("id")
        elif value_type == "string":
            return value
        elif value_type == "time":
            return value.get("time", "")[1:11]  # Extract date part
        elif value_type == "quantity":
            return f"{value.get('amount')} {value.get('unit', '').split('/')[-1]}"
        elif value_type == "monolingualtext":
            return value.get("text")
        
        return str(value) if value else None
    
    def get_related_entities(
        self,
        qid: str,
        properties: List[str] = None,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Get entities related to a given entity.
        
        Args:
            qid: Entity ID
            properties: List of property names to follow (default: key relationships)
            language: Language code
            
        Returns:
            List of related entity dictionaries
        """
        if properties is None:
            properties = ["member_of", "employer", "position_held", "subsidiary", "owner"]
        
        entity = self.get_entity(qid, language)
        
        if not entity:
            return []
        
        related = []
        
        for prop_name in properties:
            values = entity.claims.get(prop_name, [])
            
            for value in values:
                if value and value.startswith("Q"):
                    related_entity = self.get_entity(value, language)
                    if related_entity:
                        related.append({
                            "qid": value,
                            "label": related_entity.label,
                            "description": related_entity.description,
                            "relationship": prop_name,
                            "from_entity": entity.label
                        })
        
        return related
    
    def find_connections(
        self,
        entity1_name: str,
        entity2_name: str,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Find connections between two entities.
        
        Args:
            entity1_name: First entity name
            entity2_name: Second entity name
            language: Language code
            
        Returns:
            List of connection paths
        """
        # Search for both entities
        results1 = self.search(entity1_name, language, limit=1)
        results2 = self.search(entity2_name, language, limit=1)
        
        if not results1 or not results2:
            return []
        
        qid1 = results1[0].qid
        qid2 = results2[0].qid
        
        # Use SPARQL to find paths (simple 1-hop for now)
        query = f"""
        SELECT ?property ?propertyLabel ?intermediate ?intermediateLabel WHERE {{
            wd:{qid1} ?property ?intermediate .
            ?intermediate ?property2 wd:{qid2} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
        }} LIMIT 10
        """
        
        data = self._sparql_request(query)
        
        if not data:
            return []
        
        connections = []
        for binding in data.get("results", {}).get("bindings", []):
            connections.append({
                "entity1": entity1_name,
                "entity2": entity2_name,
                "intermediate": binding.get("intermediateLabel", {}).get("value", ""),
                "property": binding.get("propertyLabel", {}).get("value", "")
            })
        
        return connections
    
    def get_organization_members(self, org_name: str, language: str = "en") -> List[Dict[str, Any]]:
        """
        Get known members of an organization.
        
        Args:
            org_name: Organization name
            language: Language code
            
        Returns:
            List of member dictionaries
        """
        # First find the organization
        results = self.search(org_name, language, limit=1)
        
        if not results:
            return []
        
        org_qid = results[0].qid
        
        # SPARQL query for members
        query = f"""
        SELECT ?member ?memberLabel ?position ?positionLabel WHERE {{
            ?member wdt:P463 wd:{org_qid} .
            OPTIONAL {{ ?member wdt:P39 ?position . }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
        }} LIMIT 50
        """
        
        data = self._sparql_request(query)
        
        if not data:
            return []
        
        members = []
        for binding in data.get("results", {}).get("bindings", []):
            members.append({
                "name": binding.get("memberLabel", {}).get("value", ""),
                "qid": binding.get("member", {}).get("value", "").split("/")[-1],
                "position": binding.get("positionLabel", {}).get("value", ""),
                "organization": org_name
            })
        
        logger.info(f"Wikidata: Found {len(members)} members of '{org_name}'")
        return members
