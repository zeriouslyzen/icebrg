"""
Mapper Agent - Network and relationship mapping.
Builds network graphs of people, organizations, and their connections.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

from ...config import IceburgConfig
from ...providers.factory import provider_factory

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """An entity in the network (person, org, location)."""
    id: str
    name: str
    entity_type: str  # 'person', 'organization', 'location', 'event'
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type,
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass 
class Relationship:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relationship_type: str  # 'member_of', 'funds', 'owns', 'connected_to', 'married_to', etc.
    description: str = ""
    strength: float = 0.5  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relationship_type,
            "description": self.description,
            "strength": self.strength,
            "evidence": self.evidence
        }


@dataclass
class NetworkGraph:
    """Complete network graph with entities and relationships."""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    central_topic: str = ""
    
    def add_entity(self, entity: Entity):
        if not any(e.id == entity.id for e in self.entities):
            self.entities.append(entity)
    
    def add_relationship(self, rel: Relationship):
        self.relationships.append(rel)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None
    
    def get_connections(self, entity_id: str) -> List[Tuple[Entity, Relationship]]:
        """Get all entities connected to a given entity."""
        connections = []
        for rel in self.relationships:
            if rel.source_id == entity_id:
                target = self.get_entity(rel.target_id)
                if target:
                    connections.append((target, rel))
            elif rel.target_id == entity_id:
                source = self.get_entity(rel.source_id)
                if source:
                    connections.append((source, rel))
        return connections
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "central_topic": self.central_topic,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "stats": {
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "entity_types": self._count_entity_types(),
                "relationship_types": self._count_relationship_types()
            }
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        counts = {}
        for e in self.entities:
            counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
        return counts
    
    def _count_relationship_types(self) -> Dict[str, int]:
        counts = {}
        for r in self.relationships:
            counts[r.relationship_type] = counts.get(r.relationship_type, 0) + 1
        return counts
    
    def to_d3_format(self) -> Dict[str, Any]:
        """Convert to D3.js force-directed graph format."""
        return {
            "nodes": [
                {
                    "id": e.id,
                    "name": e.name,
                    "group": self._type_to_group(e.entity_type),
                    "type": e.entity_type
                }
                for e in self.entities
            ],
            "links": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "value": r.strength,
                    "label": r.relationship_type
                }
                for r in self.relationships
            ]
        }
    
    def _type_to_group(self, entity_type: str) -> int:
        """Map entity type to D3 group number for coloring."""
        mapping = {
            "person": 1,
            "organization": 2,
            "location": 3,
            "event": 4,
            "concept": 5
        }
        return mapping.get(entity_type, 0)


@dataclass
class MapperReport:
    """Complete mapper analysis report."""
    query: str
    network: NetworkGraph = field(default_factory=NetworkGraph)
    key_players: List[Dict[str, Any]] = field(default_factory=list)
    power_centers: List[str] = field(default_factory=list)
    hidden_connections: List[Dict[str, Any]] = field(default_factory=list)
    funding_flows: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "network": self.network.to_dict(),
            "d3_graph": self.network.to_d3_format(),
            "key_players": self.key_players,
            "power_centers": self.power_centers,
            "hidden_connections": self.hidden_connections,
            "funding_flows": self.funding_flows,
            "timestamp": self.timestamp.isoformat()
        }


class MapperAgent:
    """
    Network and relationship mapping agent.
    
    Builds graphs of:
    - People ↔ Organizations (employment, membership)
    - Funding flows (donations, grants, investments)
    - Ownership structures (parent companies, subsidiaries)
    - Historical connections (past associations, family ties)
    """
    
    # Known power structure entities
    KNOWN_POWER_NODES = {
        "cfr": "Council on Foreign Relations",
        "trilateral": "Trilateral Commission",
        "bilderberg": "Bilderberg Group",
        "wef": "World Economic Forum",
        "imf": "International Monetary Fund",
        "world_bank": "World Bank",
        "fed": "Federal Reserve",
        "cia": "Central Intelligence Agency",
        "nsa": "National Security Agency",
        "mi6": "MI6/SIS",
        "mossad": "Mossad",
        "blackrock": "BlackRock",
        "vanguard": "Vanguard Group",
        "state_street": "State Street Corporation"
    }
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.provider = None
    
    def _get_provider(self):
        if self.provider is None:
            self.provider = provider_factory(self.cfg)
        return self.provider
    
    def map_network(
        self,
        query: str,
        entities_from_gatherer: List[Dict[str, Any]] = None,
        content: str = "",
        thinking_callback: Optional[callable] = None
    ) -> MapperReport:
        """
        Build network map for a topic.
        
        Args:
            query: Research topic
            entities_from_gatherer: Pre-extracted entities from Gatherer
            content: Source content
            thinking_callback: Progress callback
            
        Returns:
            MapperReport with network graph
        """
        report = MapperReport(query=query)
        report.network.central_topic = query
        
        if thinking_callback:
            thinking_callback("🗺️ Building entity network...")
        
        # Start with entities from gatherer
        if entities_from_gatherer:
            for entity_data in entities_from_gatherer:
                entity = Entity(
                    id=self._make_id(entity_data.get("name", "")),
                    name=entity_data.get("name", "Unknown"),
                    entity_type=entity_data.get("type", "unknown"),
                    description=entity_data.get("context", "")
                )
                report.network.add_entity(entity)
        
        if thinking_callback:
            thinking_callback("🔗 Discovering relationships...")
        
        # Discover relationships using LLM
        if content or entities_from_gatherer:
            relationships = self._discover_relationships(query, report.network.entities, content)
            for rel in relationships:
                report.network.add_relationship(rel)
        
        if thinking_callback:
            thinking_callback("🏛️ Checking power structure connections...")
        
        # Check for known power structure connections
        power_connections = self._check_power_connections(report.network.entities, content)
        for entity, rel in power_connections:
            report.network.add_entity(entity)
            report.network.add_relationship(rel)
        
        # Identify key players (highest connection count)
        report.key_players = self._identify_key_players(report.network)
        
        # Identify power centers
        report.power_centers = self._identify_power_centers(report.network)
        
        # Find hidden connections (entities connected through intermediaries)
        report.hidden_connections = self._find_hidden_connections(report.network)
        
        # Analyze funding flows
        report.funding_flows = self._analyze_funding(report.network)
        
        logger.info(f"✅ Network mapping complete: {len(report.network.entities)} entities, {len(report.network.relationships)} relationships")
        return report
    
    def _make_id(self, name: str) -> str:
        """Create a clean ID from a name."""
        return name.lower().replace(" ", "_").replace("'", "").replace(".", "")[:50]
    
    def _discover_relationships(
        self,
        query: str,
        entities: List[Entity],
        content: str
    ) -> List[Relationship]:
        """Use LLM to discover relationships between entities."""
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            entity_names = [e.name for e in entities[:15]]  # Limit for prompt
            
            prompt = f"""Given these entities related to "{query}":
{json.dumps(entity_names, indent=2)}

And this context:
{content[:2000]}

Identify relationships between these entities. Return JSON array:
[{{"source": "entity name", "target": "entity name", "type": "relationship type", "description": "brief description"}}]

Relationship types: member_of, funds, owns, works_for, connected_to, married_to, partnered_with, opposes, founded, advises

Return ONLY valid JSON array."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="You are a network analyst. Identify relationships between entities. Be specific and evidence-based.",
                temperature=0.3,
                options={"max_tokens": 800}
            )
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            rel_data = json.loads(response)
            
            for rel in rel_data:
                source_id = self._make_id(rel.get("source", ""))
                target_id = self._make_id(rel.get("target", ""))
                
                # Only add if both entities exist
                if any(e.id == source_id for e in entities) and any(e.id == target_id for e in entities):
                    relationships.append(Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=rel.get("type", "connected_to"),
                        description=rel.get("description", ""),
                        strength=0.6
                    ))
                    
        except Exception as e:
            logger.warning(f"Relationship discovery failed: {e}")
        
        return relationships
    
    def _check_power_connections(
        self,
        entities: List[Entity],
        content: str
    ) -> List[Tuple[Entity, Relationship]]:
        """Check if any entities connect to known power structures."""
        connections = []
        content_lower = content.lower()
        
        for power_key, power_name in self.KNOWN_POWER_NODES.items():
            if power_key in content_lower or power_name.lower() in content_lower:
                # Add the power node as an entity
                power_entity = Entity(
                    id=power_key,
                    name=power_name,
                    entity_type="organization",
                    description="Known power structure node",
                    metadata={"is_power_node": True}
                )
                
                # Try to find which existing entities connect to it
                for entity in entities:
                    if entity.entity_type == "person":
                        # Create a weak connection that needs verification
                        rel = Relationship(
                            source_id=entity.id,
                            target_id=power_key,
                            relationship_type="possibly_connected_to",
                            description="Mentioned in same context",
                            strength=0.3
                        )
                        connections.append((power_entity, rel))
                        break  # Only add one connection per power node for now
        
        return connections
    
    def _identify_key_players(self, network: NetworkGraph) -> List[Dict[str, Any]]:
        """Identify key players based on connection count."""
        connection_counts = {}
        
        for rel in network.relationships:
            connection_counts[rel.source_id] = connection_counts.get(rel.source_id, 0) + 1
            connection_counts[rel.target_id] = connection_counts.get(rel.target_id, 0) + 1
        
        # Sort by connection count
        sorted_entities = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)
        
        key_players = []
        for entity_id, count in sorted_entities[:10]:
            entity = network.get_entity(entity_id)
            if entity:
                key_players.append({
                    "name": entity.name,
                    "type": entity.entity_type,
                    "connections": count,
                    "description": entity.description
                })
        
        return key_players
    
    def _identify_power_centers(self, network: NetworkGraph) -> List[str]:
        """Identify power centers (organizations with most connections)."""
        org_connections = {}
        
        for rel in network.relationships:
            source = network.get_entity(rel.source_id)
            target = network.get_entity(rel.target_id)
            
            if source and source.entity_type == "organization":
                org_connections[source.name] = org_connections.get(source.name, 0) + 1
            if target and target.entity_type == "organization":
                org_connections[target.name] = org_connections.get(target.name, 0) + 1
        
        # Sort and return top power centers
        sorted_orgs = sorted(org_connections.items(), key=lambda x: x[1], reverse=True)
        return [org for org, _ in sorted_orgs[:5]]
    
    def _find_hidden_connections(self, network: NetworkGraph) -> List[Dict[str, Any]]:
        """Find entities connected through intermediaries (2-hop connections)."""
        hidden = []
        
        # For each pair of entities, check if they're connected through a third
        entities = network.entities
        
        for i, e1 in enumerate(entities):
            connections_1 = {conn[0].id for conn in network.get_connections(e1.id)}
            
            for e2 in entities[i+1:]:
                if e2.id in connections_1:
                    continue  # Already directly connected
                
                connections_2 = {conn[0].id for conn in network.get_connections(e2.id)}
                
                # Find intermediaries
                intermediaries = connections_1 & connections_2
                
                if intermediaries:
                    for int_id in list(intermediaries)[:1]:  # Just take first
                        intermediary = network.get_entity(int_id)
                        if intermediary:
                            hidden.append({
                                "entity_1": e1.name,
                                "entity_2": e2.name,
                                "connected_via": intermediary.name,
                                "significance": "Two-hop connection suggests possible indirect relationship"
                            })
        
        return hidden[:10]  # Limit
    
    def _analyze_funding(self, network: NetworkGraph) -> List[Dict[str, Any]]:
        """Analyze funding relationships."""
        funding = []
        
        funding_types = {"funds", "donates_to", "invests_in", "sponsors"}
        
        for rel in network.relationships:
            if rel.relationship_type in funding_types:
                source = network.get_entity(rel.source_id)
                target = network.get_entity(rel.target_id)
                
                if source and target:
                    funding.append({
                        "source": source.name,
                        "target": target.name,
                        "type": rel.relationship_type,
                        "description": rel.description
                    })
        
        return funding


def map_network(cfg: IceburgConfig, query: str, entities: List[Dict] = None, content: str = "") -> MapperReport:
    """Convenience function for network mapping."""
    agent = MapperAgent(cfg)
    return agent.map_network(query, entities, content)
