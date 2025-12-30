"""
Influence Graph Analyzer - Phase 3
Maps power structures, identifies influence patterns, predicts cascade effects

This is network analysis for prediction markets - who influences whom,
how information spreads, where power concentrates.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in influence graph"""
    PERSON = "person"
    CORPORATION = "corporation"
    GOVERNMENT = "government"
    ORGANIZATION = "organization"
    INSTITUTION = "institution"


class EdgeType(Enum):
    """Types of relationships"""
    CONTROLS = "controls"
    INFLUENCES = "influences"
    OWNS = "owns"
    FUNDS = "funds"
    EMPLOYS = "employs"
    ALLIED_WITH = "allied_with"
    COMPETES_WITH = "competes_with"


@dataclass
class NetworkNode:
    """Node in influence graph"""
    node_id: str
    node_type: NodeType
    name: str
    influence_score: float = 0.0
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkEdge:
    """Edge in influence graph"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerCenter:
    """Identified power center in network"""
    center_id: str
    core_nodes: List[str]
    influence_radius: float
    total_influence: float
    connected_centers: List[str] = field(default_factory=list)


@dataclass
class CascadePrediction:
    """Predicted cascade effect"""
    trigger_node: str
    trigger_event: str
    affected_nodes: List[str]
    cascade_probability: float
    propagation_speed: float  # nodes/day
    max_reach: int
    timeline: Dict[str, List[str]]  # day -> affected nodes


class InfluenceGraphAnalyzer:
    """
    Analyzes network structures to identify power centers,
    predict information cascades, and map influence flows.
    
    Think: Who really controls what? How does information spread?
    Where are the hidden power structures?
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for influence
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: List[NetworkEdge] = []
        self.power_centers: List[PowerCenter] = []
        
        logger.info("Influence Graph Analyzer initialized")
    
    def add_node(self, node: NetworkNode) -> None:
        """Add node to influence graph."""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            type=node.node_type.value,
            name=node.name,
            influence=node.influence_score
        )
    
    def add_edge(self, edge: NetworkEdge) -> None:
        """Add edge to influence graph."""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            type=edge.edge_type.value,
            weight=edge.weight
        )
        
        if edge.bidirectional:
            self.graph.add_edge(
                edge.target,
                edge.source,
                type=edge.edge_type.value,
                weight=edge.weight
            )
    
    def build_influence_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> None:
        """
        Build influence graph from entities and relationships.
        
        Args:
            entities: List of entity dicts (name, type, etc.)
            relationships: List of relationship dicts (source, target, type, weight)
        """
        # Add nodes
        for entity in entities:
            node = NetworkNode(
                node_id=entity.get("id", entity["name"]),
                node_type=NodeType(entity.get("type", "organization")),
                name=entity["name"],
                metadata=entity.get("metadata", {})
            )
            self.add_node(node)
        
        # Add edges
        for rel in relationships:
            edge = NetworkEdge(
                source=rel["source"],
                target=rel["target"],
                edge_type=EdgeType(rel.get("type", "influences")),
                weight=rel.get("weight", 1.0),
                bidirectional=rel.get("bidirectional", False)
            )
            self.add_edge(edge)
        
        # Calculate influence scores
        self._calculate_centrality_metrics()
        
        logger.info(f"Built graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def identify_power_centers(
        self,
        min_influence: float = 0.5,
        cluster_threshold: float = 0.7
    ) -> List[PowerCenter]:
        """
        Identify power centers in the network.
        
        Power centers are:
        - High centrality nodes
        - Densely connected clusters
        - Control points for information flow
        
        Returns:
            List of identified power centers
        """
        if not self.graph.nodes:
            return []
        
        # Find high-influence nodes
        high_influence_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.centrality_scores.get("pagerank", 0) > min_influence
        ]
        
        # Detect communities (power clusters)
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()
            communities = list(nx.community.greedy_modularity_communities(undirected))
        except:
            communities = []
        
        power_centers = []
        
        for i, community in enumerate(communities):
            community_nodes = list(community)
            
            # Calculate total influence
            total_influence = sum(
                self.nodes[n].centrality_scores.get("pagerank", 0)
                for n in community_nodes
                if n in self.nodes
            )
            
            if total_influence > min_influence:
                # Calculate influence radius
                subgraph = self.graph.subgraph(community_nodes)
                radius = nx.diameter(subgraph.to_undirected()) if nx.is_connected(subgraph.to_undirected()) else 1
                
                power_center = PowerCenter(
                    center_id=f"center_{i}",
                    core_nodes=community_nodes[:10],  # Top 10 nodes
                    influence_radius=radius,
                    total_influence=total_influence
                )
                power_centers.append(power_center)
        
        self.power_centers = power_centers
        logger.info(f"Identified {len(power_centers)} power centers")
        
        return power_centers
    
    def predict_cascade_effects(
        self,
        trigger_node: str,
        trigger_event: str,
        cascade_threshold: float = 0.3,
        max_hops: int = 5
    ) -> CascadePrediction:
        """
        Predict cascade effects from a trigger event.
        
        Examples:
        - Bank run contagion
        - Corporate scandals spreading
        - Geopolitical tensions escalating
        - Information/rumor propagation
        
        Args:
            trigger_node: Node where event originates
            trigger_event: Description of trigger event
            cascade_threshold: Minimum influence to cascade
            max_hops: Maximum propagation distance
            
        Returns:
            Cascade prediction with affected nodes and timeline
        """
        if trigger_node not in self.graph:
            raise ValueError(f"Trigger node {trigger_node} not in graph")
        
        affected_nodes = []
        timeline = {}
        
        # BFS-style cascade simulation
        current_wave = {trigger_node}
        visited = {trigger_node}
        
        for hop in range(max_hops):
            next_wave = set()
            
            for node in current_wave:
                # Get neighbors
                neighbors = list(self.graph.successors(node))
                
                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    
                    # Check if cascade propagates based on edge weight
                    edge_data = self.graph[node][neighbor]
                    weight = edge_data.get("weight", 0.5)
                    
                    if weight >= cascade_threshold:
                        next_wave.add(neighbor)
                        visited.add(neighbor)
                        affected_nodes.append(neighbor)
            
            if next_wave:
                timeline[f"day_{hop + 1}"] = list(next_wave)
                current_wave = next_wave
            else:
                break
        
        # Calculate metrics
        cascade_prob = min(len(affected_nodes) / len(self.graph.nodes), 1.0)
        propagation_speed = len(affected_nodes) / max(len(timeline), 1)
        
        return CascadePrediction(
            trigger_node=trigger_node,
            trigger_event=trigger_event,
            affected_nodes=affected_nodes,
            cascade_probability=cascade_prob,
            propagation_speed=propagation_speed,
            max_reach=len(affected_nodes),
            timeline=timeline
        )
    
    def calculate_causal_impact(
        self,
        intervention_node: str,
        target_node: str
    ) -> float:
        """
        Calculate causal impact of intervention on target.
        
        Uses graph structure and edge weights to estimate
        causal effect (not just correlation).
        
        Returns:
            Estimated causal impact (0.0 to 1.0)
        """
        if intervention_node not in self.graph or target_node not in self.graph:
            return 0.0
        
        # Find all paths from intervention to target
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                intervention_node,
                target_node,
                cutoff=5  # Max path length
            ))
        except:
            paths = []
        
        if not paths:
            return 0.0
        
        # Calculate path strengths
        path_strengths = []
        for path in paths:
            strength = 1.0
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                strength *= edge_data.get("weight", 0.5)
            path_strengths.append(strength)
        
        # Causal impact is max path strength (strongest causal chain)
        causal_impact = max(path_strengths) if path_strengths else 0.0
        
        return causal_impact
    
    def detect_hidden_coalitions(
        self,
        min_coalition_size: int = 3
    ) -> List[List[str]]:
        """
        Detect hidden coalitions (groups that coordinate but aren't obvious).
        
        Uses community detection and structural equivalence.
        
        Returns:
            List of hidden coalitions (node groups)
        """
        if len(self.graph.nodes) < min_coalition_size:
            return []
        
        # Structural equivalence - nodes with similar connection patterns
        try:
            # Convert to undirected
            undirected = self.graph.to_undirected()
            
            # Community detection
            communities = list(nx.community.label_propagation_communities(undirected))
            
            # Filter by size
            coalitions = [
                list(community) for community in communities
                if len(community) >= min_coalition_size
            ]
            
            logger.info(f"Detected {len(coalitions)} hidden coalitions")
            return coalitions
        except:
            return []
    
    def get_shortest_influence_path(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """Get shortest path of influence from source to target."""
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except:
            return None
    
    def _calculate_centrality_metrics(self) -> None:
        """Calculate various centrality metrics for all nodes."""
        if not self.graph.nodes:
            return
        
        # PageRank (Google's algorithm - best for influence)
        pagerank = nx.pagerank(self.graph, weight="weight")
        
        # Betweenness centrality (information broker score)
        betweenness = nx.betweenness_centrality(self.graph, weight="weight")
        
        # Closeness centrality (speed of information spread)
        try:
            closeness = nx.closeness_centrality(self.graph, distance="weight")
        except:
            closeness = {}
        
        # Eigenvector centrality (connected to important nodes)
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, weight="weight", max_iter=1000)
        except:
            eigenvector = {}
        
        # Update node scores
        for node_id in self.nodes:
            if node_id in self.graph:
                self.nodes[node_id].centrality_scores = {
                    "pagerank": pagerank.get(node_id, 0.0),
                    "betweenness": betweenness.get(node_id, 0.0),
                    "closeness": closeness.get(node_id, 0.0),
                    "eigenvector": eigenvector.get(node_id, 0.0)
                }
                # Overall influence is weighted combination
                self.nodes[node_id].influence_score = (
                    pagerank.get(node_id, 0.0) * 0.4 +
                    betweenness.get(node_id, 0.0) * 0.3 +
                    eigenvector.get(node_id, 0.0) * 0.3
                )


# Global analyzer instance
_analyzer: Optional[InfluenceGraphAnalyzer] = None


def get_influence_analyzer() -> InfluenceGraphAnalyzer:
    """Get or create global influence graph analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = InfluenceGraphAnalyzer()
    return _analyzer
