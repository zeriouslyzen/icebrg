"""
V2 Network Analysis API Controller
Exposes network analysis capabilities
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..network.influence_graph_analyzer import (
    get_influence_analyzer,
    NodeType,
    EdgeType,
    NetworkNode,
    NetworkEdge
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/network", tags=["v2-network"])


@router.get("/influence-graph")
async def get_influence_graph():
    """
    Get complete influence graph data.
    
    Returns:
        {
            "nodes": [...],
            "edges": [...],
            "stats": {...}
        }
    """
    try:
        analyzer = get_influence_analyzer()
        
        nodes_data = [
            {
                "id": node.node_id,
                "type": node.node_type.value,
                "name": node.name,
                "influence_score": node.influence_score,
                "centrality": node.centrality_scores
            }
            for node in analyzer.nodes.values()
        ]
        
        edges_data = [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type.value,
                "weight": edge.weight,
                "bidirectional": edge.bidirectional
            }
            for edge in analyzer.edges
        ]
        
        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "stats": {
                "node_count": len(nodes_data),
                "edge_count": len(edges_data),
                "avg_influence": sum(n["influence_score"] for n in nodes_data) / len(nodes_data) if nodes_data else 0
            }
        }
    
    except Exception as e:
        logger.error(f"Influence graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build")
async def build_graph(request_data: Dict[str, Any]):
    """
    Build influence graph from entities and relationships.
    
    Request body:
        {
            "entities": [
                {"id": "node1", "name": "Peter Thiel", "type": "person"},
                ...
            ],
            "relationships": [
                {"source": "node1", "target": "node2", "type": "controls", "weight": 0.9},
                ...
            ]
        }
    """
    try:
        analyzer = get_influence_analyzer()
        
        entities = request_data.get("entities", [])
        relationships = request_data.get("relationships", [])
        
        analyzer.build_influence_graph(entities, relationships)
        
        return {
            "status": "built",
            "node_count": len(analyzer.nodes),
            "edge_count": len(analyzer.edges)
        }
    
    except Exception as e:
        logger.error(f"Build graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/power-centers")
async def get_power_centers(
    min_influence: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Get identified power centers in network.
    
    Returns:
        List of power centers with core nodes and influence metrics
    """
    try:
        analyzer = get_influence_analyzer()
        
        power_centers = analyzer.identify_power_centers(min_influence=min_influence)
        
        centers_data = [
            {
                "center_id": pc.center_id,
                "core_nodes": pc.core_nodes,
                "influence_radius": pc.influence_radius,
                "total_influence": pc.total_influence,
                "connected_centers": pc.connected_centers
            }
            for pc in power_centers
        ]
        
        return {
            "power_centers": centers_data,
            "count": len(centers_data)
        }
    
    except Exception as e:
        logger.error(f"Power centers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cascade-prediction")
async def predict_cascade(request_data: Dict[str, Any]):
    """
    Predict cascade effects from trigger event.
    
    Request body:
        {
            "trigger_node": "lehman_brothers",
            "trigger_event": "bankruptcy",
            "cascade_threshold": 0.3,
            "max_hops": 5
        }
        
    Returns:
        Cascade prediction with affected nodes and timeline
    """
    try:
        analyzer = get_influence_analyzer()
        
        trigger_node = request_data["trigger_node"]
        trigger_event = request_data["trigger_event"]
        threshold = request_data.get("cascade_threshold", 0.3)
        max_hops = request_data.get("max_hops", 5)
        
        prediction = analyzer.predict_cascade_effects(
            trigger_node,
            trigger_event,
            threshold,
            max_hops
        )
        
        return {
            "trigger_node": prediction.trigger_node,
            "trigger_event": prediction.trigger_event,
            "affected_nodes": prediction.affected_nodes,
            "cascade_probability": prediction.cascade_probability,
            "propagation_speed": prediction.propagation_speed,
            "max_reach": prediction.max_reach,
            "timeline": prediction.timeline
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Cascade prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-impact")
async def calculate_causal_impact(
    intervention: str = Query(..., description="Intervention node"),
    target: str = Query(..., description="Target node")
):
    """
    Calculate causal impact of intervention on target.
    
    Returns:
        Causal impact score (0.0 to 1.0)
    """
    try:
        analyzer = get_influence_analyzer()
        
        impact = analyzer.calculate_causal_impact(intervention, target)
        
        # Get influence path
        path = analyzer.get_shortest_influence_path(intervention, target)
        
        return {
            "intervention_node": intervention,
            "target_node": target,
            "causal_impact": impact,
            "influence_path": path,
            "path_length": len(path) if path else 0
        }
    
    except Exception as e:
        logger.error(f"Causal impact error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hidden-coalitions")
async def detect_hidden_coalitions(
    min_size: int = Query(3, ge=2)
):
    """
    Detect hidden coalitions in network.
    
    Returns:
        List of coalition groups
    """
    try:
        analyzer = get_influence_analyzer()
        
        coalitions = analyzer.detect_hidden_coalitions(min_coalition_size=min_size)
        
        return {
            "coalitions": coalitions,
            "count": len(coalitions)
        }
    
    except Exception as e:
        logger.error(f"Coalition detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
