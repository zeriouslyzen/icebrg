"""
Graph-Based Processing
Implements graph-based processing architecture (US20230108560A1 pattern).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import networkx as nx

logger = logging.getLogger(__name__)


class GraphProcessor:
    """
    Graph-based processor implementing US20230108560A1 pattern.
    
    Architecture:
    - Modular processors as graph nodes
    - Directed graph representation of applications
    - Continuous graph processing
    - Dynamic graph updates
    - Fine-controlled behavior execution
    """
    
    def __init__(self):
        self.processors: Dict[str, Any] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.stats = {
            "graphs_processed": 0,
            "nodes_executed": 0,
            "edges_traversed": 0,
            "execution_time_avg": 0.0
        }
        
        logger.info("GraphProcessor initialized")
    
    def add_processor(self, processor_id: str, processor: Any, dependencies: List[str] = None):
        """Add processor to graph"""
        try:
            self.processors[processor_id] = processor
            self.graph.add_node(processor_id, processor=processor)
            
            # Add dependencies as edges
            if dependencies:
                for dep in dependencies:
                    if dep in self.processors:
                        self.graph.add_edge(dep, processor_id)
            
            logger.debug(f"Added processor {processor_id} to graph")
            
        except Exception as e:
            logger.error(f"Error adding processor {processor_id}: {e}", exc_info=True)
    
    async def process_graph(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process query using graph-based execution.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Result dictionary with processed output
        """
        start_time = datetime.now()
        
        try:
            # Build graph for query
            execution_graph = self._build_execution_graph(query, context)
            
            # Execute graph nodes in topological order
            results = await self._execute_graph(execution_graph, query, context)
            
            # Aggregate results
            aggregated_result = self._aggregate_results(results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.stats["graphs_processed"] += 1
            self.stats["execution_time_avg"] = (
                (self.stats["execution_time_avg"] * (self.stats["graphs_processed"] - 1) + execution_time) /
                self.stats["graphs_processed"]
            )
            
            logger.info(f"Graph processing completed in {execution_time:.3f}s")
            
            return {
                "result": aggregated_result,
                "execution_time": execution_time,
                "nodes_executed": len(results),
                "graph_size": len(execution_graph.nodes())
            }
            
        except Exception as e:
            logger.error(f"Error processing graph: {e}", exc_info=True)
            return {
                "result": None,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _build_execution_graph(self, query: str, context: Optional[Dict[str, Any]]) -> nx.DiGraph:
        """Build execution graph for query"""
        try:
            # Create a subgraph for this query
            execution_graph = nx.DiGraph()
            
            # Add relevant processors based on query
            query_lower = query.lower()
            
            # Add processors based on query content
            if any(word in query_lower for word in ["research", "analyze", "study"]):
                if "surveyor" in self.processors:
                    execution_graph.add_node("surveyor", processor=self.processors["surveyor"])
                if "synthesist" in self.processors:
                    execution_graph.add_node("synthesist", processor=self.processors["synthesist"])
                    if "surveyor" in execution_graph:
                        execution_graph.add_edge("surveyor", "synthesist")
            
            if any(word in query_lower for word in ["code", "program", "function"]):
                if "weaver" in self.processors:
                    execution_graph.add_node("weaver", processor=self.processors["weaver"])
            
            # If no specific processors, use default
            if len(execution_graph.nodes()) == 0:
                if "surveyor" in self.processors:
                    execution_graph.add_node("surveyor", processor=self.processors["surveyor"])
            
            logger.debug(f"Built execution graph with {len(execution_graph.nodes())} nodes")
            
            return execution_graph
            
        except Exception as e:
            logger.error(f"Error building execution graph: {e}", exc_info=True)
            return nx.DiGraph()
    
    async def _execute_graph(self, graph: nx.DiGraph, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute graph nodes in topological order"""
        results = {}
        
        try:
            # Get topological order
            try:
                topo_order = list(nx.topological_sort(graph))
            except nx.NetworkXError:
                # Graph has cycles, use simple order
                topo_order = list(graph.nodes())
            
            # Execute nodes in order
            for node_id in topo_order:
                try:
                    node_data = graph.nodes[node_id]
                    processor = node_data.get("processor")
                    
                    if processor:
                        # Execute processor
                        result = await self._execute_processor(processor, query, context)
                        results[node_id] = result
                        self.stats["nodes_executed"] += 1
                        
                        # Update context with result
                        if context is None:
                            context = {}
                        context[f"{node_id}_result"] = result
                    
                except Exception as e:
                    logger.error(f"Error executing node {node_id}: {e}", exc_info=True)
                    results[node_id] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing graph: {e}", exc_info=True)
            return {}
    
    async def _execute_processor(self, processor: Any, query: str, context: Optional[Dict[str, Any]]) -> Any:
        """Execute a single processor"""
        try:
            # Check if processor is callable
            if callable(processor):
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(query, context)
                else:
                    result = processor(query, context)
            else:
                # Processor is a dict or object, return as-is
                result = processor
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing processor: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Any:
        """Aggregate results from multiple processors"""
        try:
            # Simple aggregation: combine all results
            aggregated = {
                "results": results,
                "count": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}", exc_info=True)
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            **self.stats,
            "processors_count": len(self.processors),
            "graph_nodes": len(self.graph.nodes()),
            "graph_edges": len(self.graph.edges())
        }

