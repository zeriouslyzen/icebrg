"""
Matrix-Based Reasoning
Uses matrix understanding to answer queries
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from .matrix_detection import Matrix, MatrixType, Pattern

logger = logging.getLogger(__name__)


@dataclass
class MatrixReasoningResult:
    """Result of matrix reasoning"""
    query: str
    matrices_used: List[str]
    reasoning: str
    conclusions: List[str]
    confidence: float
    metadata: Dict[str, Any]


class MatrixReasoning:
    """
    Uses matrix understanding to answer queries.
    
    For astrology: understands celestial matrix
    For marketing: understands market matrix
    For any query: understands underlying structure
    """
    
    def __init__(self):
        """Initialize matrix reasoning engine."""
        logger.info("Matrix Reasoning Engine initialized")
    
    def use_matrix_knowledge(self, query: str, matrix: Matrix) -> Dict[str, Any]:
        """
        Use matrix knowledge to answer query.
        
        Args:
            query: User query
            matrix: Matrix to use for reasoning
            
        Returns:
            Dictionary with reasoning results
        """
        result = {
            "query": query,
            "matrix": matrix.matrix_id,
            "matrix_type": matrix.matrix_type.value,
            "reasoning": [],
            "conclusions": [],
            "confidence": matrix.confidence
        }
        
        # Reason about query using matrix structure
        if matrix.graph:
            # Analyze graph structure
            nodes = list(matrix.graph.nodes())
            edges = list(matrix.graph.edges())
            
            # Find relevant nodes
            query_lower = query.lower()
            relevant_nodes = [
                node for node in nodes
                if any(word in str(node).lower() for word in query_lower.split())
            ]
            
            if relevant_nodes:
                result["reasoning"].append(f"Found {len(relevant_nodes)} relevant nodes in matrix: {', '.join(relevant_nodes[:5])}")
                
                # Find connections
                connections = []
                for node in relevant_nodes:
                    neighbors = list(matrix.graph.neighbors(node))
                    if neighbors:
                        connections.extend([(node, neighbor) for neighbor in neighbors[:3]])
                
                if connections:
                    result["reasoning"].append(f"Found {len(connections)} connections: {', '.join([f'{c[0]}->{c[1]}' for c in connections[:3]])}")
            
            # Use patterns
            if matrix.patterns:
                relevant_patterns = [
                    pattern for pattern in matrix.patterns
                    if any(word in pattern.description.lower() for word in query_lower.split())
                ]
                
                if relevant_patterns:
                    result["reasoning"].append(f"Found {len(relevant_patterns)} relevant patterns")
                    for pattern in relevant_patterns:
                        result["conclusions"].append(f"Pattern: {pattern.description} (confidence: {pattern.confidence:.2f})")
        
        # Domain-specific reasoning
        if matrix.matrix_type == MatrixType.CELESTIAL:
            result["conclusions"].extend(self._reason_about_astrology(query, matrix))
        elif matrix.matrix_type == MatrixType.ECONOMIC:
            result["conclusions"].extend(self._reason_about_markets(query, matrix))
        elif matrix.matrix_type == MatrixType.SOCIAL:
            result["conclusions"].extend(self._reason_about_social(query, matrix))
        elif matrix.matrix_type == MatrixType.CONCEPTUAL:
            result["conclusions"].extend(self._reason_about_information(query, matrix))
        elif matrix.matrix_type == MatrixType.TEMPORAL:
            result["conclusions"].extend(self._reason_about_temporal(query, matrix))
        
        logger.info(f"Matrix reasoning complete for query: {query[:50]}...")
        return result
    
    def correlate_across_matrices(self, matrices: List[Matrix]) -> Dict[str, Any]:
        """
        Correlate patterns across multiple matrices.
        
        Args:
            matrices: List of matrices to correlate
            
        Returns:
            Dictionary with correlation results
        """
        correlation_result = {
            "matrices": [m.matrix_id for m in matrices],
            "correlations": [],
            "common_patterns": [],
            "cross_matrix_insights": []
        }
        
        # Find common nodes
        all_nodes = {}
        for matrix in matrices:
            for node in matrix.nodes:
                if node not in all_nodes:
                    all_nodes[node] = []
                all_nodes[node].append(matrix.matrix_id)
        
        common_nodes = {node: matrices for node, matrices in all_nodes.items() if len(matrices) > 1}
        if common_nodes:
            correlation_result["correlations"].append(f"Found {len(common_nodes)} common nodes across matrices")
            correlation_result["common_patterns"].extend(list(common_nodes.keys())[:5])
        
        # Find common patterns
        all_patterns = {}
        for matrix in matrices:
            for pattern in matrix.patterns:
                pattern_key = f"{pattern.pattern_type}_{pattern.description[:50]}"
                if pattern_key not in all_patterns:
                    all_patterns[pattern_key] = []
                all_patterns[pattern_key].append(matrix.matrix_id)
        
        common_patterns = {pattern: matrices for pattern, matrices in all_patterns.items() if len(matrices) > 1}
        if common_patterns:
            correlation_result["correlations"].append(f"Found {len(common_patterns)} common patterns across matrices")
            correlation_result["common_patterns"].extend(list(common_patterns.keys())[:5])
        
        # Generate cross-matrix insights
        if len(matrices) >= 2:
            insight = f"Correlating {len(matrices)} matrices reveals interconnected structures"
            correlation_result["cross_matrix_insights"].append(insight)
        
        logger.info(f"Correlated {len(matrices)} matrices, found {len(common_nodes)} common nodes, {len(common_patterns)} common patterns")
        return correlation_result
    
    def understand_matrix_interactions(self, matrix1: Matrix, matrix2: Matrix) -> Dict[str, Any]:
        """
        Understand interactions between two matrices.
        
        Args:
            matrix1: First matrix
            matrix2: Second matrix
            
        Returns:
            Dictionary with interaction analysis
        """
        interaction_result = {
            "matrix1": matrix1.matrix_id,
            "matrix2": matrix2.matrix_id,
            "interactions": [],
            "shared_nodes": [],
            "shared_patterns": [],
            "insights": []
        }
        
        # Find shared nodes
        nodes1 = set(matrix1.nodes)
        nodes2 = set(matrix2.nodes)
        shared_nodes = nodes1.intersection(nodes2)
        
        if shared_nodes:
            interaction_result["shared_nodes"] = list(shared_nodes)
            interaction_result["interactions"].append(f"Found {len(shared_nodes)} shared nodes: {', '.join(list(shared_nodes)[:5])}")
        
        # Find shared patterns
        patterns1 = {p.pattern_id: p for p in matrix1.patterns}
        patterns2 = {p.pattern_id: p for p in matrix2.patterns}
        shared_patterns = set(patterns1.keys()).intersection(set(patterns2.keys()))
        
        if shared_patterns:
            interaction_result["shared_patterns"] = list(shared_patterns)
            interaction_result["interactions"].append(f"Found {len(shared_patterns)} shared patterns")
        
        # Generate insights
        if matrix1.matrix_type == MatrixType.CELESTIAL and matrix2.matrix_type == MatrixType.TEMPORAL:
            interaction_result["insights"].append("Celestial and temporal matrices interact through time-based celestial events")
        elif matrix1.matrix_type == MatrixType.ECONOMIC and matrix2.matrix_type == MatrixType.SOCIAL:
            interaction_result["insights"].append("Economic and social matrices interact through market behavior and social dynamics")
        elif matrix1.matrix_type == MatrixType.CONCEPTUAL and matrix2.matrix_type == MatrixType.ECONOMIC:
            interaction_result["insights"].append("Information and economic matrices interact through data-driven market analysis")
        
        logger.info(f"Analyzed interactions between {matrix1.matrix_id} and {matrix2.matrix_id}")
        return interaction_result
    
    def _reason_about_astrology(self, query: str, matrix: Matrix) -> List[str]:
        """Reason about astrology using celestial matrix."""
        conclusions = []
        
        query_lower = query.lower()
        
        # Check for planet mentions
        planets = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
        mentioned_planets = [planet for planet in planets if planet in query_lower]
        
        if mentioned_planets:
            conclusions.append(f"Query involves celestial bodies: {', '.join(mentioned_planets)}")
        
        # Check for zodiac signs
        signs = ["aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]
        mentioned_signs = [sign for sign in signs if sign in query_lower]
        
        if mentioned_signs:
            conclusions.append(f"Query involves zodiac signs: {', '.join(mentioned_signs)}")
        
        # Matrix understanding
        if matrix.graph:
            conclusions.append(f"Celestial matrix contains {matrix.graph.number_of_nodes()} nodes and {matrix.graph.number_of_edges()} connections")
        
        return conclusions
    
    def _reason_about_markets(self, query: str, matrix: Matrix) -> List[str]:
        """Reason about markets using economic matrix."""
        conclusions = []
        
        query_lower = query.lower()
        
        # Check for market indicators
        indicators = ["price", "volume", "supply", "demand", "trend", "volatility"]
        mentioned_indicators = [indicator for indicator in indicators if indicator in query_lower]
        
        if mentioned_indicators:
            conclusions.append(f"Query involves market indicators: {', '.join(mentioned_indicators)}")
        
        # Matrix understanding
        if matrix.graph:
            conclusions.append(f"Market matrix contains {matrix.graph.number_of_nodes()} nodes and {matrix.graph.number_of_edges()} connections")
        
        return conclusions
    
    def _reason_about_social(self, query: str, matrix: Matrix) -> List[str]:
        """Reason about social dynamics using social matrix."""
        conclusions = []
        
        query_lower = query.lower()
        
        # Check for social concepts
        concepts = ["individual", "group", "community", "society", "communication", "influence", "trust", "cooperation"]
        mentioned_concepts = [concept for concept in concepts if concept in query_lower]
        
        if mentioned_concepts:
            conclusions.append(f"Query involves social concepts: {', '.join(mentioned_concepts)}")
        
        # Matrix understanding
        if matrix.graph:
            conclusions.append(f"Social matrix contains {matrix.graph.number_of_nodes()} nodes and {matrix.graph.number_of_edges()} connections")
        
        return conclusions
    
    def _reason_about_information(self, query: str, matrix: Matrix) -> List[str]:
        """Reason about information using conceptual matrix."""
        conclusions = []
        
        query_lower = query.lower()
        
        # Check for information concepts
        concepts = ["data", "information", "knowledge", "wisdom", "source", "processing", "storage", "retrieval", "analysis"]
        mentioned_concepts = [concept for concept in concepts if concept in query_lower]
        
        if mentioned_concepts:
            conclusions.append(f"Query involves information concepts: {', '.join(mentioned_concepts)}")
        
        # Matrix understanding
        if matrix.graph:
            conclusions.append(f"Information matrix contains {matrix.graph.number_of_nodes()} nodes and {matrix.graph.number_of_edges()} connections")
        
        return conclusions
    
    def _reason_about_temporal(self, query: str, matrix: Matrix) -> List[str]:
        """Reason about temporal patterns using temporal matrix."""
        conclusions = []
        
        query_lower = query.lower()
        
        # Check for temporal concepts
        concepts = ["past", "present", "future", "cause", "effect", "sequence", "pattern", "cycle"]
        mentioned_concepts = [concept for concept in concepts if concept in query_lower]
        
        if mentioned_concepts:
            conclusions.append(f"Query involves temporal concepts: {', '.join(mentioned_concepts)}")
        
        # Matrix understanding
        if matrix.graph:
            conclusions.append(f"Temporal matrix contains {matrix.graph.number_of_nodes()} nodes and {matrix.graph.number_of_edges()} connections")
        
        return conclusions

