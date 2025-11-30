from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List, Optional
import json
import networkx as nx

from .config import IceburgConfig


class KnowledgeGraph:
    def __init__(self, cfg: IceburgConfig):
        self._cfg = cfg
        self._graph_dir: Path = cfg.data_dir / "graph"
        self._graph_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self._graph_dir / "iceburg.graphml"
        self._G = nx.MultiDiGraph()
        if self._graph_path.exists():
            try:
                self._G = nx.read_graphml(self._graph_path)
            except Exception:
                self._G = nx.MultiDiGraph()

    def search_nodes(self, concept: str, node_type: str = None) -> List[Dict[str, Any]]:
        """Search for nodes matching the concept and optionally node_type."""
        results = []
        for node, data in self._G.nodes(data=True):
            if concept.lower() in node.lower():
                if node_type is None or data.get("type") == node_type:
                    results.append({"node": node, "data": data})
        return results

    def add_synthesis(self, title: str, domains: Iterable[str], principle: str, evidence: Iterable[Tuple[str, str]]):
        if not self._G.has_node(title):
            self._G.add_node(title, type="principle")
        for domain in domains:
            if not self._G.has_node(domain):
                self._G.add_node(domain, type="domain")
            self._G.add_edge(title, domain, type="touches")
        if principle:
            if not self._G.has_node(principle):
                self._G.add_node(principle, type="principle")
            self._G.add_edge(title, principle, type="implies")
        for a, b in evidence:
            if not self._G.has_node(a):
                self._G.add_node(a, type="evidence")
            if not self._G.has_node(b):
                self._G.add_node(b, type="evidence")
            self._G.add_edge(a, b, type="supports")
        self._persist()
    
    def add_matrix_awareness(self, matrix_id: str, matrix_type: str, nodes: List[str], edges: List[tuple], patterns: List[Dict[str, Any]] = None):
        """
        Add matrix awareness to knowledge graph.
        
        Args:
            matrix_id: Matrix identifier
            matrix_type: Type of matrix (e.g., "celestial", "economic", "social")
            nodes: List of matrix nodes
            edges: List of matrix edges (tuples)
            patterns: List of patterns detected in matrix
        """
        # Add matrix as a node
        if not self._G.has_node(matrix_id):
            self._G.add_node(matrix_id, type="matrix", matrix_type=matrix_type)
        
        # Add matrix nodes
        for node in nodes:
            if not self._G.has_node(node):
                self._G.add_node(node, type="matrix_node")
            self._G.add_edge(matrix_id, node, type="contains")
        
        # Add matrix edges
        for edge in edges:
            if len(edge) >= 2:
                source, target = edge[0], edge[1]
                if not self._G.has_node(source):
                    self._G.add_node(source, type="matrix_node")
                if not self._G.has_node(target):
                    self._G.add_node(target, type="matrix_node")
                self._G.add_edge(source, target, type="matrix_edge", matrix_id=matrix_id)
        
        # Add patterns
        if patterns:
            for pattern in patterns:
                pattern_id = pattern.get("pattern_id", f"pattern_{len(patterns)}")
                if not self._G.has_node(pattern_id):
                    self._G.add_node(pattern_id, type="pattern", **pattern)
                self._G.add_edge(matrix_id, pattern_id, type="has_pattern")
        
        self._persist()

    def recurring_principles(self, min_degree: int = 2) -> List[str]:
        result: List[str] = []
        for node, data in self._G.nodes(data=True):
            if data.get("type") == "principle" and self._G.degree(node) >= min_degree:
                result.append(node)
        return result

    def save_principle(self, oracle_json: str, concepts: Iterable[str], scrutineer_json: Optional[str] = None) -> None:
        """Persist a principle and link it to concept nodes and evidence levels.
        - oracle_json: JSON with keys including principle_name, one_sentence_summary, framing, domains
        - concepts: list of concept strings extracted from current query
        - scrutineer_json: JSON with keys original_concept, evidence_level, justification (optional)
        """
        try:
            data = json.loads(oracle_json)
        except Exception:
            return
        principle_name = str(data.get("principle_name") or "Emergent Principle").strip()
        summary = str(data.get("one_sentence_summary") or "").strip()
        framing = str(data.get("framing") or "").strip()
        domains = [str(d).strip() for d in (data.get("domains") or []) if str(d).strip()]
        # Create/update principle node
        if principle_name:
            if not self._G.has_node(principle_name):
                self._G.add_node(principle_name, type="principle")
            # store attributes
            self._G.nodes[principle_name]["summary"] = summary
            self._G.nodes[principle_name]["framing"] = framing
        # Link to domains
        for domain in domains:
            if not self._G.has_node(domain):
                self._G.add_node(domain, type="domain")
            self._G.add_edge(principle_name, domain, type="RELATES_TO")
        # Link to concept nodes
        for concept in concepts:
            concept_n = concept.strip()
            if not concept_n:
                continue
            if not self._G.has_node(concept_n):
                self._G.add_node(concept_n, type="concept")
            self._G.add_edge(principle_name, concept_n, type="DERIVED_FROM")
        # Store scrutineer linkage
        if scrutineer_json:
            try:
                sdata = json.loads(scrutineer_json)
                ev = str(sdata.get("evidence_level") or "").strip()
                just = str(sdata.get("justification") or "").strip()
                self._G.nodes[principle_name]["evidence_level"] = ev
                if just:
                    self._G.nodes[principle_name]["justification"] = just
            except Exception:
                pass
        self._persist()

    def retrieve_relevant_principles(self, concepts: Iterable[str], limit: int = 8) -> List[str]:
        """Return summaries of principles connected to given concepts or domains."""
        seen: set[str] = set()
        results: List[str] = []

        # First, try exact matches
        for concept in concepts:
            c = concept.strip().lower()
            if not c:
                continue

            # Try exact match first
            for node in self._G.nodes():
                node_lower = node.lower()
                if c == node_lower and self._G.nodes[node].get("type") == "concept":
                    # Found a matching concept node, get its principle neighbors
                    for neighbor in self._G.neighbors(node):
                        if self._G.nodes[neighbor].get("type") == "principle" and neighbor not in seen:
                            seen.add(neighbor)
                            summary = self._G.nodes[neighbor].get("summary") or neighbor
                            framing = self._G.nodes[neighbor].get("framing") or ""
                            ev = self._G.nodes[neighbor].get("evidence_level") or ""
                            results.append(f"{neighbor} — {framing} {('['+ev+']') if ev else ''}: {summary}")
                            if len(results) >= limit:
                                return results

        # If no exact matches, try partial matches
        if not results:
            for concept in concepts:
                c = concept.strip().lower()
                if not c:
                    continue

                # Try partial matches on all nodes
                for node in self._G.nodes():
                    node_lower = node.lower()
                    if c in node_lower and len(c) >= 3:  # Only match if concept is at least 3 chars
                        if self._G.nodes[node].get("type") == "principle" and node not in seen:
                            seen.add(node)
                            summary = self._G.nodes[node].get("summary") or node
                            framing = self._G.nodes[node].get("framing") or ""
                            ev = self._G.nodes[node].get("evidence_level") or ""
                            results.append(f"{node} — {framing} {('['+ev+']') if ev else ''}: {summary}")
                            if len(results) >= limit:
                                return results

        return results

    def _persist(self) -> None:
        try:
            nx.write_graphml(self._G, self._graph_path)
        except Exception:
            pass
