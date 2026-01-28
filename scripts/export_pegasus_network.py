"""
Export a sanitized Pegasus network JSON from the local Matrix database.

This is designed to produce a generic dataset for the Pegasus UI
without embedding case-specific investigations (e.g. Strombeck).

Output: frontend/public/pegasus_network.json
"""

import json
from pathlib import Path

from iceburg.colossus.matrix_store import MatrixStore


EXCLUDED_KEYWORDS = [
    "strombeck",
]


def is_excluded_entity(name: str, entity_id: str) -> bool:
    """Return True if the entity should be excluded from the public Pegasus graph."""
    key = f"{name} {entity_id}".lower()
    return any(keyword in key for keyword in EXCLUDED_KEYWORDS)


def export_sanitized_network(output_path: Path, center_limit: int = 200) -> None:
    """
    Export a compact network suitable for Pegasus static visualization.

    Strategy:
    - Take the top N central entities from COLOSSUS (via MatrixStore get_network around them)
      by simply walking networks around a seed list derived from a broad search.
    - Exclude any entities whose name/id matches EXCLUDED_KEYWORDS.
    """
    store = MatrixStore()

    # Seed: broad search for generic terms to pull in a representative slice
    seeds = []
    for term in ("LLC", "Inc", "Ltd", "Trust", "Capital", "Holdings"):
        results = store.search(term, limit=50)
        for e in results:
            if not is_excluded_entity(e.name or "", e.id or ""):
                seeds.append(e.id)
            if len(seeds) >= center_limit:
                break
        if len(seeds) >= center_limit:
            break

    nodes_map = {}
    edges = []

    for eid in seeds:
        network = store.get_network(entity_id=eid, depth=1, limit=200)
        for n in network.get("nodes", []):
            if is_excluded_entity(n.get("name", ""), n.get("id", "")):
                continue
            nodes_map[n["id"]] = {
                "id": n["id"],
                "name": n.get("name"),
                "type": n.get("type"),
                "countries": n.get("countries", []),
                "sanctions_count": n.get("sanctions_count", 0),
            }
        for e in network.get("edges", []):
            src = e.get("source")
            tgt = e.get("target")
            if not src or not tgt:
                continue
            if src not in nodes_map or tgt not in nodes_map:
                continue
            edges.append(
                {
                    "id": e.get("id"),
                    "source": src,
                    "target": tgt,
                    "type": e.get("type") or e.get("relationship_type") or "CONNECTED_TO",
                }
            )

    output = {
        "nodes": list(nodes_map.values()),
        "edges": edges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote Pegasus network: {output_path} (nodes={len(output['nodes'])}, edges={len(output['edges'])})")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output = project_root / "frontend" / "public" / "pegasus_network.json"
    export_sanitized_network(output)

