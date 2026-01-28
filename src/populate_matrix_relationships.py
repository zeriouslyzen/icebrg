#!/usr/bin/env python3
"""
Populate the relationships table in matrix.db by scanning 
the original OpenSanctions JSON for relationship entities.

OpenSanctions stores relationships as separate entities with schemas:
- Family: person -> relative
- Ownership: owner -> asset
- Directorship: director -> organization  
- Succession: predecessor -> successor
- Occupancy: holder -> post
"""

import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

DB_PATH = Path.home() / "Documents" / "iceburg_matrix" / "matrix.db"
JSON_PATH = Path.home() / "Documents" / "iceburg_matrix" / "opensanctions" / "opensanctions_sanctions.json"

# Schema -> (source_prop, target_prop, rel_type)
RELATIONSHIP_SCHEMAS = {
    "Family": [
        ("person", "relative", "FAMILY_OF"),
    ],
    "Ownership": [
        ("owner", "asset", "OWNS"),
    ],
    "Directorship": [
        ("director", "organization", "DIRECTOR_OF"),
    ],
    "Succession": [
        ("predecessor", "successor", "SUCCEEDED_BY"),
    ],
    "Occupancy": [
        ("holder", "post", "HOLDS_POSITION"),
    ],
    "Membership": [
        ("member", "organization", "MEMBER_OF"),
    ],
    "Associate": [
        ("person", "associate", "ASSOCIATED_WITH"),
    ],
    "Representation": [
        ("agent", "client", "REPRESENTS"),
    ],
    "UnknownLink": [
        ("subject", "object", "LINKED_TO"),
    ],
}


def wikidata_to_osanc(wikidata_id: str) -> str:
    """Convert Wikidata ID (Q12345) to our entity ID format."""
    if wikidata_id.startswith("Q"):
        return f"osanc_{wikidata_id}"
    return wikidata_id


def extract_relationships():
    """Extract relationships from OpenSanctions JSON and insert into SQLite."""
    
    print(f"🔗 Connecting to {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing relationships
    cursor.execute("DELETE FROM relationships")
    conn.commit()
    print("🗑️  Cleared existing relationships")
    
    print(f"📖 Scanning {JSON_PATH}")
    
    relationships_found = 0
    relationships_batch = []
    lines_read = 0
    
    start_time = datetime.now()
    
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            lines_read += 1
            
            try:
                entity = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            schema = entity.get("schema", "")
            
            # Check if this is a relationship entity
            if schema not in RELATIONSHIP_SCHEMAS:
                continue
                
            props = entity.get("properties", {})
            
            for source_prop, target_prop, rel_type in RELATIONSHIP_SCHEMAS[schema]:
                sources = props.get(source_prop, [])
                targets = props.get(target_prop, [])
                
                if isinstance(sources, str):
                    sources = [sources]
                if isinstance(targets, str):
                    targets = [targets]
                
                # Create relationship for each source-target pair
                for source_id in sources:
                    for target_id in targets:
                        if not source_id or not target_id:
                            continue
                            
                        # Convert Wikidata IDs to our format
                        source_id = wikidata_to_osanc(source_id)
                        target_id = wikidata_to_osanc(target_id)
                        
                        rel_id = f"rel_{uuid.uuid4().hex[:12]}"
                        
                        # Get additional context
                        extra_props = {}
                        if "relationship" in props:
                            extra_props["relationship"] = props["relationship"]
                        if "startDate" in props:
                            extra_props["startDate"] = props["startDate"]
                        if "endDate" in props:
                            extra_props["endDate"] = props["endDate"]
                        
                        relationships_batch.append((
                            rel_id,
                            source_id,
                            target_id,
                            rel_type,
                            json.dumps(extra_props) if extra_props else None,
                            "opensanctions",
                        ))
                        relationships_found += 1
            
            # Insert batch
            if len(relationships_batch) >= 10000:
                cursor.executemany("""
                    INSERT OR IGNORE INTO relationships 
                    (relationship_id, source_id, target_id, relationship_type, properties, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, relationships_batch)
                conn.commit()
                relationships_batch = []
                
            if lines_read % 100000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"  Scanned {lines_read:,} lines | {relationships_found:,} relationships | {elapsed:.0f}s")
    
    # Insert remaining
    if relationships_batch:
        cursor.executemany("""
            INSERT OR IGNORE INTO relationships 
            (relationship_id, source_id, target_id, relationship_type, properties, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, relationships_batch)
        conn.commit()
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM relationships")
    final_count = cursor.fetchone()[0]
    
    # Show breakdown by type
    cursor.execute("SELECT relationship_type, COUNT(*) FROM relationships GROUP BY relationship_type ORDER BY COUNT(*) DESC")
    breakdown = cursor.fetchall()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"📈 Inserted {final_count:,} relationships")
    print("\n📊 Breakdown by type:")
    for rel_type, count in breakdown:
        print(f"   {rel_type}: {count:,}")
    
    conn.close()
    return final_count


if __name__ == "__main__":
    extract_relationships()
