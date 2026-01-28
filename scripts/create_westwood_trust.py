#!/usr/bin/env python3
"""
Create Westwood Court Trust Entity
Link trust to property and Erik Strombeck
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def create_westwood_trust():
    """Create trust entity and link to property."""
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    prop_id = "address_2351_westwood_ct_arcata"
    trust_id = "entity_westwood_court_trust"
    erik_id = "person_erik_strombeck"
    
    # Create trust entity
    cursor.execute("""
        INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
        VALUES (?, ?, ?, ?, ?)
    """, (
        trust_id,
        'Westwood Court Trust',
        'entity',
        'property_history',
        json.dumps({
            'property': '2351 WESTWOOD CT A1 ARCATA CA 95521-5151',
            'investigation_note': 'Trust owns $7.5M property, uses Strombeck PO Box',
            'investigation_flag': 'hidden_ownership',
            'trust_type': 'property_holding_trust'
        })
    ))
    
    # Link trust to property
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        trust_id,
        prop_id,
        'OWNS',
        json.dumps({
            'source': 'property_history',
            'investigation_note': 'Trust owns property, Strombeck PO Box confirms control',
            'investigation_flag': 'hidden_ownership'
        })
    ))
    
    # Link Erik to trust (possible beneficiary/trustee)
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        erik_id,
        trust_id,
        'BENEFICIARY_OF',
        json.dumps({
            'source': 'investigation',
            'investigation_note': 'Possible beneficiary/trustee of trust owning $7.5M property',
            'investigation_flag': 'hidden_ownership',
            'evidence': 'PO Box 37 connection'
        })
    ))
    
    # Also link as trustee
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        erik_id,
        trust_id,
        'TRUSTEE_OF',
        json.dumps({
            'source': 'investigation',
            'investigation_note': 'Possible trustee controlling trust',
            'investigation_flag': 'hidden_ownership',
            'evidence': 'PO Box 37 connection'
        })
    ))
    
    conn.commit()
    conn.close()
    
    print('✅ Created Westwood Court Trust entity')
    print('✅ Linked trust to property')
    print('✅ Linked Erik Strombeck as beneficiary/trustee')

if __name__ == "__main__":
    create_westwood_trust()
