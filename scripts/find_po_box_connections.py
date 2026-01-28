#!/usr/bin/env python3
"""
Find All Properties Using Strombeck PO Box
Uncover hidden connections through mailing address
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def find_po_box_connections():
    """Find all properties using Strombeck PO Box."""
    print('=' * 80)
    print('PO BOX 37 CONNECTION ANALYSIS')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    po_box = "PO BOX 37 EUREKA CA 95502"
    po_box_id = "address_po_box_37_eureka"
    
    # Find all properties using PO Box 37
    cursor.execute('''
        SELECT e1.name as property, e1.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.relationship_type = 'MAILING_ADDRESS'
        AND e2.entity_id = ?
    ''', (po_box_id,))
    
    po_box_properties = cursor.fetchall()
    print(f'\nProperties using PO BOX 37 ({len(po_box_properties)}):')
    total_value = 0
    for prop, props_json in po_box_properties:
        print(f'  - {prop}')
        if props_json:
            p = json.loads(props_json) if isinstance(props_json, str) else props_json
            if 'total_value' in p:
                value = p['total_value']
                total_value += value
                print(f'    Value: ${value:,}')
    
    print(f'\n  TOTAL VALUE: ${total_value:,}')
    
    # Find all Strombeck connections
    cursor.execute('''
        SELECT e1.name as entity1, e2.name as entity2, r.relationship_type
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE (e1.name LIKE '%Strombeck%' OR e2.name LIKE '%Strombeck%')
        AND (e1.name LIKE '%PO BOX%' OR e2.name LIKE '%PO BOX%')
    ''')
    
    print(f'\n\nStrombeck-PO Box connections:')
    for e1, e2, rel_type in cursor.fetchall():
        print(f'  {e1[:30]:30s} --[{rel_type}]--> {e2[:30]}')
    
    # Find trust connections
    cursor.execute('''
        SELECT e1.name as entity1, e2.name as entity2, r.relationship_type, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE e1.name LIKE '%Trust%' OR e2.name LIKE '%Trust%'
    ''')
    
    print(f'\n\nTrust Connections:')
    trusts = cursor.fetchall()
    for e1, e2, rel_type, props_json in trusts:
        print(f'  {e1[:30]:30s} --[{rel_type}]--> {e2[:30]}')
        if props_json:
            p = json.loads(props_json) if isinstance(props_json, str) else props_json
            if 'investigation_note' in p:
                note = p['investigation_note']
                print(f'    Note: {note}')
    
    conn.close()
    
    return {
        'po_box_properties': len(po_box_properties),
        'total_value': total_value,
        'trust_connections': len(trusts)
    }

if __name__ == "__main__":
    find_po_box_connections()
