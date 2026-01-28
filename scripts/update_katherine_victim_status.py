#!/usr/bin/env python3
"""
Update Katherine Strombeck Status - Mark as Victim
Refocus investigation on Erik Strombeck's asset hiding activities
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def update_katherine_status():
    """Update Katherine Strombeck status to victim."""
    print('=' * 80)
    print('UPDATING KATHERINE STROMBECK STATUS - VICTIM')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    katherine_id = 'person_katherine_strombeck'
    
    # Update Katherine's entity properties
    cursor.execute('SELECT properties FROM entities WHERE entity_id = ?', (katherine_id,))
    result = cursor.fetchone()
    
    if result:
        props = json.loads(result[0]) if isinstance(result[0], str) else result[0]
    else:
        props = {}
    
    # Update properties to mark as victim
    props.update({
        'investigation_status': 'victim',
        'investigation_note': 'Victim - Property transferred/sold as part of Erik Strombeck asset hiding scheme',
        'investigation_flag': 'victim_of_asset_hiding',
        'updated': datetime.now().isoformat()
    })
    
    cursor.execute("""
        UPDATE entities 
        SET properties = ?
        WHERE entity_id = ?
    """, (json.dumps(props), katherine_id))
    
    # Update relationships to reflect victim status
    cursor.execute("""
        SELECT source_id, target_id, relationship_type, properties
        FROM relationships
        WHERE source_id = ? OR target_id = ?
    """, (katherine_id, katherine_id))
    
    relationships = cursor.fetchall()
    
    for source_id, target_id, rel_type, rel_props_json in relationships:
        rel_props = json.loads(rel_props_json) if isinstance(rel_props_json, str) else rel_props_json
        
        # Update SOLD_TO relationship to reflect victim status
        if rel_type == 'SOLD_TO' and source_id == katherine_id:
            rel_props.update({
                'investigation_note': 'Victim property transfer - Katherine forced/coerced to transfer property to Erik',
                'investigation_flag': 'victim_transfer',
                'updated': datetime.now().isoformat()
            })
            
            cursor.execute("""
                UPDATE relationships
                SET properties = ?
                WHERE source_id = ? AND target_id = ? AND relationship_type = ?
            """, (json.dumps(rel_props), source_id, target_id, rel_type))
        
        # Update FAMILY_OF relationship
        if rel_type == 'FAMILY_OF':
            rel_props.update({
                'investigation_note': 'Spouse relationship - Katherine is Erik\'s wife and victim',
                'investigation_flag': 'spouse_victim',
                'updated': datetime.now().isoformat()
            })
            
            cursor.execute("""
                UPDATE relationships
                SET properties = ?
                WHERE source_id = ? AND target_id = ? AND relationship_type = ?
            """, (json.dumps(rel_props), source_id, target_id, rel_type))
    
    conn.commit()
    conn.close()
    
    print(f'\n✅ Updated Katherine Strombeck status to VICTIM')
    print(f'✅ Updated relationships to reflect victim status')
    print(f'✅ Investigation refocused on Erik Strombeck\'s asset hiding activities')

if __name__ == "__main__":
    update_katherine_status()
