#!/usr/bin/env python3
"""
Import Handwritten Notes Evidence
Aggregates new evidence from handwritten notes image
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def import_handwritten_evidence():
    """Import evidence from handwritten notes."""
    print('=' * 80)
    print('IMPORTING HANDWRITTEN NOTES EVIDENCE')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # New people identified
    new_people = [
        {
            'name': 'Will Startare',
            'entity_id': 'person_will_startare',
            'role': 'business_partner',
            'note': 'Bought out Erik\'s interest in 3114 Nevada St'
        },
        {
            'name': 'Corey Taylor',
            'entity_id': 'person_corey_taylor',
            'role': 'business_partner',
            'note': 'Bought out Erik\'s interest in 3114 Nevada St'
        },
        {
            'name': 'Matt Allen',
            'entity_id': 'person_matt_allen',
            'role': 'aa_sponsor',
            'note': 'Erik\'s A.A. Sponsor, first purchased 965 W Harris St'
        },
    ]
    
    # New properties identified
    new_properties = [
        {
            'address': '3114 Nevada St',
            'entity_id': 'address_3114_nevada_st',
            'city': 'Arcata',
            'state': 'CA',
            'note': 'Connected with 965 W Harris St, Erik sold interest to partners'
        },
        {
            'address': '2145 Western Avenue Arcata CA 95521',
            'entity_id': 'address_2145_western_ave_arcata',
            'city': 'Arcata',
            'state': 'CA',
            'note': 'Erik owned, sold 01/16/2026'
        },
        {
            'address': '2155 Western Avenue Arcata CA 95521',
            'entity_id': 'address_2155_western_ave_arcata',
            'city': 'Arcata',
            'state': 'CA',
            'note': 'Erik built, Erik owns, pending sale'
        },
        {
            'address': '2141 Western Avenue Arcata CA 95521',
            'entity_id': 'address_2141_western_ave_arcata',
            'city': 'Arcata',
            'state': 'CA',
            'note': 'Erik built, Erik owned, sold 12/11/2025'
        },
        {
            'address': '2149 Western Avenue Arcata CA 95521',
            'entity_id': 'address_2149_western_ave_arcata',
            'city': 'Arcata',
            'state': 'CA',
            'note': 'Erik built, Erik owned, sold 12/06/2025'
        },
    ]
    
    # Create people entities
    print('\n\nCreating people entities...')
    for person in new_people:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
                VALUES (?, ?, ?, ?, ?)
            """, (
                person['entity_id'],
                person['name'],
                'person',
                'handwritten_notes_evidence',
                json.dumps({
                    'role': person['role'],
                    'note': person['note'],
                    'investigation_relevant': True
                })
            ))
            print(f'  ✅ Created: {person["name"]} ({person["role"]})')
        except Exception as e:
            print(f'  ⚠️  Error creating {person["name"]}: {e}')
    
    # Create property entities
    print('\n\nCreating property entities...')
    for prop in new_properties:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
                VALUES (?, ?, ?, ?, ?)
            """, (
                prop['entity_id'],
                prop['address'],
                'address',
                'handwritten_notes_evidence',
                json.dumps({
                    'city': prop['city'],
                    'state': prop['state'],
                    'note': prop['note'],
                    'investigation_relevant': True
                })
            ))
            print(f'  ✅ Created: {prop["address"]}')
        except Exception as e:
            print(f'  ⚠️  Error creating {prop["address"]}: {e}')
    
    # Create relationships
    print('\n\nCreating relationships...')
    
    erik_id = 'person_erik_strombeck'
    
    # Erik OWNS properties
    erik_properties = [
        'address_2155_western_ave_arcata',  # Erik owns
        'address_2145_western_ave_arcata',  # Erik owned
        'address_2141_western_ave_arcata',  # Erik owned
        'address_2149_western_ave_arcata',  # Erik owned (we already have this)
    ]
    
    for prop_id in erik_properties:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                VALUES (?, ?, ?, ?)
            """, (
                erik_id,
                prop_id,
                'OWNS',
                json.dumps({
                    'source': 'handwritten_notes',
                    'status': 'owned_or_built',
                    'investigation_note': 'Erik owned/built these properties'
                })
            ))
            print(f'  ✅ Erik --[OWNS]--> {prop_id}')
        except Exception as e:
            pass
    
    # Erik BUILT properties
    erik_built = [
        'address_2155_western_ave_arcata',
        'address_2141_western_ave_arcata',
        'address_2149_western_ave_arcata',
    ]
    
    for prop_id in erik_built:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                VALUES (?, ?, ?, ?)
            """, (
                erik_id,
                prop_id,
                'BUILT',
                json.dumps({
                    'source': 'handwritten_notes',
                    'investigation_note': 'Erik built these properties'
                })
            ))
            print(f'  ✅ Erik --[BUILT]--> {prop_id}')
        except Exception as e:
            pass
    
    # Erik SOLD interest in 3114 Nevada St to Will Startare & Corey Taylor
    nevada_st_id = 'address_3114_nevada_st'
    will_id = 'person_will_startare'
    corey_id = 'person_corey_taylor'
    
    # Erik owned Nevada St
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        erik_id,
        nevada_st_id,
        'OWNS',
        json.dumps({
            'source': 'handwritten_notes',
            'status': 'sold_interest',
            'investigation_note': 'Erik owned, sold interest to partners',
            'sale_type': 'buyout'
        })
    ))
    
    # Erik SOLD_TO Will Startare
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        erik_id,
        will_id,
        'SOLD_TO',
        json.dumps({
            'property': '3114 Nevada St',
            'source': 'handwritten_notes',
            'investigation_note': 'Erik sold interest, Will had to come up with a lot of money',
            'transaction_type': 'buyout',
            'investigation_flag': 'asset_hiding_suspected'
        })
    ))
    
    # Erik SOLD_TO Corey Taylor
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        erik_id,
        corey_id,
        'SOLD_TO',
        json.dumps({
            'property': '3114 Nevada St',
            'source': 'handwritten_notes',
            'investigation_note': 'Erik sold interest, Corey had to come up with a lot of money',
            'transaction_type': 'buyout',
            'investigation_flag': 'asset_hiding_suspected'
        })
    ))
    
    # Will and Corey PURCHASED_FROM Erik (via Nevada St)
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        will_id,
        nevada_st_id,
        'PURCHASED_FROM',
        json.dumps({
            'source': 'handwritten_notes',
            'seller': 'Erik Strombeck',
            'investigation_note': 'Buyout of Erik\'s interest'
        })
    ))
    
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        corey_id,
        nevada_st_id,
        'PURCHASED_FROM',
        json.dumps({
            'source': 'handwritten_notes',
            'seller': 'Erik Strombeck',
            'investigation_note': 'Buyout of Erik\'s interest'
        })
    ))
    
    # Will and Corey are business partners
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        will_id,
        corey_id,
        'BUSINESS_PARTNER',
        json.dumps({
            'source': 'handwritten_notes',
            'property': '3114 Nevada St',
            'investigation_note': 'Partners who bought out Erik'
        })
    ))
    
    # Matt Allen - Erik's A.A. Sponsor relationship
    matt_id = 'person_matt_allen'
    harris_st_id = 'address_965_w_harris_eureka'
    
    # Matt Allen first purchased 965 W Harris St
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        matt_id,
        harris_st_id,
        'PURCHASED_FROM',
        json.dumps({
            'source': 'handwritten_notes',
            'investigation_note': 'First purchaser, divided into 4 parcels',
            'role': 'original_purchaser'
        })
    ))
    
    # Erik's relationship with Matt (A.A. Sponsor)
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        erik_id,
        matt_id,
        'ASSOCIATED_WITH',
        json.dumps({
            'source': 'handwritten_notes',
            'relationship_type': 'aa_sponsor',
            'investigation_note': 'Matt is Erik\'s A.A. Sponsor'
        })
    ))
    
    # Connection: 3114 Nevada St connected with 965 W Harris St
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        nevada_st_id,
        harris_st_id,
        'CONNECTED_WITH',
        json.dumps({
            'source': 'handwritten_notes',
            'investigation_note': 'Properties are connected, Erik owned both'
        })
    ))
    
    # 965 W Harris St divided into 4 parcels - Erik may still own
    cursor.execute("""
        UPDATE entities
        SET properties = json_set(properties, '$.subdivided', 'true', '$.parcels', '4', '$.status', 'may_still_own', '$.activity', 'building_units')
        WHERE entity_id = ?
    """, (harris_st_id,))
    
    print('  ✅ Created relationships')
    
    conn.commit()
    
    print(f'\n\n✅ Imported handwritten notes evidence')
    print(f'   - {len(new_people)} people')
    print(f'   - {len(new_properties)} properties')
    print(f'   - Multiple relationships created')
    
    conn.close()

if __name__ == "__main__":
    import_handwritten_evidence()
