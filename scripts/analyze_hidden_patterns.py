#!/usr/bin/env python3
"""
Analyze Hidden Patterns in Strombeck Network
Uncovers connections, asset hiding patterns, and hidden relationships
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def analyze_hidden_patterns():
    """Analyze network for hidden patterns."""
    print('=' * 80)
    print('HIDDEN PATTERN ANALYSIS - STROMBECK INVESTIGATION')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # Get Erik's complete network
    network = ms.get_network('person_erik_strombeck', depth=4, limit=200)
    
    print(f'\nNetwork Size: {len(network["nodes"])} nodes, {len(network["edges"])} edges')
    
    # Pattern 1: Property ownership chain
    print('\n\n🔍 PATTERN 1: PROPERTY OWNERSHIP CHAIN')
    print('=' * 80)
    
    cursor.execute('''
        SELECT e1.name as owner, e2.name as property, r.relationship_type, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE e1.entity_id = 'person_erik_strombeck'
        AND e2.entity_type = 'address'
        AND r.relationship_type IN ('OWNS', 'BUILT', 'SOLD_TO')
        ORDER BY r.relationship_type
    ''')
    
    erik_properties = cursor.fetchall()
    print(f'\nErik\'s Property Connections:')
    for owner, prop, rel_type, props_json in erik_properties:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        note = props.get('investigation_note', '')
        print(f'  Erik --[{rel_type:10s}]--> {prop[:50]}')
        if note:
            print(f'    Note: {note}')
    
    # Pattern 2: Business partner network
    print('\n\n🔍 PATTERN 2: BUSINESS PARTNER NETWORK')
    print('=' * 80)
    
    cursor.execute('''
        SELECT e1.name as person1, e2.name as person2, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.relationship_type IN ('BUSINESS_PARTNER', 'SOLD_TO', 'PURCHASED_FROM')
        AND (e1.name LIKE '%Strombeck%' OR e2.name LIKE '%Strombeck%'
             OR e1.name IN ('Will Startare', 'Corey Taylor', 'Matt Allen')
             OR e2.name IN ('Will Startare', 'Corey Taylor', 'Matt Allen'))
    ''')
    
    partners = cursor.fetchall()
    print(f'\nBusiness Partner Connections:')
    for p1, p2, props_json in partners:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        note = props.get('investigation_note', '')
        print(f'  {p1[:30]:30s} <--> {p2[:30]}')
        if note:
            print(f'    Note: {note}')
    
    # Pattern 3: Property subdivision pattern
    print('\n\n🔍 PATTERN 3: PROPERTY SUBDIVISION PATTERN')
    print('=' * 80)
    
    cursor.execute('''
        SELECT name, properties
        FROM entities
        WHERE entity_type = 'address'
        AND (properties LIKE '%subdivided%' OR properties LIKE '%parcels%'
             OR name LIKE '%Harris%' OR name LIKE '%Nevada%')
    ''')
    
    subdivided = cursor.fetchall()
    print(f'\nSubdivided Properties:')
    for name, props_json in subdivided:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        if props.get('subdivided') or props.get('parcels'):
            print(f'  {name}')
            print(f'    Parcels: {props.get("parcels", "N/A")}')
            print(f'    Status: {props.get("status", "N/A")}')
            print(f'    Activity: {props.get("activity", "N/A")}')
    
    # Pattern 4: Buyout transactions
    print('\n\n🔍 PATTERN 4: BUYOUT TRANSACTIONS')
    print('=' * 80)
    
    cursor.execute('''
        SELECT e1.name as seller, e2.name as buyer, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.relationship_type = 'SOLD_TO'
        AND r.properties LIKE '%buyout%'
    ''')
    
    buyouts = cursor.fetchall()
    print(f'\nBuyout Transactions:')
    for seller, buyer, props_json in buyouts:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        property_name = props.get('property', 'Unknown')
        note = props.get('investigation_note', '')
        print(f'  {seller[:30]:30s} --[BUYOUT]--> {buyer[:30]}')
        print(f'    Property: {property_name}')
        if note:
            print(f'    Note: {note}')
    
    # Pattern 5: Western Avenue property cluster
    print('\n\n🔍 PATTERN 5: WESTERN AVENUE PROPERTY CLUSTER')
    print('=' * 80)
    
    cursor.execute('''
        SELECT name, properties
        FROM entities
        WHERE entity_type = 'address'
        AND name LIKE '%Western%'
        ORDER BY name
    ''')
    
    western_props = cursor.fetchall()
    print(f'\nWestern Avenue Properties ({len(western_props)}):')
    for name, props_json in western_props:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        note = props.get('note', '')
        print(f'  {name}')
        if note:
            print(f'    {note}')
    
    # Pattern 6: Connected properties (Nevada St + Harris St)
    print('\n\n🔍 PATTERN 6: CONNECTED PROPERTIES NETWORK')
    print('=' * 80)
    
    cursor.execute('''
        SELECT e1.name as prop1, e2.name as prop2, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.relationship_type = 'CONNECTED_WITH'
        AND e1.entity_type = 'address'
        AND e2.entity_type = 'address'
    ''')
    
    connected = cursor.fetchall()
    print(f'\nConnected Properties:')
    for p1, p2, props_json in connected:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        note = props.get('investigation_note', '')
        print(f'  {p1[:40]:40s} <--> {p2[:40]}')
        if note:
            print(f'    {note}')
    
    # Pattern 7: A.A. Sponsor connection
    print('\n\n🔍 PATTERN 7: A.A. SPONSOR RELATIONSHIP')
    print('=' * 80)
    
    cursor.execute('''
        SELECT e1.name as person1, e2.name as person2, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.properties LIKE '%aa_sponsor%' OR r.properties LIKE '%A.A.%'
    ''')
    
    aa_connections = cursor.fetchall()
    print(f'\nA.A. Sponsor Connections:')
    for p1, p2, props_json in aa_connections:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        rel_type = props.get('relationship_type', 'ASSOCIATED_WITH')
        note = props.get('investigation_note', '')
        print(f'  {p1[:30]:30s} --[{rel_type}]--> {p2[:30]}')
        if note:
            print(f'    {note}')
    
    # Pattern 8: Transaction timeline analysis
    print('\n\n🔍 PATTERN 8: TRANSACTION TIMELINE ANALYSIS')
    print('=' * 80)
    
    cursor.execute('''
        SELECT r.properties, r.relationship_type, e1.name as source, e2.name as target
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.relationship_type IN ('SOLD_TO', 'PURCHASED_FROM')
        AND (e1.name LIKE '%Strombeck%' OR e2.name LIKE '%Strombeck%'
             OR e1.name IN ('Will Startare', 'Corey Taylor', 'Matt Allen')
             OR e2.name IN ('Will Startare', 'Corey Taylor', 'Matt Allen'))
        ORDER BY r.properties
    ''')
    
    transactions = cursor.fetchall()
    print(f'\nTransaction Timeline:')
    
    pre_incident = []
    post_incident = []
    
    for props_json, rel_type, source, target in transactions:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        date = props.get('date', 'Unknown')
        
        tx_info = f'{source[:25]:25s} --[{rel_type}]--> {target[:25]} ({date})'
        
        if date != 'Unknown' and date < '2022-11':
            pre_incident.append((date, tx_info))
        elif date != 'Unknown' and date >= '2022-11':
            post_incident.append((date, tx_info))
        else:
            # Check investigation notes for timing clues
            note = props.get('investigation_note', '')
            if 'just sold' in note.lower() or 'buyout' in note.lower():
                post_incident.append(('Recent', tx_info))
    
    print('\n  BEFORE NOVEMBER 2022:')
    for date, info in sorted(pre_incident):
        print(f'    {info}')
    
    print('\n  AFTER NOVEMBER 2022:')
    for date, info in sorted(post_incident):
        print(f'    🚨 {info}')
    
    # Pattern 9: Hidden ownership through partners
    print('\n\n🔍 PATTERN 9: HIDDEN OWNERSHIP THROUGH PARTNERS')
    print('=' * 80)
    
    print('\nAnalysis:')
    print('  - Erik sold interest in 3114 Nevada St to Will Startare & Corey Taylor')
    print('  - Partners had to "come up with a lot of money" to buy him out')
    print('  - This could be:')
    print('    1. Legitimate buyout')
    print('    2. Asset transfer disguised as buyout')
    print('    3. Erik extracting cash while maintaining hidden control')
    print('    4. Preparation for loss (creditor protection)')
    
    # Pattern 10: Property development pattern
    print('\n\n🔍 PATTERN 10: PROPERTY DEVELOPMENT PATTERN')
    print('=' * 80)
    
    print('\nAnalysis:')
    print('  - Erik BUILT multiple Western Avenue properties')
    print('  - 965 W Harris St divided into 4 parcels')
    print('  - "Building units on them now"')
    print('  - Pattern suggests:')
    print('    1. Property development business')
    print('    2. Subdivision strategy (create multiple assets)')
    print('    3. Ongoing development (income generation)')
    print('    4. Complex ownership structure (harder to track)')
    
    conn.close()
    
    # Save analysis
    analysis = {
        'patterns_identified': 10,
        'erik_properties': len(erik_properties),
        'business_partners': len(partners),
        'buyout_transactions': len(buyouts),
        'western_ave_cluster': len(western_props),
        'pre_incident_transactions': len(pre_incident),
        'post_incident_transactions': len(post_incident),
    }
    
    with open('HIDDEN_PATTERNS_ANALYSIS.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f'\n\n✅ Hidden pattern analysis complete')
    print(f'   Saved to: HIDDEN_PATTERNS_ANALYSIS.json')

if __name__ == "__main__":
    analyze_hidden_patterns()
