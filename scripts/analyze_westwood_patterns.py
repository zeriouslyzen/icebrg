#!/usr/bin/env python3
"""
Analyze Westwood Court Property - Find Hidden Connections
Synthesize with existing network to uncover patterns
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def analyze_westwood_patterns():
    """Analyze Westwood property and find hidden connections."""
    print('=' * 80)
    print('WESTWOOD COURT - HIDDEN PATTERN ANALYSIS')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    prop_id = "address_2351_westwood_ct_arcata"
    po_box_id = "address_po_box_37_eureka"
    
    # Get property details
    cursor.execute('SELECT name, properties FROM entities WHERE entity_id = ?', (prop_id,))
    prop_data = cursor.fetchone()
    
    if prop_data:
        prop_name, props_json = prop_data
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        print(f'\nProperty: {prop_name}')
        print(f'  Value: ${props.get("total_value", 0):,}')
        print(f'  Sale Price (2021): ${props.get("sale_price_2021", 0):,}')
    
    # Find all properties using same PO Box
    print(f'\n\n🔍 PROPERTIES USING STROMBECK PO BOX:')
    cursor.execute('''
        SELECT e1.entity_id, e1.name, e1.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.relationship_type = 'MAILING_ADDRESS'
        AND e2.entity_id = ?
    ''', (po_box_id,))
    
    po_box_properties = cursor.fetchall()
    total_portfolio_value = 0
    
    for prop_eid, prop_name, props_json in po_box_properties:
        if props_json:
            p = json.loads(props_json) if isinstance(props_json, str) else props_json
            value = p.get('total_value', 0)
            total_portfolio_value += value
            print(f'  - {prop_name}')
            if value > 0:
                print(f'    Value: ${value:,}')
    
    print(f'\n  TOTAL PORTFOLIO VALUE: ${total_portfolio_value:,}')
    
    # Find Erik's property network
    print(f'\n\n🔍 ERIK STROMBECK PROPERTY NETWORK:')
    cursor.execute('''
        SELECT e2.name, e2.properties, r.properties as rel_props
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE e1.entity_id = 'person_erik_strombeck'
        AND e2.entity_type = 'address'
        AND r.relationship_type IN ('OWNS', 'BUILT', 'SOLD_TO', 'PURCHASED_FROM')
    ''')
    
    erik_properties = cursor.fetchall()
    erik_portfolio_value = 0
    
    for prop_name, props_json, rel_props_json in erik_properties:
        if props_json:
            p = json.loads(props_json) if isinstance(props_json, str) else props_json
            value = p.get('total_value', 0)
            erik_portfolio_value += value
            print(f'  - {prop_name}')
            if value > 0:
                print(f'    Value: ${value:,}')
            if rel_props_json:
                rp = json.loads(rel_props_json) if isinstance(rel_props_json, str) else rel_props_json
                if 'investigation_flag' in rp:
                    print(f'    Flag: {rp["investigation_flag"]}')
    
    print(f'\n  ERIK TOTAL PORTFOLIO VALUE: ${erik_portfolio_value:,}')
    
    # Find trust connections
    print(f'\n\n🔍 TRUST OWNERSHIP ANALYSIS:')
    cursor.execute('''
        SELECT e1.name, e2.name, r.relationship_type, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE (e1.name LIKE '%Westwood%Trust%' OR e2.name LIKE '%Westwood%Trust%')
        OR (e1.entity_id LIKE '%trust%' AND e2.entity_type = 'address')
        OR (e2.entity_id LIKE '%trust%' AND e1.entity_type = 'address')
    ''')
    
    trust_connections = cursor.fetchall()
    if trust_connections:
        for e1, e2, rel_type, props_json in trust_connections:
            print(f'  {e1[:40]:40s} --[{rel_type}]--> {e2[:40]}')
            if props_json:
                p = json.loads(props_json) if isinstance(props_json, str) else props_json
                if 'investigation_note' in p:
                    print(f'    Note: {p["investigation_note"]}')
    else:
        print('  ⚠️  No trust entities found - need to create')
    
    # Find business partner connections
    print(f'\n\n🔍 BUSINESS PARTNER NETWORK:')
    cursor.execute('''
        SELECT e1.name, e2.name, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.relationship_type = 'BUSINESS_PARTNER'
        AND (e1.name LIKE '%Strombeck%' OR e2.name LIKE '%Strombeck%'
             OR e1.name IN ('Will Startare', 'Corey Taylor', 'Matt Allen')
             OR e2.name IN ('Will Startare', 'Corey Taylor', 'Matt Allen'))
    ''')
    
    partners = cursor.fetchall()
    for e1, e2, props_json in partners:
        print(f'  {e1[:30]:30s} <--BUSINESS_PARTNER--> {e2[:30]}')
        if props_json:
            p = json.loads(props_json) if isinstance(props_json, str) else props_json
            if 'investigation_note' in p:
                print(f'    Note: {p["investigation_note"]}')
    
    # Timeline analysis
    print(f'\n\n🔍 TIMELINE ANALYSIS:')
    cursor.execute('''
        SELECT r.properties, r.relationship_type, e1.name, e2.name
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE e2.entity_id = ?
        AND r.properties LIKE '%date%'
    ''', (prop_id,))
    
    transactions = cursor.fetchall()
    pre_2022_11 = []
    post_2022_11 = []
    
    for props_json, rel_type, e1, e2 in transactions:
        props = json.loads(props_json) if isinstance(props_json, str) else props_json
        date = props.get('date', '')
        
        tx_info = f'{e1[:25]:25s} --[{rel_type}]--> {e2[:25]} ({date})'
        
        if date and date < '2022-11':
            pre_2022_11.append((date, tx_info))
        elif date and date >= '2022-11':
            post_2022_11.append((date, tx_info))
    
    print(f'\n  Pre-November 2022 ({len(pre_2022_11)} transactions):')
    for date, info in sorted(pre_2022_11):
        print(f'    {date}: {info}')
    
    print(f'\n  Post-November 2022 ({len(post_2022_11)} transactions):')
    for date, info in sorted(post_2022_11):
        print(f'    {date}: {info}')
    
    # Hidden patterns
    print(f'\n\n🔍 HIDDEN PATTERNS IDENTIFIED:')
    
    patterns = {
        'trust_ownership': len(trust_connections) > 0,
        'po_box_connection': len(po_box_properties) > 1,
        'major_asset': total_portfolio_value > 10000000,
        'refinancing_pattern': len(post_2022_11) > len(pre_2022_11),
        'business_partner_network': len(partners) > 0,
    }
    
    for pattern, found in patterns.items():
        status = '✅' if found else '❌'
        print(f'  {status} {pattern.replace("_", " ").title()}')
    
    # Generate synthesis
    synthesis = {
        'property': prop_id,
        'total_value': props.get('total_value', 0) if prop_data else 0,
        'po_box_properties_count': len(po_box_properties),
        'total_portfolio_value': total_portfolio_value,
        'erik_portfolio_value': erik_portfolio_value,
        'trust_connections': len(trust_connections),
        'business_partners': len(partners),
        'pre_incident_transactions': len(pre_2022_11),
        'post_incident_transactions': len(post_2022_11),
        'patterns': patterns
    }
    
    with open('WESTWOOD_PATTERN_ANALYSIS.json', 'w') as f:
        json.dump(synthesis, f, indent=2)
    
    print(f'\n✅ Analysis saved to: WESTWOOD_PATTERN_ANALYSIS.json')
    
    conn.close()
    return synthesis

if __name__ == "__main__":
    analyze_westwood_patterns()
