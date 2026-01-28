#!/usr/bin/env python3
"""
Synthesize All Westwood Court Data
Aggregate with existing investigation and generate final report
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def synthesize_all_data():
    """Synthesize all Westwood data with existing investigation."""
    print('=' * 80)
    print('COMPLETE SYNTHESIS - WESTWOOD COURT + EXISTING INVESTIGATION')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # Get complete property portfolio
    cursor.execute('''
        SELECT e.name, e.properties, r.relationship_type
        FROM relationships r
        INNER JOIN entities e ON e.entity_id = r.target_id
        WHERE r.source_id = 'person_erik_strombeck'
        AND e.entity_type = 'address'
        AND r.relationship_type IN ('OWNS', 'BUILT', 'SOLD_TO', 'PURCHASED_FROM')
    ''')
    
    properties = cursor.fetchall()
    
    portfolio = {}
    total_value = 0
    
    print(f'\n\nCOMPLETE PROPERTY PORTFOLIO:')
    print('=' * 80)
    
    for prop_name, props_json, rel_type in properties:
        if props_json:
            p = json.loads(props_json) if isinstance(props_json, str) else props_json
            value = p.get('total_value', 0)
            if value > 0:
                total_value += value
                portfolio[prop_name] = {
                    'value': value,
                    'relationship': rel_type,
                    'properties': p
                }
                print(f'\n  {prop_name}')
                print(f'    Value: ${value:,}')
                print(f'    Relationship: {rel_type}')
                if 'investigation_flag' in p:
                    print(f'    Flag: {p["investigation_flag"]}')
    
    print(f'\n\n  TOTAL PORTFOLIO VALUE: ${total_value:,}')
    
    # Get trust connections
    cursor.execute('''
        SELECT e1.name, e2.name, r.relationship_type, r.properties
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE (e1.name LIKE '%Trust%' OR e2.name LIKE '%Trust%')
        AND (e1.name LIKE '%Westwood%' OR e2.name LIKE '%Westwood%'
             OR e1.name LIKE '%Strombeck%' OR e2.name LIKE '%Strombeck%')
    ''')
    
    trust_connections = cursor.fetchall()
    
    print(f'\n\nTRUST OWNERSHIP NETWORK:')
    print('=' * 80)
    
    for e1, e2, rel_type, props_json in trust_connections:
        print(f'\n  {e1[:40]:40s} --[{rel_type}]--> {e2[:40]}')
        if props_json:
            p = json.loads(props_json) if isinstance(props_json, str) else props_json
            if 'investigation_note' in p:
                print(f'    Note: {p["investigation_note"]}')
    
    # Get PO Box connections
    cursor.execute('''
        SELECT e1.name, e2.name, r.relationship_type
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e2.entity_id = r.target_id
        WHERE r.relationship_type = 'MAILING_ADDRESS'
        AND e2.name LIKE '%PO BOX 37%'
    ''')
    
    po_box_properties = cursor.fetchall()
    
    print(f'\n\nPO BOX 37 NETWORK:')
    print('=' * 80)
    
    po_box_value = 0
    for prop, po_box, rel_type in po_box_properties:
        print(f'  {prop}')
        # Get property value
        cursor.execute('SELECT properties FROM entities WHERE name = ?', (prop,))
        result = cursor.fetchone()
        if result:
            p = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            value = p.get('total_value', 0)
            if value > 0:
                po_box_value += value
                print(f'    Value: ${value:,}')
    
    print(f'\n  TOTAL PO BOX 37 PORTFOLIO: ${po_box_value:,}')
    
    # Get business partner network
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
    
    print(f'\n\nBUSINESS PARTNER NETWORK:')
    print('=' * 80)
    
    for e1, e2, props_json in partners:
        print(f'  {e1[:30]:30s} <--BUSINESS_PARTNER--> {e2[:30]}')
        if props_json:
            p = json.loads(props_json) if isinstance(props_json, str) else props_json
            if 'investigation_note' in p:
                print(f'    Note: {p["investigation_note"]}')
    
    # Generate synthesis
    synthesis = {
        'total_portfolio_value': total_value,
        'po_box_37_portfolio_value': po_box_value,
        'property_count': len(portfolio),
        'trust_connections': len(trust_connections),
        'business_partners': len(partners),
        'properties': portfolio,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('COMPLETE_WESTWOOD_SYNTHESIS.json', 'w') as f:
        json.dump(synthesis, f, indent=2)
    
    print(f'\n\n✅ Complete synthesis saved to: COMPLETE_WESTWOOD_SYNTHESIS.json')
    print(f'\n  Total Portfolio Value: ${total_value:,}')
    print(f'  PO Box 37 Portfolio: ${po_box_value:,}')
    print(f'  Properties Identified: {len(portfolio)}')
    print(f'  Trust Connections: {len(trust_connections)}')
    print(f'  Business Partners: {len(partners)}')
    
    conn.close()
    return synthesis

if __name__ == "__main__":
    synthesize_all_data()
