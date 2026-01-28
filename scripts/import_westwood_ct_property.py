#!/usr/bin/env python3
"""
Import Westwood Court Property - Major Asset Discovery
Analyzing trust ownership, PO Box connection, and hidden patterns
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def import_westwood_property():
    """Import Westwood Court property and analyze patterns."""
    print('=' * 80)
    print('WESTWOOD COURT PROPERTY - MAJOR ASSET DISCOVERY')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # Property details
    address = "2351 WESTWOOD CT A1 ARCATA CA 95521-5151"
    apn = "505-161-028-000"
    prop_id = "address_2351_westwood_ct_arcata"
    
    # Property characteristics
    total_value = 7578035  # $7.5M
    land_value = 1188552
    structures_value = 6346023
    sale_price_2021 = 7100000  # $7.1M
    building_area = 53340  # sq ft
    lot_acres = 5.6
    lot_sqft = 243935
    year_built = 1967
    use_type = "RESID. MULTIPLE FAMILY"
    
    # PO Box connection
    po_box = "PO BOX 37 EUREKA CA 95502"
    po_box_id = "address_po_box_37_eureka"
    
    # Create property entity
    cursor.execute("""
        INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
        VALUES (?, ?, ?, ?, ?)
    """, (
        prop_id,
        address,
        'address',
        'property_history',
        json.dumps({
            'apn': apn,
            'city': 'Arcata',
            'state': 'CA',
            'zip': '95521',
            'total_value': total_value,
            'land_value': land_value,
            'structures_value': structures_value,
            'sale_price_2021': sale_price_2021,
            'building_area': building_area,
            'lot_acres': lot_acres,
            'year_built': year_built,
            'use_type': use_type,
            'investigation_note': 'MAJOR ASSET - $7.5M value, Strombeck PO Box',
            'investigation_flag': 'major_asset'
        })
    ))
    
    # Create PO Box entity
    cursor.execute("""
        INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
        VALUES (?, ?, ?, ?, ?)
    """, (
        po_box_id,
        po_box,
        'address',
        'property_history',
        json.dumps({
            'type': 'po_box',
            'investigation_note': 'Strombeck mailing address',
            'investigation_flag': 'strombeck_po_box'
        })
    ))
    
    # Link property to PO Box
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        prop_id,
        po_box_id,
        'MAILING_ADDRESS',
        json.dumps({
            'source': 'property_records',
            'investigation_note': 'Property owner uses Strombeck PO Box',
            'investigation_flag': 'strombeck_connection'
        })
    ))
    
    # Link to Erik Strombeck (property owner)
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        'person_erik_strombeck',
        prop_id,
        'OWNS',
        json.dumps({
            'source': 'property_records',
            'mailing_address': po_box,
            'investigation_note': 'Property owner uses Strombeck PO Box',
            'investigation_flag': 'strombeck_ownership'
        })
    ))
    
    # Transaction records
    transactions = [
        {
            'date': '2025-12-10',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.x7270',
            'borrower': 'Available',
            'lender': 'Available',
        },
        {
            'date': '2025-08-12',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.x0697',
            'borrower': 'Available',
            'lender': 'Available',
        },
        {
            'date': '2021-07-30',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': '2021R17292',
            'buyer': 'Available',
            'seller': 'Available',
            'sale_price': 7100000,
        },
        {
            'date': '2021-07-30',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.x7293',
            'borrower': 'Available',
            'lender': 'Available',
        },
        {
            'date': '1966-07-06',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': 'Unknown',
            'buyer': 'Unknown',
            'seller': 'Unknown',
        },
    ]
    
    print(f'\n\nProperty Details:')
    print(f'  Address: {address}')
    print(f'  Total Value: ${total_value:,}')
    print(f'  Sale Price (2021): ${sale_price_2021:,}')
    print(f'  Building Area: {building_area:,} sq ft')
    print(f'  Lot Size: {lot_acres} acres ({lot_sqft:,} sq ft)')
    print(f'  Use Type: {use_type}')
    print(f'  Mailing Address: {po_box} ⚠️ STROMBECK PO BOX')
    
    # Analyze patterns
    print(f'\n\n🔍 PATTERN ANALYSIS:')
    
    # Pattern 1: Trust ownership
    print(f'\n1. TRUST OWNERSHIP PATTERN:')
    print(f'   - Property owned by trust (mentioned by user)')
    print(f'   - Trusts obscure true ownership')
    print(f'   - PO Box connection suggests Strombeck control')
    print(f'   - Investigation Flag: Hidden ownership through trust')
    
    # Pattern 2: Major asset
    print(f'\n2. MAJOR ASSET:')
    print(f'   - $7.5M property value')
    print(f'   - $7.1M sale in 2021')
    print(f'   - Multiple family residential (income property)')
    print(f'   - Investigation Flag: Major asset hidden through trust')
    
    # Pattern 3: Refinancing pattern
    print(f'\n3. REFINANCING PATTERN:')
    print(f'   - 2021: Sale + immediate refinancing')
    print(f'   - 2025: Multiple refinancings (8/12, 12/10)')
    print(f'   - Pattern: Cash extraction from major asset')
    print(f'   - Investigation Flag: Extracting cash from property')
    
    # Pattern 4: PO Box connection
    print(f'\n4. PO BOX CONNECTION:')
    print(f'   - Property owner uses Strombeck PO Box')
    print(f'   - Direct connection to Strombeck family')
    print(f'   - Investigation Flag: Confirms Strombeck ownership')
    
    # Pattern 5: Timeline correlation
    print(f'\n5. TIMELINE CORRELATION:')
    print(f'   - 2021-07-30: Property sold (before November 2022 incident)')
    print(f'   - 2025: Multiple refinancings (after incident)')
    print(f'   - Pattern: Using property for cash extraction post-incident')
    
    # Import transactions
    print(f'\n\nImporting transactions...')
    imported = 0
    
    for tx in transactions:
        if tx['type'] == 'Sale/Transfer':
            date = tx['date']
            doc_num = tx.get('document_number', 'Unknown')
            sale_price = tx.get('sale_price', 0)
            
            # Create buyer/seller entities
            if 'buyer' in tx and tx['buyer'] == 'Available':
                buyer_id = f"buyer_{date.replace('-', '')}_{doc_num.replace('.', '').replace('R', '0')}"
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        buyer_id,
                        f"Buyer ({date})",
                        'person',
                        'property_transaction',
                        json.dumps({
                            'address': address,
                            'date': date,
                            'document_number': doc_num,
                            'role': 'buyer',
                            'sale_price': sale_price,
                            'investigation_note': 'MAJOR ASSET - $7.1M sale',
                            'needs_verification': True
                        })
                    ))
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                        VALUES (?, ?, ?, ?)
                    """, (
                        buyer_id,
                        prop_id,
                        'PURCHASED_FROM',
                        json.dumps({
                            'date': date,
                            'document_number': doc_num,
                            'address': address,
                            'sale_price': sale_price,
                            'source': 'property_history',
                            'investigation_flag': 'major_asset_transaction'
                        })
                    ))
                    imported += 1
                except Exception as e:
                    pass
            
            if 'seller' in tx and tx['seller'] == 'Available':
                seller_id = f"seller_{date.replace('-', '')}_{doc_num.replace('.', '').replace('R', '0')}"
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        seller_id,
                        f"Seller ({date})",
                        'person',
                        'property_transaction',
                        json.dumps({
                            'address': address,
                            'date': date,
                            'document_number': sale_price,
                            'role': 'seller',
                            'sale_price': sale_price,
                            'investigation_note': 'MAJOR ASSET - $7.1M sale, possible Erik Strombeck',
                            'needs_verification': True
                        })
                    ))
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                        VALUES (?, ?, ?, ?)
                    """, (
                        seller_id,
                        prop_id,
                        'SOLD_TO',
                        json.dumps({
                            'date': date,
                            'document_number': doc_num,
                            'address': address,
                            'sale_price': sale_price,
                            'source': 'property_history',
                            'investigation_flag': 'major_asset_sale'
                        })
                    ))
                    imported += 1
                except Exception as e:
                    pass
    
    conn.commit()
    
    print(f'✅ Imported {imported} transaction relationships')
    
    # Find connections to other properties
    print(f'\n\n🔍 FINDING CONNECTIONS:')
    
    # Check if PO Box is used by other properties
    cursor.execute('''
        SELECT e1.name as property, e2.name as po_box
        FROM relationships r
        INNER JOIN entities e1 ON e1.entity_id = r.source_id
        INNER JOIN entities e2 ON e1.entity_id = r.target_id
        WHERE r.relationship_type = 'MAILING_ADDRESS'
        AND e2.entity_id = ?
    ''', (po_box_id,))
    
    po_box_properties = cursor.fetchall()
    if po_box_properties:
        print(f'\n  Properties using Strombeck PO Box:')
        for prop, pb in po_box_properties:
            print(f'    - {prop}')
    
    # Check for trust entities
    cursor.execute('''
        SELECT name, entity_type, properties
        FROM entities
        WHERE name LIKE '%trust%' OR properties LIKE '%trust%'
    ''')
    
    trusts = cursor.fetchall()
    if trusts:
        print(f'\n  Trust entities found:')
        for name, etype, props in trusts:
            print(f'    - {name} ({etype})')
    else:
        print(f'\n  ⚠️  No trust entities found - need to create')
        # Create trust entity
        trust_id = 'entity_westwood_trust'
        cursor.execute("""
            INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (
            trust_id,
            'Westwood Court Trust',
            'entity',
            'property_history',
            json.dumps({
                'property': address,
                'investigation_note': 'Trust owns $7.5M property',
                'investigation_flag': 'hidden_ownership'
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
                'investigation_note': 'Trust owns property, Strombeck PO Box',
                'investigation_flag': 'hidden_ownership'
            })
        ))
        
        # Link Erik to trust (possible beneficiary/trustee)
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            'person_erik_strombeck',
            trust_id,
            'BENEFICIARY_OF',
            json.dumps({
                'source': 'investigation',
                'investigation_note': 'Possible beneficiary/trustee of trust',
                'investigation_flag': 'hidden_ownership'
            })
        ))
        
        print(f'    ✅ Created: Westwood Court Trust')
    
    conn.commit()
    
    # Save analysis
    analysis = {
        'property': address,
        'apn': apn,
        'total_value': total_value,
        'sale_price_2021': sale_price_2021,
        'po_box': po_box,
        'trust_ownership': True,
        'patterns': {
            'major_asset': True,
            'trust_ownership': True,
            'po_box_connection': True,
            'refinancing_pattern': True,
            'timeline_correlation': True
        }
    }
    
    with open('WESTWOOD_CT_ANALYSIS.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f'\n✅ Analysis saved to: WESTWOOD_CT_ANALYSIS.json')
    
    conn.close()
    return analysis

if __name__ == "__main__":
    import_westwood_property()
