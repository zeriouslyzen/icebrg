#!/usr/bin/env python3
"""
Import Property Transaction Data
Processes property history records and creates relationships
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def import_transaction_data():
    """Import the property transaction data."""
    print('=' * 80)
    print('IMPORTING PROPERTY TRANSACTION DATA')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # Property details
    address = "965 W HARRIS ST EUREKA CA 95503-3926"
    apn = "008-182-024-000"
    addr_id = "address_965_w_harris_eureka"
    
    # Ensure address entity exists
    cursor.execute("""
        INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
        VALUES (?, ?, ?, ?, ?)
    """, (
        addr_id,
        address,
        'address',
        'property_history',
        json.dumps({'apn': apn, 'city': 'Eureka', 'state': 'CA', 'zip': '95503'})
    ))
    
    # Transaction records from the data
    transactions = [
        {
            'date': '2025-06-10',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.7492',
            'lender': 'Available',
            'borrower': 'Available',
        },
        {
            'date': '2025-06-10',
            'type': 'Assignment',
            'document_type': 'Subordination',
            'document_number': 'xxxx.7491',
            'new_lender': 'Available',
            'previous_lender': 'Available',
            'borrower': 'Available',
        },
        {
            'date': '2025-02-28',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.2287',
            'lender': 'Available',
            'borrower': 'Available',
        },
        {
            'date': '2024-04-22',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': 'xxxx.4916',
            'buyer': 'Available',
            'seller': 'Available',
            'sale_price': 'Available',
            'sale_type': 'Available',
        },
        {
            'date': '2024-04-22',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.4917',
            'lender': 'Available',
            'borrower': 'Available',
        },
        {
            'date': '2023-02-01',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': 'xxxx.1472',
            'buyers': ['Available', 'Available', 'Available'],  # Multiple buyers
            'seller': 'Available',
        },
        {
            'date': '2016-12-19',
            'type': 'Release',
            'document_type': 'Release',
            'document_number': 'xxxx.x4795',
        },
        {
            'date': '2006-10-23',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.x0555',
            'lender': 'Available',
            'borrower': 'Available',
        },
        {
            'date': '1999-01-28',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': 'xxxx.2762',
            'buyer': 'Available',
        },
    ]
    
    print(f'\nProcessing {len(transactions)} transaction records for: {address}')
    
    imported = 0
    
    # Analyze patterns
    sale_transfers = [t for t in transactions if t['type'] == 'Sale/Transfer']
    finance_records = [t for t in transactions if t['type'] == 'Finance']
    
    print(f'\nPATTERNS IDENTIFIED:')
    print(f'  - Sale/Transfer records: {len(sale_transfers)}')
    print(f'  - Finance/Mortgage records: {len(finance_records)}')
    print(f'  - Most recent sale: {sale_transfers[0]["date"] if sale_transfers else "N/A"}')
    print(f'  - Oldest sale: {sale_transfers[-1]["date"] if sale_transfers else "N/A"}')
    print(f'  - Multiple buyers in 2023 transaction: {len(transactions[5].get("buyers", []))}')
    
    # Key patterns
    patterns = {
        'frequent_refinancing': len(finance_records) > 3,
        'recent_activity': any(t['date'].startswith('2024') or t['date'].startswith('2025') for t in transactions),
        'multiple_sales': len(sale_transfers) >= 3,
        'multiple_buyers_2023': len(transactions[5].get('buyers', [])) > 1,
        'recent_sale_2024': any(t['date'].startswith('2024') and t['type'] == 'Sale/Transfer' for t in transactions),
    }
    
    print(f'\nKEY PATTERNS:')
    for pattern, value in patterns.items():
        print(f'  - {pattern}: {value}')
    
    # Import sale/transfer transactions
    print(f'\n\nImporting Sale/Transfer transactions...')
    
    for tx in sale_transfers:
        date = tx['date']
        doc_num = tx['document_number']
        
        # Note: Buyer/Seller are "Available" - need actual names
        # For now, create placeholder entities that can be updated
        
        if 'buyer' in tx:
            buyer_id = f"buyer_{date.replace('-', '')}_{doc_num.replace('.', '').replace('x', '0')}"
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
                        'needs_verification': True
                    })
                ))
                
                # Create PURCHASED_FROM relationship
                cursor.execute("""
                    INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                    VALUES (?, ?, ?, ?)
                """, (
                    buyer_id,
                    addr_id,
                    'PURCHASED_FROM',
                    json.dumps({
                        'date': date,
                        'document_number': doc_num,
                        'address': address,
                        'source': 'property_history'
                    })
                ))
                
                print(f'  ✅ Buyer --[PURCHASED_FROM]--> {address} ({date})')
                imported += 1
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        if 'buyers' in tx:
            for i, buyer in enumerate(tx['buyers']):
                buyer_id = f"buyer_{date.replace('-', '')}_{doc_num.replace('.', '').replace('x', '0')}_{i}"
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        buyer_id,
                        f"Buyer {i+1} ({date})",
                        'person',
                        'property_transaction',
                        json.dumps({
                            'address': address,
                            'date': date,
                            'document_number': doc_num,
                            'role': 'buyer',
                            'buyer_number': i+1,
                            'needs_verification': True
                        })
                    ))
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                        VALUES (?, ?, ?, ?)
                    """, (
                        buyer_id,
                        addr_id,
                        'PURCHASED_FROM',
                        json.dumps({
                            'date': date,
                            'document_number': doc_num,
                            'address': address,
                            'source': 'property_history'
                        })
                    ))
                    
                    print(f'  ✅ Buyer {i+1} --[PURCHASED_FROM]--> {address} ({date})')
                    imported += 1
                except Exception as e:
                    pass
        
        if 'seller' in tx:
            seller_id = f"seller_{date.replace('-', '')}_{doc_num.replace('.', '').replace('x', '0')}"
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
                        'document_number': doc_num,
                        'role': 'seller',
                        'needs_verification': True
                    })
                ))
                
                # Create SOLD_TO relationship
                cursor.execute("""
                    INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                    VALUES (?, ?, ?, ?)
                """, (
                    seller_id,
                    addr_id,
                    'SOLD_TO',
                    json.dumps({
                        'date': date,
                        'document_number': doc_num,
                        'address': address,
                        'source': 'property_history'
                    })
                ))
                
                print(f'  ✅ Seller --[SOLD_TO]--> {address} ({date})')
                imported += 1
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
    
    conn.commit()
    
    print(f'\n\n✅ Imported {imported} transaction relationships')
    print(f'\n⚠️  NOTE: Buyer/Seller names show as "Available" - need actual names')
    print(f'   These are placeholder entities that need verification')
    
    # Save analysis
    analysis = {
        'address': address,
        'apn': apn,
        'total_transactions': len(transactions),
        'sale_transfers': len(sale_transfers),
        'finance_records': len(finance_records),
        'patterns': patterns,
        'transactions': transactions
    }
    
    with open('PROPERTY_TRANSACTION_ANALYSIS.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f'\n✅ Analysis saved to: PROPERTY_TRANSACTION_ANALYSIS.json')
    
    conn.close()
    return analysis

if __name__ == "__main__":
    import_transaction_data()
