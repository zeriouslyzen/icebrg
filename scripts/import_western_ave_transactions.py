#!/usr/bin/env python3
"""
Import Western Avenue Property Transactions
Analyzing for asset hiding/income concealment patterns
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def import_and_analyze():
    """Import transactions and analyze for asset hiding patterns."""
    print('=' * 80)
    print('WESTERN AVENUE PROPERTY - ASSET HIDING ANALYSIS')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # Property details
    address = "2149 WESTERN AVE A ARCATA CA 95521-5349"
    apn = "505-095-038-000"
    addr_id = "address_western_avenue_arcata"
    
    # Ensure address entity exists
    cursor.execute("""
        INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
        VALUES (?, ?, ?, ?, ?)
    """, (
        addr_id,
        address,
        'address',
        'property_history',
        json.dumps({'apn': apn, 'city': 'Arcata', 'state': 'CA', 'zip': '95521', 'note': 'From handwritten notes'})
    ))
    
    # Link to Strombeck if this is their property
    cursor.execute("""
        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
        VALUES (?, ?, ?, ?)
    """, (
        'person_erik_strombeck',
        addr_id,
        'OWNS',
        json.dumps({'source': 'investigation', 'status': 'needs_verification'})
    ))
    
    # Transaction records
    transactions = [
        {
            'date': '2025-12-29',
            'type': 'Release',
            'document_type': 'Release',
            'document_number': 'xxxx.x8898',
        },
        {
            'date': '2025-12-01',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': 'xxxx.x6065',
            'buyers': ['Available', 'Available'],  # 2 buyers
            'seller': 'Available',
            'sale_price': 'Available',
        },
        {
            'date': '2025-12-01',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.x6066',
            'borrowers': ['Available', 'Available'],  # 2 borrowers
            'lender': 'Available',
        },
        {
            'date': '2024-11-12',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': 'xxxx.x6254',
            'buyer': 'Available',
            'seller': 'Available',
        },
        {
            'date': '2024-11-12',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.x6255',
            'borrower': 'Available',
            'lender': 'Available',
        },
        {
            'date': '2024-10-29',
            'type': 'Release',
            'document_type': 'Release',
            'document_number': 'xxxx.x4421',
        },
        {
            'date': '2024-10-29',
            'type': 'Release',
            'document_type': 'Release',
            'document_number': 'xxxx.x4418',
        },
        {
            'date': '2022-06-06',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': 'xxxx.x0868',
            'buyers': ['Available', 'Available'],  # 2 buyers
            'seller': 'Available',
        },
        {
            'date': '2022-06-06',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.x0869',
            'borrowers': ['Available', 'Available'],  # 2 borrowers
            'lender': 'Available',
        },
        {
            'date': '2021-07-02',
            'type': 'Sale/Transfer',
            'document_type': 'Deed Transfer',
            'document_number': 'xxxx.x5197',
            'buyer': 'Available',
            'seller': 'Available',
        },
        {
            'date': '2021-07-02',
            'type': 'Finance',
            'document_type': 'Trust Deed/Mortgage',
            'document_number': 'xxxx.x5198',
            'borrower': 'Available',
            'lender': 'Available',
        },
    ]
    
    print(f'\nProcessing {len(transactions)} transaction records')
    
    # Analyze for asset hiding patterns
    sale_transfers = [t for t in transactions if t['type'] == 'Sale/Transfer']
    finance_records = [t for t in transactions if t['type'] == 'Finance']
    releases = [t for t in transactions if t['type'] == 'Release']
    
    print(f'\n📊 TRANSACTION BREAKDOWN:')
    print(f'  - Sale/Transfer: {len(sale_transfers)}')
    print(f'  - Finance/Mortgage: {len(finance_records)}')
    print(f'  - Releases: {len(releases)}')
    
    # Asset hiding indicators
    patterns = {
        'frequent_sales': len(sale_transfers) >= 3,
        'multiple_buyers': any('buyers' in t and len(t.get('buyers', [])) > 1 for t in sale_transfers),
        'multiple_borrowers': any('borrowers' in t and len(t.get('borrowers', [])) > 1 for t in finance_records),
        'recent_activity': any(t['date'].startswith('2024') or t['date'].startswith('2025') for t in transactions),
        'sale_immediate_refinance': False,
        'rapid_turnover': False,
    }
    
    # Check for sale + immediate refinance
    for i, tx in enumerate(transactions):
        if tx['type'] == 'Sale/Transfer' and i + 1 < len(transactions):
            next_tx = transactions[i + 1]
            if next_tx['type'] == 'Finance' and next_tx['date'] == tx['date']:
                patterns['sale_immediate_refinance'] = True
    
    # Check for rapid turnover (sales within short timeframes)
    sale_dates = [datetime.strptime(t['date'], '%Y-%m-%d') for t in sale_transfers]
    sale_dates.sort()
    if len(sale_dates) >= 2:
        time_diffs = [(sale_dates[i+1] - sale_dates[i]).days for i in range(len(sale_dates)-1)]
        patterns['rapid_turnover'] = any(diff < 365 for diff in time_diffs)  # Sales within 1 year
    
    print(f'\n🔍 ASSET HIDING INDICATORS:')
    for pattern, value in patterns.items():
        status = '✅ YES' if value else '❌ NO'
        pattern_display = pattern.replace('_', ' ').title()
        print(f'  {status} - {pattern_display}')
    
    # Red flags
    red_flags = []
    if patterns['frequent_sales']:
        red_flags.append('Frequent sales (4 sales in 4 years) - possible asset churning')
    if patterns['multiple_buyers']:
        red_flags.append('Multiple buyers in transactions - possible asset transfer/hiding')
    if patterns['multiple_borrowers']:
        red_flags.append('Multiple borrowers - possible joint ownership to obscure')
    if patterns['sale_immediate_refinance']:
        red_flags.append('Sale + immediate refinancing - possible cash extraction')
    if patterns['rapid_turnover']:
        red_flags.append('Rapid property turnover - possible asset hiding strategy')
    
    print(f'\n🚩 RED FLAGS FOR ASSET HIDING:')
    if red_flags:
        for i, flag in enumerate(red_flags, 1):
            print(f'  {i}. {flag}')
    else:
        print('  None identified')
    
    # Import transactions
    print(f'\n\nImporting transactions...')
    imported = 0
    
    for tx in sale_transfers:
        date = tx['date']
        doc_num = tx['document_number']
        
        # Multiple buyers pattern
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
                            'investigation_note': 'Multiple buyers - asset hiding indicator',
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
                            'source': 'property_history',
                            'investigation_flag': 'multiple_buyers'
                        })
                    ))
                    
                    imported += 1
                except Exception as e:
                    pass
        
        # Single buyer
        elif 'buyer' in tx:
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
                
                imported += 1
            except Exception as e:
                pass
        
        # Seller
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
                        'investigation_note': 'Possible Erik Strombeck - needs verification',
                        'needs_verification': True
                    })
                ))
                
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
                        'source': 'property_history',
                        'investigation_flag': 'asset_hiding_suspected'
                    })
                ))
                
                imported += 1
            except Exception as e:
                pass
    
    conn.commit()
    
    print(f'✅ Imported {imported} transaction relationships')
    
    # Save analysis
    analysis = {
        'address': address,
        'apn': apn,
        'investigation_focus': 'Asset hiding / Income concealment',
        'total_transactions': len(transactions),
        'sale_transfers': len(sale_transfers),
        'finance_records': len(finance_records),
        'releases': len(releases),
        'patterns': patterns,
        'red_flags': red_flags,
        'transactions': transactions
    }
    
    with open('WESTERN_AVE_ASSET_HIDING_ANALYSIS.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f'\n✅ Analysis saved to: WESTERN_AVE_ASSET_HIDING_ANALYSIS.json')
    
    conn.close()
    return analysis

if __name__ == "__main__":
    import_and_analyze()
