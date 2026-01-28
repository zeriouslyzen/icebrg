#!/usr/bin/env python3
"""
Get Property Transactions for Strombeck Family
Searches for actual buy/sell/purchase records
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore
from src.iceburg.search.web_search import WebSearchAggregator
import os
os.environ['ICEBURG_ENABLE_WEB'] = '1'

def search_property_transactions():
    """Search for property transactions."""
    print('=' * 80)
    print('SEARCHING FOR PROPERTY TRANSACTIONS')
    print('=' * 80)
    
    search = WebSearchAggregator()
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # Search queries for each person
    targets = [
        ('Steven Mark Strombeck', 'person_steven_mark_strombeck'),
        ('Erik Strombeck', 'person_erik_strombeck'),
        ('Waltina Martha Strombeck', 'person_waltina_martha_strombeck'),
        ('Strombeck Properties', 'company_strombeck_properties'),
    ]
    
    transactions_found = []
    
    for name, entity_id in targets:
        print(f'\n\nSearching transactions for: {name}')
        
        queries = [
            f'{name} property sold purchased Arcata Eureka Humboldt',
            f'{name} real estate transactions Humboldt County',
            f'{name} property deed records',
            f'{name} bought sold property',
            f'{name} property ownership transfers',
        ]
        
        for query in queries:
            print(f'  Query: {query}')
            try:
                results = search.search(query, sources=['ddg'], max_results_per_source=5)
                
                for result in results:
                    text = f"{result.title} {result.snippet}".lower()
                    
                    # Look for transaction keywords
                    if any(word in text for word in ['sold', 'purchased', 'bought', 'deed', 'transfer', 'property']):
                        # Extract potential buyer/seller names
                        # Pattern: "sold to X" or "purchased by X" or "bought from X"
                        buyer_pattern = r'(?:sold to|purchased by|bought by|acquired by)[:\s,]+([A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?)'
                        seller_pattern = r'(?:sold by|purchased from|bought from|acquired from)[:\s,]+([A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?)'
                        
                        buyers = re.findall(buyer_pattern, result.title + ' ' + result.snippet, re.IGNORECASE)
                        sellers = re.findall(seller_pattern, result.title + ' ' + result.snippet, re.IGNORECASE)
                        
                        # Extract addresses
                        addr_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:St|Street|Ave|Avenue|Court|Rd|Road|Blvd)[\s,]*Arcata|Eureka'
                        addresses = re.findall(addr_pattern, result.title + ' ' + result.snippet, re.IGNORECASE)
                        
                        # Extract dates
                        date_pattern = r'\b(20\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s,]+(?:20\d{2})?\b'
                        dates = re.findall(date_pattern, result.title + ' ' + result.snippet, re.IGNORECASE)
                        
                        if buyers or sellers or addresses:
                            transaction = {
                                'entity': name,
                                'entity_id': entity_id,
                                'url': result.url,
                                'title': result.title,
                                'snippet': result.snippet[:200],
                                'buyers': buyers,
                                'sellers': sellers,
                                'addresses': addresses,
                                'dates': dates
                            }
                            transactions_found.append(transaction)
                            print(f'    ✅ Found transaction data')
                            if buyers:
                                print(f'      Buyers: {buyers}')
                            if sellers:
                                print(f'      Sellers: {sellers}')
                            if addresses:
                                print(f'      Addresses: {addresses}')
            except Exception as e:
                print(f'    ⚠️  Error: {e}')
    
    # Save findings
    with open('PROPERTY_TRANSACTIONS_FOUND.json', 'w') as f:
        json.dump(transactions_found, f, indent=2)
    
    print(f'\n\n✅ Found {len(transactions_found)} transaction records')
    print('Saved to: PROPERTY_TRANSACTIONS_FOUND.json')
    
    # Import to database
    print('\n\nImporting transactions to database...')
    imported = 0
    
    for tx in transactions_found:
        entity_id = tx['entity_id']
        
        # Create buyer entities and relationships
        for buyer in tx.get('buyers', []):
            if 'strombeck' not in buyer.lower():
                buyer_id = f"person_{buyer.lower().replace(' ', '_')[:50]}"
                
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        buyer_id,
                        buyer,
                        'person',
                        'property_transaction',
                        json.dumps({'transaction_source': tx['url'], 'role': 'buyer'})
                    ))
                    
                    # Create PURCHASED_FROM relationship
                    cursor.execute("""
                        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                        VALUES (?, ?, ?, ?)
                    """, (
                        buyer_id,
                        entity_id,
                        'PURCHASED_FROM',
                        json.dumps({
                            'source': 'property_transaction',
                            'url': tx['url'],
                            'addresses': tx.get('addresses', []),
                            'dates': tx.get('dates', [])
                        })
                    ))
                    
                    print(f'  ✅ {buyer} --[PURCHASED_FROM]--> {tx["entity"]}')
                    imported += 1
                except Exception as e:
                    print(f'  ⚠️  Error importing buyer {buyer}: {e}')
        
        # Create seller entities and relationships
        for seller in tx.get('sellers', []):
            if 'strombeck' not in seller.lower():
                seller_id = f"person_{seller.lower().replace(' ', '_')[:50]}"
                
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        seller_id,
                        seller,
                        'person',
                        'property_transaction',
                        json.dumps({'transaction_source': tx['url'], 'role': 'seller'})
                    ))
                    
                    # Create SOLD_TO relationship
                    cursor.execute("""
                        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                        VALUES (?, ?, ?, ?)
                    """, (
                        entity_id,
                        seller_id,
                        'SOLD_TO',
                        json.dumps({
                            'source': 'property_transaction',
                            'url': tx['url'],
                            'addresses': tx.get('addresses', []),
                            'dates': tx.get('dates', [])
                        })
                    ))
                    
                    print(f'  ✅ {tx["entity"]} --[SOLD_TO]--> {seller}')
                    imported += 1
                except Exception as e:
                    print(f'  ⚠️  Error importing seller {seller}: {e}')
        
        # Create address entities if found
        for address in tx.get('addresses', []):
            addr_id = f"address_{address.lower().replace(' ', '_').replace('+', '').replace(',', '')[:50]}"
            
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    addr_id,
                    address,
                    'address',
                    'property_transaction',
                    json.dumps({'transaction_source': tx['url']})
                ))
                
                # Link entity to address
                cursor.execute("""
                    INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                    VALUES (?, ?, ?, ?)
                """, (
                    entity_id,
                    addr_id,
                    'OWNS',
                    json.dumps({'source': 'property_transaction', 'url': tx['url']})
                ))
                
                print(f'  ✅ {tx["entity"]} --[OWNS]--> {address}')
                imported += 1
            except Exception as e:
                pass
    
    conn.commit()
    conn.close()
    
    print(f'\n✅ Imported {imported} transaction relationships')
    return transactions_found

if __name__ == "__main__":
    search_property_transactions()
