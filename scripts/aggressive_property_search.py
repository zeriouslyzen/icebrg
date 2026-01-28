#!/usr/bin/env python3
"""
Aggressive Property Transaction Search
Uses multiple methods to find property buy/sell records
"""

import sys
import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def search_all_sources():
    """Search all available sources for property transactions."""
    print('=' * 80)
    print('AGGRESSIVE PROPERTY TRANSACTION SEARCH')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    targets = [
        'Steven Mark Strombeck',
        'Erik Strombeck',
        'Waltina Martha Strombeck',
        'Strombeck Properties',
    ]
    
    addresses = [
        '960 S G St Arcata',
        'Todd Court Arcata',
        'Western Avenue Arcata',
        '7th P St Eureka',
        '965 W Harris Eureka',
        '4422 Westwood Gardens',
    ]
    
    transactions = []
    
    # Method 1: Direct county portal searches
    print('\n=== METHOD 1: COUNTY PORTAL SEARCHES ===\n')
    
    base_urls = [
        'https://www.co.humboldt.ca.us/assessor/search',
        'https://www.co.humboldt.ca.us/recorder',
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    for url in base_urls:
        print(f'\nSearching: {url}')
        for target in targets:
            try:
                # Try GET with search params
                params = {'q': target, 'owner': target, 'name': target}
                response = session.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text().lower()
                    
                    if target.lower() in text or 'strombeck' in text:
                        print(f'  ✅ Found mention of: {target}')
                        
                        # Extract any property/deed info
                        # Look for dates, addresses, names
                        date_pattern = r'\b(20\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s,]+(?:20\d{2})?\b'
                        dates = re.findall(date_pattern, response.text, re.IGNORECASE)
                        
                        if dates:
                            transactions.append({
                                'source': url,
                                'target': target,
                                'type': 'county_portal',
                                'dates': dates,
                                'url': response.url
                            })
                            print(f'    Found dates: {dates[:3]}')
                
                time.sleep(1)
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
    
    # Method 2: Search addresses
    print('\n=== METHOD 2: ADDRESS SEARCHES ===\n')
    
    for address in addresses:
        print(f'\nSearching: {address}')
        try:
            # Search assessor by address
            search_url = 'https://www.co.humboldt.ca.us/assessor/search'
            params = {'address': address, 'q': address}
            response = session.get(search_url, params=params, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text().lower()
                
                if address.lower().replace(' ', '') in text.replace(' ', ''):
                    print(f'  ✅ Found address: {address}')
                    
                    # Extract owner names
                    name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?)\b'
                    names = re.findall(name_pattern, response.text)
                    strombeck_names = [n for n in names if 'strombeck' in n.lower()]
                    
                    if strombeck_names:
                        transactions.append({
                            'source': search_url,
                            'address': address,
                            'owners': strombeck_names,
                            'type': 'address_search'
                        })
                        print(f'    Owners: {strombeck_names}')
            
            time.sleep(1)
        except Exception as e:
            print(f'  ⚠️  Error: {e}')
    
    # Method 3: Try property data sites
    print('\n=== METHOD 3: PROPERTY DATA SITES ===\n')
    
    property_sites = [
        f'https://www.zillow.com/homes/{addr.replace(" ", "-")}_rb/' for addr in addresses[:2]
    ]
    
    for site_url in property_sites:
        try:
            print(f'Checking: {site_url[:60]}...')
            response = session.get(site_url, timeout=10)
            if response.status_code == 200:
                if 'strombeck' in response.text.lower():
                    print(f'  ✅ Found Strombeck mention')
        except Exception as e:
            pass
    
    # Save findings
    with open('AGGRESSIVE_PROPERTY_SEARCH.json', 'w') as f:
        json.dump(transactions, f, indent=2)
    
    print(f'\n\n✅ Search complete! Found {len(transactions)} potential transaction sources')
    print('Saved to: AGGRESSIVE_PROPERTY_SEARCH.json')
    
    # Import any findings
    if transactions:
        print('\n\nImporting findings to database...')
        imported = 0
        
        for tx in transactions:
            if 'owners' in tx:
                for owner in tx['owners']:
                    owner_id = f"person_{owner.lower().replace(' ', '_')[:50]}"
                    addr_id = f"address_{tx['address'].lower().replace(' ', '_').replace('+', '').replace(',', '')[:50]}"
                    
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                            VALUES (?, ?, ?, ?, ?)
                        """, (owner_id, owner, 'person', 'property_search', json.dumps({'address': tx['address']})))
                        
                        cursor.execute("""
                            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                            VALUES (?, ?, ?, ?)
                        """, (owner_id, addr_id, 'OWNS', json.dumps({'source': 'property_search'})))
                        
                        imported += 1
                    except Exception as e:
                        pass
        
        conn.commit()
        print(f'✅ Imported {imported} relationships')
    
    conn.close()
    return transactions

if __name__ == "__main__":
    search_all_sources()
