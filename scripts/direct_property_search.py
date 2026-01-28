#!/usr/bin/env python3
"""
Direct Property Record Search
Uses multiple methods to find property transactions
"""

import sys
import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

def search_county_portals():
    """Search county property portals directly."""
    print('=' * 80)
    print('SEARCHING COUNTY PROPERTY PORTALS')
    print('=' * 80)
    
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # Humboldt County Assessor URLs
    urls = [
        'https://www.co.humboldt.ca.us/assessor',
        'https://www.co.humboldt.ca.us/assessor/search',
        'https://humboldt.assessor.gisworkshop.com',
    ]
    
    search_terms = [
        'Strombeck',
        'Steven Mark Strombeck',
        'Erik Strombeck',
        'Waltina Strombeck',
        'Strombeck Properties',
    ]
    
    findings = []
    
    for url in urls:
        print(f'\nTrying: {url}')
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for search forms
                forms = soup.find_all('form')
                inputs = soup.find_all('input', {'type': ['text', 'search']})
                
                print(f'  Found {len(forms)} forms, {len(inputs)} inputs')
                
                # Try to find property search functionality
                for term in search_terms:
                    # Try POST with search term
                    try:
                        search_url = f"{url}?q={term}&owner={term}"
                        search_response = requests.get(search_url, timeout=10)
                        if search_response.status_code == 200:
                            content = search_response.text.lower()
                            if term.lower() in content:
                                print(f'  ✅ Found results for: {term}')
                                findings.append({
                                    'url': search_url,
                                    'term': term,
                                    'found': True
                                })
                    except Exception as e:
                        pass
        except Exception as e:
            print(f'  ⚠️  Error: {e}')
    
    # Also try property data APIs/services
    print('\n\nTrying property data services...')
    
    # Zillow API (if available)
    # PropertyRadar (if available)
    # CoreLogic (if available)
    
    # For now, save what we found
    with open('COUNTY_PORTAL_SEARCH.json', 'w') as f:
        json.dump(findings, f, indent=2)
    
    print(f'\n✅ Searched county portals')
    print(f'   Found {len(findings)} potential sources')
    
    conn.close()
    return findings

if __name__ == "__main__":
    search_county_portals()
