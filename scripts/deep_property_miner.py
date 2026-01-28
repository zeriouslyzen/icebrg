#!/usr/bin/env python3
"""
Deep Property Miner - Aggressively queries county property databases
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

class DeepPropertyMiner:
    """Deep mining of property records."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.matrix_store = MatrixStore()
        self.findings = []
    
    def query_humboldt_assessor(self, search_term="Strombeck"):
        """Query Humboldt County Assessor property search."""
        print(f"\n=== QUERYING HUMBOLDT ASSESSOR: {search_term} ===\n")
        
        # Try the property search portal
        search_url = "https://www.co.humboldt.ca.us/assessor/search"
        
        try:
            # First, get the search page to see form structure
            response = self.session.get(search_url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for search forms
                forms = soup.find_all('form')
                print(f"Found {len(forms)} forms")
                
                # Try different search approaches
                search_params = [
                    {'owner': search_term},
                    {'name': search_term},
                    {'q': search_term},
                    {'search': search_term},
                ]
                
                for params in search_params:
                    try:
                        resp = self.session.get(search_url, params=params, timeout=15)
                        if resp.status_code == 200:
                            if search_term.lower() in resp.text.lower():
                                print(f"  ✅ Found matches with params: {params}")
                                # Extract property data
                                self._extract_property_data(resp.text, search_term)
                    except Exception as e:
                        print(f"  ⚠️  Error with {params}: {e}")
        except Exception as e:
            print(f"  ⚠️  Error accessing assessor: {e}")
    
    def _extract_property_data(self, html, search_term):
        """Extract property data from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for property listings, tables, or data
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    text = ' '.join([c.get_text(strip=True) for c in cells])
                    if search_term.lower() in text.lower():
                        print(f"    Found: {text[:100]}")
                        self.findings.append({
                            'type': 'property_record',
                            'data': text,
                            'search_term': search_term
                        })
    
    def query_addresses(self):
        """Query specific addresses from notes."""
        addresses = [
            "960 S G St Arcata",
            "Todd Court Arcata",
            "Western Avenue Arcata",
            "7th P St Eureka",
            "965 W Harris Eureka",
        ]
        
        print("\n=== QUERYING SPECIFIC ADDRESSES ===\n")
        
        for address in addresses:
            print(f"Searching: {address}")
            # Try assessor search
            self.query_humboldt_assessor(address)
            time.sleep(2)
    
    def run(self):
        """Run deep property mining."""
        print("=" * 80)
        print("DEEP PROPERTY MINING")
        print("=" * 80)
        
        # Search for Strombeck
        self.query_humboldt_assessor("Strombeck")
        
        # Search specific addresses
        self.query_addresses()
        
        # Save findings
        report_file = Path(__file__).parent.parent / "DEEP_PROPERTY_MINING.json"
        with open(report_file, 'w') as f:
            json.dump({
                'findings': self.findings,
                'count': len(self.findings)
            }, f, indent=2)
        
        print(f"\n✅ Deep mining complete! Found {len(self.findings)} property records")

if __name__ == "__main__":
    miner = DeepPropertyMiner()
    miner.run()
