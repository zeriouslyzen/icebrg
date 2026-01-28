#!/usr/bin/env python3
"""
Strombeck Intelligence Mining Script
Aggressively gathers data from multiple sources and imports to PEGASUS Matrix database.
"""

import os
import sys
import json
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import time
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore
from src.iceburg.search.web_search import WebSearchAggregator

# Enable web search
os.environ['ICEBURG_ENABLE_WEB'] = '1'

class StrombeckIntelligenceMiner:
    """Aggressive intelligence mining for Strombeck properties."""
    
    def __init__(self):
        self.matrix_store = MatrixStore()
        self.web_search = WebSearchAggregator()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.findings = []
        
    def mine_humboldt_county_property_records(self):
        """Attempt to access Humboldt County property databases."""
        print("\n=== MINING HUMBOLDT COUNTY PROPERTY RECORDS ===\n")
        
        # County assessor portal URLs
        urls = [
            "https://humboldt.assessor.gisworkshop.com/",
            "https://www.co.humboldt.ca.us/assessor",
            "https://www.co.humboldt.ca.us/recorder",
            "https://humboldtcountyca.gov/assessor",
        ]
        
        for url in urls:
            try:
                print(f"Attempting: {url}")
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ Accessible: {url}")
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for search forms or property search links
                    search_links = soup.find_all('a', href=True)
                    for link in search_links:
                        href = link.get('href', '').lower()
                        text = link.get_text().lower()
                        if any(term in href or term in text for term in ['property', 'search', 'parcel', 'assessor']):
                            print(f"    Found search link: {link.get_text()[:60]}")
                            self.findings.append({
                                'type': 'county_portal',
                                'url': url,
                                'search_link': link.get('href'),
                                'text': link.get_text()
                            })
                else:
                    print(f"  ❌ Status {response.status_code}")
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
            
            time.sleep(1)  # Rate limiting
        
        # Try direct property search endpoints
        search_endpoints = [
            "https://humboldt.assessor.gisworkshop.com/search",
            "https://www.co.humboldt.ca.us/assessor/search",
        ]
        
        for endpoint in search_endpoints:
            try:
                # Try searching for Strombeck
                params = {'q': 'Strombeck', 'owner': 'Strombeck'}
                response = self.session.get(endpoint, params=params, timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ Search endpoint accessible: {endpoint}")
                    self.findings.append({
                        'type': 'search_endpoint',
                        'url': endpoint,
                        'status': 'accessible'
                    })
            except Exception as e:
                print(f"  ⚠️  Search endpoint error: {e}")
    
    def mine_california_business_records(self):
        """Mine California Secretary of State business records."""
        print("\n=== MINING CALIFORNIA BUSINESS RECORDS ===\n")
        
        # CA SOS business search
        sos_url = "https://bizfileonline.sos.ca.gov/search/business"
        
        searches = [
            "Strombeck Properties",
            "Strombeck Properties LLC",
            "STEATA LLC",
            "STEATA",
        ]
        
        for search_term in searches:
            try:
                print(f"Searching: {search_term}")
                # CA SOS uses POST with form data
                data = {
                    'SearchType': 'CORP',
                    'SearchCriteria': search_term,
                    'SearchSubType': 'Keyword'
                }
                
                response = self.session.post(sos_url, data=data, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for results table
                    results = soup.find_all('tr', class_=lambda x: x and 'result' in x.lower())
                    if results:
                        print(f"  ✅ Found {len(results)} results")
                        for result in results[:5]:
                            cells = result.find_all('td')
                            if cells:
                                entity_name = cells[0].get_text(strip=True) if len(cells) > 0 else ""
                                entity_num = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                                print(f"    - {entity_name} ({entity_num})")
                                self.findings.append({
                                    'type': 'business_record',
                                    'entity_name': entity_name,
                                    'entity_number': entity_num,
                                    'search_term': search_term
                                })
                    else:
                        print(f"  ⚠️  No results found")
                else:
                    print(f"  ❌ Status {response.status_code}")
                    
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
            
            time.sleep(2)  # Rate limiting
    
    def mine_property_data_services(self):
        """Mine property data from various services."""
        print("\n=== MINING PROPERTY DATA SERVICES ===\n")
        
        # Try various property search sites
        property_sites = [
            ("PropertyRadar", "https://www.propertyradar.com"),
            ("Zillow", "https://www.zillow.com/homes/Strombeck-Arcata-CA_rb/"),
            ("Realtor.com", "https://www.realtor.com/realestateandhomes-search/Arcata_CA"),
            ("Redfin", "https://www.redfin.com/city/307/CA/Arcata"),
        ]
        
        for site_name, base_url in property_sites:
            try:
                print(f"Checking {site_name}...")
                response = self.session.get(base_url, timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ {site_name} accessible")
                    # Look for Strombeck mentions
                    if 'strombeck' in response.text.lower():
                        print(f"    ⚠️  Found Strombeck mention!")
                        self.findings.append({
                            'type': 'property_service',
                            'service': site_name,
                            'url': base_url,
                            'has_data': True
                        })
            except Exception as e:
                print(f"  ⚠️  {site_name} error: {e}")
            
            time.sleep(1)
    
    def mine_web_intelligence(self):
        """Mine web intelligence using ICEBURG search."""
        print("\n=== MINING WEB INTELLIGENCE ===\n")
        
        queries = [
            "Strombeck Properties Arcata property records",
            "Strombeck Properties Humboldt County assessor",
            "Strombeck property sales Arcata California",
            "Steven Strombeck property ownership Humboldt",
            "STEATA LLC California business records",
            "Strombeck Properties LLC California",
            "960 S G St Arcata property owner",
            "Todd Court Arcata property records",
            "Western Avenue Arcata property owner",
            "7th P St Eureka property owner",
            "965 W Harris Eureka property owner",
            "Strombeck Properties tenant complaints",
            "Strombeck Properties lawsuits",
            "Strombeck Properties business license",
        ]
        
        all_results = []
        for query in queries:
            print(f"Searching: {query[:60]}...")
            try:
                results = self.web_search.search(query, sources=['ddg'], max_results_per_source=5)
                all_results.extend(results)
                print(f"  Found {len(results)} results")
                
                # Check for relevant results
                for r in results:
                    if any(term in r.title.lower() or term in r.snippet.lower() 
                           for term in ['strombeck', 'arcata', 'eureka', 'property', 'humboldt']):
                        self.findings.append({
                            'type': 'web_intelligence',
                            'title': r.title,
                            'url': r.url,
                            'snippet': r.snippet[:200],
                            'query': query
                        })
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
            
            time.sleep(1)  # Rate limiting
        
        print(f"\nTotal web intelligence gathered: {len(all_results)} sources")
    
    def mine_court_records(self):
        """Mine court records for Strombeck-related cases."""
        print("\n=== MINING COURT RECORDS ===\n")
        
        # Humboldt County Superior Court
        court_urls = [
            "https://www.humboldt.courts.ca.gov/",
            "https://www.humboldt.courts.ca.gov/online-services",
        ]
        
        for url in court_urls:
            try:
                print(f"Checking: {url}")
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ Court portal accessible")
                    # Look for case search links
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link.get('href', '').lower()
                        text = link.get_text().lower()
                        if any(term in href or term in text for term in ['case', 'search', 'records']):
                            print(f"    Found: {link.get_text()[:60]}")
                            self.findings.append({
                                'type': 'court_portal',
                                'url': url,
                                'search_link': link.get('href')
                            })
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
    
    def import_to_matrix_database(self):
        """Import findings into Matrix database."""
        print("\n=== IMPORTING TO MATRIX DATABASE ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        imported_count = 0
        
        # Create entities for findings
        entities_to_add = []
        
        # Strombeck Properties company
        entities_to_add.append({
            'entity_id': 'strombeck_properties_company',
            'name': 'Strombeck Properties',
            'entity_type': 'company',
            'source': 'intelligence_mining',
            'properties': json.dumps({
                'phone': '(707) 822-4557',
                'established': '1990',
                'location': 'Arcata, CA',
                'website': 'strombeckproperties.com',
                'properties_count': 6,
                'focus': 'Student housing'
            })
        })
        
        # Steven Mark Strombeck
        entities_to_add.append({
            'entity_id': 'steven_mark_strombeck',
            'name': 'Steven Mark Strombeck',
            'entity_type': 'person',
            'source': 'intelligence_mining',
            'properties': json.dumps({
                'phone': '(707) 822-4557',
                'education': 'Sacramento State University (1982)',
                'connections': ['Redwood Capital Bank', 'Strombeck Properties']
            })
        })
        
        # Add other family members from notes
        for name, entity_id in [
            ('Waltina Martha Strombeck', 'waltina_martha_strombeck'),
            ('Erik Strombeck', 'erik_strombeck'),
        ]:
            entities_to_add.append({
                'entity_id': entity_id,
                'name': name,
                'entity_type': 'person',
                'source': 'intelligence_mining',
                'properties': json.dumps({'family': 'Strombeck'})
            })
        
        # Add properties/addresses
        addresses = [
            ('960 S G St Arcata', 'address_960_s_g_st_arcata'),
            ('Todd Court Arcata', 'address_todd_court_arcata'),
            ('Western Avenue Arcata', 'address_western_ave_arcata'),
            ('7th + P St Eureka', 'address_7th_p_st_eureka'),
            ('965 W Harris Eureka', 'address_965_w_harris_eureka'),
            ('4422 Westwood Gardens', 'address_4422_westwood_gardens'),
        ]
        
        for address, entity_id in addresses:
            entities_to_add.append({
                'entity_id': entity_id,
                'name': address,
                'entity_type': 'address',
                'source': 'intelligence_mining',
                'properties': json.dumps({'location': address})
            })
        
        # STEATA LLC
        entities_to_add.append({
            'entity_id': 'steata_llc',
            'name': 'STEATA LLC',
            'entity_type': 'company',
            'source': 'intelligence_mining',
            'properties': json.dumps({'status': 'investigating'})
        })
        
        # Insert entities
        for entity in entities_to_add:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO entities 
                    (entity_id, name, entity_type, source, properties)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entity['entity_id'],
                    entity['name'],
                    entity['entity_type'],
                    entity['source'],
                    entity['properties']
                ))
                imported_count += 1
                print(f"  ✅ Added: {entity['name']}")
            except Exception as e:
                print(f"  ⚠️  Error adding {entity['name']}: {e}")
        
        # Create relationships
        relationships_to_add = [
            ('steven_mark_strombeck', 'strombeck_properties_company', 'OWNS'),
            ('steven_mark_strombeck', 'waltina_martha_strombeck', 'FAMILY_OF'),
            ('steven_mark_strombeck', 'erik_strombeck', 'FAMILY_OF'),
            ('strombeck_properties_company', 'address_960_s_g_st_arcata', 'OWNS'),
            ('strombeck_properties_company', 'address_todd_court_arcata', 'OWNS'),
            ('strombeck_properties_company', 'address_western_ave_arcata', 'OWNS'),
        ]
        
        for source_id, target_id, rel_type in relationships_to_add:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO relationships
                    (source_id, target_id, relationship_type)
                    VALUES (?, ?, ?)
                """, (source_id, target_id, rel_type))
                print(f"  ✅ Relationship: {source_id} --[{rel_type}]--> {target_id}")
            except Exception as e:
                print(f"  ⚠️  Relationship error: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Imported {imported_count} entities to Matrix database")
    
    def run_full_mining(self):
        """Run all mining operations."""
        print("=" * 80)
        print("STROMBECK INTELLIGENCE MINING - FULL OPERATION")
        print("=" * 80)
        
        self.mine_humboldt_county_property_records()
        self.mine_california_business_records()
        self.mine_property_data_services()
        self.mine_web_intelligence()
        self.mine_court_records()
        
        # Import to database
        self.import_to_matrix_database()
        
        # Save findings report
        report_file = Path(__file__).parent.parent / "STROMBECK_MINING_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'findings_count': len(self.findings),
                'findings': self.findings
            }, f, indent=2)
        
        print(f"\n✅ Mining complete! Findings saved to: {report_file}")
        print(f"Total findings: {len(self.findings)}")


if __name__ == "__main__":
    miner = StrombeckIntelligenceMiner()
    miner.run_full_mining()
