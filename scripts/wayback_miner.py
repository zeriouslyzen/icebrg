#!/usr/bin/env python3
"""
Wayback Machine / Internet Archive Miner
Gathers historical data from last 3 years for Strombeck investigation
"""

import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

class WaybackMiner:
    """Mine historical data from Wayback Machine."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.matrix_store = MatrixStore()
        self.findings = []
        self.wayback_api = "https://web.archive.org"
        
        # Calculate date range (last 3 years)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=3*365)
        
    def get_wayback_snapshots(self, url, start_date=None, end_date=None):
        """Get all Wayback Machine snapshots for a URL."""
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        
        # Wayback CDX API
        cdx_url = f"{self.wayback_api}/cdx/search/cdx"
        params = {
            'url': url,
            'output': 'json',
            'from': start_date.strftime('%Y%m%d'),
            'to': end_date.strftime('%Y%m%d'),
            'collapse': 'urlkey'  # One snapshot per URL
        }
        
        try:
            print(f"  Fetching snapshots for: {url}")
            response = self.session.get(cdx_url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:  # First item is header
                    snapshots = data[1:]
                    print(f"    Found {len(snapshots)} snapshots")
                    return snapshots
            return []
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            return []
    
    def get_snapshot_content(self, timestamp, url):
        """Get the actual content of a Wayback snapshot."""
        wayback_url = f"{self.wayback_api}/web/{timestamp}/{url}"
        
        try:
            response = self.session.get(wayback_url, timeout=30, allow_redirects=True)
            if response.status_code == 200:
                return response.text
        except requests.exceptions.Timeout:
            # Try with longer timeout
            try:
                response = self.session.get(wayback_url, timeout=60, allow_redirects=True)
                if response.status_code == 200:
                    return response.text
            except Exception as e2:
                print(f"    ⚠️  Timeout even with extended timeout: {e2}")
        except Exception as e:
            print(f"    ⚠️  Error fetching snapshot: {e}")
        return None
    
    def mine_strombeck_website(self):
        """Mine historical versions of Strombeck Properties website."""
        print("\n=== MINING STROMBECK PROPERTIES WEBSITE HISTORY ===\n")
        
        urls = [
            "strombeckproperties.com",
            "www.strombeckproperties.com",
            "strombeckproperties.com/staff.html",
            "strombeckproperties.com/about.html",
        ]
        
        for url in urls:
            full_url = f"http://{url}" if not url.startswith('http') else url
            snapshots = self.get_wayback_snapshots(full_url)
            
            if snapshots:
                # Get most recent and oldest snapshots
                recent = snapshots[-1] if snapshots else None
                oldest = snapshots[0] if snapshots else None
                
                if recent:
                    timestamp = recent[1]  # CDX format: [urlkey, timestamp, original, ...]
                    print(f"  Most recent: {timestamp}")
                    content = self.get_snapshot_content(timestamp, full_url)
                    if content:
                        self._extract_website_data(content, url, timestamp)
                
                if oldest and oldest != recent:
                    timestamp = oldest[1]
                    print(f"  Oldest: {timestamp}")
                    content = self.get_snapshot_content(timestamp, full_url)
                    if content:
                        self._extract_website_data(content, url, timestamp)
            
            time.sleep(2)  # Rate limiting
    
    def _extract_website_data(self, html, url, timestamp):
        """Extract data from website HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract text content
        text = soup.get_text()
        
        # Look for specific information
        findings = {
            'url': url,
            'timestamp': timestamp,
            'phone_numbers': [],
            'addresses': [],
            'names': [],
            'properties': []
        }
        
        # Find phone numbers
        import re
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        findings['phone_numbers'] = list(set(phones))
        
        # Find addresses (Arcata, Eureka patterns)
        address_pattern = r'\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Court|Rd|Road|Blvd|Boulevard)[\s,]*Arcata|Eureka'
        addresses = re.findall(address_pattern, text, re.IGNORECASE)
        findings['addresses'] = list(set(addresses))
        
        # Find names (Strombeck variations)
        name_pattern = r'(?:Steven|Mark|Waltina|Martha|Erik|Christian)\s+Strombeck'
        names = re.findall(name_pattern, text, re.IGNORECASE)
        findings['names'] = list(set(names))
        
        # Look for property listings
        property_keywords = ['property', 'rental', 'apartment', 'unit', 'location']
        for keyword in property_keywords:
            if keyword in text.lower():
                # Try to extract property info
                pass
        
        if any([findings['phone_numbers'], findings['addresses'], findings['names']]):
            self.findings.append({
                'type': 'website_snapshot',
                'data': findings
            })
            print(f"    Extracted: {len(findings['phone_numbers'])} phones, {len(findings['addresses'])} addresses, {len(findings['names'])} names")
    
    def mine_redwood_capital_bank(self):
        """Mine historical data about Redwood Capital Bank and Steven Strombeck."""
        print("\n=== MINING REDWOOD CAPITAL BANK HISTORY ===\n")
        
        urls = [
            "redwoodcapitalbank.com",
            "investor.redwoodcapitalbank.com",
        ]
        
        for url in urls:
            full_url = f"http://{url}" if not url.startswith('http') else url
            snapshots = self.get_wayback_snapshots(full_url)
            
            if snapshots:
                # Get a few snapshots across the 3-year period
                sample_size = min(5, len(snapshots))
                step = len(snapshots) // sample_size if sample_size > 0 else 1
                
                for i in range(0, len(snapshots), step):
                    snapshot = snapshots[i]
                    timestamp = snapshot[1]
                    print(f"  Snapshot {timestamp}")
                    content = self.get_snapshot_content(timestamp, full_url)
                    if content:
                        if 'strombeck' in content.lower():
                            print(f"    ✅ Found Strombeck mention!")
                            self.findings.append({
                                'type': 'bank_snapshot',
                                'url': url,
                                'timestamp': timestamp,
                                'has_strombeck': True
                            })
            
            time.sleep(2)
    
    def mine_property_listings(self):
        """Mine historical property listing sites."""
        print("\n=== MINING PROPERTY LISTING HISTORY ===\n")
        
        # Search for property listings that may have been removed
        search_queries = [
            "strombeck properties arcata",
            "960 S G St Arcata",
            "strombeck rental arcata",
        ]
        
        for query in search_queries:
            # Wayback doesn't have direct search, but we can try common property sites
            property_sites = [
                f"zillow.com/homes/{query.replace(' ', '-')}",
                f"realtor.com/realestateandhomes-search/{query.replace(' ', '-')}",
            ]
            
            for site_url in property_sites:
                full_url = f"https://{site_url}"
                snapshots = self.get_wayback_snapshots(full_url)
                if snapshots:
                    print(f"  Found {len(snapshots)} snapshots for {site_url}")
                    # Get most recent
                    if snapshots:
                        recent = snapshots[-1]
                        timestamp = recent[1]
                        content = self.get_snapshot_content(timestamp, full_url)
                        if content and 'strombeck' in content.lower():
                            print(f"    ✅ Found Strombeck property listing!")
                            self.findings.append({
                                'type': 'property_listing',
                                'url': site_url,
                                'timestamp': timestamp
                            })
    
    def mine_news_articles(self):
        """Mine historical news articles about Strombeck."""
        print("\n=== MINING NEWS ARTICLE HISTORY ===\n")
        
        # Try local news sites
        news_sites = [
            "times-standard.com",
            "northcoastjournal.com",
            "lostcoastoutpost.com",
        ]
        
        for site in news_sites:
            # Search for Strombeck articles
            search_url = f"https://{site}/search?q=strombeck"
            snapshots = self.get_wayback_snapshots(search_url)
            
            if snapshots:
                print(f"  Found {len(snapshots)} snapshots for {site}")
                # Get a few snapshots
                for snapshot in snapshots[-3:]:  # Last 3
                    timestamp = snapshot[1]
                    content = self.get_snapshot_content(timestamp, search_url)
                    if content:
                        soup = BeautifulSoup(content, 'html.parser')
                        # Look for article links
                        links = soup.find_all('a', href=True)
                        for link in links:
                            href = link.get('href', '')
                            text = link.get_text()
                            if 'strombeck' in text.lower() or 'strombeck' in href.lower():
                                print(f"    Found article: {text[:60]}")
                                self.findings.append({
                                    'type': 'news_article',
                                    'site': site,
                                    'title': text,
                                    'url': href,
                                    'timestamp': timestamp
                                })
            
            time.sleep(2)
    
    def mine_yelp_reviews(self):
        """Mine historical Yelp reviews."""
        print("\n=== MINING YELP REVIEW HISTORY ===\n")
        
        yelp_url = "https://www.yelp.com/biz/strombeck-properties-arcata"
        snapshots = self.get_wayback_snapshots(yelp_url)
        
        if snapshots:
            print(f"  Found {len(snapshots)} Yelp snapshots")
            # Get snapshots over time to see review changes
            sample_size = min(10, len(snapshots))
            step = len(snapshots) // sample_size if sample_size > 0 else 1
            
            for i in range(0, len(snapshots), step):
                snapshot = snapshots[i]
                timestamp = snapshot[1]
                content = self.get_snapshot_content(timestamp, yelp_url)
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    # Extract review count, rating
                    review_text = soup.get_text()
                    if 'review' in review_text.lower():
                        # Try to extract review count
                        import re
                        review_count_match = re.search(r'(\d+)\s+review', review_text, re.IGNORECASE)
                        rating_match = re.search(r'(\d+\.?\d*)\s+star', review_text, re.IGNORECASE)
                        
                        review_count = review_count_match.group(1) if review_count_match else None
                        rating = rating_match.group(1) if rating_match else None
                        
                        self.findings.append({
                            'type': 'yelp_snapshot',
                            'timestamp': timestamp,
                            'review_count': review_count,
                            'rating': rating
                        })
                        print(f"    {timestamp}: {review_count} reviews, {rating} stars")
    
    def import_findings_to_database(self):
        """Import historical findings to Matrix database."""
        print("\n=== IMPORTING HISTORICAL DATA TO DATABASE ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        imported = 0
        
        for finding in self.findings:
            if finding['type'] == 'website_snapshot':
                data = finding['data']
                
                # Add phone numbers as entities if new
                for phone in data.get('phone_numbers', []):
                    entity_id = f"phone_{phone.replace('-', '').replace('(', '').replace(')', '').replace(' ', '')}"
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                            VALUES (?, ?, ?, ?, ?)
                        """, (entity_id, phone, 'phone', 'wayback_machine', json.dumps({
                            'timestamp': data['timestamp'],
                            'source_url': data['url']
                        })))
                        imported += 1
                    except Exception as e:
                        print(f"  ⚠️  Error adding phone: {e}")
                
                # Add addresses
                for address in data.get('addresses', []):
                    entity_id = f"address_{address.lower().replace(' ', '_').replace(',', '')[:50]}"
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                            VALUES (?, ?, ?, ?, ?)
                        """, (entity_id, address, 'address', 'wayback_machine', json.dumps({
                            'timestamp': data['timestamp'],
                            'source_url': data['url']
                        })))
                        imported += 1
                    except Exception as e:
                        print(f"  ⚠️  Error adding address: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Imported {imported} historical entities")
    
    def run(self):
        """Run full Wayback mining operation."""
        print("=" * 80)
        print("WAYBACK MACHINE / INTERNET ARCHIVE MINING")
        print(f"Date Range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print("=" * 80)
        
        self.mine_strombeck_website()
        self.mine_redwood_capital_bank()
        self.mine_property_listings()
        self.mine_news_articles()
        self.mine_yelp_reviews()
        
        # Import to database
        self.import_findings_to_database()
        
        # Save report
        report_file = Path(__file__).parent.parent / "WAYBACK_MINING_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'date_range': {
                    'start': self.start_date.isoformat(),
                    'end': self.end_date.isoformat()
                },
                'findings_count': len(self.findings),
                'findings': self.findings
            }, f, indent=2)
        
        print(f"\n✅ Wayback mining complete! Found {len(self.findings)} historical records")
        print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    miner = WaybackMiner()
    miner.run()
