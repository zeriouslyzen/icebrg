#!/usr/bin/env python3
"""
Selenium-based County Portal Scraper
Handles JavaScript-heavy portals and form submissions for property records
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not installed. Install with: pip install selenium")
    print("   Also need ChromeDriver: brew install chromedriver")

class SeleniumCountyScraper:
    """Selenium-based scraper for county property portals."""
    
    def __init__(self, headless: bool = True):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available")
        
        self.matrix_store = MatrixStore()
        self.findings = []
        self.driver = None
        self.headless = headless
        
    def setup_driver(self):
        """Setup Chrome WebDriver."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            print("✅ ChromeDriver initialized")
        except Exception as e:
            print(f"❌ Failed to initialize ChromeDriver: {e}")
            print("   Install ChromeDriver: brew install chromedriver")
            raise
    
    def scrape_humboldt_assessor(self, search_term: str = "Strombeck"):
        """Scrape Humboldt County Assessor portal."""
        print(f"\n=== SCRAPING HUMBOLDT ASSESSOR: {search_term} ===\n")
        
        try:
            # Navigate to assessor portal
            url = "https://www.co.humboldt.ca.us/assessor"
            print(f"Navigating to: {url}")
            self.driver.get(url)
            time.sleep(3)
            
            # Wait for page to load
            time.sleep(5)
            
            # Look for property search link
            try:
                # Try to find search link - wait for it to be clickable
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                
                wait = WebDriverWait(self.driver, 10)
                
                # Try multiple ways to find search
                search_links = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "Search")
                if not search_links:
                    search_links = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "Property")
                if not search_links:
                    search_links = self.driver.find_elements(By.XPATH, "//a[contains(text(), 'search') or contains(text(), 'Search')]")
                
                if search_links:
                    print(f"Found {len(search_links)} search links")
                    # Try to click first clickable link
                    for link in search_links:
                        try:
                            wait.until(EC.element_to_be_clickable(link))
                            link.click()
                            time.sleep(5)
                            break
                        except:
                            continue
                
                # Look for search form - wait for inputs to be visible
                search_inputs = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input[type='text'], input[type='search'], input[name*='search'], input[id*='search']")))
                
                if search_inputs:
                    print(f"Found {len(search_inputs)} search inputs")
                    # Try to enter search term in first visible input
                    for inp in search_inputs:
                        try:
                            # Scroll to element
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", inp)
                            time.sleep(1)
                            
                            # Wait for element to be visible and interactable
                            wait.until(EC.element_to_be_clickable(inp))
                            
                            inp.clear()
                            inp.send_keys(search_term)
                            time.sleep(2)
                            
                            # Look for submit button
                            submit_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], button[class*='submit'], button[id*='submit']")
                            if not submit_buttons:
                                # Try pressing Enter
                                from selenium.webdriver.common.keys import Keys
                                inp.send_keys(Keys.RETURN)
                                time.sleep(5)
                            else:
                                for btn in submit_buttons:
                                    try:
                                        wait.until(EC.element_to_be_clickable(btn))
                                        btn.click()
                                        time.sleep(5)
                                        break
                                    except:
                                        continue
                            
                            # Extract results
                            self._extract_property_results(search_term)
                            break
                        except Exception as e:
                            print(f"  ⚠️  Error with input: {e}")
                            continue
                
            except Exception as e:
                print(f"  ⚠️  Error navigating: {e}")
                # Try direct search URL
                search_url = f"https://www.co.humboldt.ca.us/assessor/search?q={search_term}"
                print(f"Trying direct URL: {search_url}")
                self.driver.get(search_url)
                time.sleep(5)
                self._extract_property_results(search_term)
                
        except Exception as e:
            print(f"❌ Scraping error: {e}")
    
    def _extract_property_results(self, search_term: str):
        """Extract property data from search results."""
        try:
            # Get page source
            page_source = self.driver.page_source
            
            # Look for property data in various formats
            # Tables
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        row_text = ' '.join([c.text for c in cells])
                        if search_term.lower() in row_text.lower():
                            print(f"  ✅ Found property data: {row_text[:100]}")
                            self.findings.append({
                                'type': 'property_record',
                                'data': row_text,
                                'search_term': search_term,
                                'source': 'humboldt_assessor'
                            })
            
            # Lists/divs with property info
            property_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                "div.property, div.result, li.property, .property-item")
            for elem in property_elements:
                text = elem.text
                if search_term.lower() in text.lower() and len(text) > 20:
                    print(f"  ✅ Found property: {text[:100]}")
                    self.findings.append({
                        'type': 'property_record',
                        'data': text,
                        'search_term': search_term,
                        'source': 'humboldt_assessor'
                    })
            
            # Look for links to property details
            links = self.driver.find_elements(By.TAG_NAME, "a")
            for link in links:
                href = link.get_attribute('href') or ''
                text = link.text
                if search_term.lower() in text.lower() or 'property' in href.lower():
                    print(f"  Found property link: {text[:60]} -> {href[:60]}")
                    # Could navigate to detail page here
            
        except Exception as e:
            print(f"  ⚠️  Error extracting results: {e}")
    
    def scrape_humboldt_recorder(self, search_term: str = "Strombeck"):
        """Scrape Humboldt County Recorder portal for property deeds."""
        print(f"\n=== SCRAPING HUMBOLDT RECORDER: {search_term} ===\n")
        
        try:
            url = "https://www.co.humboldt.ca.us/recorder"
            print(f"Navigating to: {url}")
            self.driver.get(url)
            time.sleep(3)
            
            # Look for search/self-service link
            try:
                search_links = self.driver.find_elements(By.PARTIAL_LINK_TEXT, "Search")
                if search_links:
                    search_links[0].click()
                    time.sleep(3)
                    
                    # Look for grantor/grantee search
                    inputs = self.driver.find_elements(By.CSS_SELECTOR, "input")
                    for inp in inputs:
                        try:
                            placeholder = inp.get_attribute('placeholder') or ''
                            name = inp.get_attribute('name') or ''
                            if 'grantor' in placeholder.lower() or 'grantor' in name.lower() or 'name' in placeholder.lower():
                                inp.clear()
                                inp.send_keys(search_term)
                                time.sleep(1)
                                
                                # Submit
                                submit = self.driver.find_elements(By.CSS_SELECTOR, "button[type='submit'], input[type='submit']")
                                if submit:
                                    submit[0].click()
                                    time.sleep(5)
                                    self._extract_deed_results(search_term)
                                    break
                        except Exception as e:
                            continue
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                
        except Exception as e:
            print(f"❌ Recorder scraping error: {e}")
    
    def _extract_deed_results(self, search_term: str):
        """Extract deed/transaction data."""
        try:
            # Look for deed records
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 3:
                        row_data = {
                            'grantor': cells[0].text if len(cells) > 0 else '',
                            'grantee': cells[1].text if len(cells) > 1 else '',
                            'date': cells[2].text if len(cells) > 2 else '',
                            'document': cells[3].text if len(cells) > 3 else ''
                        }
                        
                        if search_term.lower() in str(row_data).lower():
                            print(f"  ✅ Found deed record: {row_data}")
                            self.findings.append({
                                'type': 'deed_record',
                                'data': row_data,
                                'search_term': search_term,
                                'source': 'humboldt_recorder'
                            })
        except Exception as e:
            print(f"  ⚠️  Error extracting deeds: {e}")
    
    def scrape_addresses(self, addresses: List[str]):
        """Scrape specific addresses."""
        print("\n=== SCRAPING SPECIFIC ADDRESSES ===\n")
        
        for address in addresses:
            print(f"\nSearching: {address}")
            self.scrape_humboldt_assessor(address)
            time.sleep(3)
    
    def import_findings_to_database(self):
        """Import findings to Matrix database with transaction relationships."""
        print("\n=== IMPORTING TO DATABASE ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        imported = 0
        
        for finding in self.findings:
            if finding['type'] == 'deed_record':
                data = finding['data']
                grantor = data.get('grantor', '').strip()
                grantee = data.get('grantee', '').strip()
                date = data.get('date', '').strip()
                
                if grantor and grantee:
                    # Create entities if needed
                    grantor_id = f"person_{grantor.lower().replace(' ', '_')[:50]}"
                    grantee_id = f"person_{grantee.lower().replace(' ', '_')[:50]}"
                    
                    # Add grantor
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                            VALUES (?, ?, ?, ?, ?)
                        """, (grantor_id, grantor, 'person', 'county_recorder', json.dumps({'role': 'grantor'})))
                    except Exception as e:
                        pass
                    
                    # Add grantee
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                            VALUES (?, ?, ?, ?, ?)
                        """, (grantee_id, grantee, 'person', 'county_recorder', json.dumps({'role': 'grantee'})))
                    except Exception as e:
                        pass
                    
                    # Create SOLD_TO relationship
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                            VALUES (?, ?, ?, ?)
                        """, (grantor_id, grantee_id, 'SOLD_TO', json.dumps({'date': date, 'source': 'county_recorder'})))
                        print(f"  ✅ Relationship: {grantor} --[SOLD_TO]--> {grantee} ({date})")
                        imported += 1
                    except Exception as e:
                        print(f"  ⚠️  Relationship error: {e}")
                    
                    # Create PURCHASED_FROM relationship (reverse)
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                            VALUES (?, ?, ?, ?)
                        """, (grantee_id, grantor_id, 'PURCHASED_FROM', json.dumps({'date': date, 'source': 'county_recorder'})))
                        imported += 1
                    except Exception as e:
                        pass
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Imported {imported} transaction relationships")
    
    def run(self):
        """Run full scraping operation."""
        print("=" * 80)
        print("SELENIUM COUNTY PORTAL SCRAPER")
        print("=" * 80)
        
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium not available. Install: pip install selenium")
            return
        
        try:
            self.setup_driver()
            
            # Scrape assessor
            self.scrape_humboldt_assessor("Strombeck")
            
            # Scrape recorder for transactions
            self.scrape_humboldt_recorder("Strombeck")
            
            # Scrape specific addresses
            addresses = [
                "960 S G St Arcata",
                "Todd Court Arcata",
                "Western Avenue Arcata",
            ]
            self.scrape_addresses(addresses)
            
            # Import to database
            self.import_findings_to_database()
            
            # Save report
            report_file = Path(__file__).parent.parent / "SELENIUM_SCRAPING_REPORT.json"
            with open(report_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'findings_count': len(self.findings),
                    'findings': self.findings
                }, f, indent=2)
            
            print(f"\n✅ Scraping complete! Found {len(self.findings)} records")
            print(f"Report saved to: {report_file}")
            
        finally:
            if self.driver:
                self.driver.quit()
                print("\n✅ Browser closed")


if __name__ == "__main__":
    scraper = SeleniumCountyScraper(headless=True)
    scraper.run()
