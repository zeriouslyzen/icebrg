#!/usr/bin/env python3
"""
Fix Assessor Website Access Issues
Diagnose and fix JavaScript errors on county assessor portals
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not installed")

class AssessorWebsiteFixer:
    """Fix assessor website access issues."""
    
    def __init__(self):
        self.driver = None
        
    def setup_driver(self, headless=False):
        """Setup Chrome WebDriver with proper configuration."""
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium not available")
            return False
        
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            print("✅ ChromeDriver initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize ChromeDriver: {e}")
            return False
    
    def diagnose_website_error(self, url):
        """Diagnose JavaScript errors on website."""
        print('=' * 80)
        print(f'DIAGNOSING WEBSITE: {url}')
        print('=' * 80)
        
        if not self.driver:
            if not self.setup_driver(headless=False):
                return
        
        try:
            print(f'\n🔍 Navigating to: {url}')
            self.driver.get(url)
            time.sleep(5)
            
            # Check for JavaScript errors in console
            print('\n📊 Checking browser console for errors...')
            logs = self.driver.get_log('browser')
            
            errors = [log for log in logs if log['level'] == 'SEVERE']
            if errors:
                print(f'\n⚠️  Found {len(errors)} JavaScript errors:')
                for error in errors[:10]:
                    print(f'  - {error["message"]}')
            else:
                print('  ✅ No JavaScript errors found in console')
            
            # Check page title
            print(f'\n📄 Page Title: {self.driver.title}')
            
            # Check if page loaded
            page_source_length = len(self.driver.page_source)
            print(f'📏 Page Source Length: {page_source_length} characters')
            
            # Look for common assessor portal elements
            print('\n🔍 Looking for common elements...')
            
            # Check for search forms
            search_inputs = self.driver.find_elements(By.CSS_SELECTOR, 'input[type="text"], input[type="search"]')
            print(f'  Search inputs found: {len(search_inputs)}')
            
            # Check for buttons
            buttons = self.driver.find_elements(By.CSS_SELECTOR, 'button, input[type="submit"]')
            print(f'  Buttons found: {len(buttons)}')
            
            # Check for iframes (common in government portals)
            iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
            print(f'  Iframes found: {len(iframes)}')
            
            # Try to find the problematic element
            print('\n🔍 Looking for AJAX-related code...')
            page_source = self.driver.page_source
            
            if 'response is not defined' in page_source.lower() or 'response' in page_source.lower():
                print('  ⚠️  Found "response" references in page source')
                # Try to find the problematic script
                scripts = self.driver.find_elements(By.TAG_NAME, 'script')
                print(f'  Scripts found: {len(scripts)}')
                
                for i, script in enumerate(scripts[:5]):
                    script_text = script.get_attribute('innerHTML') or ''
                    if 'response' in script_text.lower() and 'success' in script_text.lower():
                        print(f'\n  ⚠️  Found potential problematic script #{i}:')
                        print(f'      {script_text[:200]}...')
            
            # Check network requests
            print('\n🌐 Checking network requests...')
            performance_log = self.driver.get_log('performance')
            failed_requests = [log for log in performance_log if 'failed' in str(log).lower() or 'error' in str(log).lower()]
            if failed_requests:
                print(f'  ⚠️  Found {len(failed_requests)} failed network requests')
            
            return {
                'url': url,
                'title': self.driver.title,
                'errors': errors,
                'search_inputs': len(search_inputs),
                'buttons': len(buttons),
                'iframes': len(iframes)
            }
            
        except Exception as e:
            print(f'\n❌ Error diagnosing website: {e}')
            return None
    
    def test_humboldt_assessor_portals(self):
        """Test different Humboldt County Assessor portal URLs."""
        print('=' * 80)
        print('TESTING HUMBOLDT COUNTY ASSESSOR PORTALS')
        print('=' * 80)
        
        urls = [
            'https://www.co.humboldt.ca.us/assessor',
            'https://www.co.humboldt.ca.us/assessor/search',
            'https://humboldt.assessor.gisworkshop.com',
            'https://humboldtgov.org/220/Assessor',
        ]
        
        results = []
        
        for url in urls:
            print(f'\n\n{"="*80}')
            print(f'TESTING: {url}')
            print(f'{"="*80}')
            
            result = self.diagnose_website_error(url)
            if result:
                results.append(result)
            
            time.sleep(2)
        
        return results
    
    def suggest_fixes(self, diagnosis_results):
        """Suggest fixes based on diagnosis."""
        print('\n\n' + '=' * 80)
        print('SUGGESTED FIXES')
        print('=' * 80)
        
        fixes = []
        
        for result in diagnosis_results:
            if result['errors']:
                fixes.append({
                    'issue': 'JavaScript errors detected',
                    'fix': 'Website may have been updated or has a bug. Try: 1) Clear browser cache, 2) Try different browser, 3) Wait and retry later, 4) Contact website administrator'
                })
            
            if result['iframes'] > 0:
                fixes.append({
                    'issue': 'Website uses iframes',
                    'fix': 'May need to switch to iframe context. Try: driver.switch_to.frame(iframe_element)'
                })
            
            if result['search_inputs'] == 0:
                fixes.append({
                    'issue': 'No search inputs found',
                    'fix': 'Website structure may have changed. Try manual navigation or check for JavaScript-rendered content'
                })
        
        if not fixes:
            fixes.append({
                'issue': 'No specific issues detected',
                'fix': 'Website may be temporarily down or experiencing issues. Try again later or use alternative access methods'
            })
        
        print('\n🔧 FIX SUGGESTIONS:')
        for i, fix in enumerate(fixes, 1):
            print(f'\n{i}. {fix["issue"]}')
            print(f'   Fix: {fix["fix"]}')
        
        return fixes
    
    def cleanup(self):
        """Close browser."""
        if self.driver:
            self.driver.quit()
            print('\n✅ Browser closed')

if __name__ == "__main__":
    fixer = AssessorWebsiteFixer()
    
    try:
        # Test assessor portals
        results = fixer.test_humboldt_assessor_portals()
        
        # Suggest fixes
        fixes = fixer.suggest_fixes(results)
        
        # Save results
        report_file = f'ASSESSOR_WEBSITE_DIAGNOSIS_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'fixes': fixes
            }, f, indent=2)
        
        print(f'\n✅ Diagnosis complete! Results saved to: {report_file}')
        
    finally:
        fixer.cleanup()
