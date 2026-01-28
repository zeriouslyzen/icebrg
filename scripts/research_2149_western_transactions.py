#!/usr/bin/env python3
"""
Research 2149 Western Ave Transactions
Deep dive into buyers, sellers, and transaction patterns
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore
from src.iceburg.search.web_search import WebSearchAggregator
import os
os.environ['ICEBURG_ENABLE_WEB'] = '1'

class WesternAveTransactionResearcher:
    """Research 2149 Western Ave transactions and parties."""
    
    def __init__(self):
        self.search = WebSearchAggregator()
        self.ms = MatrixStore()
        self.findings = []
        
    def research_katherine_strombeck(self):
        """Research Katherine Strombeck."""
        print('=' * 80)
        print('RESEARCHING KATHERINE STROMBECK')
        print('=' * 80)
        
        queries = [
            'Katherine Strombeck Arcata Humboldt',
            'Katherine Strombeck Erik Strombeck relationship',
            'Katherine Strombeck property owner',
            'Katherine Strombeck Strombeck family',
        ]
        
        results = {
            'name': 'Katherine Strombeck',
            'properties': [],
            'relationships': [],
            'findings': []
        }
        
        for query in queries:
            print(f'\n🔍 Searching: {query}')
            try:
                search_results = self.search.search(
                    query,
                    sources=['ddg', 'brave'],
                    max_results_per_source=10
                )
                
                for result in search_results:
                    text = f"{result.title} {result.snippet}".lower()
                    full_text = f"{result.title} {result.snippet}"
                    
                    # Extract property addresses
                    address_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Ct|Court|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place)'
                    addresses = re.findall(address_pattern, full_text, re.IGNORECASE)
                    for addr in addresses:
                        if addr not in results['properties']:
                            results['properties'].append(addr)
                    
                    # Store relevant findings
                    if any(keyword in text for keyword in ['katherine', 'strombeck', 'property', 'arcata', 'erik']):
                        results['findings'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                        self.findings.append({
                            'person': 'Katherine Strombeck',
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        return results
    
    def research_phillips_joseph(self):
        """Research Phillips Joseph L."""
        print('\n\n' + '=' * 80)
        print('RESEARCHING PHILLIPS JOSEPH L')
        print('=' * 80)
        
        queries = [
            'Phillips Joseph L Arcata Humboldt',
            'Phillips Joseph L property buyer',
            'Phillips Joseph L Simmons Lorenza',
            'Joseph Phillips Arcata California',
        ]
        
        results = {
            'name': 'Phillips Joseph L',
            'properties': [],
            'relationships': [],
            'findings': []
        }
        
        for query in queries:
            print(f'\n🔍 Searching: {query}')
            try:
                search_results = self.search.search(
                    query,
                    sources=['ddg', 'brave'],
                    max_results_per_source=10
                )
                
                for result in search_results:
                    text = f"{result.title} {result.snippet}".lower()
                    full_text = f"{result.title} {result.snippet}"
                    
                    # Extract property addresses
                    address_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Ct|Court|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place)'
                    addresses = re.findall(address_pattern, full_text, re.IGNORECASE)
                    for addr in addresses:
                        if addr not in results['properties']:
                            results['properties'].append(addr)
                    
                    # Store relevant findings
                    if any(keyword in text for keyword in ['phillips', 'joseph', 'property', 'arcata', 'buyer', 'simmons']):
                        results['findings'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                        self.findings.append({
                            'person': 'Phillips Joseph L',
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        return results
    
    def research_simmons_lorenza(self):
        """Research Simmons Lorenza."""
        print('\n\n' + '=' * 80)
        print('RESEARCHING SIMMONS LORENZA')
        print('=' * 80)
        
        queries = [
            'Simmons Lorenza Arcata Humboldt',
            'Simmons Lorenza property buyer',
            'Simmons Lorenza Phillips Joseph',
            'Lorenza Simmons Arcata California',
        ]
        
        results = {
            'name': 'Simmons Lorenza',
            'properties': [],
            'relationships': [],
            'findings': []
        }
        
        for query in queries:
            print(f'\n🔍 Searching: {query}')
            try:
                search_results = self.search.search(
                    query,
                    sources=['ddg', 'brave'],
                    max_results_per_source=10
                )
                
                for result in search_results:
                    text = f"{result.title} {result.snippet}".lower()
                    full_text = f"{result.title} {result.snippet}"
                    
                    # Extract property addresses
                    address_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Ct|Court|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place)'
                    addresses = re.findall(address_pattern, full_text, re.IGNORECASE)
                    for addr in addresses:
                        if addr not in results['properties']:
                            results['properties'].append(addr)
                    
                    # Store relevant findings
                    if any(keyword in text for keyword in ['simmons', 'lorenza', 'property', 'arcata', 'buyer', 'phillips']):
                        results['findings'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                        self.findings.append({
                            'person': 'Simmons Lorenza',
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        return results
    
    def research_united_wholesale_mortgage(self):
        """Research United Wholesale Mortgage LLC."""
        print('\n\n' + '=' * 80)
        print('RESEARCHING UNITED WHOLESALE MORTGAGE LLC')
        print('=' * 80)
        
        queries = [
            'United Wholesale Mortgage LLC',
            'United Wholesale Mortgage LLC lender',
            'UWM LLC mortgage lender',
        ]
        
        results = {
            'name': 'United Wholesale Mortgage LLC',
            'findings': []
        }
        
        for query in queries:
            print(f'\n🔍 Searching: {query}')
            try:
                search_results = self.search.search(
                    query,
                    sources=['ddg', 'brave'],
                    max_results_per_source=5
                )
                
                for result in search_results:
                    text = f"{result.title} {result.snippet}".lower()
                    
                    if any(keyword in text for keyword in ['united wholesale', 'mortgage', 'lender', 'uwm']):
                        results['findings'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        return results
    
    def analyze_transaction_pattern(self, katherine_results, phillips_results, simmons_results):
        """Analyze transaction patterns."""
        print('\n\n' + '=' * 80)
        print('ANALYZING TRANSACTION PATTERNS')
        print('=' * 80)
        
        analysis = {
            'transaction_chain': [],
            'red_flags': [],
            'patterns': []
        }
        
        # Transaction chain
        analysis['transaction_chain'].append({
            'date': '2024-11-12',
            'seller': 'Katherine Strombeck',
            'buyer': 'Erik Strombeck',
            'property': '2149 Western Ave A Arcata CA',
            'type': 'Family transfer'
        })
        
        analysis['transaction_chain'].append({
            'date': '2025-12-01',
            'seller': 'Erik Strombeck',
            'buyer': 'Phillips Joseph L & Simmons Lorenza',
            'property': '2149 Western Ave A Arcata CA',
            'sale_price': 404000,
            'loan_amount': 222200,
            'type': 'Sale to third parties'
        })
        
        # Red flags
        analysis['red_flags'].append('Family transfer (Katherine → Erik) followed by quick sale to third parties')
        analysis['red_flags'].append('Property held for only 1 year before sale')
        analysis['red_flags'].append('Sale price $404K, loan $222K (55% LTV - suggests down payment)')
        
        # Patterns
        analysis['patterns'].append('Family property transfer pattern')
        analysis['patterns'].append('Quick turnaround sale (1 year)')
        analysis['patterns'].append('Third-party buyers with mortgage financing')
        
        print(f'\n🔍 TRANSACTION CHAIN:')
        for tx in analysis['transaction_chain']:
            print(f'\n  {tx["date"]}:')
            print(f'    Seller: {tx["seller"]}')
            print(f'    Buyer: {tx["buyer"]}')
            print(f'    Type: {tx["type"]}')
            if 'sale_price' in tx:
                print(f'    Sale Price: ${tx["sale_price"]:,}')
                print(f'    Loan Amount: ${tx["loan_amount"]:,}')
        
        print(f'\n\n⚠️  RED FLAGS:')
        for flag in analysis['red_flags']:
            print(f'  - {flag}')
        
        print(f'\n\n📊 PATTERNS:')
        for pattern in analysis['patterns']:
            print(f'  - {pattern}')
        
        return analysis
    
    def import_to_database(self, katherine_results, phillips_results, simmons_results, uwm_results, analysis):
        """Import findings to database."""
        print('\n\n' + '=' * 80)
        print('IMPORTING TO DATABASE')
        print('=' * 80)
        
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
        prop_id = 'address_2149_western_ave_arcata'
        
        # Create Katherine Strombeck entity
        katherine_id = 'person_katherine_strombeck'
        cursor.execute("""
            INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (
            katherine_id,
            'Katherine Strombeck',
            'person',
            'property_transaction',
            json.dumps({
                'research_date': datetime.now().isoformat(),
                'investigation_note': 'Sold 2149 Western Ave to Erik Strombeck in 2024',
                'investigation_flag': 'family_property_transfer'
            })
        ))
        
        # Create Phillips Joseph L entity
        phillips_id = 'person_phillips_joseph_l'
        cursor.execute("""
            INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (
            phillips_id,
            'Phillips Joseph L',
            'person',
            'property_transaction',
            json.dumps({
                'research_date': datetime.now().isoformat(),
                'investigation_note': 'Purchased 2149 Western Ave from Erik Strombeck in 2025',
                'investigation_flag': 'property_buyer'
            })
        ))
        
        # Create Simmons Lorenza entity
        simmons_id = 'person_simmons_lorenza'
        cursor.execute("""
            INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (
            simmons_id,
            'Simmons Lorenza',
            'person',
            'property_transaction',
            json.dumps({
                'research_date': datetime.now().isoformat(),
                'investigation_note': 'Co-purchased 2149 Western Ave from Erik Strombeck in 2025',
                'investigation_flag': 'property_buyer'
            })
        ))
        
        # Create United Wholesale Mortgage LLC entity
        uwm_id = 'company_united_wholesale_mortgage_llc'
        cursor.execute("""
            INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (
            uwm_id,
            'United Wholesale Mortgage LLC',
            'company',
            'property_transaction',
            json.dumps({
                'research_date': datetime.now().isoformat(),
                'investigation_note': 'Lender for 2149 Western Ave purchase',
                'investigation_flag': 'mortgage_lender'
            })
        ))
        
        # Create relationships
        # Katherine sold to Erik
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            katherine_id,
            'person_erik_strombeck',
            'SOLD_TO',
            json.dumps({
                'property': '2149 Western Ave A Arcata CA',
                'date': '2024-11-12',
                'document_number': '2024.16254',
                'source': 'property_history',
                'investigation_flag': 'family_property_transfer'
            })
        ))
        
        # Erik sold to Phillips & Simmons
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            'person_erik_strombeck',
            phillips_id,
            'SOLD_TO',
            json.dumps({
                'property': '2149 Western Ave A Arcata CA',
                'date': '2025-12-01',
                'sale_price': 404000,
                'document_number': '2025.16065',
                'source': 'property_history',
                'investigation_flag': 'quick_sale_after_family_transfer'
            })
        ))
        
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            'person_erik_strombeck',
            simmons_id,
            'SOLD_TO',
            json.dumps({
                'property': '2149 Western Ave A Arcata CA',
                'date': '2025-12-01',
                'sale_price': 404000,
                'document_number': '2025.16065',
                'source': 'property_history',
                'investigation_flag': 'quick_sale_after_family_transfer'
            })
        ))
        
        # Phillips & Simmons relationship
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            phillips_id,
            simmons_id,
            'CO_PURCHASER',
            json.dumps({
                'property': '2149 Western Ave A Arcata CA',
                'date': '2025-12-01',
                'source': 'property_history'
            })
        ))
        
        # Link to property
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            phillips_id,
            prop_id,
            'PURCHASED_FROM',
            json.dumps({
                'date': '2025-12-01',
                'sale_price': 404000,
                'source': 'property_history'
            })
        ))
        
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            simmons_id,
            prop_id,
            'PURCHASED_FROM',
            json.dumps({
                'date': '2025-12-01',
                'sale_price': 404000,
                'source': 'property_history'
            })
        ))
        
        # Link lender
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            uwm_id,
            prop_id,
            'LENDER_FOR',
            json.dumps({
                'date': '2025-12-01',
                'loan_amount': 222200,
                'source': 'property_history'
            })
        ))
        
        # Link Katherine to Erik as family
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            katherine_id,
            'person_erik_strombeck',
            'FAMILY_OF',
            json.dumps({
                'source': 'property_transaction',
                'investigation_note': 'Property transfer suggests family relationship',
                'investigation_flag': 'family_transfer'
            })
        ))
        
        conn.commit()
        conn.close()
        
        print(f'\n✅ Imported Katherine Strombeck')
        print(f'✅ Imported Phillips Joseph L')
        print(f'✅ Imported Simmons Lorenza')
        print(f'✅ Imported United Wholesale Mortgage LLC')
        print(f'✅ Created transaction relationships')
    
    def generate_report(self, katherine_results, phillips_results, simmons_results, uwm_results, analysis):
        """Generate research report."""
        report = []
        report.append("# 2149 Western Ave Transaction Research Report")
        report.append(f"\n**Research Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n## TRANSACTION SUMMARY")
        report.append(f"\n### Property: 2149 Western Ave A Arcata CA 95521-5349")
        report.append(f"**APN**: 505-095-038-000")
        
        report.append(f"\n### Transaction Chain")
        for tx in analysis['transaction_chain']:
            report.append(f"\n**{tx['date']}**:")
            report.append(f"- Seller: {tx['seller']}")
            report.append(f"- Buyer: {tx['buyer']}")
            report.append(f"- Type: {tx['type']}")
            if 'sale_price' in tx:
                report.append(f"- Sale Price: ${tx['sale_price']:,}")
                report.append(f"- Loan Amount: ${tx['loan_amount']:,}")
        
        report.append(f"\n## KATHERINE STROMBECK")
        report.append(f"\n### Findings: {len(katherine_results['findings'])}")
        report.append(f"\n### Properties: {len(katherine_results['properties'])}")
        for prop in katherine_results['properties'][:5]:
            report.append(f"- {prop}")
        
        report.append(f"\n## PHILLIPS JOSEPH L")
        report.append(f"\n### Findings: {len(phillips_results['findings'])}")
        report.append(f"\n### Properties: {len(phillips_results['properties'])}")
        for prop in phillips_results['properties'][:5]:
            report.append(f"- {prop}")
        
        report.append(f"\n## SIMMONS LORENZA")
        report.append(f"\n### Findings: {len(simmons_results['findings'])}")
        report.append(f"\n### Properties: {len(simmons_results['properties'])}")
        for prop in simmons_results['properties'][:5]:
            report.append(f"- {prop}")
        
        report.append(f"\n## UNITED WHOLESALE MORTGAGE LLC")
        report.append(f"\n### Findings: {len(uwm_results['findings'])}")
        
        report.append(f"\n## RED FLAGS")
        for flag in analysis['red_flags']:
            report.append(f"- ⚠️  {flag}")
        
        report.append(f"\n## PATTERNS")
        for pattern in analysis['patterns']:
            report.append(f"- {pattern}")
        
        return '\n'.join(report)
    
    def run_research(self):
        """Run complete research."""
        print('=' * 80)
        print('2149 WESTERN AVE TRANSACTION RESEARCH')
        print('=' * 80)
        
        # Research all parties
        katherine_results = self.research_katherine_strombeck()
        phillips_results = self.research_phillips_joseph()
        simmons_results = self.research_simmons_lorenza()
        uwm_results = self.research_united_wholesale_mortgage()
        
        # Analyze patterns
        analysis = self.analyze_transaction_pattern(katherine_results, phillips_results, simmons_results)
        
        # Import to database
        self.import_to_database(katherine_results, phillips_results, simmons_results, uwm_results, analysis)
        
        # Generate report
        report = self.generate_report(katherine_results, phillips_results, simmons_results, uwm_results, analysis)
        
        # Save report
        report_file = f'WESTERN_AVE_TRANSACTION_RESEARCH_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save data
        data_file = f'WESTERN_AVE_TRANSACTION_DATA_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(data_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'katherine_strombeck': katherine_results,
                'phillips_joseph': phillips_results,
                'simmons_lorenza': simmons_results,
                'united_wholesale_mortgage': uwm_results,
                'analysis': analysis,
                'findings': self.findings
            }, f, indent=2)
        
        print(f'\n\n✅ Research Complete!')
        print(f'📄 Report saved to: {report_file}')
        print(f'📊 Data saved to: {data_file}')
        
        return {
            'katherine': katherine_results,
            'phillips': phillips_results,
            'simmons': simmons_results,
            'uwm': uwm_results,
            'analysis': analysis
        }

if __name__ == "__main__":
    researcher = WesternAveTransactionResearcher()
    researcher.run_research()
