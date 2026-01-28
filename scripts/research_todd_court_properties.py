#!/usr/bin/env python3
"""
Research Todd Court Properties - 2535, 2565, 2567
Get assessor information and property details
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

class ToddCourtPropertyResearcher:
    """Research Todd Court properties and assessor information."""
    
    def __init__(self):
        self.search = WebSearchAggregator()
        self.ms = MatrixStore()
        self.findings = []
        
    def research_todd_court_properties(self):
        """Research specific Todd Court addresses."""
        print('=' * 80)
        print('RESEARCHING TODD COURT PROPERTIES')
        print('=' * 80)
        
        addresses = [
            '2535 Todd Court Arcata CA',
            '2565 Todd Court Arcata CA',
            '2567 Todd Court Arcata CA',
        ]
        
        results = {}
        
        for address in addresses:
            print(f'\n{"="*80}')
            print(f'RESEARCHING: {address}')
            print(f'{"="*80}')
            
            queries = [
                f'{address} assessor',
                f'{address} APN',
                f'{address} property owner',
                f'{address} Strombeck',
                f'{address} property records',
                f'{address} Humboldt County',
            ]
            
            address_results = {
                'address': address,
                'assessor_info': {},
                'apn': None,
                'owner': None,
                'value': None,
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
                        
                        # Extract APN
                        apn_pattern = r'APN[:\s]+([0-9]{3}-[0-9]{3}-[0-9]{3}-[0-9]{3})'
                        apn_match = re.search(apn_pattern, full_text, re.IGNORECASE)
                        if apn_match and not address_results['apn']:
                            address_results['apn'] = apn_match.group(1)
                            print(f'  ✅ Found APN: {address_results["apn"]}')
                        
                        # Extract property value
                        value_patterns = [
                            r'\$([0-9,]+)',
                            r'value[:\s]+\$?([0-9,]+)',
                            r'assessed[:\s]+\$?([0-9,]+)',
                        ]
                        for pattern in value_patterns:
                            value_match = re.search(pattern, full_text, re.IGNORECASE)
                            if value_match:
                                value_str = value_match.group(1).replace(',', '')
                                try:
                                    value = int(value_str)
                                    if value > 10000:  # Reasonable property value
                                        address_results['value'] = value
                                        print(f'  ✅ Found Value: ${value:,}')
                                        break
                                except:
                                    pass
                        
                        # Look for owner information
                        if 'strombeck' in text or 'owner' in text:
                            owner_patterns = [
                                r'owner[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                                r'owned by[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                            ]
                            for pattern in owner_patterns:
                                owner_match = re.search(pattern, full_text, re.IGNORECASE)
                                if owner_match:
                                    address_results['owner'] = owner_match.group(1)
                                    print(f'  ✅ Found Owner: {address_results["owner"]}')
                                    break
                        
                        # Store relevant findings
                        if any(keyword in text for keyword in ['todd', 'court', 'arcata', 'property', 'assessor', 'apn', 'strombeck']):
                            address_results['findings'].append({
                                'url': result.url,
                                'title': result.title,
                                'snippet': result.snippet
                            })
                            self.findings.append({
                                'address': address,
                                'url': result.url,
                                'title': result.title,
                                'snippet': result.snippet
                            })
                    
                    print(f'  ✅ Found {len(search_results)} results')
                    
                except Exception as e:
                    print(f'  ⚠️  Error: {e}')
            
            results[address] = address_results
        
        return results
    
    def check_database_for_todd_court(self):
        """Check database for existing Todd Court information."""
        print('\n\n' + '=' * 80)
        print('CHECKING DATABASE FOR TODD COURT PROPERTIES')
        print('=' * 80)
        
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
        # Search for Todd Court entities
        cursor.execute("""
            SELECT entity_id, name, properties
            FROM entities
            WHERE name LIKE '%Todd%' OR name LIKE '%todd%'
            ORDER BY name
        """)
        
        db_properties = cursor.fetchall()
        
        print(f'\nFound {len(db_properties)} Todd Court entities in database:')
        for eid, name, props_json in db_properties:
            print(f'\n  {name} ({eid})')
            if props_json:
                props = json.loads(props_json) if isinstance(props_json, str) else props_json
                if 'apn' in props:
                    print(f'    APN: {props["apn"]}')
                if 'total_value' in props:
                    print(f'    Value: ${props["total_value"]:,}')
                if 'owner' in props:
                    print(f'    Owner: {props["owner"]}')
        
        # Check relationships
        cursor.execute("""
            SELECT e1.name, e2.name, r.relationship_type, r.properties
            FROM relationships r
            INNER JOIN entities e1 ON e1.entity_id = r.source_id
            INNER JOIN entities e2 ON e2.entity_id = r.target_id
            WHERE (e1.name LIKE '%Todd%' OR e2.name LIKE '%Todd%'
                   OR e1.name LIKE '%todd%' OR e2.name LIKE '%todd%')
            AND e1.entity_type = 'person'
        """)
        
        relationships = cursor.fetchall()
        
        print(f'\n\nFound {len(relationships)} relationships:')
        for e1, e2, rel_type, props_json in relationships:
            print(f'  {e1[:30]:30s} --[{rel_type}]--> {e2[:30]}')
        
        conn.close()
        
        return {
            'db_properties': db_properties,
            'relationships': relationships
        }
    
    def import_to_database(self, research_results):
        """Import findings to database."""
        print('\n\n' + '=' * 80)
        print('IMPORTING TODD COURT PROPERTIES TO DATABASE')
        print('=' * 80)
        
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
        imported = 0
        
        for address, results in research_results.items():
            # Create entity ID
            address_clean = address.lower().replace(' ', '_').replace(',', '').replace('.', '')
            entity_id = f"address_{address_clean}"
            
            # Create properties
            props = {
                'address': address,
                'city': 'Arcata',
                'state': 'CA',
                'research_date': datetime.now().isoformat()
            }
            
            if results['apn']:
                props['apn'] = results['apn']
            if results['value']:
                props['total_value'] = results['value']
            if results['owner']:
                props['owner'] = results['owner']
            
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    address,
                    'address',
                    'property_research',
                    json.dumps(props)
                ))
                
                # Link to Erik if owner is Strombeck
                if results['owner'] and 'strombeck' in results['owner'].lower():
                    cursor.execute("""
                        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                        VALUES (?, ?, ?, ?)
                    """, (
                        'person_erik_strombeck',
                        entity_id,
                        'OWNS',
                        json.dumps({
                            'source': 'property_research',
                            'investigation_note': f'Property owner: {results["owner"]}',
                            'investigation_flag': 'todd_court_property'
                        })
                    ))
                
                imported += 1
                print(f'  ✅ Imported: {address}')
                if results['apn']:
                    print(f'     APN: {results["apn"]}')
                if results['value']:
                    print(f'     Value: ${results["value"]:,}')
                
            except Exception as e:
                print(f'  ⚠️  Error importing {address}: {e}')
        
        conn.commit()
        conn.close()
        
        print(f'\n✅ Imported {imported} properties')
    
    def generate_report(self, research_results, db_info):
        """Generate research report."""
        report = []
        report.append("# Todd Court Properties Research Report")
        report.append(f"\n**Research Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n## PROPERTIES RESEARCHED")
        report.append(f"\n- 2535 Todd Court Arcata CA")
        report.append(f"- 2565 Todd Court Arcata CA")
        report.append(f"- 2567 Todd Court Arcata CA")
        
        report.append(f"\n## RESEARCH FINDINGS")
        
        for address, results in research_results.items():
            report.append(f"\n### {address}")
            report.append(f"\n**APN**: {results['apn'] if results['apn'] else 'Not Found'}")
            report.append(f"**Owner**: {results['owner'] if results['owner'] else 'Not Found'}")
            report.append(f"**Value**: ${results['value']:,}" if results['value'] else "**Value**: Not Found")
            report.append(f"**Findings**: {len(results['findings'])}")
            
            if results['findings']:
                report.append(f"\n**Key Findings**:")
                for finding in results['findings'][:5]:
                    report.append(f"- **{finding['title']}**")
                    report.append(f"  - URL: {finding['url']}")
                    report.append(f"  - Snippet: {finding['snippet'][:200]}...")
        
        report.append(f"\n## DATABASE STATUS")
        report.append(f"\n**Properties in Database**: {len(db_info['db_properties'])}")
        for eid, name, props_json in db_info['db_properties']:
            report.append(f"- {name}")
        
        report.append(f"\n**Relationships**: {len(db_info['relationships'])}")
        for e1, e2, rel_type, props_json in db_info['relationships']:
            report.append(f"- {e1} --[{rel_type}]--> {e2}")
        
        report.append(f"\n## ASSESSOR INFORMATION")
        report.append(f"\n**Status**: Research conducted for assessor information")
        report.append(f"\n**Note**: Specific APN and assessor details require direct access to Humboldt County Assessor database")
        
        return '\n'.join(report)
    
    def run_research(self):
        """Run complete research."""
        print('=' * 80)
        print('TODD COURT PROPERTIES RESEARCH')
        print('=' * 80)
        
        # Check database first
        db_info = self.check_database_for_todd_court()
        
        # Research properties
        research_results = self.research_todd_court_properties()
        
        # Import to database
        self.import_to_database(research_results)
        
        # Generate report
        report = self.generate_report(research_results, db_info)
        
        # Save report
        report_file = f'TODD_COURT_PROPERTIES_RESEARCH_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save data
        data_file = f'TODD_COURT_PROPERTIES_DATA_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(data_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'research_results': research_results,
                'database_info': {
                    'properties': [(eid, name) for eid, name, _ in db_info['db_properties']],
                    'relationships': [(e1, e2, rel_type) for e1, e2, rel_type, _ in db_info['relationships']]
                },
                'findings': self.findings
            }, f, indent=2)
        
        print(f'\n\n✅ Research Complete!')
        print(f'📄 Report saved to: {report_file}')
        print(f'📊 Data saved to: {data_file}')
        
        return research_results

if __name__ == "__main__":
    researcher = ToddCourtPropertyResearcher()
    researcher.run_research()
