#!/usr/bin/env python3
"""
Investigate Phillips & Simmons Connections to Erik Strombeck
Determine if buyers are legitimate or connected parties
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

class PhillipsSimmonsConnectionInvestigator:
    """Investigate connections between Phillips/Simmons and Erik Strombeck."""
    
    def __init__(self):
        self.search = WebSearchAggregator()
        self.ms = MatrixStore()
        self.findings = []
        self.connections_found = []
        
    def research_phillips_deep(self):
        """Deep research on Phillips Joseph L."""
        print('=' * 80)
        print('DEEP RESEARCH: PHILLIPS JOSEPH L')
        print('=' * 80)
        
        queries = [
            'Phillips Joseph L Arcata California',
            'Joseph Phillips Arcata Humboldt',
            'Phillips Joseph L Erik Strombeck',
            'Joseph Phillips property buyer Arcata',
            'Phillips Joseph L Simmons Lorenza relationship',
            'Joseph Phillips phone number Arcata',
            'Phillips Joseph L address Arcata',
            'Joseph Phillips business Arcata',
        ]
        
        results = {
            'name': 'Phillips Joseph L',
            'addresses': [],
            'phone_numbers': [],
            'businesses': [],
            'connections_to_erik': [],
            'connections_to_simmons': [],
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
                    
                    # Extract addresses
                    address_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Ct|Court|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place)'
                    addresses = re.findall(address_pattern, full_text, re.IGNORECASE)
                    for addr in addresses:
                        if addr not in results['addresses']:
                            results['addresses'].append(addr)
                    
                    # Extract phone numbers
                    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
                    phones = re.findall(phone_pattern, full_text)
                    for phone in phones:
                        if phone not in results['phone_numbers']:
                            results['phone_numbers'].append(phone)
                    
                    # Look for connections to Erik
                    if any(keyword in text for keyword in ['erik', 'strombeck', 'erik strombeck']):
                        results['connections_to_erik'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet,
                            'connection_type': 'potential_connection'
                        })
                        self.connections_found.append({
                            'person1': 'Phillips Joseph L',
                            'person2': 'Erik Strombeck',
                            'evidence': result.snippet[:200],
                            'url': result.url
                        })
                    
                    # Look for connections to Simmons
                    if 'simmons' in text or 'lorenza' in text:
                        results['connections_to_simmons'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
                    # Store relevant findings
                    if any(keyword in text for keyword in ['phillips', 'joseph', 'arcata', 'property', 'buyer']):
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
    
    def research_simmons_deep(self):
        """Deep research on Simmons Lorenza."""
        print('\n\n' + '=' * 80)
        print('DEEP RESEARCH: SIMMONS LORENZA')
        print('=' * 80)
        
        queries = [
            'Simmons Lorenza Arcata California',
            'Lorenza Simmons Arcata Humboldt',
            'Simmons Lorenza Erik Strombeck',
            'Lorenza Simmons property buyer Arcata',
            'Simmons Lorenza Phillips Joseph',
            'Lorenza Simmons phone number Arcata',
            'Simmons Lorenza address Arcata',
            'Lorenza Simmons business Arcata',
        ]
        
        results = {
            'name': 'Simmons Lorenza',
            'addresses': [],
            'phone_numbers': [],
            'businesses': [],
            'connections_to_erik': [],
            'connections_to_phillips': [],
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
                    
                    # Extract addresses
                    address_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Ct|Court|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place)'
                    addresses = re.findall(address_pattern, full_text, re.IGNORECASE)
                    for addr in addresses:
                        if addr not in results['addresses']:
                            results['addresses'].append(addr)
                    
                    # Extract phone numbers
                    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
                    phones = re.findall(phone_pattern, full_text)
                    for phone in phones:
                        if phone not in results['phone_numbers']:
                            results['phone_numbers'].append(phone)
                    
                    # Look for connections to Erik
                    if any(keyword in text for keyword in ['erik', 'strombeck', 'erik strombeck']):
                        results['connections_to_erik'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet,
                            'connection_type': 'potential_connection'
                        })
                        self.connections_found.append({
                            'person1': 'Simmons Lorenza',
                            'person2': 'Erik Strombeck',
                            'evidence': result.snippet[:200],
                            'url': result.url
                        })
                    
                    # Look for connections to Phillips
                    if 'phillips' in text or 'joseph' in text:
                        results['connections_to_phillips'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
                    # Store relevant findings
                    if any(keyword in text for keyword in ['simmons', 'lorenza', 'arcata', 'property', 'buyer']):
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
    
    def check_shared_addresses_phones(self, phillips_results, simmons_results):
        """Check for shared addresses and phone numbers with Erik Strombeck."""
        print('\n\n' + '=' * 80)
        print('CHECKING SHARED ADDRESSES & PHONE NUMBERS')
        print('=' * 80)
        
        # Get Erik's addresses and phones from database
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
        # Get Erik's addresses
        cursor.execute("""
            SELECT e2.name, e2.properties
            FROM relationships r
            INNER JOIN entities e1 ON e1.entity_id = r.source_id
            INNER JOIN entities e2 ON e2.entity_id = r.target_id
            WHERE e1.entity_id = 'person_erik_strombeck'
            AND e2.entity_type = 'address'
            AND r.relationship_type IN ('OWNS', 'LOCATED_AT', 'MAILING_ADDRESS')
        """)
        
        erik_addresses = []
        for name, props_json in cursor.fetchall():
            erik_addresses.append(name.lower())
            if props_json:
                props = json.loads(props_json) if isinstance(props_json, str) else props_json
                if 'address' in props:
                    erik_addresses.append(props['address'].lower())
        
        # Get Erik's phone numbers
        cursor.execute("""
            SELECT e2.name, e2.properties
            FROM relationships r
            INNER JOIN entities e1 ON e1.entity_id = r.source_id
            INNER JOIN entities e2 ON e2.entity_id = r.target_id
            WHERE e1.entity_id = 'person_erik_strombeck'
            AND e2.entity_type = 'phone'
            AND r.relationship_type = 'HAS_PHONE'
        """)
        
        erik_phones = []
        for name, props_json in cursor.fetchall():
            erik_phones.append(name.lower())
        
        conn.close()
        
        # Check for matches
        shared_addresses = []
        shared_phones = []
        
        # Check Phillips addresses
        for addr in phillips_results['addresses']:
            addr_lower = addr.lower()
            for erik_addr in erik_addresses:
                if addr_lower in erik_addr or erik_addr in addr_lower:
                    shared_addresses.append({
                        'person': 'Phillips Joseph L',
                        'address': addr,
                        'erik_address': erik_addr
                    })
        
        # Check Simmons addresses
        for addr in simmons_results['addresses']:
            addr_lower = addr.lower()
            for erik_addr in erik_addresses:
                if addr_lower in erik_addr or erik_addr in addr_lower:
                    shared_addresses.append({
                        'person': 'Simmons Lorenza',
                        'address': addr,
                        'erik_address': erik_addr
                    })
        
        # Check Phillips phones
        for phone in phillips_results['phone_numbers']:
            phone_clean = re.sub(r'[^0-9]', '', phone)
            for erik_phone in erik_phones:
                erik_phone_clean = re.sub(r'[^0-9]', '', erik_phone)
                if phone_clean == erik_phone_clean:
                    shared_phones.append({
                        'person': 'Phillips Joseph L',
                        'phone': phone,
                        'erik_phone': erik_phone
                    })
        
        # Check Simmons phones
        for phone in simmons_results['phone_numbers']:
            phone_clean = re.sub(r'[^0-9]', '', phone)
            for erik_phone in erik_phones:
                erik_phone_clean = re.sub(r'[^0-9]', '', erik_phone)
                if phone_clean == erik_phone_clean:
                    shared_phones.append({
                        'person': 'Simmons Lorenza',
                        'phone': phone,
                        'erik_phone': erik_phone
                    })
        
        print(f'\n  Shared Addresses Found: {len(shared_addresses)}')
        for match in shared_addresses:
            print(f'    ⚠️  {match["person"]}: {match["address"]} matches Erik\'s {match["erik_address"]}')
        
        print(f'\n  Shared Phone Numbers Found: {len(shared_phones)}')
        for match in shared_phones:
            print(f'    ⚠️  {match["person"]}: {match["phone"]} matches Erik\'s {match["erik_phone"]}')
        
        return {
            'shared_addresses': shared_addresses,
            'shared_phones': shared_phones
        }
    
    def analyze_connections(self, phillips_results, simmons_results, shared_data):
        """Analyze all connections found."""
        print('\n\n' + '=' * 80)
        print('CONNECTION ANALYSIS')
        print('=' * 80)
        
        analysis = {
            'phillips_to_erik': len(phillips_results['connections_to_erik']),
            'simmons_to_erik': len(simmons_results['connections_to_erik']),
            'shared_addresses': len(shared_data['shared_addresses']),
            'shared_phones': len(shared_data['shared_phones']),
            'total_connections': len(self.connections_found),
            'assessment': 'unknown',
            'red_flags': [],
            'conclusion': ''
        }
        
        # Assess connection level
        total_indicators = (
            analysis['phillips_to_erik'] +
            analysis['simmons_to_erik'] +
            analysis['shared_addresses'] +
            analysis['shared_phones']
        )
        
        if total_indicators == 0:
            analysis['assessment'] = 'no_connection_found'
            analysis['conclusion'] = 'No direct connections found - may be legitimate third-party buyers'
        elif total_indicators <= 2:
            analysis['assessment'] = 'weak_connection'
            analysis['conclusion'] = 'Weak connections found - requires further investigation'
            analysis['red_flags'].append('Some connection indicators found')
        else:
            analysis['assessment'] = 'strong_connection'
            analysis['conclusion'] = 'Strong connections found - likely associates/friends, not legitimate buyers'
            analysis['red_flags'].append('Multiple connection indicators suggest relationship')
            analysis['red_flags'].append('May be nominee buyers or associates')
        
        print(f'\n🔍 CONNECTION ASSESSMENT:')
        print(f'  Phillips → Erik Connections: {analysis["phillips_to_erik"]}')
        print(f'  Simmons → Erik Connections: {analysis["simmons_to_erik"]}')
        print(f'  Shared Addresses: {analysis["shared_addresses"]}')
        print(f'  Shared Phone Numbers: {analysis["shared_phones"]}')
        print(f'  Total Indicators: {total_indicators}')
        print(f'\n  Assessment: {analysis["assessment"]}')
        print(f'  Conclusion: {analysis["conclusion"]}')
        
        if analysis['red_flags']:
            print(f'\n  ⚠️  RED FLAGS:')
            for flag in analysis['red_flags']:
                print(f'    - {flag}')
        
        return analysis
    
    def import_to_database(self, phillips_results, simmons_results, analysis):
        """Import findings to database."""
        print('\n\n' + '=' * 80)
        print('IMPORTING TO DATABASE')
        print('=' * 80)
        
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
        phillips_id = 'person_phillips_joseph_l'
        simmons_id = 'person_simmons_lorenza'
        
        # Update Phillips entity
        cursor.execute('SELECT properties FROM entities WHERE entity_id = ?', (phillips_id,))
        result = cursor.fetchone()
        props = json.loads(result[0]) if result and result[0] else {}
        
        props.update({
            'connection_investigation': {
                'connections_to_erik': len(phillips_results['connections_to_erik']),
                'addresses_found': phillips_results['addresses'],
                'phone_numbers_found': phillips_results['phone_numbers'],
                'assessment': analysis['assessment'],
                'investigation_date': datetime.now().isoformat()
            },
            'investigation_flag': 'buyer_connection_check'
        })
        
        cursor.execute("""
            UPDATE entities SET properties = ? WHERE entity_id = ?
        """, (json.dumps(props), phillips_id))
        
        # Update Simmons entity
        cursor.execute('SELECT properties FROM entities WHERE entity_id = ?', (simmons_id,))
        result = cursor.fetchone()
        props = json.loads(result[0]) if result and result[0] else {}
        
        props.update({
            'connection_investigation': {
                'connections_to_erik': len(simmons_results['connections_to_erik']),
                'addresses_found': simmons_results['addresses'],
                'phone_numbers_found': simmons_results['phone_numbers'],
                'assessment': analysis['assessment'],
                'investigation_date': datetime.now().isoformat()
            },
            'investigation_flag': 'buyer_connection_check'
        })
        
        cursor.execute("""
            UPDATE entities SET properties = ? WHERE entity_id = ?
        """, (json.dumps(props), simmons_id))
        
        # Create connection relationships if found
        if analysis['phillips_to_erik'] > 0:
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                VALUES (?, ?, ?, ?)
            """, (
                phillips_id,
                'person_erik_strombeck',
                'ASSOCIATED_WITH',
                json.dumps({
                    'source': 'connection_investigation',
                    'connection_type': 'potential_associate',
                    'investigation_note': 'Potential connection found - requires verification',
                    'investigation_flag': 'buyer_connection'
                })
            ))
        
        if analysis['simmons_to_erik'] > 0:
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                VALUES (?, ?, ?, ?)
            """, (
                simmons_id,
                'person_erik_strombeck',
                'ASSOCIATED_WITH',
                json.dumps({
                    'source': 'connection_investigation',
                    'connection_type': 'potential_associate',
                    'investigation_note': 'Potential connection found - requires verification',
                    'investigation_flag': 'buyer_connection'
                })
            ))
        
        conn.commit()
        conn.close()
        
        print(f'\n✅ Updated Phillips Joseph L entity')
        print(f'✅ Updated Simmons Lorenza entity')
        if analysis['phillips_to_erik'] > 0 or analysis['simmons_to_erik'] > 0:
            print(f'✅ Created ASSOCIATED_WITH relationships')
    
    def generate_report(self, phillips_results, simmons_results, shared_data, analysis):
        """Generate investigation report."""
        report = []
        report.append("# Phillips & Simmons Connection Investigation Report")
        report.append(f"\n**Investigation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n## INVESTIGATION PURPOSE")
        report.append(f"\nDetermine if Phillips Joseph L and Simmons Lorenza are legitimate third-party buyers or connected to Erik Strombeck (friends/associates).")
        
        report.append(f"\n## PHILLIPS JOSEPH L")
        report.append(f"\n### Connections to Erik Strombeck: {len(phillips_results['connections_to_erik'])}")
        if phillips_results['connections_to_erik']:
            for conn in phillips_results['connections_to_erik'][:5]:
                report.append(f"\n- **{conn['title']}**")
                report.append(f"  - URL: {conn['url']}")
                report.append(f"  - Evidence: {conn['snippet'][:200]}...")
        
        report.append(f"\n### Addresses Found: {len(phillips_results['addresses'])}")
        for addr in phillips_results['addresses'][:10]:
            report.append(f"- {addr}")
        
        report.append(f"\n### Phone Numbers Found: {len(phillips_results['phone_numbers'])}")
        for phone in phillips_results['phone_numbers'][:10]:
            report.append(f"- {phone}")
        
        report.append(f"\n## SIMMONS LORENZA")
        report.append(f"\n### Connections to Erik Strombeck: {len(simmons_results['connections_to_erik'])}")
        if simmons_results['connections_to_erik']:
            for conn in simmons_results['connections_to_erik'][:5]:
                report.append(f"\n- **{conn['title']}**")
                report.append(f"  - URL: {conn['url']}")
                report.append(f"  - Evidence: {conn['snippet'][:200]}...")
        
        report.append(f"\n### Addresses Found: {len(simmons_results['addresses'])}")
        for addr in simmons_results['addresses'][:10]:
            report.append(f"- {addr}")
        
        report.append(f"\n### Phone Numbers Found: {len(simmons_results['phone_numbers'])}")
        for phone in simmons_results['phone_numbers'][:10]:
            report.append(f"- {phone}")
        
        report.append(f"\n## SHARED DATA ANALYSIS")
        report.append(f"\n### Shared Addresses: {len(shared_data['shared_addresses'])}")
        for match in shared_data['shared_addresses']:
            report.append(f"- ⚠️  {match['person']}: {match['address']} matches Erik's address")
        
        report.append(f"\n### Shared Phone Numbers: {len(shared_data['shared_phones'])}")
        for match in shared_data['shared_phones']:
            report.append(f"- ⚠️  {match['person']}: {match['phone']} matches Erik's phone")
        
        report.append(f"\n## CONNECTION ASSESSMENT")
        report.append(f"\n### Assessment: **{analysis['assessment'].upper()}**")
        report.append(f"\n### Conclusion: {analysis['conclusion']}")
        
        if analysis['red_flags']:
            report.append(f"\n### Red Flags:")
            for flag in analysis['red_flags']:
                report.append(f"- ⚠️  {flag}")
        
        report.append(f"\n## INVESTIGATION RECOMMENDATIONS")
        if analysis['assessment'] == 'strong_connection':
            report.append(f"\n1. **VERIFY CONNECTION**: Strong indicators suggest relationship - verify through additional sources")
            report.append(f"2. **NOMINEE BUYER INVESTIGATION**: If connected, investigate if they are nominee buyers")
            report.append(f"3. **ASSET HIDING PATTERN**: Connected buyers would indicate asset hiding through nominee purchases")
        elif analysis['assessment'] == 'weak_connection':
            report.append(f"\n1. **FURTHER INVESTIGATION**: Weak connections found - continue investigation")
            report.append(f"2. **VERIFY INDEPENDENTLY**: Check connections through additional sources")
        else:
            report.append(f"\n1. **LEGITIMATE BUYERS**: No connections found - appear to be legitimate third-party buyers")
            report.append(f"2. **MONITOR**: Continue monitoring for any future connections")
        
        return '\n'.join(report)
    
    def run_investigation(self):
        """Run complete connection investigation."""
        print('=' * 80)
        print('PHILLIPS & SIMMONS CONNECTION INVESTIGATION')
        print('=' * 80)
        
        # Deep research on both parties
        phillips_results = self.research_phillips_deep()
        simmons_results = self.research_simmons_deep()
        
        # Check for shared addresses/phones
        shared_data = self.check_shared_addresses_phones(phillips_results, simmons_results)
        
        # Analyze connections
        analysis = self.analyze_connections(phillips_results, simmons_results, shared_data)
        
        # Import to database
        self.import_to_database(phillips_results, simmons_results, analysis)
        
        # Generate report
        report = self.generate_report(phillips_results, simmons_results, shared_data, analysis)
        
        # Save report
        report_file = f'PHILLIPS_SIMMONS_CONNECTION_INVESTIGATION_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save data
        data_file = f'PHILLIPS_SIMMONS_CONNECTION_DATA_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(data_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'phillips_results': phillips_results,
                'simmons_results': simmons_results,
                'shared_data': shared_data,
                'analysis': analysis,
                'connections_found': self.connections_found,
                'findings': self.findings
            }, f, indent=2)
        
        print(f'\n\n✅ Investigation Complete!')
        print(f'📄 Report saved to: {report_file}')
        print(f'📊 Data saved to: {data_file}')
        
        return {
            'phillips': phillips_results,
            'simmons': simmons_results,
            'shared': shared_data,
            'analysis': analysis
        }

if __name__ == "__main__":
    investigator = PhillipsSimmonsConnectionInvestigator()
    investigator.run_investigation()
