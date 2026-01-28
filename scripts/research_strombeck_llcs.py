#!/usr/bin/env python3
"""
Research Strombeck LLCs Across Different States
Investigate Adam Strombeck and Erik Strombeck LLC structures
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

class StrombeckLLCResearcher:
    """Research Strombeck LLCs across different states."""
    
    def __init__(self):
        self.search = WebSearchAggregator()
        self.ms = MatrixStore()
        self.findings = []
        
    def research_adam_strombeck(self):
        """Research Adam Strombeck and his LLCs."""
        print('=' * 80)
        print('RESEARCHING ADAM STROMBECK')
        print('=' * 80)
        
        queries = [
            'Adam Strombeck LLC corporationwiki',
            'Adam Strombeck LLC different states',
            'Adam Strombeck business registration',
            'Adam Strombeck Erik Strombeck relationship',
            'Adam Strombeck Strombeck Properties',
        ]
        
        results = {
            'name': 'Adam Strombeck',
            'llcs': [],
            'states': set(),
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
                    
                    # Extract LLC information
                    llc_patterns = [
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+LLC)',
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+L\.L\.C\.)',
                    ]
                    
                    for pattern in llc_patterns:
                        llcs = re.findall(pattern, full_text, re.IGNORECASE)
                        for llc in llcs:
                            if llc not in results['llcs']:
                                results['llcs'].append(llc)
                    
                    # Extract state information
                    state_pattern = r'\b(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)\b'
                    states = re.findall(state_pattern, full_text, re.IGNORECASE)
                    for state in states:
                        results['states'].add(state.title())
                    
                    # Store relevant findings
                    if any(keyword in text for keyword in ['adam', 'strombeck', 'llc', 'corporation', 'business']):
                        results['findings'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                        self.findings.append({
                            'person': 'Adam Strombeck',
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        results['states'] = list(results['states'])
        return results
    
    def research_erik_strombeck_llcs(self):
        """Research Erik Strombeck LLCs."""
        print('\n\n' + '=' * 80)
        print('RESEARCHING ERIK STROMBECK LLCs')
        print('=' * 80)
        
        queries = [
            'Erik Strombeck LLC different states',
            'Erik Strombeck LLC Delaware Nevada Wyoming',
            'Erik Strombeck business registration multiple states',
            'Erik Strombeck asset protection LLC',
        ]
        
        results = {
            'name': 'Erik Strombeck',
            'llcs': [],
            'states': set(),
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
                    
                    # Extract LLC information
                    llc_patterns = [
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+LLC)',
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+L\.L\.C\.)',
                        r'(Erik\s+Strombeck\s+LLC)',
                        r'(Strombeck\s+Properties\s+LLC)',
                    ]
                    
                    for pattern in llc_patterns:
                        llcs = re.findall(pattern, full_text, re.IGNORECASE)
                        for llc in llcs:
                            if llc not in results['llcs']:
                                results['llcs'].append(llc)
                    
                    # Extract state information
                    state_pattern = r'\b(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)\b'
                    states = re.findall(state_pattern, full_text, re.IGNORECASE)
                    for state in states:
                        results['states'].add(state.title())
                    
                    # Store relevant findings
                    if any(keyword in text for keyword in ['erik', 'strombeck', 'llc', 'corporation', 'business', 'delaware', 'nevada', 'wyoming']):
                        results['findings'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                        self.findings.append({
                            'person': 'Erik Strombeck',
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        results['states'] = list(results['states'])
        return results
    
    def analyze_asset_hiding_pattern(self, adam_results, erik_results):
        """Analyze why LLCs are in different states."""
        print('\n\n' + '=' * 80)
        print('ANALYZING ASSET HIDING PATTERN')
        print('=' * 80)
        
        analysis = {
            'pattern': 'multi_state_llc_structure',
            'adam_states': adam_results['states'],
            'erik_states': erik_results['states'],
            'all_states': list(set(adam_results['states'] + erik_results['states'])),
            'reasons': [],
            'red_flags': []
        }
        
        # Common asset protection states
        asset_protection_states = ['Delaware', 'Nevada', 'Wyoming', 'South Dakota']
        
        print(f'\nAdam Strombeck LLC States: {adam_results["states"]}')
        print(f'Erik Strombeck LLC States: {erik_results["states"]}')
        print(f'All States: {analysis["all_states"]}')
        
        # Analyze reasons for different states
        if len(analysis['all_states']) > 1:
            analysis['reasons'].append('Multiple state registrations suggest asset protection strategy')
        
        for state in analysis['all_states']:
            if state in asset_protection_states:
                analysis['reasons'].append(f'{state} is a known asset protection state (strong privacy laws, favorable LLC statutes)')
                analysis['red_flags'].append(f'LLC registered in {state} - asset protection jurisdiction')
        
        if 'California' in analysis['all_states'] and len(analysis['all_states']) > 1:
            analysis['reasons'].append('California LLCs likely for local operations, out-of-state LLCs for asset protection')
            analysis['red_flags'].append('Dual-state structure: California operations + out-of-state asset protection')
        
        print(f'\n\n🔍 ANALYSIS:')
        print(f'  Pattern: Multi-state LLC structure')
        print(f'  States Used: {", ".join(analysis["all_states"])}')
        print(f'\n  Reasons:')
        for reason in analysis['reasons']:
            print(f'    - {reason}')
        print(f'\n  Red Flags:')
        for flag in analysis['red_flags']:
            print(f'    ⚠️  {flag}')
        
        return analysis
    
    def import_to_database(self, adam_results, erik_results, analysis):
        """Import findings to database."""
        print('\n\n' + '=' * 80)
        print('IMPORTING TO DATABASE')
        print('=' * 80)
        
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
        # Create Adam Strombeck entity
        adam_id = 'person_adam_strombeck'
        cursor.execute("""
            INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (
            adam_id,
            'Adam Strombeck',
            'person',
            'web_research',
            json.dumps({
                'research_date': datetime.now().isoformat(),
                'llcs': adam_results['llcs'],
                'states': adam_results['states'],
                'investigation_note': 'Strombeck family member with LLCs in multiple states',
                'investigation_flag': 'multi_state_llc_structure'
            })
        ))
        
        # Create Erik Strombeck entity (if not exists)
        erik_id = 'person_erik_strombeck'
        cursor.execute("""
            INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (
            erik_id,
            'Erik Strombeck',
            'person',
            'web_research',
            json.dumps({
                'research_date': datetime.now().isoformat(),
                'llcs': erik_results['llcs'],
                'states': erik_results['states'],
                'investigation_note': 'LLCs registered in multiple states - asset protection pattern',
                'investigation_flag': 'multi_state_llc_structure'
            })
        ))
        
        # Link Adam and Erik as family
        cursor.execute("""
            INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
            VALUES (?, ?, ?, ?)
        """, (
            adam_id,
            erik_id,
            'FAMILY_OF',
            json.dumps({
                'source': 'web_research',
                'investigation_note': 'Both have LLCs in different states - family asset protection strategy',
                'investigation_flag': 'multi_state_llc_pattern'
            })
        ))
        
        # Create LLC entities
        all_llcs = list(set(adam_results['llcs'] + erik_results['llcs']))
        for llc_name in all_llcs:
            llc_id = f"company_{llc_name.lower().replace(' ', '_').replace('.', '').replace(',', '')}"
            llc_id = re.sub(r'[^a-z0-9_]', '', llc_id)
            
            cursor.execute("""
                INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                VALUES (?, ?, ?, ?, ?)
            """, (
                llc_id,
                llc_name,
                'company',
                'web_research',
                json.dumps({
                    'research_date': datetime.now().isoformat(),
                    'investigation_flag': 'multi_state_llc'
                })
            ))
        
        conn.commit()
        conn.close()
        
        print(f'\n✅ Imported Adam Strombeck entity')
        print(f'✅ Updated Erik Strombeck entity')
        print(f'✅ Created {len(all_llcs)} LLC entities')
        print(f'✅ Linked Adam and Erik as family')
    
    def generate_report(self, adam_results, erik_results, analysis):
        """Generate research report."""
        report = []
        report.append("# Strombeck Multi-State LLC Research Report")
        report.append(f"\n**Research Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n## ADAM STROMBECK")
        report.append(f"\n### LLCs Found")
        for llc in adam_results['llcs']:
            report.append(f"- {llc}")
        
        report.append(f"\n### States")
        for state in adam_results['states']:
            report.append(f"- {state}")
        
        report.append(f"\n### Findings")
        report.append(f"- Total findings: {len(adam_results['findings'])}")
        
        report.append(f"\n## ERIK STROMBECK")
        report.append(f"\n### LLCs Found")
        for llc in erik_results['llcs']:
            report.append(f"- {llc}")
        
        report.append(f"\n### States")
        for state in erik_results['states']:
            report.append(f"- {state}")
        
        report.append(f"\n### Findings")
        report.append(f"- Total findings: {len(erik_results['findings'])}")
        
        report.append(f"\n## ASSET HIDING ANALYSIS")
        report.append(f"\n### Pattern Identified")
        report.append(f"- **Multi-State LLC Structure**")
        report.append(f"- All States Used: {', '.join(analysis['all_states'])}")
        
        report.append(f"\n### Reasons for Different States")
        for reason in analysis['reasons']:
            report.append(f"- {reason}")
        
        report.append(f"\n### Red Flags")
        for flag in analysis['red_flags']:
            report.append(f"- ⚠️  {flag}")
        
        report.append(f"\n## INVESTIGATION IMPLICATIONS")
        report.append(f"\n### Asset Protection Strategy")
        report.append(f"- Multiple state registrations suggest systematic asset protection")
        report.append(f"- Out-of-state LLCs likely used to obscure ownership")
        report.append(f"- California LLCs for local operations, other states for asset protection")
        
        report.append(f"\n### Next Steps")
        report.append(f"1. Verify actual LLC registrations in each state")
        report.append(f"2. Identify all LLCs owned by Adam and Erik Strombeck")
        report.append(f"3. Map property ownership through LLC structures")
        report.append(f"4. Analyze asset transfers between LLCs")
        
        return '\n'.join(report)
    
    def run_research(self):
        """Run complete research."""
        print('=' * 80)
        print('STROMBECK MULTI-STATE LLC RESEARCH')
        print('=' * 80)
        
        # Research Adam Strombeck
        adam_results = self.research_adam_strombeck()
        
        # Research Erik Strombeck LLCs
        erik_results = self.research_erik_strombeck_llcs()
        
        # Analyze pattern
        analysis = self.analyze_asset_hiding_pattern(adam_results, erik_results)
        
        # Import to database
        self.import_to_database(adam_results, erik_results, analysis)
        
        # Generate report
        report = self.generate_report(adam_results, erik_results, analysis)
        
        # Save report
        report_file = f'STROMBECK_MULTI_STATE_LLC_RESEARCH_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save data
        data_file = f'STROMBECK_MULTI_STATE_LLC_DATA_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(data_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'adam_strombeck': adam_results,
                'erik_strombeck': erik_results,
                'analysis': analysis,
                'findings': self.findings
            }, f, indent=2)
        
        print(f'\n\n✅ Research Complete!')
        print(f'📄 Report saved to: {report_file}')
        print(f'📊 Data saved to: {data_file}')
        
        return {
            'adam': adam_results,
            'erik': erik_results,
            'analysis': analysis
        }

if __name__ == "__main__":
    researcher = StrombeckLLCResearcher()
    researcher.run_research()
