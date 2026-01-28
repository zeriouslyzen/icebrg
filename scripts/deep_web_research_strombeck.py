#!/usr/bin/env python3
"""
Deep Web Research - Strombeck Investigation
Comprehensive web research across all investigation areas
"""

import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore
from src.iceburg.search.web_search import WebSearchAggregator
import os
os.environ['ICEBURG_ENABLE_WEB'] = '1'

class DeepStrombeckResearcher:
    """Deep web research for Strombeck investigation."""
    
    def __init__(self):
        self.search = WebSearchAggregator()
        self.ms = MatrixStore()
        self.findings = []
        self.entities_found = set()
        self.relationships_found = []
        
    def generate_research_queries(self) -> List[Dict[str, str]]:
        """Generate comprehensive research queries."""
        queries = []
        
        # November 2022 Incident
        queries.extend([
            {'category': 'incident', 'query': 'Erik Strombeck November 2022 incident Arcata Humboldt'},
            {'category': 'incident', 'query': 'Erik Strombeck court case November 2022'},
            {'category': 'incident', 'query': 'Erik Strombeck legal action November 2022'},
            {'category': 'incident', 'query': 'Strombeck Properties lawsuit November 2022'},
            {'category': 'incident', 'query': 'Arcata Humboldt November 2022 property dispute'},
        ])
        
        # Trust Documents
        queries.extend([
            {'category': 'trust', 'query': 'Westwood Court Trust Arcata California'},
            {'category': 'trust', 'query': '2351 Westwood Court Arcata trust documents'},
            {'category': 'trust', 'query': 'Erik Strombeck trust beneficiary trustee'},
            {'category': 'trust', 'query': 'Strombeck family trust California'},
        ])
        
        # Property Transactions
        queries.extend([
            {'category': 'property', 'query': '2351 Westwood Court Arcata sale 2021'},
            {'category': 'property', 'query': '2149 Western Avenue Arcata property transactions'},
            {'category': 'property', 'query': '965 W Harris St Eureka property sale'},
            {'category': 'property', 'query': '3114 Nevada St Arcata property sale'},
            {'category': 'property', 'query': 'PO Box 37 Eureka property owner'},
        ])
        
        # Business Partners
        queries.extend([
            {'category': 'partners', 'query': 'Will Startare Arcata Humboldt'},
            {'category': 'partners', 'query': 'Corey Taylor Arcata Humboldt property'},
            {'category': 'partners', 'query': 'Matt Allen Eureka A.A. sponsor'},
            {'category': 'partners', 'query': 'Will Startare Corey Taylor business partners'},
        ])
        
        # Court Records
        queries.extend([
            {'category': 'court', 'query': 'Erik Strombeck Humboldt County court records'},
            {'category': 'court', 'query': 'Steven Mark Strombeck court records Humboldt'},
            {'category': 'court', 'query': 'Strombeck Properties court cases'},
            {'category': 'court', 'query': 'Erik Strombeck bankruptcy filing'},
        ])
        
        # Business Records
        queries.extend([
            {'category': 'business', 'query': 'Strombeck Properties LLC California Secretary of State'},
            {'category': 'business', 'query': 'STEATA LLC California business records'},
            {'category': 'business', 'query': 'Erik Strombeck business registration California'},
        ])
        
        # Property Records
        queries.extend([
            {'category': 'property', 'query': 'Humboldt County property records PO Box 37'},
            {'category': 'property', 'query': 'Arcata property owner Erik Strombeck'},
            {'category': 'property', 'query': 'Eureka property owner Strombeck'},
        ])
        
        # Background Research
        queries.extend([
            {'category': 'background', 'query': 'Erik Strombeck Arcata Humboldt background'},
            {'category': 'background', 'query': 'Steven Mark Strombeck Redwood Capital Bank'},
            {'category': 'background', 'query': 'Strombeck Properties student housing Arcata'},
        ])
        
        return queries
    
    def extract_entities(self, text: str, url: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        entities = []
        
        # Extract names (Person names)
        name_pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b'
        names = re.findall(name_pattern, text)
        
        for name in names:
            name_lower = name.lower()
            # Filter for relevant names
            if any(keyword in name_lower for keyword in [
                'strombeck', 'startare', 'taylor', 'allen', 'matt', 'will', 'corey',
                'erik', 'steven', 'waltina', 'martha'
            ]):
                if name not in self.entities_found:
                    entities.append({
                        'type': 'person',
                        'name': name,
                        'source': url,
                        'context': text[:200]
                    })
                    self.entities_found.add(name)
        
        # Extract addresses
        address_patterns = [
            r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Ct|Court|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place)',
            r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Ct|Court|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place)\s+[A-Z][a-z]+',
            r'PO\s+BOX\s+\d+',
        ]
        
        for pattern in address_patterns:
            addresses = re.findall(pattern, text, re.IGNORECASE)
            for addr in addresses:
                addr_clean = addr.strip()
                if addr_clean not in self.entities_found:
                    entities.append({
                        'type': 'address',
                        'name': addr_clean,
                        'source': url,
                        'context': text[:200]
                    })
                    self.entities_found.add(addr_clean)
        
        # Extract companies/trusts
        company_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:LLC|Inc|Corp|Corporation|Trust|Company|Properties))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Trust)',
        ]
        
        for pattern in company_patterns:
            companies = re.findall(pattern, text)
            for company in companies:
                if company not in self.entities_found:
                    entities.append({
                        'type': 'company',
                        'name': company,
                        'source': url,
                        'context': text[:200]
                    })
                    self.entities_found.add(company)
        
        return entities
    
    def extract_relationships(self, text: str, url: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Extract relationships from text."""
        relationships = []
        
        text_lower = text.lower()
        
        # Family relationships
        if any(word in text_lower for word in ['son', 'father', 'mother', 'wife', 'husband', 'child', 'parent']):
            for entity1 in entities:
                for entity2 in entities:
                    if entity1['name'] != entity2['name']:
                        if 'strombeck' in entity1['name'].lower() or 'strombeck' in entity2['name'].lower():
                            if 'son' in text_lower or 'child' in text_lower:
                                relationships.append({
                                    'source': entity1['name'],
                                    'target': entity2['name'],
                                    'type': 'CHILD_OF',
                                    'source_url': url,
                                    'context': text[:200]
                                })
                            elif 'wife' in text_lower or 'husband' in text_lower:
                                relationships.append({
                                    'source': entity1['name'],
                                    'target': entity2['name'],
                                    'type': 'SPOUSE_OF',
                                    'source_url': url,
                                    'context': text[:200]
                                })
        
        # Business relationships
        if any(word in text_lower for word in ['partner', 'business', 'bought', 'sold', 'purchased']):
            for entity1 in entities:
                for entity2 in entities:
                    if entity1['name'] != entity2['name']:
                        if any(name in entity1['name'].lower() or name in entity2['name'].lower() 
                               for name in ['strombeck', 'startare', 'taylor', 'allen']):
                            if 'partner' in text_lower or 'business' in text_lower:
                                relationships.append({
                                    'source': entity1['name'],
                                    'target': entity2['name'],
                                    'type': 'BUSINESS_PARTNER',
                                    'source_url': url,
                                    'context': text[:200]
                                })
                            elif 'bought' in text_lower or 'purchased' in text_lower:
                                relationships.append({
                                    'source': entity1['name'],
                                    'target': entity2['name'],
                                    'type': 'PURCHASED_FROM',
                                    'source_url': url,
                                    'context': text[:200]
                                })
                            elif 'sold' in text_lower:
                                relationships.append({
                                    'source': entity1['name'],
                                    'target': entity2['name'],
                                    'type': 'SOLD_TO',
                                    'source_url': url,
                                    'context': text[:200]
                                })
        
        return relationships
    
    def search_category(self, category: str, queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Search a specific category."""
        print(f'\n\n{"="*80}')
        print(f'RESEARCHING CATEGORY: {category.upper()}')
        print(f'{"="*80}')
        
        category_queries = [q for q in queries if q['category'] == category]
        results = {
            'category': category,
            'queries': [],
            'entities': [],
            'relationships': [],
            'findings': []
        }
        
        for query_info in category_queries:
            query = query_info['query']
            print(f'\n🔍 Searching: {query}')
            
            try:
                search_results = self.search.search(
                    query,
                    sources=['ddg', 'brave'],
                    max_results_per_source=10
                )
                
                query_results = {
                    'query': query,
                    'results_count': len(search_results),
                    'urls': []
                }
                
                for result in search_results:
                    text = f"{result.title} {result.snippet}".lower()
                    full_text = f"{result.title} {result.snippet}"
                    
                    query_results['urls'].append({
                        'url': result.url,
                        'title': result.title,
                        'snippet': result.snippet
                    })
                    
                    # Extract entities
                    entities = self.extract_entities(full_text, result.url)
                    results['entities'].extend(entities)
                    
                    # Extract relationships
                    relationships = self.extract_relationships(full_text, result.url, entities)
                    results['relationships'].extend(relationships)
                    
                    # Store finding if relevant
                    if any(keyword in text for keyword in [
                        'strombeck', 'erik', 'steven', 'waltina', 'startare', 'taylor',
                        'allen', 'westwood', 'western', 'harris', 'nevada', 'po box 37',
                        'trust', 'november 2022', 'court', 'property', 'sale', 'bought'
                    ]):
                        results['findings'].append({
                            'query': query,
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet,
                            'relevance': 'high'
                        })
                        self.findings.append({
                            'category': category,
                            'query': query,
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                
                results['queries'].append(query_results)
                print(f'  ✅ Found {len(search_results)} results')
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
                results['queries'].append({
                    'query': query,
                    'error': str(e)
                })
        
        print(f'\n  Total entities found: {len(results["entities"])}')
        print(f'  Total relationships found: {len(results["relationships"])}')
        print(f'  Total findings: {len(results["findings"])}')
        
        return results
    
    def import_to_database(self, all_results: Dict[str, Any]):
        """Import findings to database."""
        print(f'\n\n{"="*80}')
        print('IMPORTING FINDINGS TO DATABASE')
        print(f'{"="*80}')
        
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
        imported_entities = 0
        imported_relationships = 0
        
        # Import entities
        all_entities = []
        for category_results in all_results.values():
            all_entities.extend(category_results.get('entities', []))
        
        # Deduplicate entities
        seen_entities = set()
        for entity in all_entities:
            entity_key = f"{entity['type']}_{entity['name']}"
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                
                entity_id = f"{entity['type']}_{entity['name'].lower().replace(' ', '_').replace('.', '').replace(',', '')}"
                entity_id = re.sub(r'[^a-z0-9_]', '', entity_id)
                
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        entity_id,
                        entity['name'],
                        entity['type'],
                        'web_research',
                        json.dumps({
                            'source_url': entity.get('source', ''),
                            'context': entity.get('context', ''),
                            'research_date': datetime.now().isoformat()
                        })
                    ))
                    imported_entities += 1
                except Exception as e:
                    pass
        
        # Import relationships
        all_relationships = []
        for category_results in all_results.values():
            all_relationships.extend(category_results.get('relationships', []))
        
        # Deduplicate relationships
        seen_relationships = set()
        for rel in all_relationships:
            rel_key = f"{rel['source']}_{rel['type']}_{rel['target']}"
            if rel_key not in seen_relationships:
                seen_relationships.add(rel_key)
                
                source_id = f"person_{rel['source'].lower().replace(' ', '_')}"
                target_id = f"person_{rel['target'].lower().replace(' ', '_')}"
                
                # Handle different entity types
                if rel['source'].lower().startswith(('po box', 'address')):
                    source_id = f"address_{rel['source'].lower().replace(' ', '_')}"
                if rel['target'].lower().startswith(('po box', 'address')):
                    target_id = f"address_{rel['target'].lower().replace(' ', '_')}"
                
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                        VALUES (?, ?, ?, ?)
                    """, (
                        source_id,
                        target_id,
                        rel['type'],
                        json.dumps({
                            'source_url': rel.get('source_url', ''),
                            'context': rel.get('context', ''),
                            'research_date': datetime.now().isoformat(),
                            'source': 'web_research'
                        })
                    ))
                    imported_relationships += 1
                except Exception as e:
                    pass
        
        conn.commit()
        conn.close()
        
        print(f'\n✅ Imported {imported_entities} entities')
        print(f'✅ Imported {imported_relationships} relationships')
    
    def generate_report(self, all_results: Dict[str, Any]) -> str:
        """Generate comprehensive research report."""
        report = []
        report.append("# Deep Web Research Report - Strombeck Investigation")
        report.append(f"\n**Research Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Total Categories Researched**: {len(all_results)}")
        
        total_findings = sum(len(r.get('findings', [])) for r in all_results.values())
        total_entities = sum(len(r.get('entities', [])) for r in all_results.values())
        total_relationships = sum(len(r.get('relationships', [])) for r in all_results.values())
        
        report.append(f"\n**Total Findings**: {total_findings}")
        report.append(f"**Total Entities Found**: {total_entities}")
        report.append(f"**Total Relationships Found**: {total_relationships}")
        
        for category, results in all_results.items():
            report.append(f"\n\n## {category.upper()} RESEARCH")
            report.append(f"\n### Summary")
            report.append(f"- Queries Executed: {len(results.get('queries', []))}")
            report.append(f"- Entities Found: {len(results.get('entities', []))}")
            report.append(f"- Relationships Found: {len(results.get('relationships', []))}")
            report.append(f"- Findings: {len(results.get('findings', []))}")
            
            if results.get('findings'):
                report.append(f"\n### Key Findings")
                for finding in results['findings'][:10]:  # Top 10
                    report.append(f"\n**{finding['title']}**")
                    report.append(f"- URL: {finding['url']}")
                    report.append(f"- Snippet: {finding['snippet'][:200]}...")
            
            if results.get('entities'):
                report.append(f"\n### Entities Discovered")
                for entity in results['entities'][:20]:  # Top 20
                    report.append(f"- **{entity['name']}** ({entity['type']})")
        
        return '\n'.join(report)
    
    def run_comprehensive_research(self):
        """Run comprehensive deep web research."""
        print('=' * 80)
        print('DEEP WEB RESEARCH - STROMBECK INVESTIGATION')
        print('=' * 80)
        
        queries = self.generate_research_queries()
        print(f'\nGenerated {len(queries)} research queries across 8 categories')
        
        all_results = {}
        
        categories = ['incident', 'trust', 'property', 'partners', 'court', 'business', 'background']
        
        for category in categories:
            try:
                results = self.search_category(category, queries)
                all_results[category] = results
            except Exception as e:
                print(f'\n⚠️  Error researching {category}: {e}')
                all_results[category] = {'category': category, 'error': str(e)}
        
        # Import to database
        self.import_to_database(all_results)
        
        # Generate report
        report = self.generate_report(all_results)
        
        # Save report
        report_file = f'DEEP_WEB_RESEARCH_REPORT_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save raw data
        data_file = f'DEEP_WEB_RESEARCH_DATA_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(data_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
                'findings': self.findings
            }, f, indent=2)
        
        print(f'\n\n✅ Research Complete!')
        print(f'📄 Report saved to: {report_file}')
        print(f'📊 Data saved to: {data_file}')
        
        return all_results

if __name__ == "__main__":
    researcher = DeepStrombeckResearcher()
    researcher.run_comprehensive_research()
