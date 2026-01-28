#!/usr/bin/env python3
"""
Get REAL Strombeck Family Data
Actually searches and imports real information
"""

import sys
import json
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore
from src.iceburg.search.web_search import WebSearchAggregator
import os
os.environ['ICEBURG_ENABLE_WEB'] = '1'

def search_real_family_data():
    """Search for actual family relationship data."""
    print('=' * 80)
    print('SEARCHING FOR REAL STROMBECK FAMILY DATA')
    print('=' * 80)
    
    search = WebSearchAggregator()
    ms = MatrixStore()
    conn = ms.get_connection()
    cursor = conn.cursor()
    
    # Real search queries
    queries = [
        'Steven Mark Strombeck Erik Strombeck relationship Arcata',
        'Waltina Martha Strombeck Steven Mark Strombeck wife mother',
        'Strombeck Properties owner family members',
        'Erik Strombeck Arcata Humboldt',
        'Steven Strombeck children kids family',
    ]
    
    findings = []
    
    for query in queries:
        print(f'\nSearching: {query}')
        try:
            results = search.search(query, sources=['ddg'], max_results_per_source=5)
            
            for result in results:
                text = f"{result.title} {result.snippet}".lower()
                
                # Look for actual family relationship indicators
                if any(word in text for word in ['son', 'father', 'mother', 'wife', 'husband', 'child', 'parent']):
                    # Extract names
                    name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?)\b'
                    names = re.findall(name_pattern, result.title + ' ' + result.snippet)
                    
                    # Filter for Strombeck names
                    strombeck_names = [n for n in names if 'strombeck' in n.lower()]
                    
                    if strombeck_names:
                        findings.append({
                            'query': query,
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet,
                            'names': strombeck_names
                        })
                        print(f'  ✅ Found: {strombeck_names}')
        except Exception as e:
            print(f'  ⚠️  Error: {e}')
    
    # Save findings
    with open('STROMBECK_REAL_DATA_FINDINGS.json', 'w') as f:
        json.dump(findings, f, indent=2)
    
    print(f'\n\n✅ Found {len(findings)} potential data points')
    print('Saved to: STROMBECK_REAL_DATA_FINDINGS.json')
    
    conn.close()
    return findings

if __name__ == "__main__":
    search_real_family_data()
