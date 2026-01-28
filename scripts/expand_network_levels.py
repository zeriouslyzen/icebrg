#!/usr/bin/env python3
"""
Multi-Level Network Expansion
Uses handwritten notes data to find next-level connections systematically
"""

import json
import sys
from pathlib import Path
from typing import List, Set, Dict
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore
from src.iceburg.search.web_search import WebSearchAggregator
import os
os.environ['ICEBURG_ENABLE_WEB'] = '1'

class NetworkExpander:
    """Expand network in levels from seed data."""
    
    def __init__(self):
        self.matrix_store = MatrixStore()
        self.web_search = WebSearchAggregator()
        self.processed_entities = set()
        
    def expand_level_1(self):
        """Level 1: Direct connections from handwritten notes."""
        print("\n=== LEVEL 1: DIRECT CONNECTIONS FROM NOTES ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        # Get all entities from handwritten notes
        cursor.execute("""
            SELECT entity_id, name, entity_type, properties
            FROM entities
            WHERE source = 'handwritten_notes'
        """)
        
        seed_entities = cursor.fetchall()
        print(f"Found {len(seed_entities)} seed entities from handwritten notes")
        
        # For each seed entity, find connections
        for entity in seed_entities:
            entity_id, name, entity_type, props_json = entity
            props = json.loads(props_json) if props_json else {}
            
            if entity_id in self.processed_entities:
                continue
            
            print(f"\nExpanding from: {name} ({entity_type})")
            
            # Search for connections
            if entity_type == 'person':
                self._find_person_connections(name, entity_id, cursor)
            elif entity_type == 'company':
                self._find_company_connections(name, entity_id, cursor)
            elif entity_type == 'address':
                self._find_address_connections(name, entity_id, cursor)
            elif entity_type == 'phone':
                self._find_phone_connections(name, entity_id, cursor)
            
            self.processed_entities.add(entity_id)
        
        conn.commit()
        conn.close()
    
    def _find_person_connections(self, name: str, entity_id: str, cursor):
        """Find connections for a person."""
        queries = [
            f"{name} property owner Arcata Eureka",
            f"{name} business partner Humboldt",
            f"{name} real estate transactions",
            f"{name} sold property",
        ]
        
        for query in queries:
            results = self.web_search.search(query, sources=['ddg'], max_results_per_source=3)
            for result in results:
                self._extract_connections_from_result(result, entity_id, cursor, 'person_search')
    
    def _find_company_connections(self, name: str, entity_id: str, cursor):
        """Find connections for a company."""
        queries = [
            f"{name} owners directors Humboldt",
            f"{name} properties owned",
            f"{name} business partners",
            f"{name} transactions sales",
        ]
        
        for query in queries:
            results = self.web_search.search(query, sources=['ddg'], max_results_per_source=3)
            for result in results:
                self._extract_connections_from_result(result, entity_id, cursor, 'company_search')
    
    def _find_address_connections(self, address: str, entity_id: str, cursor):
        """Find connections for an address."""
        queries = [
            f"{address} property owner",
            f"{address} sold purchased",
            f"{address} deed records",
            f"{address} tenant rental",
        ]
        
        for query in queries:
            results = self.web_search.search(query, sources=['ddg'], max_results_per_source=3)
            for result in results:
                self._extract_connections_from_result(result, entity_id, cursor, 'address_search')
    
    def _find_phone_connections(self, phone: str, entity_id: str, cursor):
        """Find connections for a phone number."""
        queries = [
            f"{phone} owner business",
            f"{phone} property management",
        ]
        
        for query in queries:
            results = self.web_search.search(query, sources=['ddg'], max_results_per_source=3)
            for result in results:
                self._extract_connections_from_result(result, entity_id, cursor, 'phone_search')
    
    def _extract_connections_from_result(self, result, source_entity_id: str, cursor, search_type: str):
        """Extract entities and relationships from search result."""
        full_text = f"{result.title} {result.snippet}"
        
        # Extract person names (better filtering)
        person_pattern = r'\b([A-Z][a-z]{2,} [A-Z][a-z]{2,})\b'
        persons = re.findall(person_pattern, full_text)
        
        # Extract company names
        company_pattern = r'\b([A-Z][a-z]+ (?:Properties|LLC|Inc|Corp|Company|Realty|Housing|Agency|Bank|Capital))\b'
        companies = re.findall(company_pattern, full_text)
        
        # Extract addresses
        addr_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:St|Street|Ave|Avenue|Court|Rd|Road|Blvd|Boulevard)[\s,]*Arcata|Eureka'
        addresses = re.findall(addr_pattern, full_text, re.IGNORECASE)
        
        # Create entities and relationships
        skip_words = {'property', 'management', 'housing', 'apartments'}
        
        for person in persons:
            if any(word in person.lower() for word in skip_words):
                continue
            if 'strombeck' in person.lower():
                continue
            
            person_id = f"person_{person.lower().replace(' ', '_')[:50]}"
            self._create_entity_and_link(person_id, person, 'person', source_entity_id, cursor, search_type, 'ASSOCIATED_WITH')
        
        for company in companies:
            if 'strombeck' in company.lower():
                continue
            
            company_id = f"company_{company.lower().replace(' ', '_').replace('.', '')[:50]}"
            self._create_entity_and_link(company_id, company, 'company', source_entity_id, cursor, search_type, 'ASSOCIATED_WITH')
        
        for address in addresses:
            addr_id = f"address_{address.lower().replace(' ', '_').replace('+', '').replace(',', '')[:50]}"
            self._create_entity_and_link(addr_id, address, 'address', source_entity_id, cursor, search_type, 'LOCATED_AT')
    
    def _create_entity_and_link(self, entity_id: str, name: str, entity_type: str, 
                                source_entity_id: str, cursor, source: str, rel_type: str):
        """Create entity and link to source."""
        try:
            # Create entity
            cursor.execute("""
                INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                VALUES (?, ?, ?, ?, ?)
            """, (
                entity_id,
                name,
                entity_type,
                source,
                json.dumps({'discovered_via': source_entity_id})
            ))
            
            # Create relationship
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                VALUES (?, ?, ?, ?)
            """, (
                source_entity_id,
                entity_id,
                rel_type,
                json.dumps({'source': source, 'discovered_from': source_entity_id})
            ))
            
            print(f"    ✅ {name} --[{rel_type}]--> (from {source_entity_id})")
        except Exception as e:
            pass  # Skip duplicates/errors
    
    def expand_level_2(self):
        """Level 2: Connections from Level 1 entities."""
        print("\n=== LEVEL 2: SECOND-DEGREE CONNECTIONS ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        # Get Level 1 entities (connected to handwritten notes)
        cursor.execute("""
            SELECT DISTINCT e.entity_id, e.name, e.entity_type
            FROM entities e
            INNER JOIN relationships r ON r.target_id = e.entity_id
            INNER JOIN entities seed ON seed.entity_id = r.source_id
            WHERE seed.source = 'handwritten_notes'
            AND e.source != 'handwritten_notes'
            LIMIT 20
        """)
        
        level1_entities = cursor.fetchall()
        print(f"Found {len(level1_entities)} Level 1 entities to expand")
        
        for entity_id, name, entity_type in level1_entities:
            if entity_id in self.processed_entities:
                continue
            
            print(f"\nExpanding Level 2 from: {name}")
            
            if entity_type == 'person':
                self._find_person_connections(name, entity_id, cursor)
            elif entity_type == 'company':
                self._find_company_connections(name, entity_id, cursor)
            
            self.processed_entities.add(entity_id)
        
        conn.commit()
        conn.close()
    
    def find_property_transaction_chains(self):
        """Find property transaction chains by following SOLD_TO/PURCHASED_FROM."""
        print("\n=== FINDING PROPERTY TRANSACTION CHAINS ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        # Get all addresses from handwritten notes
        cursor.execute("""
            SELECT entity_id, name
            FROM entities
            WHERE entity_type = 'address' AND source = 'handwritten_notes'
        """)
        
        addresses = cursor.fetchall()
        
        for addr_id, address in addresses:
            print(f"\nFinding transactions for: {address}")
            
            # Search for transactions
            queries = [
                f"{address} sold to purchased by",
                f"{address} property transaction deed",
                f"{address} ownership transfer",
            ]
            
            for query in queries:
                results = self.web_search.search(query, sources=['ddg'], max_results_per_source=3)
                
                for result in results:
                    text = result.title + ' ' + result.snippet
                    
                    # Look for buyer/seller patterns
                    buyer_match = re.search(r'(?:sold to|purchased by|bought by)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', text, re.IGNORECASE)
                    seller_match = re.search(r'(?:sold by|purchased from)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', text, re.IGNORECASE)
                    
                    if buyer_match:
                        buyer_name = buyer_match.group(1)
                        buyer_id = f"person_{buyer_name.lower().replace(' ', '_')[:50]}"
                        
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                buyer_id,
                                buyer_name,
                                'person',
                                'transaction_chain',
                                json.dumps({'address': address, 'role': 'buyer'})
                            ))
                            
                            cursor.execute("""
                                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                                VALUES (?, ?, ?, ?)
                            """, (
                                buyer_id,
                                addr_id,
                                'PURCHASED_FROM',
                                json.dumps({'address': address, 'source': 'transaction_search'})
                            ))
                            
                            print(f"  ✅ Transaction: {buyer_name} --[PURCHASED_FROM]--> {address}")
                        except Exception as e:
                            pass
                    
                    if seller_match:
                        seller_name = seller_match.group(1)
                        seller_id = f"person_{seller_name.lower().replace(' ', '_')[:50]}"
                        
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                seller_id,
                                seller_name,
                                'person',
                                'transaction_chain',
                                json.dumps({'address': address, 'role': 'seller'})
                            ))
                            
                            cursor.execute("""
                                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                                VALUES (?, ?, ?, ?)
                            """, (
                                seller_id,
                                addr_id,
                                'SOLD_TO',
                                json.dumps({'address': address, 'source': 'transaction_search'})
                            ))
                            
                            print(f"  ✅ Transaction: {seller_name} --[SOLD_TO]--> {address}")
                        except Exception as e:
                            pass
        
        conn.commit()
        conn.close()
    
    def run_full_expansion(self):
        """Run complete multi-level expansion."""
        print("=" * 80)
        print("MULTI-LEVEL NETWORK EXPANSION")
        print("=" * 80)
        
        # Level 1: Direct from handwritten notes
        self.expand_level_1()
        
        # Level 2: From Level 1 connections
        self.expand_level_2()
        
        # Property transaction chains
        self.find_property_transaction_chains()
        
        # Report
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(DISTINCT e.entity_id)
            FROM entities e
            WHERE e.source IN ('handwritten_notes', 'person_search', 'company_search', 
                              'address_search', 'phone_search', 'transaction_chain')
        """)
        total_entities = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM relationships
            WHERE properties LIKE '%person_search%' 
               OR properties LIKE '%company_search%'
               OR properties LIKE '%transaction_chain%'
        """)
        total_rels = cursor.fetchone()[0]
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("EXPANSION COMPLETE")
        print("=" * 80)
        print(f"Total entities: {total_entities}")
        print(f"Total relationships: {total_rels}")
        print("\n✅ Network expanded and ready for PEGASUS visualization!")


if __name__ == "__main__":
    expander = NetworkExpander()
    expander.run_full_expansion()
