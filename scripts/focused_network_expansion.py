#!/usr/bin/env python3
"""
Focused Network Expansion
Uses handwritten notes data systematically to find REAL next-level connections
"""

import json
import sys
from pathlib import Path
from typing import List, Set, Dict
import re
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore
from src.iceburg.search.web_search import WebSearchAggregator
import os
os.environ['ICEBURG_ENABLE_WEB'] = '1'

class FocusedNetworkExpander:
    """Focused expansion finding real connections."""
    
    def __init__(self):
        self.matrix_store = MatrixStore()
        self.web_search = WebSearchAggregator()
        
        # Real entity filters (common words to skip)
        self.skip_words = {
            'property', 'management', 'housing', 'apartments', 'affordable',
            'list', 'lost', 'coast', 'union', 'county', 'city', 'state',
            'california', 'official', 'website', 'contact', 'decorating',
            'ownership', 'general', 'building', 'parkway', 'arrowhead',
            'vista', 'realty', 'action', 'agency', 'authority', 'license',
            'corporation', 'rental', 'managers', 'poly', 'cal', 'overview',
            'details', 'discover', 'company', 'info', 'registered', 'agent',
            'sacramento', 'arts', 'degree', 'contractors', 'alumni', 'owned',
            'forever', 'humboldt', 'answer', 'your', 'our', 'staff', 'real',
            'estate', 'transfer', 'records', 'phone', 'number', 'contact',
            'bizapedia', 'salvation', 'army', 'eureka'  # Common false positives
        }
    
    def find_who_else_has_phone_numbers(self):
        """Find who else uses the phone numbers from notes."""
        print("\n=== FINDING WHO ELSE HAS THESE PHONE NUMBERS ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        phones = ['707-822-4557']  # Known phone
        
        for phone in phones:
            print(f"\nSearching: {phone}")
            
            # Reverse phone lookup queries
            queries = [
                f"{phone} owner business",
                f"{phone} who owns",
                f"{phone} property management",
                f"reverse phone lookup {phone}",
            ]
            
            for query in queries:
                results = self.web_search.search(query, sources=['ddg'], max_results_per_source=5)
                
                for result in results:
                    text = result.title + ' ' + result.snippet
                    
                    # Extract actual names (better filtering)
                    # Look for "Owner: Name" or "Business: Name" patterns
                    owner_patterns = [
                        r'owner[:\s]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})',
                        r'business[:\s]+([A-Z][a-z]+ (?:Properties|LLC|Inc|Corp|Company))',
                        r'([A-Z][a-z]{2,} [A-Z][a-z]{2,})[\s,]+(?:owns|operates|manages)[\s,]+(?:this|the)[\s,]+(?:phone|number|business)',
                    ]
                    
                    for pattern in owner_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            if isinstance(match, tuple):
                                match = ' '.join(match)
                            
                            # Filter false positives
                            parts = match.split()
                            if len(parts) >= 2 and all(part.lower() not in self.skip_words for part in parts):
                                self._create_phone_connection(phone, match, cursor, result.url)
                
                time.sleep(1)
        
        conn.commit()
        conn.close()
    
    def _create_phone_connection(self, phone: str, name: str, cursor, source_url: str):
        """Create entity and link to phone number."""
        phone_id = f"phone_{phone.replace('-', '').replace('(', '').replace(')', '').replace(' ', '')}"
        
        # Determine entity type
        if any(word in name.lower() for word in ['llc', 'inc', 'corp', 'company', 'properties']):
            entity_type = 'company'
            entity_id = f"company_{name.lower().replace(' ', '_').replace('.', '')[:50]}"
        else:
            entity_type = 'person'
            entity_id = f"person_{name.lower().replace(' ', '_')[:50]}"
        
        try:
            # Create entity
            cursor.execute("""
                INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                VALUES (?, ?, ?, ?, ?)
            """, (
                entity_id,
                name,
                entity_type,
                'phone_reverse_lookup',
                json.dumps({'phone': phone, 'source_url': source_url})
            ))
            
            # Create relationship
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                VALUES (?, ?, ?, ?)
            """, (
                entity_id,
                phone_id,
                'HAS_PHONE',
                json.dumps({'source': 'reverse_lookup', 'phone': phone})
            ))
            
            print(f"  ✅ {name} ({entity_type}) --[HAS_PHONE]--> {phone}")
        except Exception as e:
            pass
    
    def find_who_else_owns_addresses(self):
        """Find who else owns the addresses from notes."""
        print("\n=== FINDING WHO ELSE OWNS THESE ADDRESSES ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        addresses = [
            '960 S G St Arcata',
            'Todd Court Arcata',
            'Western Avenue Arcata',
            '7th + P St Eureka',
            '965 W Harris Eureka',
        ]
        
        for address in addresses:
            print(f"\nSearching ownership for: {address}")
            
            queries = [
                f"{address} property owner",
                f"{address} who owns",
                f"{address} deed owner",
                f"{address} assessor records",
            ]
            
            for query in queries:
                results = self.web_search.search(query, sources=['ddg'], max_results_per_source=5)
                
                for result in results:
                    text = result.title + ' ' + result.snippet
                    
                    # Look for owner patterns
                    owner_patterns = [
                        r'owner[:\s]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})',
                        r'owned by[:\s]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})',
                        r'property of[:\s]+([A-Z][a-z]+ (?:Properties|LLC|Inc))',
                        r'([A-Z][a-z]{2,} [A-Z][a-z]{2,})[\s,]+(?:owns|owned)[\s,]+(?:this|the|property)',
                    ]
                    
                    for pattern in owner_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            if isinstance(match, tuple):
                                match = ' '.join(match)
                            
                            parts = match.split()
                            if len(parts) >= 2 and all(part.lower() not in self.skip_words for part in parts):
                                self._create_address_ownership(address, match, cursor, result.url)
                
                time.sleep(1)
        
        conn.commit()
        conn.close()
    
    def _create_address_ownership(self, address: str, owner_name: str, cursor, source_url: str):
        """Create ownership relationship."""
        addr_id = f"address_{address.lower().replace(' ', '_').replace('+', '').replace(',', '').replace('.', '')[:50]}"
        
        # Determine entity type
        if any(word in owner_name.lower() for word in ['llc', 'inc', 'corp', 'company', 'properties']):
            entity_type = 'company'
            entity_id = f"company_{owner_name.lower().replace(' ', '_').replace('.', '')[:50]}"
        else:
            entity_type = 'person'
            entity_id = f"person_{owner_name.lower().replace(' ', '_')[:50]}"
        
        try:
            # Create entity
            cursor.execute("""
                INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                VALUES (?, ?, ?, ?, ?)
            """, (
                entity_id,
                owner_name,
                entity_type,
                'address_ownership_search',
                json.dumps({'address': address, 'source_url': source_url})
            ))
            
            # Create OWNS relationship
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                VALUES (?, ?, ?, ?)
            """, (
                entity_id,
                addr_id,
                'OWNS',
                json.dumps({'source': 'ownership_search', 'address': address})
            ))
            
            print(f"  ✅ {owner_name} ({entity_type}) --[OWNS]--> {address}")
        except Exception as e:
            pass
    
    def find_property_sales_purchases(self):
        """Find actual property sales and purchases."""
        print("\n=== FINDING PROPERTY SALES AND PURCHASES ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        # Get Strombeck addresses
        cursor.execute("""
            SELECT entity_id, name
            FROM entities
            WHERE entity_type = 'address' 
            AND (source = 'handwritten_notes' OR name LIKE '%Arcata%' OR name LIKE '%Eureka%')
        """)
        
        addresses = cursor.fetchall()
        
        for addr_id, address in addresses:
            print(f"\nFinding transactions for: {address}")
            
            queries = [
                f"{address} sold to",
                f"{address} purchased by",
                f"{address} property sale",
                f"{address} deed transfer",
                f"{address} transaction records",
            ]
            
            for query in queries:
                results = self.web_search.search(query, sources=['ddg'], max_results_per_source=5)
                
                for result in results:
                    text = result.title + ' ' + result.snippet
                    
                    # Look for transaction patterns
                    # "Sold to John Smith" or "Purchased by Jane Doe"
                    buyer_pattern = r'(?:sold to|purchased by|bought by|acquired by)[:\s,]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})'
                    seller_pattern = r'(?:sold by|purchased from|bought from)[:\s,]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})'
                    
                    buyers = re.findall(buyer_pattern, text, re.IGNORECASE)
                    sellers = re.findall(seller_pattern, text, re.IGNORECASE)
                    
                    # Also look for "X sold Y to Z" pattern
                    transaction_pattern = r'([A-Z][a-z]{2,} [A-Z][a-z]{2,})[\s,]+(?:sold|purchased)[\s,]+(?:property|this|it)[\s,]+(?:to|from)[\s,]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})'
                    tx_matches = re.findall(transaction_pattern, text, re.IGNORECASE)
                    
                    for buyer in buyers:
                        parts = buyer.split()
                        if len(parts) >= 2 and all(p.lower() not in self.skip_words for p in parts):
                            self._create_transaction(addr_id, address, buyer, 'buyer', cursor)
                    
                    for seller in sellers:
                        parts = seller.split()
                        if len(parts) >= 2 and all(p.lower() not in self.skip_words for p in parts):
                            self._create_transaction(addr_id, address, seller, 'seller', cursor)
                    
                    for tx_match in tx_matches:
                        if len(tx_match) == 2:
                            seller_name, buyer_name = tx_match
                            for name in [seller_name, buyer_name]:
                                parts = name.split()
                                if len(parts) >= 2 and all(p.lower() not in self.skip_words for p in parts):
                                    role = 'seller' if name == seller_name else 'buyer'
                                    self._create_transaction(addr_id, address, name, role, cursor)
                
                time.sleep(1)
        
        conn.commit()
        conn.close()
    
    def _create_transaction(self, addr_id: str, address: str, person_name: str, role: str, cursor):
        """Create transaction relationship."""
        person_id = f"person_{person_name.lower().replace(' ', '_')[:50]}"
        
        try:
            # Create person entity
            cursor.execute("""
                INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                VALUES (?, ?, ?, ?, ?)
            """, (
                person_id,
                person_name,
                'person',
                'property_transaction',
                json.dumps({'address': address, 'role': role})
            ))
            
            # Create transaction relationship
            if role == 'buyer':
                rel_type = 'PURCHASED_FROM'
            else:
                rel_type = 'SOLD_TO'
            
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                VALUES (?, ?, ?, ?)
            """, (
                person_id,
                addr_id,
                rel_type,
                json.dumps({'address': address, 'role': role, 'source': 'transaction_search'})
            ))
            
            print(f"  ✅ {person_name} --[{rel_type}]--> {address}")
        except Exception as e:
            pass
    
    def find_business_partners(self):
        """Find business partners and associates."""
        print("\n=== FINDING BUSINESS PARTNERS ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        # Search for Strombeck Properties partners
        queries = [
            "Strombeck Properties business partners Arcata",
            "Strombeck Properties associates Humboldt",
            "Strombeck Properties joint ventures",
            "STEATA LLC owners members Humboldt",
        ]
        
        for query in queries:
            print(f"\nSearching: {query}")
            results = self.web_search.search(query, sources=['ddg'], max_results_per_source=5)
            
            for result in results:
                text = result.title + ' ' + result.snippet
                
                # Look for partner names
                partner_patterns = [
                    r'partner[:\s]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})',
                    r'associate[:\s]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})',
                    r'([A-Z][a-z]{2,} [A-Z][a-z]{2,})[\s,]+(?:and|&)[\s,]+(?:Strombeck|Steven)',
                ]
                
                for pattern in partner_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = ' '.join(match)
                        
                        parts = match.split()
                        if (len(parts) >= 2 and 
                            all(p.lower() not in self.skip_words for p in parts) and
                            'strombeck' not in match.lower()):
                            
                            partner_id = f"person_{match.lower().replace(' ', '_')[:50]}"
                            
                            try:
                                cursor.execute("""
                                    INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    partner_id,
                                    match,
                                    'person',
                                    'business_partner_search',
                                    json.dumps({'source_url': result.url})
                                ))
                                
                                # Link to Strombeck Properties
                                strombeck_id = "company_strombeck_properties"
                                cursor.execute("""
                                    INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                                    VALUES (?, ?, ?, ?)
                                """, (
                                    partner_id,
                                    strombeck_id,
                                    'ASSOCIATED_WITH',
                                    json.dumps({'type': 'business_partner', 'source': 'web_search'})
                                ))
                                
                                print(f"  ✅ Business partner: {match} --[ASSOCIATED_WITH]--> Strombeck Properties")
                            except Exception as e:
                                pass
            
            time.sleep(1)
        
        conn.commit()
        conn.close()
    
    def run(self):
        """Run focused expansion."""
        print("=" * 80)
        print("FOCUSED NETWORK EXPANSION - FINDING REAL CONNECTIONS")
        print("=" * 80)
        
        self.find_who_else_has_phone_numbers()
        self.find_who_else_owns_addresses()
        self.find_property_sales_purchases()
        self.find_business_partners()
        
        # Report
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(DISTINCT entity_id)
            FROM entities
            WHERE source IN ('phone_reverse_lookup', 'address_ownership_search', 
                            'property_transaction', 'business_partner_search')
        """)
        new_entities = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM relationships
            WHERE relationship_type IN ('SOLD_TO', 'PURCHASED_FROM', 'OWNS', 'HAS_PHONE', 'ASSOCIATED_WITH')
            AND properties LIKE '%reverse_lookup%' OR properties LIKE '%ownership_search%' 
               OR properties LIKE '%transaction%' OR properties LIKE '%business_partner%'
        """)
        new_rels = cursor.fetchone()[0]
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("FOCUSED EXPANSION COMPLETE")
        print("=" * 80)
        print(f"New entities found: {new_entities}")
        print(f"New relationships created: {new_rels}")
        print("\n✅ Real connections mapped! View in PEGASUS.")


if __name__ == "__main__":
    expander = FocusedNetworkExpander()
    expander.run()
