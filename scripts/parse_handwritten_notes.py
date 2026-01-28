#!/usr/bin/env python3
"""
Parse Handwritten Notes and Build Network
Uses the handwritten notes as seed data, then expands network to find connections
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.colossus.matrix_store import MatrixStore
from src.iceburg.search.web_search import WebSearchAggregator
import os
os.environ['ICEBURG_ENABLE_WEB'] = '1'

class HandwrittenNotesParser:
    """Parse handwritten notes and build network."""
    
    def __init__(self):
        self.matrix_store = MatrixStore()
        self.web_search = WebSearchAggregator()
        
        # Data from handwritten notes image
        self.notes_data = {
            'phone_numbers': [
                '707-822-4557',
                '707-527-XXXX',  # Steven Mark Strombeck
                '707-293-XXXX',
                '707-499-XXXX',
            ],
            'people': [
                {'name': 'Steven Mark Strombeck', 'phone': '707-527-XXXX'},
                {'name': 'Waltina Martha Strombeck', 'extension': '2204'},
                {'name': 'Erik Strombeck', 'extension': '4186'},
            ],
            'companies': [
                'Strombeck Properties',
                'STEATA LLC',
            ],
            'addresses': [
                {'address': '960 S G St Arcata', 'company': 'Strombeck Properties'},
                {'address': 'Todd Court Arcata'},
                {'address': 'Western Avenue Arcata'},
                {'address': '7th + P St Eureka'},
                {'address': '965 W Harris Eureka'},
                {'address': '4422 Westwood Gardens', 'type': 'apartment'},
            ],
            'business_address': '960 S G St Arcata',
        }
    
    def create_base_entities(self):
        """Create all entities from handwritten notes."""
        print("\n=== CREATING BASE ENTITIES FROM HANDWRITTEN NOTES ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        created = []
        
        # Create people
        for person in self.notes_data['people']:
            entity_id = f"person_{person['name'].lower().replace(' ', '_')}"
            try:
                props = {'source': 'handwritten_notes'}
                if 'phone' in person:
                    props['phone'] = person['phone']
                if 'extension' in person:
                    props['extension'] = person['extension']
                
                cursor.execute("""
                    INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    person['name'],
                    'person',
                    'handwritten_notes',
                    json.dumps(props)
                ))
                created.append(f"Person: {person['name']}")
                print(f"  ✅ Created: {person['name']}")
            except Exception as e:
                print(f"  ⚠️  Error creating {person['name']}: {e}")
        
        # Create companies
        for company in self.notes_data['companies']:
            entity_id = f"company_{company.lower().replace(' ', '_').replace('.', '')}"
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    company,
                    'company',
                    'handwritten_notes',
                    json.dumps({'source': 'handwritten_notes'})
                ))
                created.append(f"Company: {company}")
                print(f"  ✅ Created: {company}")
            except Exception as e:
                print(f"  ⚠️  Error creating {company}: {e}")
        
        # Create addresses
        for addr_data in self.notes_data['addresses']:
            address = addr_data['address'] if isinstance(addr_data, dict) else addr_data
            entity_id = f"address_{address.lower().replace(' ', '_').replace('+', '').replace(',', '').replace('.', '')[:50]}"
            try:
                props = {'source': 'handwritten_notes'}
                if isinstance(addr_data, dict):
                    if 'type' in addr_data:
                        props['type'] = addr_data['type']
                    if 'company' in addr_data:
                        props['company'] = addr_data['company']
                
                cursor.execute("""
                    INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    address,
                    'address',
                    'handwritten_notes',
                    json.dumps(props)
                ))
                created.append(f"Address: {address}")
                print(f"  ✅ Created: {address}")
            except Exception as e:
                print(f"  ⚠️  Error creating {address}: {e}")
        
        # Create phone numbers as entities
        for phone in self.notes_data['phone_numbers']:
            if phone != '707-527-XXXX':  # Already linked to Steven
                entity_id = f"phone_{phone.replace('-', '').replace('(', '').replace(')', '').replace(' ', '').replace('X', '0')}"
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO entities (entity_id, name, entity_type, source, properties)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        entity_id,
                        phone,
                        'phone',
                        'handwritten_notes',
                        json.dumps({'source': 'handwritten_notes', 'area_code': '707'})
                    ))
                    created.append(f"Phone: {phone}")
                    print(f"  ✅ Created: {phone}")
                except Exception as e:
                    print(f"  ⚠️  Error creating {phone}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Created {len(created)} base entities")
        return created
    
    def create_base_relationships(self):
        """Create relationships from handwritten notes."""
        print("\n=== CREATING BASE RELATIONSHIPS ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        relationships = []
        
        # Steven Mark Strombeck relationships
        steven_id = "person_steven_mark_strombeck"
        strombeck_props_id = "company_strombeck_properties"
        
        # Steven OWNS Strombeck Properties
        relationships.append((steven_id, strombeck_props_id, 'OWNS'))
        
        # Strombeck Properties OWNS addresses
        for addr_data in self.notes_data['addresses']:
            address = addr_data['address'] if isinstance(addr_data, dict) else addr_data
            addr_id = f"address_{address.lower().replace(' ', '_').replace('+', '').replace(',', '').replace('.', '')[:50]}"
            
            # If address is linked to Strombeck Properties
            if isinstance(addr_data, dict) and addr_data.get('company') == 'Strombeck Properties':
                relationships.append((strombeck_props_id, addr_id, 'OWNS'))
            elif '960 S G St' in address:  # Known Strombeck Properties address
                relationships.append((strombeck_props_id, addr_id, 'OWNS'))
        
        # Family relationships
        waltina_id = "person_waltina_martha_strombeck"
        erik_id = "person_erik_strombeck"
        
        relationships.append((steven_id, waltina_id, 'FAMILY_OF'))
        relationships.append((steven_id, erik_id, 'FAMILY_OF'))
        relationships.append((waltina_id, erik_id, 'FAMILY_OF'))
        
        # Phone number relationships
        main_phone_id = "phone_7078224557"
        relationships.append((strombeck_props_id, main_phone_id, 'HAS_PHONE'))
        relationships.append((steven_id, main_phone_id, 'HAS_PHONE'))
        
        # STEATA LLC relationships (need to find connections)
        steata_id = "company_steata_llc"
        
        # Insert relationships
        for source_id, target_id, rel_type in relationships:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                    VALUES (?, ?, ?, ?)
                """, (
                    source_id,
                    target_id,
                    rel_type,
                    json.dumps({'source': 'handwritten_notes'})
                ))
                print(f"  ✅ {source_id} --[{rel_type}]--> {target_id}")
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ Created {len(relationships)} base relationships")
    
    def expand_network_from_phone_numbers(self):
        """Use phone numbers to find next-level connections."""
        print("\n=== EXPANDING NETWORK FROM PHONE NUMBERS ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        # Get all phone numbers from notes
        phones = [p for p in self.notes_data['phone_numbers'] if 'XXXX' not in p]
        
        for phone in phones:
            print(f"\nSearching connections for: {phone}")
            
            # Search web for phone number
            query = f"{phone} Arcata Eureka Humboldt"
            results = self.web_search.search(query, sources=['ddg'], max_results_per_source=5)
            
            found_entities = set()
            
            for result in results:
                # Extract names/entities from search results
                full_text = f"{result.title} {result.snippet}"
                text = full_text.lower()
                
                # Look for actual person names (more specific patterns)
                # Skip common false positives
                skip_words = {'property', 'management', 'housing', 'apartments', 'affordable', 
                             'list', 'lost', 'coast', 'union', 'st', 'street', 'avenue',
                             'county', 'city', 'state', 'california', 'arcata', 'eureka',
                             'official', 'website', 'contact', 'decorating', 'ownership',
                             'general', 'building', 'parkway', 'arrowhead', 'vista', 'realty',
                             'action', 'agency', 'authority', 'license', 'corporation',
                             'rental', 'managers', 'poly', 'cal'}
                
                # Better name patterns - must be actual names
                name_patterns = [
                    r'\b([A-Z][a-z]{2,} [A-Z][a-z]{2,})\b',  # First Last (min 3 chars each)
                    r'\b([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]{2,})\b',  # First M. Last
                ]
                
                for pattern in name_patterns:
                    matches = re.findall(pattern, full_text)
                    for match in matches:
                        parts = match.split()
                        # Filter out false positives
                        if (len(parts) == 2 and 
                            parts[0].lower() not in skip_words and 
                            parts[1].lower() not in skip_words and
                            'strombeck' not in match.lower() and
                            len(parts[0]) >= 3 and len(parts[1]) >= 3):
                            found_entities.add(match)
                
                # Look for company names
                company_patterns = [
                    r'\b([A-Z][a-z]+ (?:Properties|LLC|Inc|Corp|Company|Realty|Housing|Agency))\b',
                ]
                
                for pattern in company_patterns:
                    matches = re.findall(pattern, full_text)
                    for match in matches:
                        if 'strombeck' not in match.lower():
                            found_entities.add(match)
                
                # Look for addresses
                addr_pattern = r'\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Court|Rd|Road|Blvd)[\s,]*Arcata|Eureka'
                addr_matches = re.findall(addr_pattern, text, re.IGNORECASE)
                for addr in addr_matches:
                    found_entities.add(addr)
            
            # Create entities for found connections
            phone_id = f"phone_{phone.replace('-', '').replace('(', '').replace(')', '').replace(' ', '')}"
            
            for entity_name in found_entities:
                if len(entity_name) < 5:  # Skip short matches
                    continue
                
                # Determine entity type more accurately
                entity_name_lower = entity_name.lower()
                
                # Check if it's a company/organization
                if any(word in entity_name_lower for word in ['llc', 'inc', 'corp', 'company', 'properties', 
                                                              'realty', 'housing', 'agency', 'authority']):
                    entity_type = 'company'
                    entity_id = f"company_{entity_name_lower.replace(' ', '_').replace('.', '').replace(',', '')[:50]}"
                # Check if it's an address
                elif any(word in entity_name_lower for word in ['st', 'street', 'ave', 'avenue', 'court', 'road', 'blvd', 'boulevard']):
                    entity_type = 'address'
                    entity_id = f"address_{entity_name_lower.replace(' ', '_').replace('+', '').replace(',', '').replace('.', '')[:50]}"
                # Otherwise assume person
                else:
                    entity_type = 'person'
                    entity_id = f"person_{entity_name_lower.replace(' ', '_')[:50]}"
                
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        entity_id,
                        entity_name,
                        entity_type,
                        'phone_reverse_lookup',
                        json.dumps({'phone': phone, 'source_url': 'web_search'})
                    ))
                    
                    # Create relationship: phone -> entity
                    cursor.execute("""
                        INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                        VALUES (?, ?, ?, ?)
                    """, (
                        phone_id,
                        entity_id,
                        'ASSOCIATED_WITH',
                        json.dumps({'via': 'phone_number', 'phone': phone})
                    ))
                    
                    print(f"  ✅ Found connection: {entity_name} (via {phone})")
                except Exception as e:
                    pass  # Skip duplicates
        
        conn.commit()
        conn.close()
    
    def expand_network_from_addresses(self):
        """Use addresses to find property ownership and connections."""
        print("\n=== EXPANDING NETWORK FROM ADDRESSES ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        for addr_data in self.notes_data['addresses']:
            address = addr_data['address'] if isinstance(addr_data, dict) else addr_data
            print(f"\nSearching connections for: {address}")
            
            # Search for property ownership
            query = f"{address} property owner Arcata Eureka"
            results = self.web_search.search(query, sources=['ddg'], max_results_per_source=5)
            
            addr_id = f"address_{address.lower().replace(' ', '_').replace('+', '').replace(',', '').replace('.', '')[:50]}"
            
            for result in results:
                text = f"{result.title} {result.snippet}".lower()
                
                # Look for owner names
                owner_patterns = [
                    r'owner[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',
                    r'owned by[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',
                    r'property of[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',
                ]
                
                for pattern in owner_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for owner_name in matches:
                        if 'strombeck' not in owner_name.lower():
                            owner_id = f"person_{owner_name.lower().replace(' ', '_')[:50]}"
                            
                            try:
                                # Create owner entity
                                cursor.execute("""
                                    INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    owner_id,
                                    owner_name,
                                    'person',
                                    'address_property_search',
                                    json.dumps({'address': address})
                                ))
                                
                                # Create OWNS relationship
                                cursor.execute("""
                                    INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                                    VALUES (?, ?, ?, ?)
                                """, (
                                    owner_id,
                                    addr_id,
                                    'OWNS',
                                    json.dumps({'source': 'web_search', 'address': address})
                                ))
                                
                                print(f"  ✅ Found owner: {owner_name} owns {address}")
                            except Exception as e:
                                pass
        
        conn.commit()
        conn.close()
    
    def find_company_connections(self):
        """Find connections through companies."""
        print("\n=== FINDING COMPANY CONNECTIONS ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        # Search for STEATA LLC connections
        steata_id = "company_steata_llc"
        
        query = "STEATA LLC Arcata Eureka Humboldt California"
        results = self.web_search.search(query, sources=['ddg'], max_results_per_source=10)
        
        for result in results:
            text = f"{result.title} {result.snippet}".lower()
            
            # Look for associated names
            if 'steata' in text or 'steata llc' in text:
                # Extract names from context
                name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
                matches = re.findall(name_pattern, result.title + ' ' + result.snippet)
                
                for name in matches:
                    if 'steata' not in name.lower():
                        person_id = f"person_{name.lower().replace(' ', '_')[:50]}"
                        
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                person_id,
                                name,
                                'person',
                                'company_search',
                                json.dumps({'company': 'STEATA LLC'})
                            ))
                            
                            # Create relationship
                            cursor.execute("""
                                INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                                VALUES (?, ?, ?, ?)
                            """, (
                                person_id,
                                steata_id,
                                'ASSOCIATED_WITH',
                                json.dumps({'source': 'web_search'})
                            ))
                            
                            print(f"  ✅ Found connection: {name} --[ASSOCIATED_WITH]--> STEATA LLC")
                        except Exception as e:
                            pass
        
        conn.commit()
        conn.close()
    
    def find_property_transaction_chains(self):
        """Find property transaction chains by searching for sales/purchases."""
        print("\n=== FINDING PROPERTY TRANSACTION CHAINS ===\n")
        
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        # Search for property transactions involving Strombeck addresses
        for addr_data in self.notes_data['addresses']:
            address = addr_data['address'] if isinstance(addr_data, dict) else addr_data
            
            queries = [
                f"{address} sold purchased transaction",
                f"{address} property sale Arcata Eureka",
                f"{address} deed transfer",
            ]
            
            for query in queries:
                results = self.web_search.search(query, sources=['ddg'], max_results_per_source=3)
                
                for result in results:
                    text = f"{result.title} {result.snippet}".lower()
                    
                    # Look for transaction keywords
                    if any(word in text for word in ['sold', 'purchased', 'bought', 'deed', 'transfer']):
                        # Extract buyer/seller names
                        buyer_pattern = r'(?:sold to|purchased by|bought by)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)'
                        seller_pattern = r'(?:sold by|purchased from)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)'
                        
                        buyers = re.findall(buyer_pattern, text, re.IGNORECASE)
                        sellers = re.findall(seller_pattern, text, re.IGNORECASE)
                        
                        addr_id = f"address_{address.lower().replace(' ', '_').replace('+', '').replace(',', '').replace('.', '')[:50]}"
                        
                        for buyer in buyers:
                            buyer_id = f"person_{buyer.lower().replace(' ', '_')[:50]}"
                            try:
                                cursor.execute("""
                                    INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    buyer_id,
                                    buyer,
                                    'person',
                                    'transaction_search',
                                    json.dumps({'address': address, 'role': 'buyer'})
                                ))
                                
                                cursor.execute("""
                                    INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                                    VALUES (?, ?, ?, ?)
                                """, (
                                    buyer_id,
                                    addr_id,
                                    'PURCHASED_FROM',
                                    json.dumps({'source': 'web_search', 'address': address})
                                ))
                                
                                print(f"  ✅ Transaction: {buyer} --[PURCHASED_FROM]--> {address}")
                            except Exception as e:
                                pass
                        
                        for seller in sellers:
                            seller_id = f"person_{seller.lower().replace(' ', '_')[:50]}"
                            try:
                                cursor.execute("""
                                    INSERT OR IGNORE INTO entities (entity_id, name, entity_type, source, properties)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    seller_id,
                                    seller,
                                    'person',
                                    'transaction_search',
                                    json.dumps({'address': address, 'role': 'seller'})
                                ))
                                
                                cursor.execute("""
                                    INSERT OR IGNORE INTO relationships (source_id, target_id, relationship_type, properties)
                                    VALUES (?, ?, ?, ?)
                                """, (
                                    seller_id,
                                    addr_id,
                                    'SOLD_TO',
                                    json.dumps({'source': 'web_search', 'address': address})
                                ))
                                
                                print(f"  ✅ Transaction: {seller} --[SOLD_TO]--> {address}")
                            except Exception as e:
                                pass
        
        conn.commit()
        conn.close()
    
    def run_full_expansion(self):
        """Run complete network expansion."""
        print("=" * 80)
        print("HANDWRITTEN NOTES NETWORK EXPANSION")
        print("=" * 80)
        
        # Step 1: Create base entities from notes
        self.create_base_entities()
        
        # Step 2: Create base relationships
        self.create_base_relationships()
        
        # Step 3: Expand from phone numbers (reverse lookup)
        self.expand_network_from_phone_numbers()
        
        # Step 4: Expand from addresses (property ownership)
        self.expand_network_from_addresses()
        
        # Step 5: Find company connections
        self.find_company_connections()
        
        # Step 6: Find property transaction chains
        self.find_property_transaction_chains()
        
        # Report
        conn = self.matrix_store.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM entities WHERE source IN ('handwritten_notes', 'phone_reverse_lookup', 'address_property_search', 'company_search', 'transaction_search')")
        total_entities = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM relationships WHERE properties LIKE '%handwritten_notes%' OR properties LIKE '%web_search%'")
        total_rels = cursor.fetchone()[0]
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("NETWORK EXPANSION COMPLETE")
        print("=" * 80)
        print(f"Total entities created: {total_entities}")
        print(f"Total relationships created: {total_rels}")
        print("\n✅ Network ready for visualization in PEGASUS!")


if __name__ == "__main__":
    parser = HandwrittenNotesParser()
    parser.run_full_expansion()
