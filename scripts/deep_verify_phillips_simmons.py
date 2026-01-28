#!/usr/bin/env python3
"""
Deep Verification Investigation - Phillips & Simmons
Comprehensive investigation across all verification areas
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

class DeepVerificationInvestigator:
    """Deep verification investigation for Phillips & Simmons."""
    
    def __init__(self):
        self.search = WebSearchAggregator()
        self.ms = MatrixStore()
        self.findings = []
        self.verification_results = {
            'joseph_strombeck_verification': {},
            'simmons_phillips_relationship': {},
            'property_records': {},
            'social_media_connections': {},
            'phone_verification': {}
        }
        
    def verify_joseph_strombeck_identity(self):
        """Verify if Phillips Joseph L is actually Joseph Strombeck."""
        print('=' * 80)
        print('VERIFICATION 1: JOSEPH STROMBECK IDENTITY')
        print('=' * 80)
        
        queries = [
            'Joseph Strombeck Arcata Humboldt',
            'Joseph Strombeck Erik Strombeck family',
            'Joseph Strombeck Phillips',
            'Joseph Strombeck property owner Arcata',
            'Joseph Strombeck phone number',
            'Joseph Strombeck address Arcata',
        ]
        
        results = {
            'name_variations': [],
            'addresses': [],
            'phone_numbers': [],
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
                    
                    # Extract name variations
                    if 'joseph strombeck' in text:
                        results['name_variations'].append({
                            'variation': 'Joseph Strombeck',
                            'source': result.url,
                            'context': result.snippet[:200]
                        })
                    
                    if 'joseph phillips' in text and 'strombeck' in text:
                        results['connections_to_phillips'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
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
                    if 'erik' in text and 'strombeck' in text:
                        results['connections_to_erik'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
                    results['findings'].append({
                        'url': result.url,
                        'title': result.title,
                        'snippet': result.snippet
                    })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        # Assessment
        assessment = {
            'name_variations_found': len(results['name_variations']),
            'connections_to_erik': len(results['connections_to_erik']),
            'connections_to_phillips': len(results['connections_to_phillips']),
            'verification_status': 'inconclusive'
        }
        
        if results['connections_to_phillips']:
            assessment['verification_status'] = 'possible_match'
            assessment['evidence'] = 'Found references linking Joseph Strombeck and Phillips'
        elif results['connections_to_erik']:
            assessment['verification_status'] = 'strombeck_family_member'
            assessment['evidence'] = 'Joseph Strombeck appears to be family member'
        else:
            assessment['verification_status'] = 'no_match_found'
        
        self.verification_results['joseph_strombeck_verification'] = {
            'results': results,
            'assessment': assessment
        }
        
        print(f'\n📊 ASSESSMENT:')
        print(f'  Name Variations Found: {assessment["name_variations_found"]}')
        print(f'  Connections to Erik: {assessment["connections_to_erik"]}')
        print(f'  Connections to Phillips: {assessment["connections_to_phillips"]}')
        print(f'  Verification Status: {assessment["verification_status"]}')
        
        return results, assessment
    
    def investigate_simmons_phillips_relationship(self):
        """Investigate if Simmons and Phillips are married/related."""
        print('\n\n' + '=' * 80)
        print('VERIFICATION 2: SIMMONS-PHILLIPS RELATIONSHIP')
        print('=' * 80)
        
        queries = [
            'Lorenza Simmons Phillips married',
            'Lorenza Simmons-Phillips',
            'Simmons Phillips couple Arcata',
            'Lorenza Phillips Simmons relationship',
            'Joseph Phillips Lorenza Simmons',
        ]
        
        results = {
            'combined_names_found': [],
            'marriage_records': [],
            'shared_addresses': [],
            'relationship_indicators': [],
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
                    
                    # Look for combined names
                    if 'simmons-phillips' in text or 'simmons phillips' in text:
                        results['combined_names_found'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
                    # Look for marriage indicators
                    if any(word in text for word in ['married', 'wife', 'husband', 'spouse', 'wedding']):
                        results['marriage_records'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
                    # Extract addresses
                    address_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Ct|Court|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place)'
                    addresses = re.findall(address_pattern, full_text, re.IGNORECASE)
                    for addr in addresses:
                        if addr not in results['shared_addresses']:
                            results['shared_addresses'].append(addr)
                    
                    results['findings'].append({
                        'url': result.url,
                        'title': result.title,
                        'snippet': result.snippet
                    })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        # Assessment
        assessment = {
            'combined_names_found': len(results['combined_names_found']),
            'marriage_indicators': len(results['marriage_records']),
            'relationship_status': 'unknown'
        }
        
        if results['combined_names_found']:
            assessment['relationship_status'] = 'likely_married_or_related'
            assessment['evidence'] = 'Found combined name "Simmons-Phillips"'
        elif results['marriage_records']:
            assessment['relationship_status'] = 'marriage_indicated'
            assessment['evidence'] = 'Found marriage indicators'
        else:
            assessment['relationship_status'] = 'no_relationship_found'
        
        self.verification_results['simmons_phillips_relationship'] = {
            'results': results,
            'assessment': assessment
        }
        
        print(f'\n📊 ASSESSMENT:')
        print(f'  Combined Names Found: {assessment["combined_names_found"]}')
        print(f'  Marriage Indicators: {assessment["marriage_indicators"]}')
        print(f'  Relationship Status: {assessment["relationship_status"]}')
        
        return results, assessment
    
    def cross_reference_property_records(self):
        """Cross-reference property records for other purchases."""
        print('\n\n' + '=' * 80)
        print('VERIFICATION 3: PROPERTY RECORDS CROSS-REFERENCE')
        print('=' * 80)
        
        queries = [
            'Phillips Joseph L property owner Arcata',
            'Simmons Lorenza property owner Arcata',
            'Joseph Phillips property purchases Arcata',
            'Lorenza Simmons property purchases Arcata',
            'Phillips Simmons property transactions Arcata',
        ]
        
        results = {
            'phillips_properties': [],
            'simmons_properties': [],
            'shared_properties': [],
            'other_purchases': [],
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
                    
                    if 'phillips' in text and 'joseph' in text:
                        for addr in addresses:
                            if addr not in results['phillips_properties']:
                                results['phillips_properties'].append(addr)
                    
                    if 'simmons' in text and 'lorenza' in text:
                        for addr in addresses:
                            if addr not in results['simmons_properties']:
                                results['simmons_properties'].append(addr)
                    
                    # Look for other purchases
                    if any(word in text for word in ['purchased', 'bought', 'sale', 'property', 'real estate']):
                        results['other_purchases'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
                    results['findings'].append({
                        'url': result.url,
                        'title': result.title,
                        'snippet': result.snippet
                    })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        # Check database for properties
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
        # Get properties purchased by Phillips/Simmons
        cursor.execute("""
            SELECT e1.name, e2.name, r.properties
            FROM relationships r
            INNER JOIN entities e1 ON e1.entity_id = r.source_id
            INNER JOIN entities e2 ON e2.entity_id = r.target_id
            WHERE e1.entity_id IN ('person_phillips_joseph_l', 'person_simmons_lorenza')
            AND e2.entity_type = 'address'
            AND r.relationship_type = 'PURCHASED_FROM'
        """)
        
        db_properties = cursor.fetchall()
        for buyer, property_name, props_json in db_properties:
            if buyer == 'Phillips Joseph L':
                results['phillips_properties'].append(property_name)
            elif buyer == 'Simmons Lorenza':
                results['simmons_properties'].append(property_name)
        
        conn.close()
        
        # Assessment
        assessment = {
            'phillips_properties_count': len(results['phillips_properties']),
            'simmons_properties_count': len(results['simmons_properties']),
            'other_purchases_found': len(results['other_purchases']),
            'pattern': 'unknown'
        }
        
        if len(results['phillips_properties']) > 1 or len(results['simmons_properties']) > 1:
            assessment['pattern'] = 'multiple_properties'
            assessment['red_flag'] = 'Multiple property purchases suggest possible investor or connected buyer'
        elif len(results['phillips_properties']) == 1 and len(results['simmons_properties']) == 1:
            assessment['pattern'] = 'single_purchase_together'
            assessment['note'] = 'Only one property purchase found - the 2149 Western Ave purchase'
        
        self.verification_results['property_records'] = {
            'results': results,
            'assessment': assessment
        }
        
        print(f'\n📊 ASSESSMENT:')
        print(f'  Phillips Properties: {assessment["phillips_properties_count"]}')
        print(f'  Simmons Properties: {assessment["simmons_properties_count"]}')
        print(f'  Other Purchases Found: {assessment["other_purchases_found"]}')
        print(f'  Pattern: {assessment["pattern"]}')
        
        return results, assessment
    
    def investigate_social_media_connections(self):
        """Deep dive into social media for mutual connections."""
        print('\n\n' + '=' * 80)
        print('VERIFICATION 4: SOCIAL MEDIA CONNECTIONS')
        print('=' * 80)
        
        queries = [
            'Phillips Joseph L Facebook friends Erik Strombeck',
            'Simmons Lorenza Facebook Erik Strombeck',
            'Joseph Phillips social media Arcata',
            'Lorenza Simmons social media Arcata',
            'Phillips Simmons mutual friends Erik Strombeck',
        ]
        
        results = {
            'facebook_profiles': [],
            'mutual_connections': [],
            'social_media_links': [],
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
                    
                    # Look for Facebook profiles
                    if 'facebook' in text:
                        results['facebook_profiles'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
                    # Look for mutual connections
                    if any(word in text for word in ['mutual', 'friend', 'connection', 'knows', 'linked']):
                        results['mutual_connections'].append({
                            'url': result.url,
                            'title': result.title,
                            'snippet': result.snippet
                        })
                    
                    # Look for social media links
                    if any(domain in result.url for domain in ['facebook', 'linkedin', 'twitter', 'instagram']):
                        results['social_media_links'].append({
                            'platform': 'facebook' if 'facebook' in result.url else 'other',
                            'url': result.url,
                            'title': result.title
                        })
                    
                    results['findings'].append({
                        'url': result.url,
                        'title': result.title,
                        'snippet': result.snippet
                    })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        # Assessment
        assessment = {
            'facebook_profiles_found': len(results['facebook_profiles']),
            'mutual_connections_found': len(results['mutual_connections']),
            'social_media_links': len(results['social_media_links']),
            'connection_level': 'unknown'
        }
        
        if results['mutual_connections']:
            assessment['connection_level'] = 'mutual_connections_found'
            assessment['evidence'] = 'Found indicators of mutual connections'
        elif results['facebook_profiles']:
            assessment['connection_level'] = 'profiles_found'
            assessment['evidence'] = 'Found social media profiles'
        else:
            assessment['connection_level'] = 'no_connections_found'
        
        self.verification_results['social_media_connections'] = {
            'results': results,
            'assessment': assessment
        }
        
        print(f'\n📊 ASSESSMENT:')
        print(f'  Facebook Profiles: {assessment["facebook_profiles_found"]}')
        print(f'  Mutual Connections: {assessment["mutual_connections_found"]}')
        print(f'  Social Media Links: {assessment["social_media_links"]}')
        print(f'  Connection Level: {assessment["connection_level"]}')
        
        return results, assessment
    
    def verify_phone_numbers(self):
        """Verify phone numbers against Erik's known contacts."""
        print('\n\n' + '=' * 80)
        print('VERIFICATION 5: PHONE NUMBER VERIFICATION')
        print('=' * 80)
        
        # Get Erik's phone numbers from database
        conn = self.ms.get_connection()
        cursor = conn.cursor()
        
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
            phone_clean = re.sub(r'[^0-9]', '', name)
            erik_phones.append(phone_clean)
        
        # Known Phillips/Simmons phones
        phillips_phones = ['7078224722', '8607399701']
        simmons_phones = []
        
        # Search for phone number connections
        queries = [
            '707-822-4722 Erik Strombeck',
            '860-739-9701 Erik Strombeck',
            '7078224722 property owner',
            '8607399701 property owner',
        ]
        
        results = {
            'phillips_phones': phillips_phones,
            'simmons_phones': simmons_phones,
            'erik_phones': erik_phones,
            'matches_found': [],
            'phone_connections': [],
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
                    
                    # Check for matches
                    for ph_phone in phillips_phones:
                        if ph_phone in text.replace('-', '').replace(' ', '').replace('(', '').replace(')', ''):
                            results['phone_connections'].append({
                                'phone': ph_phone,
                                'url': result.url,
                                'title': result.title,
                                'snippet': result.snippet
                            })
                    
                    results['findings'].append({
                        'url': result.url,
                        'title': result.title,
                        'snippet': result.snippet
                    })
                
                print(f'  ✅ Found {len(search_results)} results')
                
            except Exception as e:
                print(f'  ⚠️  Error: {e}')
        
        # Check for matches
        for ph_phone in phillips_phones:
            for erik_phone in erik_phones:
                if ph_phone == erik_phone:
                    results['matches_found'].append({
                        'phone': ph_phone,
                        'match_type': 'exact_match'
                    })
        
        conn.close()
        
        # Assessment
        assessment = {
            'phillips_phones_checked': len(phillips_phones),
            'simmons_phones_checked': len(simmons_phones),
            'erik_phones_checked': len(erik_phones),
            'matches_found': len(results['matches_found']),
            'phone_connections_found': len(results['phone_connections']),
            'verification_status': 'no_match'
        }
        
        if results['matches_found']:
            assessment['verification_status'] = 'match_found'
            assessment['red_flag'] = 'Phone number matches Erik\'s contacts'
        elif results['phone_connections']:
            assessment['verification_status'] = 'connections_found'
            assessment['evidence'] = 'Found phone number connections'
        else:
            assessment['verification_status'] = 'no_match'
        
        self.verification_results['phone_verification'] = {
            'results': results,
            'assessment': assessment
        }
        
        print(f'\n📊 ASSESSMENT:')
        print(f'  Phillips Phones Checked: {assessment["phillips_phones_checked"]}')
        print(f'  Erik Phones Checked: {assessment["erik_phones_checked"]}')
        print(f'  Matches Found: {assessment["matches_found"]}')
        print(f'  Phone Connections: {assessment["phone_connections_found"]}')
        print(f'  Verification Status: {assessment["verification_status"]}')
        
        return results, assessment
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        report = []
        report.append("# Deep Verification Investigation - Final Report")
        report.append(f"\n**Investigation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n## EXECUTIVE SUMMARY")
        report.append(f"\nComprehensive investigation across 5 verification areas to determine if Phillips Joseph L and Simmons Lorenza are connected to Erik Strombeck.")
        
        # Joseph Strombeck Verification
        joseph = self.verification_results['joseph_strombeck_verification']
        report.append(f"\n## 1. JOSEPH STROMBECK IDENTITY VERIFICATION")
        report.append(f"\n### Status: **{joseph['assessment']['verification_status'].upper()}**")
        report.append(f"- Name Variations Found: {joseph['assessment']['name_variations_found']}")
        report.append(f"- Connections to Erik: {joseph['assessment']['connections_to_erik']}")
        report.append(f"- Connections to Phillips: {joseph['assessment']['connections_to_phillips']}")
        if 'evidence' in joseph['assessment']:
            report.append(f"- Evidence: {joseph['assessment']['evidence']}")
        
        # Simmons-Phillips Relationship
        relationship = self.verification_results['simmons_phillips_relationship']
        report.append(f"\n## 2. SIMMONS-PHILLIPS RELATIONSHIP INVESTIGATION")
        report.append(f"\n### Status: **{relationship['assessment']['relationship_status'].upper()}**")
        report.append(f"- Combined Names Found: {relationship['assessment']['combined_names_found']}")
        report.append(f"- Marriage Indicators: {relationship['assessment']['marriage_indicators']}")
        if 'evidence' in relationship['assessment']:
            report.append(f"- Evidence: {relationship['assessment']['evidence']}")
        
        # Property Records
        properties = self.verification_results['property_records']
        report.append(f"\n## 3. PROPERTY RECORDS CROSS-REFERENCE")
        report.append(f"\n### Status: **{properties['assessment']['pattern'].upper()}**")
        report.append(f"- Phillips Properties: {properties['assessment']['phillips_properties_count']}")
        report.append(f"- Simmons Properties: {properties['assessment']['simmons_properties_count']}")
        report.append(f"- Other Purchases Found: {properties['assessment']['other_purchases_found']}")
        if 'red_flag' in properties['assessment']:
            report.append(f"- ⚠️  Red Flag: {properties['assessment']['red_flag']}")
        
        # Social Media
        social = self.verification_results['social_media_connections']
        report.append(f"\n## 4. SOCIAL MEDIA CONNECTIONS")
        report.append(f"\n### Status: **{social['assessment']['connection_level'].upper()}**")
        report.append(f"- Facebook Profiles: {social['assessment']['facebook_profiles_found']}")
        report.append(f"- Mutual Connections: {social['assessment']['mutual_connections_found']}")
        report.append(f"- Social Media Links: {social['assessment']['social_media_links']}")
        if 'evidence' in social['assessment']:
            report.append(f"- Evidence: {social['assessment']['evidence']}")
        
        # Phone Verification
        phones = self.verification_results['phone_verification']
        report.append(f"\n## 5. PHONE NUMBER VERIFICATION")
        report.append(f"\n### Status: **{phones['assessment']['verification_status'].upper()}**")
        report.append(f"- Phillips Phones Checked: {phones['assessment']['phillips_phones_checked']}")
        report.append(f"- Erik Phones Checked: {phones['assessment']['erik_phones_checked']}")
        report.append(f"- Matches Found: {phones['assessment']['matches_found']}")
        report.append(f"- Phone Connections: {phones['assessment']['phone_connections_found']}")
        if 'red_flag' in phones['assessment']:
            report.append(f"- ⚠️  Red Flag: {phones['assessment']['red_flag']}")
        
        # Overall Assessment
        report.append(f"\n## OVERALL ASSESSMENT")
        
        total_indicators = (
            (1 if joseph['assessment']['verification_status'] != 'no_match_found' else 0) +
            (1 if relationship['assessment']['relationship_status'] != 'no_relationship_found' else 0) +
            (1 if properties['assessment']['pattern'] == 'multiple_properties' else 0) +
            (1 if social['assessment']['connection_level'] != 'no_connections_found' else 0) +
            (1 if phones['assessment']['verification_status'] != 'no_match' else 0)
        )
        
        if total_indicators >= 3:
            overall_status = 'STRONG_CONNECTION_INDICATORS'
            conclusion = 'Multiple indicators suggest Phillips/Simmons are connected to Erik Strombeck'
        elif total_indicators >= 2:
            overall_status = 'MODERATE_CONNECTION_INDICATORS'
            conclusion = 'Some indicators found - requires further investigation'
        else:
            overall_status = 'WEAK_CONNECTION_INDICATORS'
            conclusion = 'Limited indicators - may be legitimate buyers'
        
        report.append(f"\n### Overall Status: **{overall_status}**")
        report.append(f"\n### Conclusion: {conclusion}")
        report.append(f"\n### Total Indicators: {total_indicators}/5")
        
        report.append(f"\n## INVESTIGATION RECOMMENDATIONS")
        if overall_status == 'STRONG_CONNECTION_INDICATORS':
            report.append(f"\n1. **VERIFY CONNECTIONS**: Strong indicators suggest relationship - verify through additional sources")
            report.append(f"2. **NOMINEE BUYER INVESTIGATION**: Investigate if they are nominee buyers")
            report.append(f"3. **ASSET HIDING PATTERN**: Connected buyers would indicate asset hiding through nominee purchases")
        elif overall_status == 'MODERATE_CONNECTION_INDICATORS':
            report.append(f"\n1. **FURTHER INVESTIGATION**: Continue investigation with additional sources")
            report.append(f"2. **VERIFY INDEPENDENTLY**: Cross-reference findings with property records")
        else:
            report.append(f"\n1. **MONITOR**: Continue monitoring for any future connections")
            report.append(f"2. **VERIFY LEGITIMACY**: Confirm they are legitimate third-party buyers")
        
        return '\n'.join(report)
    
    def run_investigation(self):
        """Run complete deep verification investigation."""
        print('=' * 80)
        print('DEEP VERIFICATION INVESTIGATION')
        print('=' * 80)
        
        # Run all 5 verifications
        self.verify_joseph_strombeck_identity()
        self.investigate_simmons_phillips_relationship()
        self.cross_reference_property_records()
        self.investigate_social_media_connections()
        self.verify_phone_numbers()
        
        # Generate final report
        report = self.generate_final_report()
        
        # Save report
        report_file = f'DEEP_VERIFICATION_REPORT_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save data
        data_file = f'DEEP_VERIFICATION_DATA_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(data_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'verification_results': self.verification_results,
                'findings': self.findings
            }, f, indent=2)
        
        print(f'\n\n✅ Deep Verification Investigation Complete!')
        print(f'📄 Report saved to: {report_file}')
        print(f'📊 Data saved to: {data_file}')
        
        return self.verification_results

if __name__ == "__main__":
    investigator = DeepVerificationInvestigator()
    investigator.run_investigation()
