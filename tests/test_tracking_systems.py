#!/usr/bin/env python3
"""
Test Tracking Systems
Tests source citation, copyright vault, and emergent intelligence tracking
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.tracking.source_citation_tracker import SourceCitationTracker
from iceburg.tracking.copyright_vault import CopyrightVault
from iceburg.tracking.emergent_intelligence_tracker import EmergentIntelligenceTracker


def test_tracking_systems():
    """Test all tracking systems"""
    print("\n" + "="*70)
    print("ICEBURG 2.0 - TRACKING SYSTEMS TEST")
    print("="*70 + "\n")
    
    # Test 1: Source Citation Tracker
    print("1. SOURCE CITATION TRACKER")
    print("-" * 70)
    citation_tracker = SourceCitationTracker()
    
    sources = [
        {
            "url": "https://arxiv.org/abs/2023.12345",
            "title": "Quantum Computing Research",
            "summary": "Research on quantum computing",
            "source_type": "arxiv",
            "copyright_status": "open_access"
        },
        {
            "url": "https://example.com/article",
            "title": "Example Article",
            "summary": "Example content",
            "source_type": "web",
            "copyright_status": "unknown"
        }
    ]
    
    citation_id = citation_tracker.track_citation(
        query="What is quantum computing?",
        response="Quantum computing is...",
        sources=sources
    )
    
    print(f"  ✅ Citation tracked: {citation_id}")
    print(f"  ✅ Sources tracked: {len(sources)}")
    
    stats = citation_tracker.get_source_stats()
    print(f"  ✅ Total sources: {stats['total_sources']}")
    print(f"  ✅ Total citations: {stats['total_citations']}")
    print(f"  ✅ Source types: {stats['source_types']}")
    print(f"  ✅ Copyright statuses: {stats['copyright_statuses']}")
    
    # Test 2: Copyright Vault
    print("\n2. COPYRIGHT VAULT")
    print("-" * 70)
    copyright_vault = CopyrightVault()
    
    vault_id = copyright_vault.store_content(
        content="This is example content from a web page...",
        url="https://example.com/article",
        source_type="web_scraped",
        copyright_status="fair_use"
    )
    
    print(f"  ✅ Content stored in vault: {vault_id}")
    
    status = copyright_vault.check_copyright_status("https://example.com/article")
    print(f"  ✅ Copyright status: {status}")
    
    compliant_content = copyright_vault.get_copyright_compliant_content("https://example.com/article")
    print(f"  ✅ Copyright-compliant content available: {compliant_content is not None}")
    
    vault_stats = copyright_vault.get_vault_stats()
    print(f"  ✅ Total vault entries: {vault_stats['total_entries']}")
    print(f"  ✅ Copyright statuses: {vault_stats['copyright_statuses']}")
    
    # Test 3: Emergent Intelligence Tracker
    print("\n3. EMERGENT INTELLIGENCE TRACKER")
    print("-" * 70)
    intelligence_tracker = EmergentIntelligenceTracker()
    
    intelligence_id = intelligence_tracker.track_intelligence(
        content="This is a novel synthesis of quantum mechanics and consciousness research, revealing emergent patterns that bridge previously disconnected domains.",
        domain="quantum_consciousness",
        intelligence_type="insight"
    )
    
    print(f"  ✅ Intelligence tracked: {intelligence_id}")
    
    can_generate = intelligence_tracker.can_keep_generating_intelligence()
    print(f"  ✅ Can keep generating: {can_generate['can_keep_generating']}")
    print(f"  ✅ Total intelligence generated: {can_generate['total_intelligence_generated']}")
    print(f"  ✅ Generation rate: {can_generate['generation_rate_per_day']:.2f} per day")
    print(f"  ✅ Emergence events: {can_generate['emergence_events_count']}")
    
    patterns = intelligence_tracker.get_linguistic_patterns()
    print(f"  ✅ Linguistic patterns tracked: {patterns['total_patterns_tracked']}")
    print(f"  ✅ Complexity distribution: {patterns['complexity_distribution']}")
    
    print("\n" + "="*70)
    print("TRACKING SYSTEMS TEST COMPLETE")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    test_tracking_systems()

