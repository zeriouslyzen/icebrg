"""
Test Deep Knowledge Decoding System
Tests etymology tracing, occult connections, secret societies, and total knowledge accumulation
"""

import asyncio
from src.iceburg.config import load_config
from src.iceburg.knowledge.deep_etymology_tracing import DeepEtymologyTracing
from src.iceburg.knowledge.occult_knowledge_database import OccultKnowledgeDatabase
from src.iceburg.knowledge.predictive_history import PredictiveHistorySystem
from src.iceburg.knowledge.total_knowledge_accumulator import TotalKnowledgeAccumulator
from src.iceburg.gnosis.unified_gnosis_interface import UnifiedGnosisInterface
from src.iceburg.unified_interface import UnifiedICEBURG


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_deep_etymology():
    """Test deep etymology tracing."""
    print_section("TEST 1: Deep Etymology Tracing")
    
    try:
        etymology_tracing = DeepEtymologyTracing()
        
        test_terms = ["astrology", "star", "constellation"]
        
        for term in test_terms:
            print(f"\nTerm: '{term}'")
            print("-" * 60)
            
            trace = etymology_tracing.trace_deep_etymology(term)
            
            print(f"Complete Chain: {trace.complete_chain}")
            print(f"Layers: {len(trace.layers)}")
            for i, layer in enumerate(trace.layers, 1):
                print(f"  Layer {i}: {layer.term} ({layer.language}: {layer.meaning})")
            
            print(f"\nOccult Path: {trace.occult_path}")
            print(f"Secret Society Path: {trace.secret_society_path}")
            print(f"Suppressed Knowledge Path: {trace.suppressed_knowledge_path}")
            print(f"Confidence: {trace.confidence:.2f}")
        
        print("\n✓ Deep etymology tracing test complete")
        return True
    except Exception as e:
        print(f"\n✗ Deep etymology tracing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_occult_database():
    """Test occult knowledge database."""
    print_section("TEST 2: Occult Knowledge Database")
    
    try:
        occult_db = OccultKnowledgeDatabase()
        
        test_terms = ["astrology", "star"]
        
        for term in test_terms:
            print(f"\nTerm: '{term}'")
            print("-" * 60)
            
            # Find occult connections
            occult_conns = occult_db.find_occult_connections(term)
            print(f"Occult Connections: {len(occult_conns)}")
            for conn in occult_conns[:3]:
                print(f"  - {conn.target_term} ({conn.connection_type}): {conn.description}")
            
            # Find secret society connections
            secret_societies = occult_db.find_secret_society_connections(term)
            print(f"\nSecret Society Connections: {len(secret_societies)}")
            for society in secret_societies[:2]:
                print(f"  - {society.name}: {society.description}")
                print(f"    Connections: {', '.join(society.connections[:3])}")
            
            # Find suppressed knowledge
            suppressed = occult_db.find_suppressed_knowledge(term)
            print(f"\nSuppressed Knowledge: {len(suppressed)}")
            for supp in suppressed[:3]:
                print(f"  - {supp}")
            
            # Decode hidden structure
            hidden_structure = occult_db.decode_hidden_structure(term)
            print(f"\nHidden Structure Path: {' → '.join(hidden_structure['complete_path'][:5])}")
        
        print("\n✓ Occult knowledge database test complete")
        return True
    except Exception as e:
        print(f"\n✗ Occult knowledge database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predictive_history():
    """Test predictive history system."""
    print_section("TEST 3: Predictive History System")
    
    try:
        predictive_history = PredictiveHistorySystem()
        
        test_terms = ["astrology", "star"]
        
        for term in test_terms:
            print(f"\nTerm: '{term}'")
            print("-" * 60)
            
            # Match historical patterns
            patterns = predictive_history.match_historical_patterns(term)
            print(f"Historical Patterns: {len(patterns)}")
            for pattern in patterns[:2]:
                print(f"  - {pattern.pattern_type}: {pattern.description}")
                print(f"    Time Period: {pattern.time_period}")
                print(f"    Connections: {', '.join(pattern.connections[:3])}")
            
            # Predict patterns
            predictions = predictive_history.predict_pattern(term, patterns)
            print(f"\nPredicted Patterns: {len(predictions['predicted_patterns'])}")
            for pred in predictions['predicted_patterns'][:3]:
                print(f"  - {pred}")
            
            # Correlate patterns
            correlation = predictive_history.correlate_patterns_across_time(patterns)
            print(f"\nPattern Correlation:")
            print(f"  Time Periods: {', '.join(correlation['time_periods'])}")
            print(f"  Common Patterns: {', '.join(correlation['common_patterns'][:3])}")
            print(f"  Correlation Score: {correlation['correlation_score']:.2f}")
        
        print("\n✓ Predictive history test complete")
        return True
    except Exception as e:
        print(f"\n✗ Predictive history test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_total_knowledge():
    """Test total knowledge accumulator."""
    print_section("TEST 4: Total Knowledge Accumulator")
    
    try:
        cfg = load_config()
        total_knowledge = TotalKnowledgeAccumulator(cfg)
        
        test_queries = [
            "What is astrology?",
            "How does astrology work?",
            "Tell me about stars and constellations"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            # Decode complete knowledge
            decoding = total_knowledge.decode_complete_knowledge(query)
            
            print(f"Terms Extracted: {decoding['terms']}")
            print(f"Etymology Traces: {len(decoding['etymology_traces'])}")
            for trace in decoding['etymology_traces'][:2]:
                print(f"  - {trace['term']}: {trace['trace'][:100]}...")
                if trace.get('occult_path'):
                    print(f"    Occult Path: {', '.join(trace['occult_path'][:3])}")
            
            print(f"\nOccult Connections: {len(decoding['occult_connections'])}")
            for conn in decoding['occult_connections'][:2]:
                print(f"  - {conn['term']} → {conn['connection']['target']} ({conn['connection']['type']})")
            
            print(f"\nSecret Society Connections: {len(decoding['secret_society_connections'])}")
            for society in decoding['secret_society_connections'][:2]:
                print(f"  - {society['term']} → {society['society']['name']}")
            
            print(f"\nSuppressed Knowledge: {len(decoding['suppressed_knowledge'])}")
            for supp in decoding['suppressed_knowledge'][:2]:
                print(f"  - {supp['term']}: {supp['knowledge']}")
            
            print(f"\nHistorical Patterns: {len(decoding['historical_patterns'])}")
            for pattern in decoding['historical_patterns'][:2]:
                print(f"  - {pattern['term']}: {pattern['pattern']['description'][:60]}...")
            
            print(f"\nComplete Knowledge Chains: {len(decoding['complete_knowledge_chains'])}")
            for chain in decoding['complete_knowledge_chains'][:1]:
                print(f"  - {chain[:200]}...")
        
        print("\n✓ Total knowledge accumulator test complete")
        return True
    except Exception as e:
        print(f"\n✗ Total knowledge accumulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversations():
    """Test conversations with ICEBURG."""
    print_section("TEST 5: Conversations with ICEBURG")
    
    try:
        iceburg = UnifiedICEBURG()
        
        test_queries = [
            "What is astrology?",
            "How does astrology relate to stars and constellations?",
            "Tell me about the occult connections of astrology",
            "What secret societies are connected to astrology?",
            "What suppressed knowledge exists about astrology?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"CONVERSATION {i}")
            print(f"{'='*80}\n")
            
            print(f"USER: {query}\n")
            
            result = await iceburg.process(query, {"user_id": "test_user"})
            
            print(f"ICEBURG:")
            print(f"  Mode: {result.get('mode', 'unknown')}")
            print(f"  Complexity: {result.get('complexity', 0):.2f}")
            
            if result.get('gnosis'):
                gnosis = result['gnosis']
                print(f"\n  Gnosis Integration:")
                print(f"    - Knowledge items: {gnosis.get('knowledge_items', 0)}")
                print(f"    - Matrices identified: {gnosis.get('matrices_identified', 0)}")
                print(f"    - Tools discovered: {gnosis.get('tools_discovered', 0)}")
            
            # Check for total knowledge
            if result.get('result'):
                result_text = str(result['result'])
                if len(result_text) > 500:
                    result_text = result_text[:500] + "..."
                print(f"\n  Response preview: {result_text}")
            
            print("\n" + "-"*80)
        
        print("\n✓ Conversations test complete")
        return True
    except Exception as e:
        print(f"\n✗ Conversations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  DEEP KNOWLEDGE DECODING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Run tests
    results['deep_etymology'] = test_deep_etymology()
    results['occult_database'] = test_occult_database()
    results['predictive_history'] = test_predictive_history()
    results['total_knowledge'] = test_total_knowledge()
    results['conversations'] = asyncio.run(test_conversations())
    
    # Summary
    print_section("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nTest Results:")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "=" * 80)
    print("  TESTING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

