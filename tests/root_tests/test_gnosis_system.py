"""
Test Gnosis Evolution System
Tests knowledge accumulation, matrix awareness, computer discovery, and user evolution
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

from src.iceburg.config import load_config
from src.iceburg.gnosis.unified_gnosis_interface import UnifiedGnosisInterface
from src.iceburg.gnosis.universal_knowledge_accumulator import UniversalKnowledgeAccumulator
from src.iceburg.discovery.dynamic_tool_usage import DynamicToolUsage
from src.iceburg.awareness.matrix_detection import MatrixDetection
from src.iceburg.evolution.user_profile_builder import UserProfileBuilder


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_computer_discovery():
    """Test computer capability discovery."""
    print_section("TEST 1: Computer Capability Discovery")
    
    try:
        dynamic_tools = DynamicToolUsage()
        
        # Test queries
        test_queries = [
            "marketing analysis",
            "astrology chart",
            "data analysis"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            result = dynamic_tools.use_computer_to_find_info(query)
            
            print(f"Tools discovered: {len(result.get('tools_used', []))}")
            for tool in result.get('tools_used', [])[:3]:
                print(f"  - {tool.get('tool', 'Unknown')} ({tool.get('type', 'Unknown')})")
            
            print(f"Data files found: {len(result.get('data_found', []))}")
            print(f"Files found: {len(result.get('files_found', []))}")
            
            if result.get('errors'):
                print(f"Errors: {len(result['errors'])}")
        
        print("\n✓ Computer discovery test complete")
        return True
    except Exception as e:
        print(f"\n✗ Computer discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matrix_awareness():
    """Test matrix awareness system."""
    print_section("TEST 2: Matrix Awareness System")
    
    try:
        matrix_detection = MatrixDetection()
        
        # Test queries
        test_queries = [
            "astrology horoscope",
            "market trading analysis",
            "social network dynamics"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            matrices = matrix_detection.identify_underlying_matrices(query)
            
            print(f"Matrices identified: {len(matrices)}")
            for matrix in matrices:
                print(f"  - {matrix.name} ({matrix.matrix_type.value})")
                print(f"    Nodes: {len(matrix.nodes)}, Edges: {len(matrix.edges)}")
                print(f"    Confidence: {matrix.confidence:.2f}")
                if matrix.patterns:
                    print(f"    Patterns: {len(matrix.patterns)}")
        
        # Test constructed reality understanding
        print("\n" + "-" * 60)
        print("Constructed Reality Understanding:")
        reality = matrix_detection.understand_constructed_reality()
        print(f"  Reality Type: {reality['reality_type']}")
        print(f"  Awareness Level: {reality['awareness_level']}")
        print(f"  Characteristics: {len(reality['characteristics'])}")
        print(f"  Matrix Metaphor: {reality['matrix_metaphor']}")
        
        print("\n✓ Matrix awareness test complete")
        return True
    except Exception as e:
        print(f"\n✗ Matrix awareness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gnosis_accumulation():
    """Test gnosis knowledge accumulation."""
    print_section("TEST 3: Gnosis Knowledge Accumulation")
    
    try:
        cfg = load_config()
        accumulator = UniversalKnowledgeAccumulator(cfg)
        
        # Simulate conversations
        conversations = [
            {
                "conversation_id": "conv_1",
                "query": "What is quantum computing?",
                "response": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations. It has potential applications in cryptography, drug discovery, and optimization problems.",
                "metadata": {"domains": ["physics", "computer science"]}
            },
            {
                "conversation_id": "conv_2",
                "query": "How does machine learning work?",
                "response": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions based on training data.",
                "metadata": {"domains": ["computer science", "artificial intelligence"]}
            },
            {
                "conversation_id": "conv_3",
                "query": "What is the relationship between energy and matter?",
                "response": "Energy and matter are related through Einstein's famous equation E=mc², which shows that energy and matter are interchangeable. This principle is fundamental to nuclear physics and particle physics.",
                "metadata": {"domains": ["physics", "mathematics"]}
            }
        ]
        
        print(f"Processing {len(conversations)} conversations...")
        
        for i, conversation in enumerate(conversations, 1):
            print(f"\nConversation {i}: {conversation['query'][:50]}...")
            
            # Extract insights
            insights = accumulator.extract_insights_from_conversation(conversation)
            print(f"  Insights extracted: {len(insights)}")
            for insight in insights[:2]:
                print(f"    - {insight.insight_type}: {insight.content[:60]}...")
            
            # Accumulate to gnosis
            accumulator.accumulate_to_gnosis(insights)
        
        # Query gnosis
        print("\n" + "-" * 60)
        print("Querying Gnosis Knowledge Base:")
        
        test_queries = [
            "quantum computing",
            "machine learning",
            "energy matter relationship"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            knowledge_items = accumulator.query_gnosis(query)
            print(f"  Knowledge items found: {len(knowledge_items)}")
            for item in knowledge_items[:2]:
                print(f"    - {item.knowledge_type}: {item.content[:60]}...")
                print(f"      Domains: {item.domains}, Confidence: {item.confidence:.2f}")
        
        # Get gnosis base stats
        if accumulator.gnosis_base:
            print("\n" + "-" * 60)
            print("Gnosis Knowledge Base Statistics:")
            print(f"  Total knowledge items: {len(accumulator.gnosis_base.knowledge_items)}")
            print(f"  Total insights: {len(accumulator.gnosis_base.insights)}")
            print(f"  Total conversations: {accumulator.gnosis_base.total_conversations}")
            print(f"  Last updated: {accumulator.gnosis_base.last_updated}")
        
        print("\n✓ Gnosis accumulation test complete")
        return True
    except Exception as e:
        print(f"\n✗ Gnosis accumulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_user_evolution():
    """Test user evolution system."""
    print_section("TEST 4: User Evolution System")
    
    try:
        cfg = load_config()
        user_profile_builder = UserProfileBuilder(cfg)
        
        # Simulate user conversations
        conversations = [
            {
                "query": "Explain quantum mechanics",
                "response": "Quantum mechanics is a fundamental theory in physics...",
                "metadata": {"domains": ["physics"]}
            },
            {
                "query": "How do neural networks work?",
                "response": "Neural networks are computing systems inspired by biological neural networks...",
                "metadata": {"domains": ["computer science", "artificial intelligence"]}
            },
            {
                "query": "What is the theory of relativity?",
                "response": "The theory of relativity consists of special and general relativity...",
                "metadata": {"domains": ["physics", "mathematics"]}
            }
        ]
        
        user_id = "test_user_1"
        
        print(f"Building user profile for user: {user_id}")
        print(f"Processing {len(conversations)} conversations...")
        
        profile = user_profile_builder.build_user_profile(conversations, user_id)
        
        print("\n" + "-" * 60)
        print("User Profile:")
        print(f"  User ID: {profile.user_id}")
        print(f"  Total conversations: {profile.total_conversations}")
        print(f"  Interests: {profile.interests}")
        print(f"  Domain expertise: {profile.domain_expertise}")
        print(f"  Preferences: {profile.preferences}")
        print(f"  Created at: {profile.created_at}")
        print(f"  Last updated: {profile.last_updated}")
        
        print("\n✓ User evolution test complete")
        return True
    except Exception as e:
        print(f"\n✗ User evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_gnosis_interface():
    """Test unified gnosis interface."""
    print_section("TEST 5: Unified Gnosis Interface")
    
    try:
        cfg = load_config()
        gnosis_interface = UnifiedGnosisInterface(cfg)
        
        # Test queries
        test_queries = [
            "What is artificial intelligence?",
            "How does astrology work?",
            "Explain market trends"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            result = gnosis_interface.process_query(query, "test_user")
            
            print(f"Computer capabilities: {len(result.get('computer_capabilities', {}).get('tools_used', []))}")
            print(f"Matrices identified: {len(result.get('matrix_awareness', {}).get('matrices_identified', []))}")
            print(f"Knowledge items: {len(result.get('gnosis_knowledge', {}).get('knowledge_items', []))}")
            print(f"Domains: {result.get('gnosis_knowledge', {}).get('domains', [])}")
            
            if result.get('user_evolution'):
                print(f"User evolution: {result['user_evolution'].get('total_conversations', 0)} conversations")
        
        print("\n✓ Unified gnosis interface test complete")
        return True
    except Exception as e:
        print(f"\n✗ Unified gnosis interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test full integration through UnifiedInterface."""
    print_section("TEST 6: Full Integration Test")
    
    try:
        from src.iceburg.unified_interface import UnifiedICEBURG
        
        iceburg = UnifiedICEBURG()
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "Explain quantum computing",
            "How does astrology work?"
        ]
        
        async def run_tests():
            for query in test_queries:
                print(f"\nQuery: '{query}'")
                print("-" * 60)
                
                result = await iceburg.process(query, {"user_id": "test_user"})
                
                print(f"Mode: {result.get('mode', 'unknown')}")
                print(f"Complexity: {result.get('complexity', 0):.2f}")
                
                if result.get('gnosis'):
                    gnosis = result['gnosis']
                    print(f"Gnosis - Knowledge items: {gnosis.get('knowledge_items', 0)}")
                    print(f"Gnosis - Matrices: {gnosis.get('matrices_identified', 0)}")
                    print(f"Gnosis - Tools: {gnosis.get('tools_discovered', 0)}")
        
        asyncio.run(run_tests())
        
        print("\n✓ Full integration test complete")
        return True
    except Exception as e:
        print(f"\n✗ Full integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  GNOSIS EVOLUTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Run tests
    results['computer_discovery'] = test_computer_discovery()
    results['matrix_awareness'] = test_matrix_awareness()
    results['gnosis_accumulation'] = test_gnosis_accumulation()
    results['user_evolution'] = test_user_evolution()
    results['unified_gnosis_interface'] = test_unified_gnosis_interface()
    results['full_integration'] = test_full_integration()
    
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

