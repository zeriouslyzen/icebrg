"""
Detailed Gnosis Test - Shows prompts and replies
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

from src.iceburg.config import load_config
from src.iceburg.gnosis.unified_gnosis_interface import UnifiedGnosisInterface
from src.iceburg.unified_interface import UnifiedICEBURG


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def test_with_prompts_and_replies():
    """Test with detailed prompts and replies."""
    print_section("GNOSIS SYSTEM - PROMPTS AND REPLIES TEST")
    
    cfg = load_config()
    gnosis_interface = UnifiedGnosisInterface(cfg)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does astrology work?",
        "What are market trends?",
        "Tell me about social networks"
    ]
    
    print("Testing queries through gnosis system...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*80}\n")
        
        # Process through gnosis interface
        result = gnosis_interface.process_query(query, f"test_user_{i}")
        
        print("PROMPT (Query):")
        print(f"  {query}\n")
        
        print("REPLY (Gnosis Results):")
        print(f"  Computer Capabilities Discovered: {len(result.get('computer_capabilities', {}).get('tools_used', []))}")
        if result.get('computer_capabilities', {}).get('tools_used'):
            for tool in result['computer_capabilities']['tools_used'][:3]:
                print(f"    - {tool.get('tool', 'Unknown')} ({tool.get('type', 'Unknown')})")
        
        print(f"\n  Matrices Identified: {len(result.get('matrix_awareness', {}).get('matrices_identified', []))}")
        if result.get('matrix_awareness', {}).get('matrices_identified'):
            matrices = result['matrix_awareness']['matrices_identified']
            matrix_types = result['matrix_awareness'].get('matrix_types', [])
            for matrix_id, matrix_type in zip(matrices[:3], matrix_types[:3]):
                print(f"    - {matrix_id} ({matrix_type})")
        
        print(f"\n  Knowledge Items Found: {len(result.get('gnosis_knowledge', {}).get('knowledge_items', []))}")
        if result.get('gnosis_knowledge', {}).get('knowledge_items'):
            for item in result['gnosis_knowledge']['knowledge_items'][:2]:
                print(f"    - {item.get('knowledge_type', 'unknown')}: {item.get('content', '')[:80]}...")
        
        print(f"\n  Domains: {result.get('gnosis_knowledge', {}).get('domains', [])[:5]}")
        
        print(f"\n  User Evolution:")
        if result.get('user_evolution'):
            evo = result['user_evolution']
            print(f"    - Total conversations: {evo.get('total_conversations', 0)}")
            print(f"    - Interests: {evo.get('interests', [])[:3]}")
            print(f"    - Domain expertise: {list(evo.get('domain_expertise', {}).keys())[:3]}")
        
        print("\n" + "-"*80)
    
    # Test through UnifiedInterface
    print_section("UNIFIED INTERFACE TEST - WITH PROMPTS AND REPLIES")
    
    try:
        iceburg = UnifiedICEBURG(config_path=None)
        
        test_queries_2 = [
            "What is artificial intelligence?",
            "How does quantum computing work?",
            "Explain astrology"
        ]
        
        for i, query in enumerate(test_queries_2, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {i}: {query}")
            print(f"{'='*80}\n")
            
            print("PROMPT (Query):")
            print(f"  {query}\n")
            
            result = await iceburg.process(query, {"user_id": f"test_user_{i}"})
            
            print("REPLY (ICEBURG Response):")
            print(f"  Mode: {result.get('mode', 'unknown')}")
            print(f"  Complexity: {result.get('complexity', 0):.2f}")
            
            if result.get('gnosis'):
                gnosis = result['gnosis']
                print(f"\n  Gnosis Integration:")
                print(f"    - Knowledge items: {gnosis.get('knowledge_items', 0)}")
                print(f"    - Matrices identified: {gnosis.get('matrices_identified', 0)}")
                print(f"    - Tools discovered: {gnosis.get('tools_discovered', 0)}")
            
            if result.get('result'):
                result_text = str(result['result'])
                if len(result_text) > 200:
                    result_text = result_text[:200] + "..."
                print(f"\n  Response preview: {result_text}")
            
            print("\n" + "-"*80)
        
        print("\n✓ Unified interface test complete")
        return True
    except Exception as e:
        print(f"\n✗ Unified interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run detailed test."""
    asyncio.run(test_with_prompts_and_replies())


if __name__ == "__main__":
    main()

