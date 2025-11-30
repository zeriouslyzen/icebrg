"""
Test Conversations with ICEBURG
Have conversations with ICEBURG to see how deep knowledge decoding unfolds
"""

import asyncio
from src.iceburg.unified_interface import UnifiedICEBURG


async def main():
    """Run conversations with ICEBURG."""
    print("\n" + "=" * 80)
    print("  ICEBURG DEEP KNOWLEDGE DECODING - CONVERSATIONS")
    print("=" * 80 + "\n")
    
    iceburg = UnifiedICEBURG()
    
    conversations = [
        {
            "query": "What is astrology?",
            "description": "Basic astrology question"
        },
        {
            "query": "How does astrology relate to stars and constellations?",
            "description": "Astrology connections question"
        },
        {
            "query": "Tell me about the occult connections of astrology",
            "description": "Occult connections question"
        },
        {
            "query": "What secret societies are connected to astrology?",
            "description": "Secret society connections question"
        },
        {
            "query": "What suppressed knowledge exists about astrology?",
            "description": "Suppressed knowledge question"
        },
        {
            "query": "Trace the etymology of astrology from its origins to the present",
            "description": "Deep etymology tracing question"
        }
    ]
    
    for i, conv in enumerate(conversations, 1):
        print(f"\n{'='*80}")
        print(f"CONVERSATION {i}: {conv['description']}")
        print(f"{'='*80}\n")
        
        print(f"USER: {conv['query']}\n")
        print("ICEBURG:")
        print("-" * 80)
        
        try:
            result = await iceburg.process(conv['query'], {"user_id": "test_user"})
            
            # Display results
            if result.get('mode'):
                print(f"Mode: {result['mode']}")
            
            if result.get('complexity'):
                print(f"Complexity: {result['complexity']:.2f}")
            
            # Display gnosis results
            if result.get('gnosis'):
                gnosis = result['gnosis']
                print(f"\nGnosis Integration:")
                if gnosis.get('knowledge_items'):
                    print(f"  - Knowledge items found: {gnosis['knowledge_items']}")
                if gnosis.get('matrices_identified'):
                    print(f"  - Matrices identified: {gnosis['matrices_identified']}")
                if gnosis.get('tools_discovered'):
                    print(f"  - Tools discovered: {gnosis['tools_discovered']}")
            
            # Display total knowledge results
            if result.get('result'):
                result_text = str(result['result'])
                if len(result_text) > 1000:
                    result_text = result_text[:1000] + "..."
                print(f"\nResponse:\n{result_text}")
            
            # Display total knowledge decoding
            if hasattr(iceburg, 'gnosis_interface') and iceburg.gnosis_interface:
                try:
                    gnosis_result = iceburg.gnosis_interface.process_query(conv['query'], "test_user")
                    if gnosis_result.get('total_knowledge'):
                        total_knowledge = gnosis_result['total_knowledge']
                        print(f"\nTotal Knowledge Decoding:")
                        if total_knowledge.get('etymology_traces'):
                            print(f"  - Etymology traces: {len(total_knowledge['etymology_traces'])}")
                            for trace in total_knowledge['etymology_traces'][:1]:
                                print(f"    * {trace['term']}: {trace['trace'][:100]}...")
                        if total_knowledge.get('occult_connections'):
                            print(f"  - Occult connections: {len(total_knowledge['occult_connections'])}")
                            for conn in total_knowledge['occult_connections'][:2]:
                                print(f"    * {conn['term']} → {conn['connection']['target']} ({conn['connection']['type']})")
                        if total_knowledge.get('secret_society_connections'):
                            print(f"  - Secret society connections: {len(total_knowledge['secret_society_connections'])}")
                            for society in total_knowledge['secret_society_connections'][:2]:
                                print(f"    * {society['term']} → {society['society']['name']}")
                        if total_knowledge.get('suppressed_knowledge'):
                            print(f"  - Suppressed knowledge: {len(total_knowledge['suppressed_knowledge'])}")
                            for supp in total_knowledge['suppressed_knowledge'][:2]:
                                print(f"    * {supp['term']}: {supp['knowledge']}")
                        if total_knowledge.get('historical_patterns'):
                            print(f"  - Historical patterns: {len(total_knowledge['historical_patterns'])}")
                            for pattern in total_knowledge['historical_patterns'][:1]:
                                print(f"    * {pattern['term']}: {pattern['pattern']['description'][:80]}...")
                except Exception as e:
                    print(f"\nNote: Total knowledge decoding error: {e}")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-" * 80)
    
    print("\n" + "=" * 80)
    print("  CONVERSATIONS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

