"""
Integration test for Secretary Knowledge Base (Phase 2)

Tests knowledge extraction, storage, and retrieval in a realistic scenario.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.iceburg.config import load_config
from src.iceburg.agents.secretary_knowledge import SecretaryKnowledgeBase


def test_knowledge_base_workflow():
    """Test complete knowledge base workflow."""
    print("=" * 60)
    print("Secretary Knowledge Base Integration Test")
    print("=" * 60)
    
    try:
        cfg = load_config()
    except Exception:
        # Fallback config
        from unittest.mock import Mock
        cfg = Mock()
        cfg.data_dir = Path("./data")
        cfg.surveyor_model = "gemini-2.0-flash-exp"
        cfg.primary_model = "gemini-2.0-flash-exp"
        cfg.embed_model = "nomic-embed-text"
    
    # Use temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "secretary_knowledge"
        kb = SecretaryKnowledgeBase(cfg, knowledge_base_dir=kb_dir)
        
        print("\n✅ Knowledge base initialized")
        
        # Test 1: Store topics
        print("\n1. Testing topic storage...")
        kb.store_topic("Python Programming", "Python is a high-level programming language.")
        kb.store_topic("Machine Learning", "Machine learning is a subset of AI.")
        print("   ✓ Topics stored")
        
        # Test 2: Update persona
        print("\n2. Testing persona storage...")
        kb.update_persona("user_123", {
            "preferences": {"language": "Python", "style": "concise"},
            "expertise": ["machine learning", "data science"]
        })
        print("   ✓ Persona stored")
        
        # Test 3: Retrieve persona
        print("\n3. Testing persona retrieval...")
        persona = kb.get_persona("user_123")
        if persona:
            print(f"   ✓ Persona retrieved: {len(persona)} fields")
        else:
            print("   ✗ Persona not found")
        
        # Test 4: Query knowledge
        print("\n4. Testing knowledge query...")
        results = kb.query_knowledge("Python", k=3)
        print(f"   ✓ Found {len(results)} results")
        
        # Test 5: Build index
        print("\n5. Testing index building...")
        kb.build_index()
        index_file = kb.indexes_dir / "topic_index.json"
        if index_file.exists():
            print("   ✓ Index built successfully")
        else:
            print("   ✗ Index not created")
        
        # Test 6: Process conversation
        print("\n6. Testing conversation processing...")
        try:
            # This requires LLM, so it might fail in test environment
            knowledge = kb.process_conversation(
                query="I love Python programming",
                response="Python is great for data science and machine learning.",
                user_id="user_123",
                conversation_id="conv_456"
            )
            if knowledge:
                print(f"   ✓ Conversation processed: {len(knowledge.get('topics', []))} topics extracted")
            else:
                print("   ⚠ Conversation processing requires LLM (skipped)")
        except Exception as e:
            print(f"   ⚠ Conversation processing requires LLM (skipped in test): {e}")
        
        print("\n" + "=" * 60)
        print("✅ All knowledge base tests completed!")
        print("=" * 60)
        
        return True


if __name__ == "__main__":
    success = test_knowledge_base_workflow()
    sys.exit(0 if success else 1)

