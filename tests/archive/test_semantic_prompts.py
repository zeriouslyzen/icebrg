#!/usr/bin/env python3
"""
Quick test of semantic prompts for Secretary agent.
Tests: date/time awareness and nickname recognition.
"""

import sys
sys.path.insert(0, '/Users/jackdanger/Desktop/Projects/iceburg/src')

from iceburg.config import IceburgConfig
from iceburg.agents.secretary import SecretaryAgent

def test_semantic_prompts():
    print("=" * 60)
    print("TESTING SEMANTIC PROMPTS")
    print("=" * 60)
    
    # Initialize config with required parameters
    cfg = IceburgConfig(
        data_dir="/Users/jackdanger/Desktop/Projects/iceburg/data",
        surveyor_model="llama3.1:8b",
        dissident_model="llama3.1:8b",
        synthesist_model="llama3.1:8b",
        oracle_model="llama3.1:8b",
        embed_model="nomic-embed-text",
        llm_provider="ollama"
    )
    secretary = SecretaryAgent(cfg)
    
    # Test 1: Date/Time Awareness
    print("\n📅 TEST 1: Date/Time Awareness")
    print("-" * 60)
    print("Query: 'what's the date today'")
    try:
        response = secretary.run(
            query="what's the date today",
            conversation_id="test_semantic_1",
            user_id="test_user",
            mode="chat"
        )
        print(f"\nResponse:\n{response}\n")
        
        # Check if response contains actual date
        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")
        if today[:3] in response or "Monday" in response or "Tuesday" in response:
            print("✅ PASS: Response contains current date context")
        else:
            print("⚠️  WARNING: Response may not have date context")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    # Test 2: Nickname Recognition ("ice")
    print("\n🧊 TEST 2: Nickname Recognition - 'ice'")
    print("-" * 60)
    print("Query: 'yo ice what's up'")
    try:
        response = secretary.run(
            query="yo ice what's up",
            conversation_id="test_semantic_2",
            user_id="test_user",
            mode="chat"
        )
        print(f"\nResponse:\n{response}\n")
        
        # Check if response is conversational (not literal)
        if "iceberg" in response.lower() or "frozen water" in response.lower():
            print("❌ FAIL: Interpreted 'ice' literally")
        else:
            print("✅ PASS: Recognized 'ice' as nickname")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    # Test 3: Nickname Recognition ("berg")
    print("\n⛰️  TEST 3: Nickname Recognition - 'berg'")
    print("-" * 60)
    print("Query: 'hey berg, what can you do'")
    try:
        response = secretary.run(
            query="hey berg, what can you do",
            conversation_id="test_semantic_3",
            user_id="test_user",
            mode="chat"
        )
        print(f"\nResponse:\n{response}\n")
        
        # Check if response talks about ICEBURG capabilities
        if "research" in response.lower() or "agent" in response.lower():
            print("✅ PASS: Recognized 'berg' as nickname and responded about capabilities")
        else:
            print("⚠️  WARNING: May not have recognized 'berg' as nickname")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("SEMANTIC PROMPT TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_semantic_prompts()
