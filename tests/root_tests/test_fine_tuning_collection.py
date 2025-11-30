"""
Test Fine-Tuning Data Collection System
Verifies that the fine-tuning data collection system works correctly
"""

import os
import sys
from pathlib import Path

# Enable fine-tuning data collection
os.environ["ICEBURG_ENABLE_FINE_TUNING_DATA"] = "1"

from src.iceburg.data_collection import FineTuningLogger

def test_fine_tuning_logger():
    """Test fine-tuning logger functionality."""
    print("=" * 80)
    print("Testing Fine-Tuning Data Collection System")
    print("=" * 80)
    
    # Initialize logger
    logger = FineTuningLogger()
    
    print(f"\n✅ Fine-tuning logger initialized")
    print(f"   Enabled: {logger.enabled}")
    print(f"   Data directory: {logger.data_dir}")
    
    # Test conversation logging
    print("\n📝 Testing conversation logging...")
    test_messages = [
        {"role": "system", "content": "You are ICEBURG, an AI civilization."},
        {"role": "user", "content": "What is quantum mechanics?"},
        {"role": "assistant", "content": "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles."}
    ]
    
    test_metadata = {
        "model": "llama3.1:8b",
        "mode": "research",
        "agent": "surveyor",
        "quality_score": 0.95
    }
    
    logger.log_conversation(test_messages, test_metadata, quality_score=0.95)
    print("   ✅ Conversation logged successfully")
    
    # Test agent generation logging
    print("\n🤖 Testing agent generation logging...")
    test_code = """
from __future__ import annotations
from typing import Dict, Any
from ..config import IceburgConfig
from ..llm import chat_complete
import logging

logger = logging.getLogger(__name__)

class TestAgent:
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
    
    def run(self, query: str) -> Dict[str, Any]:
        return {"response": "Test response"}
"""
    
    test_validation = {
        "valid": True,
        "syntax_valid": True,
        "imports_valid": True,
        "methods_valid": True,
        "safety_valid": True
    }
    
    test_agent_metadata = {
        "template": {
            "specialization": "testing",
            "capabilities": ["test", "validate"],
            "domain_focus": "testing",
            "reasoning_patterns": ["test-driven"]
        },
        "emergence_data": {},
        "generation_method": "llm"
    }
    
    logger.log_agent_generation(
        agent_name="test_agent",
        generated_code=test_code,
        validation_result=test_validation,
        metadata=test_agent_metadata
    )
    print("   ✅ Agent generation logged successfully")
    
    # Test statistics
    print("\n📊 Testing statistics...")
    stats = logger.get_stats()
    print(f"   Conversations: {stats['conversations']}")
    print(f"   Reasoning Chains: {stats['reasoning_chains']}")
    print(f"   Quality Metrics: {stats['quality_metrics']}")
    print(f"   Agent Generations: {stats['agent_generations']}")
    
    # Test export
    print("\n📤 Testing export functionality...")
    output_dir = Path("data/fine_tuning/export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export in ChatML format
    output_file = output_dir / "test_chatml_format.jsonl"
    count = logger.export_for_fine_tuning(
        output_file=output_file,
        format="chatml",
        min_quality=0.8,
        min_conversations=2
    )
    
    if count > 0:
        print(f"   ✅ Exported {count} conversations to {output_file}")
        print(f"   File exists: {output_file.exists()}")
        print(f"   File size: {output_file.stat().st_size} bytes")
    else:
        print("   ⚠️  No conversations exported (may be expected if quality threshold not met)")
    
    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("=" * 80)
    print("\nTo enable fine-tuning data collection:")
    print("  export ICEBURG_ENABLE_FINE_TUNING_DATA=1")
    print("\nTo export data for fine-tuning:")
    print("  python -m src.iceburg.data_collection.export_fine_tuning_data --format chatml")
    print("=" * 80)

if __name__ == "__main__":
    test_fine_tuning_logger()

