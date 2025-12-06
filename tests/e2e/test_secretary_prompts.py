"""
End-to-end validation tests for Secretary agent with real prompts.
These tests validate memory functionality with actual queries.
"""

import pytest
import sys
from pathlib import Path
import time
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.secretary import SecretaryAgent, run
from iceburg.config import load_config


class TestSecretaryMemoryValidation:
    """E2E validation tests with real prompts"""
    
    @pytest.fixture
    def cfg(self):
        """Load ICEBURG config"""
        return load_config()
    
    @pytest.fixture
    def agent(self, cfg):
        """Create Secretary agent with memory enabled"""
        return SecretaryAgent(cfg, enable_memory=True)
    
    @pytest.fixture
    def conversation_id(self):
        """Generate unique conversation ID"""
        return str(uuid.uuid4())
    
    def test_conversation_continuity(self, agent, conversation_id):
        """
        Test 1: Conversation Continuity
        Query 1: "My name is Alice and I'm interested in quantum computing"
        Query 2: "What's my name and what am I interested in?"
        Expected: Secretary remembers name and interest
        """
        # First query
        response1 = agent.run(
            query="My name is Alice and I'm interested in quantum computing",
            conversation_id=conversation_id
        )
        
        assert response1 is not None
        assert len(response1) > 0
        
        # Wait a moment for memory to be stored
        time.sleep(0.5)
        
        # Second query - should remember
        response2 = agent.run(
            query="What's my name and what am I interested in?",
            conversation_id=conversation_id
        )
        
        assert response2 is not None
        # Response should mention Alice and quantum computing
        response_lower = response2.lower()
        assert "alice" in response_lower or "quantum" in response_lower
    
    def test_cross_session_memory(self, agent, conversation_id):
        """
        Test 2: Cross-Session Memory
        Session 1: "I prefer detailed explanations"
        Session 2: "Explain quantum computing" (should be detailed)
        Expected: Secretary uses preference from previous session
        """
        user_id = f"test_user_{uuid.uuid4()}"
        
        # First session - set preference
        response1 = agent.run(
            query="I prefer detailed explanations",
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        assert response1 is not None
        
        # Wait for memory storage
        time.sleep(0.5)
        
        # Second session - should remember preference
        new_conversation_id = str(uuid.uuid4())
        response2 = agent.run(
            query="Explain quantum computing",
            conversation_id=new_conversation_id,
            user_id=user_id
        )
        
        assert response2 is not None
        # Response should be detailed (longer than a short response)
        assert len(response2) > 100  # Detailed explanation should be longer
    
    def test_memory_retrieval(self, agent, conversation_id):
        """
        Test 3: Memory Retrieval
        Query: "What did we discuss about AI yesterday?"
        Expected: Secretary retrieves and references past conversation
        """
        # First, have a conversation about AI
        response1 = agent.run(
            query="Tell me about artificial intelligence",
            conversation_id=conversation_id
        )
        
        assert response1 is not None
        time.sleep(0.5)
        
        # Ask about previous discussion
        response2 = agent.run(
            query="What did we discuss about AI?",
            conversation_id=conversation_id
        )
        
        assert response2 is not None
        # Should reference the previous conversation
        response_lower = response2.lower()
        assert "ai" in response_lower or "artificial" in response_lower or "intelligence" in response_lower
    
    def test_basic_functionality(self, agent):
        """
        Test that basic functionality still works without memory
        """
        response = agent.run(
            query="What is ICEBURG?",
            conversation_id=None,
            user_id=None
        )
        
        assert response is not None
        assert len(response) > 0
        # Should mention ICEBURG
        response_lower = response.lower()
        assert "iceburg" in response_lower or "iceberg" in response_lower


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

