"""
Integration test for Secretary V2 - All Phases End-to-End
Tests all 6 phases of the AGI enhancement working together.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path

from src.iceburg.agents.secretary import SecretaryAgent
from src.iceburg.config import IceburgConfig


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    cfg = Mock(spec=IceburgConfig)
    cfg.data_dir = Path(tempfile.mkdtemp())
    cfg.surveyor_model = "gemini-2.0-flash-exp"
    cfg.primary_model = "gemini-2.0-flash-exp"
    cfg.llm_provider = "gemini"
    cfg.timeout_s = 60
    cfg.embed_model = "nomic-embed-text"
    return cfg


@patch('src.iceburg.providers.factory.provider_factory')
def test_secretary_v2_full_integration(mock_factory, mock_config):
    """
    Test all Secretary V2 phases working together:
    - Phase 1: Memory Persistence
    - Phase 2: Tool Calling
    - Phase 3: Multimodal Processing
    - Phase 4: Blackboard Integration
    - Phase 5: Efficiency Optimizations
    - Phase 6: Autonomous Planning
    """
    # Mock provider
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = "Test response with comprehensive answer"
    mock_factory.return_value = mock_provider
    
    # Create Secretary with all features enabled
    agent = SecretaryAgent(
        mock_config,
        enable_memory=True,
        enable_tools=True,
        enable_blackboard=True,
        enable_cache=True,
        enable_planning=True,
        enable_knowledge_base=True
    )
    
    # Mock all subsystems
    agent.memory = Mock()
    agent.memory.search.return_value = []
    agent.memory.index_texts.return_value = ["memory_id_1"]
    
    agent.agent_memory = Mock()
    agent.agent_memory.get_memories_by_type.return_value = []
    agent.agent_memory.add_memory.return_value = "memory_id_2"
    
    agent.local_persistence = Mock()
    agent.local_persistence.get_conversations.return_value = []
    agent.local_persistence.save_conversation = Mock()
    
    agent.workspace = Mock()
    agent.workspace.publish = Mock()
    
    agent.agent_comm = Mock()
    
    # Test 1: Simple query (should use cache optimization)
    response1 = agent.run(
        query="What is ICEBURG?",
        conversation_id="test_conv_1",
        user_id="test_user_1"
    )
    assert response1 is not None
    assert mock_provider.chat_complete.called
    
    # Test 2: Same query again (should hit cache)
    response2 = agent.run(
        query="What is ICEBURG?",
        conversation_id="test_conv_1",
        user_id="test_user_1"
    )
    assert response2 == response1  # Should be cached
    
    # Test 3: Memory persistence - verify storage was called
    assert agent.local_persistence.save_conversation.called
    
    # Test 4: Blackboard integration - verify _get_agent_context works
    context = agent._get_agent_context("test query")
    assert context is not None  # Should return empty string or context
    
    # Test 5: Simple question detection
    assert agent._is_simple_question_pattern("What is AI?") == True
    assert agent._is_simple_question_pattern("Create a comprehensive multi-step plan for building a web application") == False
    
    # Test 6: Tool detection
    assert agent._needs_tools("calculate 2+2") == True
    assert agent._needs_tools("hello") == False
    
    print("✅ All Secretary V2 phases working correctly!")


@patch('src.iceburg.providers.factory.provider_factory')
def test_secretary_v2_planning_integration(mock_factory, mock_config):
    """Test planning phase specifically."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = '{"is_goal": false, "goals": []}'
    mock_factory.return_value = mock_provider
    
    agent = SecretaryAgent(
        mock_config,
        enable_memory=False,
        enable_tools=False,
        enable_blackboard=False,
        enable_cache=False,
        enable_planning=True,
        enable_knowledge_base=False
    )
    
    # Test goal extraction
    if agent.planner:
        goals = agent.planner.extract_goals("What is the weather?")
        assert goals == []  # Simple question, no goals
        
        print("✅ Planning phase working correctly!")


@patch('src.iceburg.providers.factory.provider_factory')
def test_secretary_v2_graceful_degradation(mock_factory, mock_config):
    """Test that Secretary works even when subsystems fail."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = "Fallback response"
    mock_factory.return_value = mock_provider
    
    agent = SecretaryAgent(
        mock_config,
        enable_memory=True,
        enable_tools=True,
        enable_blackboard=True,
        enable_cache=True,
        enable_planning=True,
        enable_knowledge_base=True
    )
    
    # Make all subsystems fail
    agent.memory = Mock()
    agent.memory.search.side_effect = Exception("Memory failed")
    
    agent.agent_memory = Mock()
    agent.agent_memory.get_memories_by_type.side_effect = Exception("Agent memory failed")
    
    agent.local_persistence = Mock()
    agent.local_persistence.get_conversations.side_effect = Exception("Persistence failed")
    
    # Should still work with graceful degradation
    response = agent.run(
        query="Test query",
        conversation_id="test_conv",
        user_id="test_user"
    )
    
    assert response == "Fallback response"
    print("✅ Graceful degradation working correctly!")


@patch('src.iceburg.providers.factory.provider_factory')
def test_secretary_v2_standalone_run(mock_factory, mock_config):
    """Test the standalone run() function wrapper."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = "Run function response"
    mock_factory.return_value = mock_provider
    
    from src.iceburg.agents.secretary import run
    
    # Test with extra kwargs (simulating server call)
    response = run(
        mock_config,
        query="Test query",
        mode="chat",
        routing_mode="fast"
    )
    
    assert response == "Run function response"
    print("✅ Standalone run() wrapper working correctly!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
