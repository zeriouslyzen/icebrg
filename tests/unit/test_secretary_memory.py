"""
Unit tests for Secretary agent memory functionality.
Tests memory storage, retrieval, conversation continuity, and cross-session memory.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.secretary import SecretaryAgent, run
from iceburg.config import IceburgConfig


@pytest.fixture
def mock_config():
    """Create a mock ICEBURG config"""
    config = Mock(spec=IceburgConfig)
    config.data_dir = Path(tempfile.mkdtemp())
    config.surveyor_model = "llama3.1:8b"
    config.primary_model = "llama3.1:8b"
    config.llm_provider = "ollama"
    config.provider_url = "http://localhost:11434"
    config.timeout_s = 60
    config.enable_code_generation = False
    config.disable_memory = False
    config.enable_software_lab = False
    config.max_context_length = 4096
    config.fast = True
    config.embed_model = "nomic-embed-text"
    return config


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestSecretaryMemory:
    """Test memory functionality of Secretary agent"""
    
    def test_secretary_agent_initialization(self, mock_config):
        """Test that SecretaryAgent initializes correctly"""
        agent = SecretaryAgent(mock_config, enable_memory=True)
        assert agent.cfg == mock_config
        assert agent.enable_memory is True
    
    def test_secretary_agent_initialization_without_memory(self, mock_config):
        """Test that SecretaryAgent works without memory"""
        agent = SecretaryAgent(mock_config, enable_memory=False)
        assert agent.enable_memory is False
        assert agent.memory is None
        assert agent.agent_memory is None
    
    @patch('iceburg.agents.secretary.provider_factory')
    def test_memory_retrieval_called(self, mock_provider_factory, mock_config):
        """Test that memory retrieval is called when memory is enabled"""
        # Mock provider
        mock_provider = Mock()
        mock_provider.chat_complete.return_value = "Test response"
        mock_provider_factory.return_value = mock_provider
        
        # Create agent with memory
        agent = SecretaryAgent(mock_config, enable_memory=True)
        
        # Mock memory systems
        agent.memory = Mock()
        agent.memory.search.return_value = []
        agent.agent_memory = Mock()
        agent.agent_memory.get_memories_by_type.return_value = []
        agent.local_persistence = Mock()
        agent.local_persistence.get_conversations.return_value = []
        
        # Run query
        response = agent.run(
            query="Test query",
            conversation_id="test_conv_123",
            user_id="test_user_123"
        )
        
        # Verify memory retrieval was called
        assert agent.local_persistence.get_conversations.called
        assert response == "Test response"
    
    @patch('iceburg.agents.secretary.provider_factory')
    def test_memory_storage_called(self, mock_provider_factory, mock_config):
        """Test that memory storage is called after response"""
        # Mock provider
        mock_provider = Mock()
        mock_provider.chat_complete.return_value = "Test response"
        mock_provider_factory.return_value = mock_provider
        
        # Create agent with memory
        agent = SecretaryAgent(mock_config, enable_memory=True)
        
        # Mock memory systems
        agent.memory = Mock()
        agent.memory.search.return_value = []
        agent.memory.index_texts.return_value = ["memory_id_1"]
        agent.agent_memory = Mock()
        agent.agent_memory.get_memories_by_type.return_value = []
        agent.agent_memory.add_memory.return_value = "memory_id_2"
        agent.local_persistence = Mock()
        agent.local_persistence.get_conversations.return_value = []
        agent.local_persistence.save_conversation = Mock()
        
        # Run query
        response = agent.run(
            query="Test query",
            conversation_id="test_conv_123",
            user_id="test_user_123"
        )
        
        # Verify memory storage was called
        assert agent.local_persistence.save_conversation.called
        assert agent.memory.index_texts.called
        assert agent.agent_memory.add_memory.called
    
    @patch('iceburg.agents.secretary.provider_factory')
    def test_conversation_continuity(self, mock_provider_factory, mock_config):
        """Test that conversation history is retrieved and used"""
        # Mock provider
        mock_provider = Mock()
        mock_provider.chat_complete.return_value = "Test response"
        mock_provider_factory.return_value = mock_provider
        
        # Create agent with memory
        agent = SecretaryAgent(mock_config, enable_memory=True)
        
        # Mock conversation history
        agent.local_persistence = Mock()
        agent.local_persistence.get_conversations.return_value = [
            {
                "user_message": "What is AI?",
                "assistant_message": "AI is artificial intelligence.",
                "timestamp": "2025-01-01T00:00:00"
            }
        ]
        agent.memory = Mock()
        agent.memory.search.return_value = []
        agent.agent_memory = Mock()
        agent.agent_memory.get_memories_by_type.return_value = []
        
        # Run query
        response = agent.run(
            query="Tell me more",
            conversation_id="test_conv_123"
        )
        
        # Verify conversation history was retrieved
        agent.local_persistence.get_conversations.assert_called_with(
            conversation_id="test_conv_123",
            limit=10
        )
        assert response == "Test response"
    
    @patch('iceburg.agents.secretary.provider_factory')
    def test_cross_session_memory(self, mock_provider_factory, mock_config):
        """Test that cross-session memory is retrieved using user_id"""
        # Mock provider
        mock_provider = Mock()
        mock_provider.chat_complete.return_value = "Test response"
        mock_provider_factory.return_value = mock_provider
        
        # Create agent with memory
        agent = SecretaryAgent(mock_config, enable_memory=True)
        
        # Mock long-term memory
        agent.memory = Mock()
        agent.memory.search.return_value = [
            {
                "document": "User prefers detailed explanations",
                "metadata": {"user_id": "test_user_123", "timestamp": "2025-01-01T00:00:00"}
            }
        ]
        agent.agent_memory = Mock()
        agent.agent_memory.get_memories_by_type.return_value = []
        agent.local_persistence = Mock()
        agent.local_persistence.get_conversations.return_value = []
        
        # Run query
        response = agent.run(
            query="Explain quantum computing",
            user_id="test_user_123"
        )
        
        # Verify long-term memory was searched
        agent.memory.search.assert_called()
        assert response == "Test response"
    
    @patch('iceburg.agents.secretary.provider_factory')
    def test_memory_context_building(self, mock_provider_factory, mock_config):
        """Test that memory context is properly built"""
        # Mock provider
        mock_provider = Mock()
        mock_provider.chat_complete.return_value = "Test response"
        mock_provider_factory.return_value = mock_provider
        
        # Create agent with memory
        agent = SecretaryAgent(mock_config, enable_memory=True)
        
        # Mock memories
        agent.local_persistence = Mock()
        agent.local_persistence.get_conversations.return_value = [
            {
                "user_message": "Hello",
                "assistant_message": "Hi there!",
                "timestamp": "2025-01-01T00:00:00"
            }
        ]
        agent.memory = Mock()
        agent.memory.search.return_value = []
        agent.agent_memory = Mock()
        agent.agent_memory.get_memories_by_type.return_value = []
        
        # Run query and check that context is built
        response = agent.run(
            query="What did we discuss?",
            conversation_id="test_conv_123"
        )
        
        # Verify provider was called with context
        call_args = mock_provider.chat_complete.call_args
        prompt = call_args[1]["prompt"]
        assert "PREVIOUS CONTEXT" in prompt or "Recent conversation" in prompt
    
    def test_backward_compatibility_run_function(self, mock_config):
        """Test that original run() function still works without memory"""
        with patch('iceburg.agents.secretary.provider_factory') as mock_provider_factory:
            mock_provider = Mock()
            mock_provider.chat_complete.return_value = "Test response"
            mock_provider_factory.return_value = mock_provider
            
            # Call original run() function without conversation_id/user_id
            response = run(
                cfg=mock_config,
                query="Test query"
            )
            
            assert response == "Test response"
    
    @patch('iceburg.agents.secretary.provider_factory')
    def test_memory_error_handling(self, mock_provider_factory, mock_config):
        """Test that memory errors don't break the agent"""
        # Mock provider
        mock_provider = Mock()
        mock_provider.chat_complete.return_value = "Test response"
        mock_provider_factory.return_value = mock_provider
        
        # Create agent with memory
        agent = SecretaryAgent(mock_config, enable_memory=True)
        
        # Mock memory systems to raise errors
        agent.local_persistence = Mock()
        agent.local_persistence.get_conversations.side_effect = Exception("Memory error")
        agent.memory = Mock()
        agent.memory.search.side_effect = Exception("Memory error")
        agent.agent_memory = Mock()
        agent.agent_memory.get_memories_by_type.side_effect = Exception("Memory error")
        
        # Should still work without memory
        response = agent.run(
            query="Test query",
            conversation_id="test_conv_123"
        )
        
        assert response == "Test response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

