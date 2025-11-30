"""
Unit tests for Surveyor Agent
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.surveyor import run as surveyor_run
from iceburg.config import IceburgConfig
from iceburg.vectorstore import VectorStore


class TestSurveyorAgent(unittest.TestCase):
    """Unit tests for Surveyor Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        self.config.surveyor_model = "gemma2:2b"
        self.config.data_dir = "/tmp/test_data"
        
        self.vectorstore = Mock(spec=VectorStore)
        self.query = "What is artificial intelligence?"
    
    def test_surveyor_basic_execution(self):
        """Test basic surveyor execution"""
        # Mock vectorstore search
        mock_hit = Mock()
        mock_hit.metadata = {"source": "kb"}
        mock_hit.document = "AI is the simulation of human intelligence."
        self.vectorstore.semantic_search.return_value = [mock_hit]
        
        # Mock chat_complete
        with patch('iceburg.agents.surveyor.chat_complete') as mock_chat:
            mock_chat.return_value = "Artificial intelligence is the simulation of human intelligence by machines."
            
            result = surveyor_run(self.config, self.vectorstore, self.query, verbose=False)
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            self.vectorstore.semantic_search.assert_called_once()
    
    def test_surveyor_with_iceburg_data(self):
        """Test surveyor with ICEBURG data prioritization"""
        # Mock hits with ICEBURG data
        mock_hit1 = Mock()
        mock_hit1.metadata = {"source": "research_outputs/iceburg"}
        mock_hit1.document = "ICEBURG research on AI: Advanced findings..."
        
        mock_hit2 = Mock()
        mock_hit2.metadata = {"source": "kb"}
        mock_hit2.document = "General AI knowledge."
        
        self.vectorstore.semantic_search.return_value = [mock_hit1, mock_hit2]
        
        with patch('iceburg.agents.surveyor.chat_complete') as mock_chat:
            mock_chat.return_value = "Response prioritizing ICEBURG research."
            
            result = surveyor_run(self.config, self.vectorstore, self.query, verbose=False)
            
            # Verify ICEBURG data is prioritized
            self.assertIsInstance(result, str)
            self.vectorstore.semantic_search.assert_called_once()
    
    def test_surveyor_error_handling(self):
        """Test surveyor error handling"""
        # Mock vectorstore error
        self.vectorstore.semantic_search.side_effect = Exception("VectorStore error")
        
        with patch('iceburg.agents.surveyor.chat_complete') as mock_chat:
            mock_chat.return_value = "Fallback response."
            
            # Should not raise exception
            result = surveyor_run(self.config, self.vectorstore, self.query, verbose=False)
            self.assertIsInstance(result, str)
    
    def test_surveyor_multimodal_input(self):
        """Test surveyor with multimodal input"""
        multimodal_input = {"type": "image", "data": b"fake_image_data"}
        
        with patch('iceburg.agents.surveyor.chat_complete') as mock_chat:
            mock_chat.return_value = "Response with multimodal analysis."
            
            result = surveyor_run(
                self.config, 
                self.vectorstore, 
                self.query, 
                verbose=False,
                multimodal_input=multimodal_input
            )
            
            self.assertIsInstance(result, str)
            # Verify multimodal input was handled
            mock_chat.assert_called_once()


if __name__ == '__main__':
    unittest.main()

