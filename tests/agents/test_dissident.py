"""
Unit tests for Dissident Agent
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.dissident import run as dissident_run
from iceburg.config import IceburgConfig


class TestDissidentAgent(unittest.TestCase):
    """Unit tests for Dissident Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        self.config.dissident_model = "gemma2:2b"
        self.query = "What are the assumptions in AI safety research?"
        self.surveyor_output = "AI safety research assumes that AI systems can be controlled and aligned with human values."
    
    def test_dissident_basic_execution(self):
        """Test basic dissident execution"""
        with patch('iceburg.agents.dissident.chat_complete') as mock_chat:
            mock_chat.return_value = "Alternative perspective: AI systems may be fundamentally uncontrollable."
            
            result = dissident_run(self.config, self.query, self.surveyor_output, verbose=False)
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            mock_chat.assert_called_once()
    
    def test_dissident_challenges_assumptions(self):
        """Test that dissident challenges assumptions"""
        with patch('iceburg.agents.dissident.chat_complete') as mock_chat:
            mock_chat.return_value = "Assumption challenged: Control may be an illusion."
            
            result = dissident_run(self.config, self.query, self.surveyor_output, verbose=False)
            
            # Verify dissident output contains alternative perspectives
            self.assertIsInstance(result, str)
            # Check that chat_complete was called with correct system prompt
            call_args = mock_chat.call_args
            self.assertIn("DISSIDENT_SYSTEM", str(call_args))
    
    def test_dissident_with_empty_surveyor_output(self):
        """Test dissident with empty surveyor output"""
        with patch('iceburg.agents.dissident.chat_complete') as mock_chat:
            mock_chat.return_value = "Alternative analysis without consensus view."
            
            result = dissident_run(self.config, self.query, "", verbose=False)
            
            self.assertIsInstance(result, str)
            mock_chat.assert_called_once()
    
    def test_dissident_error_handling(self):
        """Test dissident error handling"""
        with patch('iceburg.agents.dissident.chat_complete') as mock_chat:
            mock_chat.side_effect = Exception("LLM error")
            
            # Should raise exception
            with self.assertRaises(Exception):
                dissident_run(self.config, self.query, self.surveyor_output, verbose=False)


if __name__ == '__main__':
    unittest.main()

