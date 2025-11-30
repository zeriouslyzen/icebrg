"""
Unit tests for Synthesist Agent
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.synthesist import run as synthesist_run
from iceburg.config import IceburgConfig


class TestSynthesistAgent(unittest.TestCase):
    """Unit tests for Synthesist Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        self.config.synthesist_model = "gemma2:2b"
        
        self.enhanced_context = {
            "surveyor": "Surveyor analysis: AI systems are designed for control.",
            "dissident": "Dissident analysis: Control may be an illusion."
        }
    
    def test_synthesist_basic_execution(self):
        """Test basic synthesist execution"""
        with patch('iceburg.agents.synthesist.chat_complete') as mock_chat:
            mock_chat.return_value = "Synthesis: AI control is complex, involving both design and emergent properties."
            
            result = synthesist_run(
                self.config,
                self.enhanced_context,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            mock_chat.assert_called_once()
    
    def test_synthesist_integrates_multiple_sources(self):
        """Test that synthesist integrates multiple agent outputs"""
        enhanced_context = {
            "surveyor": "Surveyor output 1",
            "dissident": "Dissident output 2",
            "archaeologist": "Archaeologist output 3"
        }
        
        with patch('iceburg.agents.synthesist.chat_complete') as mock_chat:
            mock_chat.return_value = "Integrated synthesis of all sources."
            
            result = synthesist_run(self.config, enhanced_context, verbose=False)
            
            self.assertIsInstance(result, str)
            # Verify all sources were considered
            call_args = str(mock_chat.call_args)
            self.assertIn("surveyor", call_args.lower())
            self.assertIn("dissident", call_args.lower())
    
    def test_synthesist_with_multimodal_evidence(self):
        """Test synthesist with multimodal evidence"""
        multimodal_evidence = [
            {"type": "image", "data": "image_data"},
            {"type": "text", "data": "text_data"}
        ]
        
        with patch('iceburg.agents.synthesist.chat_complete') as mock_chat:
            mock_chat.return_value = "Synthesis with multimodal evidence."
            
            result = synthesist_run(
                self.config,
                self.enhanced_context,
                verbose=False,
                multimodal_evidence=multimodal_evidence
            )
            
            self.assertIsInstance(result, str)
            mock_chat.assert_called_once()
    
    def test_synthesist_error_handling(self):
        """Test synthesist error handling"""
        with patch('iceburg.agents.synthesist.chat_complete') as mock_chat:
            mock_chat.side_effect = Exception("Synthesis error")
            
            with self.assertRaises(Exception):
                synthesist_run(self.config, self.enhanced_context, verbose=False)


if __name__ == '__main__':
    unittest.main()

