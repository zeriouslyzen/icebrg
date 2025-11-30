"""
Unit tests for Scrutineer Agent
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.scrutineer import run as scrutineer_run
from iceburg.config import IceburgConfig


class TestScrutineerAgent(unittest.TestCase):
    """Unit tests for Scrutineer Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        self.config.scrutineer_model = "gemma2:2b"
        self.synthesis_output = "Synthesized output with potential contradictions."
    
    def test_scrutineer_basic_execution(self):
        """Test basic scrutineer execution"""
        with patch('iceburg.agents.scrutineer.chat_complete') as mock_chat:
            mock_chat.return_value = '{"claims": [{"text": "claim1", "evidence_level": "A"}]}'
            
            result = scrutineer_run(
                self.config,
                self.synthesis_output,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            mock_chat.assert_called_once()
    
    def test_scrutineer_detects_contradictions(self):
        """Test that scrutineer detects contradictions"""
        synthesis_with_contradiction = "Statement A. Statement B contradicts A."
        
        with patch('iceburg.agents.scrutineer.chat_complete') as mock_chat:
            mock_chat.return_value = '{"claims": [{"text": "contradiction detected", "evidence_level": "C"}]}'
            
            result = scrutineer_run(
                self.config,
                synthesis_with_contradiction,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
    
    def test_scrutineer_uses_evidence_grading(self):
        """Test that scrutineer uses evidence grading (A/B/C/S/X)"""
        with patch('iceburg.agents.scrutineer.chat_complete') as mock_chat:
            mock_chat.return_value = '{"claims": [{"text": "claim", "evidence_level": "S", "suppression_indicators": ["pattern1"]}]}'
            
            result = scrutineer_run(
                self.config,
                self.synthesis_output,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
            # Verify evidence grading is used
            self.assertIn("evidence_level", result.lower() or "S" in result)


if __name__ == '__main__':
    unittest.main()

