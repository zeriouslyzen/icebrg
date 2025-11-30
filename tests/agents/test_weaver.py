"""
Unit tests for Weaver Agent
"""

import unittest
from unittest.mock import Mock, patch
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.weaver import run as weaver_run
from iceburg.config import IceburgConfig


class TestWeaverAgent(unittest.TestCase):
    """Unit tests for Weaver Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        self.config.weaver_model = "gemma2:2b"
        
        self.oracle_output = json.dumps({
            "principle_name": "Test Principle",
            "one_sentence_summary": "Test principle for code generation",
            "domains": ["software", "AI"],
            "predictions": ["prediction1"],
            "study_design": {
                "manipulation": "test",
                "measurement": "metric",
                "success_criteria": "criteria",
                "minimal_design_risk": "low"
            }
        })
    
    def test_weaver_basic_execution(self):
        """Test basic weaver execution"""
        with patch('iceburg.agents.weaver.chat_complete') as mock_chat:
            mock_chat.return_value = "def test_function():\n    return 'code'"
            
            result = weaver_run(self.config, self.oracle_output, verbose=False)
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
    
    def test_weaver_generates_code(self):
        """Test that weaver generates executable code"""
        with patch('iceburg.agents.weaver.chat_complete') as mock_chat:
            mock_chat.return_value = "def implement_principle():\n    # Implementation\n    pass"
            
            result = weaver_run(self.config, self.oracle_output, verbose=False)
            
            self.assertIsInstance(result, str)
            # Verify code-like structure
            self.assertIn("def", result.lower() or "class" in result.lower())
    
    def test_weaver_handles_invalid_json(self):
        """Test weaver handles invalid JSON gracefully"""
        invalid_oracle_output = "Invalid JSON format"
        
        with patch('iceburg.agents.weaver.chat_complete') as mock_chat:
            mock_chat.return_value = "Fallback code generation."
            
            result = weaver_run(self.config, invalid_oracle_output, verbose=False)
            
            self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()

