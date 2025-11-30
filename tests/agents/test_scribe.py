"""
Unit tests for Scribe Agent
"""

import unittest
from unittest.mock import Mock, patch
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.scribe import run as scribe_run
from iceburg.config import IceburgConfig


class TestScribeAgent(unittest.TestCase):
    """Unit tests for Scribe Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        
        self.oracle_output = json.dumps({
            "principle_name": "Test Principle",
            "one_sentence_summary": "Test principle for documentation",
            "domains": ["research", "AI"],
            "predictions": ["prediction1", "prediction2"]
        })
    
    def test_scribe_basic_execution(self):
        """Test basic scribe execution"""
        result = scribe_run(self.config, self.oracle_output, verbose=False)
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_scribe_generates_documentation(self):
        """Test that scribe generates structured documentation"""
        result = scribe_run(self.config, self.oracle_output, verbose=False)
        
        # Verify documentation structure
        self.assertIsInstance(result, str)
        # Should contain knowledge outputs
        self.assertGreater(len(result), 100)
    
    def test_scribe_handles_invalid_json(self):
        """Test scribe handles invalid JSON gracefully"""
        invalid_oracle_output = "Invalid JSON format"
        
        result = scribe_run(self.config, invalid_oracle_output, verbose=False)
        
        # Should still generate fallback output
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()

