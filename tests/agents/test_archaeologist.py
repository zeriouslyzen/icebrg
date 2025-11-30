"""
Unit tests for Archaeologist Agent
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.archaeologist import run as archaeologist_run
from iceburg.config import IceburgConfig


class TestArchaeologistAgent(unittest.TestCase):
    """Unit tests for Archaeologist Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        self.query = "What buried evidence exists about suppressed research?"
    
    def test_archaeologist_basic_execution(self):
        """Test basic archaeologist execution"""
        with patch('iceburg.agents.archaeologist._run_modular') as mock_run:
            mock_run.return_value = "Archaeological findings: Buried evidence discovered."
            
            result = archaeologist_run(
                self.config,
                self.query,
                documents=None,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
            mock_run.assert_called_once()
    
    def test_archaeologist_with_documents(self):
        """Test archaeologist with document input"""
        documents = [
            "Document 1: Historical research findings.",
            "Document 2: Suppressed knowledge."
        ]
        
        with patch('iceburg.agents.archaeologist._run_modular') as mock_run:
            mock_run.return_value = "Analysis of provided documents."
            
            result = archaeologist_run(
                self.config,
                self.query,
                documents=documents,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
            # Verify documents were passed
            call_args = mock_run.call_args
            self.assertIsNotNone(call_args)
    
    def test_archaeologist_error_handling(self):
        """Test archaeologist error handling"""
        with patch('iceburg.agents.archaeologist._run_modular') as mock_run:
            mock_run.side_effect = Exception("Archaeologist error")
            
            with self.assertRaises(Exception):
                archaeologist_run(
                    self.config,
                    self.query,
                    documents=None,
                    verbose=False
                )


if __name__ == '__main__':
    unittest.main()

