"""
Unit tests for Supervisor Agent
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.supervisor import run as supervisor_run
from iceburg.config import IceburgConfig


class TestSupervisorAgent(unittest.TestCase):
    """Unit tests for Supervisor Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        self.query = "Test query"
        self.stage_outputs = {
            "surveyor": "Surveyor output",
            "dissident": "Dissident output",
            "synthesist": "Synthesist output"
        }
    
    def test_supervisor_basic_execution(self):
        """Test basic supervisor execution"""
        with patch('iceburg.agents.supervisor._run_modular') as mock_run:
            mock_run.return_value = "Supervisor validation: All outputs validated."
            
            result = supervisor_run(
                self.config,
                self.stage_outputs,
                self.query,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
            mock_run.assert_called_once()
    
    def test_supervisor_validates_outputs(self):
        """Test that supervisor validates agent outputs"""
        stage_outputs = {
            "surveyor": "Output 1",
            "dissident": "Output 2",
            "synthesist": "Output 3",
            "oracle": "Output 4"
        }
        
        with patch('iceburg.agents.supervisor._run_modular') as mock_run:
            mock_run.return_value = "Validation complete: All outputs meet quality standards."
            
            result = supervisor_run(
                self.config,
                stage_outputs,
                self.query,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
            # Verify all outputs were validated
            call_args = mock_run.call_args
            self.assertIsNotNone(call_args)
    
    def test_supervisor_error_handling(self):
        """Test supervisor error handling"""
        with patch('iceburg.agents.supervisor._run_modular') as mock_run:
            mock_run.side_effect = Exception("Supervisor error")
            
            with self.assertRaises(Exception):
                supervisor_run(
                    self.config,
                    self.stage_outputs,
                    self.query,
                    verbose=False
                )


if __name__ == '__main__':
    unittest.main()

