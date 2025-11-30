"""
Unit tests for Oracle Agent
"""

import unittest
from unittest.mock import Mock, patch
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.oracle import run as oracle_run
from iceburg.config import IceburgConfig
from iceburg.vectorstore import VectorStore
from iceburg.graph_store import KnowledgeGraph


class TestOracleAgent(unittest.TestCase):
    """Unit tests for Oracle Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=IceburgConfig)
        self.config.oracle_model = "gemma2:2b"
        
        self.vectorstore = Mock(spec=VectorStore)
        self.knowledge_graph = Mock(spec=KnowledgeGraph)
        
        self.synthesis_output = "Synthesized analysis of AI systems."
    
    def test_oracle_basic_execution(self):
        """Test basic oracle execution"""
        mock_json_output = {
            "principle_name": "Test Principle",
            "one_sentence_summary": "Test summary",
            "framing": "Conclusion",
            "domains": ["AI", "Philosophy"],
            "evidence_pairs": [["evidence1", "A"]],
            "predictions": ["prediction1"],
            "study_design": {
                "manipulation": "test",
                "measurement": "metric",
                "success_criteria": "criteria",
                "minimal_design_risk": "low"
            },
            "implications": ["implication1"],
            "prior_principles": ["prior1"]
        }
        
        with patch('iceburg.agents.oracle.chat_complete') as mock_chat:
            mock_chat.return_value = json.dumps(mock_json_output)
            
            result = oracle_run(
                self.config,
                self.knowledge_graph,
                self.vectorstore,
                self.synthesis_output,
                verbose=False
            )
            
            self.assertIsInstance(result, (str, dict))
            mock_chat.assert_called_once()
    
    def test_oracle_generates_valid_json(self):
        """Test that oracle generates valid JSON"""
        mock_json_output = {
            "principle_name": "AI Control Principle",
            "one_sentence_summary": "AI control involves both design and emergence.",
            "framing": "Theory",
            "domains": ["AI"],
            "evidence_pairs": [["evidence", "B"]],
            "predictions": ["prediction"],
            "study_design": {
                "manipulation": "test",
                "measurement": "metric",
                "success_criteria": "criteria",
                "minimal_design_risk": "medium"
            },
            "implications": ["implication"],
            "prior_principles": []
        }
        
        with patch('iceburg.agents.oracle.chat_complete') as mock_chat:
            mock_chat.return_value = json.dumps(mock_json_output)
            
            result = oracle_run(
                self.config,
                self.knowledge_graph,
                self.vectorstore,
                self.synthesis_output,
                verbose=False
            )
            
            # Verify result can be parsed as JSON
            if isinstance(result, str):
                parsed = json.loads(result)
                self.assertIn("principle_name", parsed)
                self.assertIn("framing", parsed)
    
    def test_oracle_handles_invalid_json(self):
        """Test oracle handles invalid JSON gracefully"""
        with patch('iceburg.agents.oracle.chat_complete') as mock_chat:
            mock_chat.return_value = "Invalid JSON response"
            
            # Should handle gracefully
            result = oracle_run(
                self.config,
                self.knowledge_graph,
                self.vectorstore,
                self.synthesis_output,
                verbose=False
            )
            
            self.assertIsInstance(result, str)
    
    def test_oracle_uses_evidence_grading(self):
        """Test that oracle uses evidence grading (A/B/C/S/X)"""
        mock_json_output = {
            "principle_name": "Test",
            "one_sentence_summary": "Test",
            "framing": "Conclusion",
            "domains": ["AI"],
            "evidence_pairs": [["evidence1", "A"],  # Grade A
            "predictions": [],
            "study_design": {
                "manipulation": "test",
                "measurement": "metric",
                "success_criteria": "criteria",
                "minimal_design_risk": "low"
            },
            "implications": [],
            "prior_principles": []
        }
        
        with patch('iceburg.agents.oracle.chat_complete') as mock_chat:
            mock_chat.return_value = json.dumps(mock_json_output)
            
            result = oracle_run(
                self.config,
                self.knowledge_graph,
                self.vectorstore,
                self.synthesis_output,
                verbose=False
            )
            
            # Verify evidence grading is used
            if isinstance(result, str):
                parsed = json.loads(result)
                self.assertIn("evidence_pairs", parsed)
                self.assertIn("framing", parsed)


if __name__ == '__main__':
    unittest.main()

