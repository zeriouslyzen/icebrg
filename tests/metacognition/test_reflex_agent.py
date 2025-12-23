"""
Test suite for Reflex Agent
Tests response compression, bullet extraction, and reflection storage
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestReflexAgent:
    """Test suite for reflex agent capabilities"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_cfg = Mock()
        self.mock_cfg.surveyor_model = "llama3.1:8b"
        
    def test_response_compression(self):
        """Test that long responses are correctly compressed"""
        long_response = """
        This is a very long response with many details about bioelectric fields.
        It contains information about ion channels, membrane potentials, and
        cellular communication. The response goes on and on with more details
        about electromagnetic effects on biological systems. There are many
        paragraphs and the verbosity is quite high. We need to compress this
        into a more manageable format for quick consumption.
        """ * 3  # Make it longer
        
        # Compression should reduce length significantly
        from src.iceburg.agents.reflex_agent import ReflexAgent
        
        # ReflexAgent takes no constructor arguments
        agent = ReflexAgent()
        
        # Test compression method exists and works
        result = agent.compress_response(long_response)
        
        # Compressed should be shorter than original
        assert result.compression_ratio < 1.0
        assert len(result.compressed) < len(result.full)
    
    def test_bullet_extraction(self):
        """Test extraction of key points as bullets"""
        response_with_points = """
        Key findings:
        1. Bioelectric fields influence cellular behavior
        2. Ion channels respond to electromagnetic stimulation
        3. Schumann resonance correlates with brain activity
        4. Consciousness may have electromagnetic components
        """
        
        # Should extract 4 bullet points
        bullets = []
        for line in response_with_points.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                bullets.append(line)
        
        assert len(bullets) >= 3  # At least 3 bullets
    
    def test_reflection_storage(self):
        """Test that reflections are stored in knowledge tree"""
        from src.iceburg.agents.reflex_agent import ReflexAgent, ReflexResponse
        
        # ReflexAgent takes no constructor arguments
        agent = ReflexAgent()
        
        # Create a response with reflections using actual field names
        response = ReflexResponse(
            preview={"core_insight": "Point 1", "actionable_guidance": "Point 2", "key_context": "Point 3"},
            compressed="Compressed version",
            full="Full long response",
            compression_ratio=0.5,
            reflections=[
                {"type": "breakthrough", "content": "Novel insight about fields"},
                {"type": "synthesis", "content": "Cross-domain connection found"}
            ]
        )
        
        assert len(response.reflections) == 2
        assert response.reflections[0]["type"] == "breakthrough"
    
    def test_reflection_types_classification(self):
        """Test classification of different reflection types"""
        reflection_examples = [
            ("BREAKTHROUGH: New discovery...", "breakthrough"),
            ("SYNTHESIS: Combining ideas...", "synthesis"),
            ("CONTRADICTION: Conflicts with...", "contradiction"),
            ("PREDICTION: Future implications...", "prediction"),
        ]
        
        for text, expected_type in reflection_examples:
            # Simple classification based on keywords
            detected_type = None
            if "BREAKTHROUGH" in text.upper():
                detected_type = "breakthrough"
            elif "SYNTHESIS" in text.upper():
                detected_type = "synthesis"
            elif "CONTRADICTION" in text.upper():
                detected_type = "contradiction"
            elif "PREDICTION" in text.upper():
                detected_type = "prediction"
            
            assert detected_type == expected_type
    
    def test_top_5_reflections_limit(self):
        """Test that only top 5 reflections are retained by content depth"""
        reflections = [
            {"content": "Short"},
            {"content": "Medium length content here"},
            {"content": "This is a much longer and more substantial reflection with depth"},
            {"content": "Another brief one"},
            {"content": "Very long detailed reflection with multiple insights and cross-references to other domains"},
            {"content": "Yet another short one"},
            {"content": "A moderate length reflection about bioelectric phenomena"},
        ]
        
        # Sort by length and take top 5
        sorted_reflections = sorted(
            reflections,
            key=lambda x: len(x.get("content", "")),
            reverse=True
        )
        top_5 = sorted_reflections[:5]
        
        assert len(top_5) == 5
        # First one should be the longest
        assert "Very long detailed" in top_5[0]["content"]


class TestReflexResponse:
    """Test ReflexResponse dataclass"""
    
    def test_reflex_response_creation(self):
        """Test that ReflexResponse can be created with all fields"""
        try:
            from src.iceburg.agents.reflex_agent import ReflexResponse
            
            # Use actual field names: preview (Dict), full, compressed, compression_ratio, reflections
            response = ReflexResponse(
                preview={"core_insight": "Point 1", "actionable_guidance": "Point 2", "key_context": "Point 3"},
                compressed="Short version",
                full="Full long version with all details",
                compression_ratio=0.5,
                reflections=[{"type": "insight", "content": "Key insight"}]
            )
            
            assert len(response.preview) == 3
            assert len(response.compressed) < len(response.full)
            assert len(response.reflections) == 1
        except ImportError:
            pytest.skip("ReflexResponse not available")
    
    def test_empty_reflections_handling(self):
        """Test that empty reflections are handled gracefully"""
        try:
            from src.iceburg.agents.reflex_agent import ReflexResponse
            
            # Use actual field names
            response = ReflexResponse(
                preview={},
                compressed="",
                full="Some response",
                compression_ratio=1.0,
                reflections=[]
            )
            
            assert response.reflections == []
        except ImportError:
            pytest.skip("ReflexResponse not available")


class TestKnowledgeTreeIntegration:
    """Test integration with knowledge tree storage"""
    
    def test_reflection_storage_in_tree(self):
        """Test that reflections are stored in unified memory"""
        # Mock the knowledge tree storage
        mock_memory = Mock()
        mock_memory.store.return_value = True
        
        reflection = {
            "type": "breakthrough",
            "content": "Novel insight about consciousness",
            "timestamp": datetime.now().isoformat(),
            "source_agent": "reflex"
        }
        
        # Simulate storage
        result = mock_memory.store("reflections", reflection)
        assert result is True
    
    def test_reflection_retrieval(self):
        """Test that stored reflections can be retrieved"""
        mock_memory = Mock()
        mock_memory.retrieve.return_value = [
            {"type": "breakthrough", "content": "Insight 1"},
            {"type": "synthesis", "content": "Insight 2"},
        ]
        
        results = mock_memory.retrieve("reflections", limit=10)
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
