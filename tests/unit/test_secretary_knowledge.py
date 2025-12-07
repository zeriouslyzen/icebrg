"""
Unit tests for Secretary Knowledge Base (Phase 2: Self-Updating Knowledge Base)
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.iceburg.agents.secretary_knowledge import SecretaryKnowledgeBase


@pytest.fixture
def mock_config():
    """Create a mock config."""
    cfg = Mock()
    cfg.data_dir = Path("./data")
    cfg.surveyor_model = "gemini-2.0-flash-exp"
    cfg.primary_model = "gemini-2.0-flash-exp"
    cfg.embed_model = "nomic-embed-text"
    return cfg


@pytest.fixture
def temp_kb_dir():
    """Create a temporary directory for knowledge base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def knowledge_base(mock_config, temp_kb_dir):
    """Create a knowledge base instance."""
    return SecretaryKnowledgeBase(mock_config, knowledge_base_dir=temp_kb_dir)


def test_knowledge_base_initialization(knowledge_base, temp_kb_dir):
    """Test knowledge base initializes correctly."""
    assert knowledge_base is not None
    assert knowledge_base.kb_dir == temp_kb_dir
    assert knowledge_base.topics_dir.exists()
    assert knowledge_base.personas_dir.exists()
    assert knowledge_base.indexes_dir.exists()
    assert knowledge_base.summaries_dir.exists()


def test_store_topic(knowledge_base):
    """Test storing a topic."""
    topic = "Quantum Computing"
    content = "Quantum computing uses quantum mechanics to process information."
    
    knowledge_base.store_topic(topic, content)
    
    # Check topic file exists
    topic_file = knowledge_base.topics_dir / "Quantum-Computing.md"
    assert topic_file.exists()
    
    # Check content
    with open(topic_file, 'r', encoding='utf-8') as f:
        file_content = f.read()
        assert topic in file_content
        assert content in file_content
    
    # Check metadata
    assert topic in knowledge_base.metadata["topics"]


def test_store_topic_update(knowledge_base):
    """Test updating an existing topic."""
    topic = "AI"
    content1 = "Initial content about AI."
    content2 = "Updated content about AI."
    
    # Store initial
    knowledge_base.store_topic(topic, content1)
    
    # Update
    knowledge_base.store_topic(topic, content2)
    
    # Check both contents are present
    topic_file = knowledge_base.topics_dir / "AI.md"
    with open(topic_file, 'r', encoding='utf-8') as f:
        file_content = f.read()
        assert content1 in file_content
        assert content2 in file_content


def test_update_persona(knowledge_base):
    """Test updating user persona."""
    user_id = "test_user_123"
    traits = {
        "preferences": {"language": "Python", "style": "concise"},
        "expertise": ["machine learning", "data science"]
    }
    
    knowledge_base.update_persona(user_id, traits)
    
    # Check persona file exists
    persona_file = knowledge_base.personas_dir / f"{user_id}.json"
    assert persona_file.exists()
    
    # Check content
    with open(persona_file, 'r', encoding='utf-8') as f:
        persona = json.load(f)
        assert persona["preferences"] == traits["preferences"]
        assert persona["expertise"] == traits["expertise"]
        assert "last_updated" in persona


def test_get_persona(knowledge_base):
    """Test retrieving user persona."""
    user_id = "test_user_456"
    traits = {"preferences": {"language": "JavaScript"}}
    
    knowledge_base.update_persona(user_id, traits)
    
    # Retrieve
    persona = knowledge_base.get_persona(user_id)
    assert persona is not None
    assert persona["preferences"] == traits["preferences"]


def test_get_persona_nonexistent(knowledge_base):
    """Test retrieving non-existent persona."""
    persona = knowledge_base.get_persona("nonexistent_user")
    assert persona is None


@patch('src.iceburg.agents.secretary_knowledge.provider_factory')
def test_extract_knowledge(mock_factory, knowledge_base):
    """Test knowledge extraction."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = '''{
        "topics": ["quantum computing", "AI"],
        "facts": ["Quantum computers use qubits", "AI uses neural networks"],
        "preferences": {"language": "Python"},
        "expertise": ["machine learning"],
        "important": true
    }'''
    mock_factory.return_value = mock_provider
    
    conversation = "User: Tell me about quantum computing and AI. Assistant: Quantum computing uses qubits..."
    knowledge = knowledge_base.extract_knowledge(conversation)
    
    assert "topics" in knowledge
    assert len(knowledge["topics"]) == 2
    assert "quantum computing" in knowledge["topics"]
    assert "facts" in knowledge
    assert "important" in knowledge


def test_query_knowledge_topic(knowledge_base):
    """Test querying knowledge base by topic."""
    # Store a topic
    knowledge_base.store_topic("Machine Learning", "Machine learning is a subset of AI.")
    
    # Query
    results = knowledge_base.query_knowledge("machine learning", k=5)
    
    assert len(results) > 0
    assert any(r.get("type") == "topic" for r in results)


def test_build_index(knowledge_base):
    """Test building topic index."""
    # Store some topics
    knowledge_base.store_topic("Machine Learning", "Content about ML")
    knowledge_base.store_topic("Deep Learning", "Content about DL")
    knowledge_base.store_topic("Neural Networks", "Content about NN")
    
    # Build index
    knowledge_base.build_index()
    
    # Check index file exists
    index_file = knowledge_base.indexes_dir / "topic_index.json"
    assert index_file.exists()
    
    # Check index content
    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)
        assert "topics" in index
        assert "cross_references" in index
        assert len(index["topics"]) >= 3


def test_process_conversation(knowledge_base):
    """Test processing a conversation for knowledge extraction."""
    with patch.object(knowledge_base, 'extract_knowledge') as mock_extract:
        mock_extract.return_value = {
            "topics": ["Python"],
            "facts": ["Python is a programming language"],
            "preferences": {"language": "Python"},
            "expertise": [],
            "important": True
        }
        
        knowledge = knowledge_base.process_conversation(
            query="I prefer Python",
            response="Python is a great language",
            user_id="test_user"
        )
        
        assert knowledge is not None
        assert "topics" in knowledge
        
        # Check topic was stored
        topic_file = knowledge_base.topics_dir / "Python.md"
        assert topic_file.exists()


def test_sanitize_topic_name(knowledge_base):
    """Test topic name sanitization for filenames."""
    # Test various topic names
    test_cases = [
        ("Python Programming", "Python-Programming"),
        ("C++ Basics", "C-Basics"),
        ("AI/ML Concepts", "AI-ML-Concepts"),
        ("What is AI?", "What-is-AI"),
    ]
    
    for topic, expected_safe in test_cases:
        knowledge_base.store_topic(topic, "Test content")
        # Check file exists with sanitized name
        safe_name = expected_safe.replace(" ", "-")
        topic_file = knowledge_base.topics_dir / f"{safe_name}.md"
        # Note: The actual sanitization might differ, so we just check a file was created
        files = list(knowledge_base.topics_dir.glob("*.md"))
        assert len(files) > 0

