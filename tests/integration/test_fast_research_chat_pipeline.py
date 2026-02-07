"""
Integration test: Fast mode -> Research -> Fast Chat with last_research.
Verifies: Secretary in Fast mode can receive last_research_summary and build on it
for coherence and conversation continuity.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.iceburg.agents.secretary import run as secretary_run
from src.iceburg.config import IceburgConfig


@pytest.fixture
def mock_config():
    cfg = Mock(spec=IceburgConfig)
    cfg.data_dir = Path(tempfile.mkdtemp())
    cfg.surveyor_model = "llama3.1:8b"
    cfg.primary_model = "llama3.1:8b"
    cfg.llm_provider = "ollama"
    cfg.timeout_s = 30
    cfg.embed_model = "nomic-embed-text"
    return cfg


@patch("src.iceburg.providers.factory.provider_factory")
@patch("src.iceburg.agents.secretary.SecretaryAgent._needs_tools", return_value=False)
@patch("src.iceburg.search.answer_query", side_effect=Exception("skip search for test"))
def test_secretary_fast_then_chat_with_last_research(mock_answer, mock_needs_tools, mock_factory, mock_config):
    """Fast mode research question -> then Fast Chat with last_research should be coherent."""
    mock_provider = Mock()
    # Provide enough returns for any number of chat_complete calls (first run + second run)
    r1 = "I can run a full research pipeline with Surveyor, Dissident, Synthesist, and Oracle. Click 'Run full research' to proceed."
    r2 = "Based on the research report: the main finding was that astrology-organ links are debated. The Surveyor found historical claims; the Dissident noted confirmation bias. I'd suggest we look at peer-reviewed studies next."
    mock_provider.chat_complete.side_effect = [r1, r2, r2, r2, r2]
    mock_factory.return_value = mock_provider

    # 1) Fast mode: research-style question (no last_research)
    out1 = secretary_run(
        mock_config,
        "What do we know about connections between astrology and organs?",
        verbose=False,
        conversation_id="pipe_test_1",
        last_research_summary=None,
    )
    assert out1 is not None
    assert mock_provider.chat_complete.call_count >= 1

    # 2) Fast Chat with last_research: follow-up as if research just completed
    fake_report = (
        "Research report: Surveyor found historical links between zodiac signs and body parts (Aries/head, etc.). "
        "Dissident noted lack of controlled studies. Synthesist: correlation is cultural, not causal. "
        "Oracle: evidence is weak; more RCTs needed."
    )
    out2 = secretary_run(
        mock_config,
        "Summarize the main finding and what we should do next.",
        verbose=False,
        conversation_id="pipe_test_1",
        last_research_summary=fake_report,
    )
    assert out2 is not None
    lower = out2.lower()
    assert "research" in lower or "surveyor" in lower or "finding" in lower or "report" in lower or "evidence" in lower


@patch("src.iceburg.providers.factory.provider_factory")
@patch("src.iceburg.agents.secretary.SecretaryAgent._needs_tools", return_value=False)
def test_secretary_last_research_injected_in_prompt(mock_needs_tools, mock_factory, mock_config):
    """last_research_summary is accepted and prompt passed to provider contains it."""
    mock_provider = Mock()
    mock_provider.chat_complete.return_value = "The research shows three key points."
    mock_factory.return_value = mock_provider

    summary = "LAST RESEARCH: The United Nations is run by the Secretary-General and Member States."
    out = secretary_run(
        mock_config,
        "Who runs the UN according to the research?",
        verbose=False,
        last_research_summary=summary,
    )
    assert out is not None
    assert mock_provider.chat_complete.called
    call_kwargs = mock_provider.chat_complete.call_args
    prompt = (call_kwargs.kwargs or {}).get("prompt", "") or ""
    assert "United Nations" in prompt or "Secretary-General" in prompt or "LAST RESEARCH" in prompt
