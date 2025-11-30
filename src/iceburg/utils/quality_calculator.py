"""
Quality Score Calculator
Calculates dynamic quality scores for ICEBURG responses based on multiple factors.
"""

from typing import Dict, Any, Optional, List
import re
import logging

logger = logging.getLogger(__name__)


def calculate_quality_score(
    response_text: str,
    query_text: str,
    agent_results: Optional[Dict[str, Any]] = None,
    response_time: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate quality score (0.0-1.0) for ICEBURG response.
    
    Factors:
    - Response completeness (length, structure)
    - Quality indicators (evidence, analysis, conclusions)
    - Agent coordination (multi-agent results)
    - Response time (efficiency)
    - Query complexity matching
    
    Args:
        response_text: The response text to evaluate
        query_text: The original query
        agent_results: Results from multiple agents (optional)
        response_time: Response time in seconds (optional)
        metadata: Additional metadata (optional)
    
    Returns:
        Quality score between 0.0 and 1.0
    """
    if not response_text or not response_text.strip():
        return 0.0
    
    score = 0.0
    factors = {}
    
    # Factor 1: Response Completeness (0-0.3)
    completeness_score = _calculate_completeness(response_text, query_text)
    score += completeness_score * 0.3
    factors["completeness"] = completeness_score
    
    # Factor 2: Quality Indicators (0-0.3)
    quality_indicators_score = _calculate_quality_indicators(response_text)
    score += quality_indicators_score * 0.3
    factors["quality_indicators"] = quality_indicators_score
    
    # Factor 3: Agent Coordination (0-0.2)
    if agent_results:
        coordination_score = _calculate_agent_coordination(agent_results)
        score += coordination_score * 0.2
        factors["agent_coordination"] = coordination_score
    else:
        factors["agent_coordination"] = 0.5  # Default if no agent results
    
    # Factor 4: Response Efficiency (0-0.1)
    if response_time:
        efficiency_score = _calculate_efficiency(response_time, len(response_text))
        score += efficiency_score * 0.1
        factors["efficiency"] = efficiency_score
    else:
        factors["efficiency"] = 0.5  # Default if no time data
    
    # Factor 5: Query Complexity Matching (0-0.1)
    complexity_match_score = _calculate_complexity_match(response_text, query_text)
    score += complexity_match_score * 0.1
    factors["complexity_match"] = complexity_match_score
    
    # Clamp to [0.0, 1.0]
    final_score = max(0.0, min(1.0, score))
    
    if metadata:
        metadata["quality_factors"] = factors
    
    return final_score


def _calculate_completeness(response_text: str, query_text: str) -> float:
    """Calculate completeness score based on response length and structure."""
    score = 0.0
    
    # Length factor (0-0.5)
    word_count = len(response_text.split())
    if word_count < 50:
        length_score = 0.2
    elif word_count < 200:
        length_score = 0.4
    elif word_count < 500:
        length_score = 0.6
    elif word_count < 1000:
        length_score = 0.8
    else:
        length_score = 1.0
    
    # Structure factor (0-0.5)
    structure_indicators = [
        r"\n\n",  # Paragraphs
        r"##|###",  # Headers
        r"[-*]",  # Lists
        r"\d+\.",  # Numbered lists
    ]
    
    structure_count = sum(1 for pattern in structure_indicators if re.search(pattern, response_text))
    structure_score = min(1.0, structure_count / 3.0)  # Normalize to 0-1
    
    # Combine length and structure
    score = (length_score * 0.6) + (structure_score * 0.4)
    
    return score


def _calculate_quality_indicators(response_text: str) -> float:
    """Calculate quality based on presence of quality indicators."""
    response_lower = response_text.lower()
    
    # Evidence indicators
    evidence_indicators = [
        "evidence", "research", "study", "data", "analysis",
        "findings", "results", "conclusion", "summary",
        "according to", "based on", "shows that"
    ]
    
    evidence_count = sum(1 for indicator in evidence_indicators if indicator in response_lower)
    evidence_score = min(1.0, evidence_count / 5.0)  # Normalize to 0-1
    
    # Reasoning indicators
    reasoning_indicators = [
        "because", "therefore", "however", "furthermore",
        "in addition", "consequently", "thus", "hence"
    ]
    
    reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
    reasoning_score = min(1.0, reasoning_count / 4.0)  # Normalize to 0-1
    
    # Synthesis indicators
    synthesis_indicators = [
        "synthesize", "combine", "integrate", "unify",
        "merge", "consolidate", "coordinate"
    ]
    
    synthesis_count = sum(1 for indicator in synthesis_indicators if indicator in response_lower)
    synthesis_score = min(1.0, synthesis_count / 3.0)  # Normalize to 0-1
    
    # Combine scores
    quality_score = (evidence_score * 0.5) + (reasoning_score * 0.3) + (synthesis_score * 0.2)
    
    return quality_score


def _calculate_agent_coordination(agent_results: Dict[str, Any]) -> float:
    """Calculate score based on multi-agent coordination."""
    if not agent_results:
        return 0.5
    
    # Count successful agents
    successful_agents = sum(1 for result in agent_results.values() 
                          if result and (isinstance(result, str) and result.strip() or 
                                        isinstance(result, dict) and result.get("success", False)))
    
    total_agents = len(agent_results)
    
    if total_agents == 0:
        return 0.5
    
    # Success rate
    success_rate = successful_agents / total_agents
    
    # Diversity score (different agents contributing)
    unique_contributions = len(set(str(result)[:100] for result in agent_results.values() if result))
    diversity_score = min(1.0, unique_contributions / max(1, total_agents))
    
    # Combine success rate and diversity
    coordination_score = (success_rate * 0.7) + (diversity_score * 0.3)
    
    return coordination_score


def _calculate_efficiency(response_time: float, response_length: int) -> float:
    """Calculate efficiency score based on response time and length."""
    # Target: 100 words per second
    words_per_second = (response_length / 5.0) / max(response_time, 0.1)  # Approximate words
    
    # Normalize to 0-1 (target: 10-50 words/second is good)
    if words_per_second < 1:
        efficiency_score = 0.2
    elif words_per_second < 5:
        efficiency_score = 0.5
    elif words_per_second < 10:
        efficiency_score = 0.7
    elif words_per_second < 50:
        efficiency_score = 0.9
    else:
        efficiency_score = 1.0
    
    # Penalize very slow responses
    if response_time > 60:
        efficiency_score *= 0.7
    elif response_time > 120:
        efficiency_score *= 0.5
    
    return efficiency_score


def _calculate_complexity_match(response_text: str, query_text: str) -> float:
    """Calculate how well response matches query complexity."""
    query_words = len(query_text.split())
    response_words = len(response_text.split())
    
    # Simple queries (< 10 words) should have shorter responses
    if query_words < 10:
        if response_words < 200:
            return 1.0
        elif response_words < 500:
            return 0.7
        else:
            return 0.4
    
    # Complex queries (> 30 words) should have longer responses
    elif query_words > 30:
        if response_words > 500:
            return 1.0
        elif response_words > 200:
            return 0.7
        else:
            return 0.4
    
    # Medium queries (10-30 words)
    else:
        if 100 < response_words < 1000:
            return 1.0
        elif 50 < response_words < 2000:
            return 0.7
        else:
            return 0.5

