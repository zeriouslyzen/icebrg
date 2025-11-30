"""
Proactive Knowledge Gap Filling
Identifies gaps and asks follow-up questions to build complete knowledge
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ..curiosity.curiosity_engine import CuriosityEngine, KnowledgeGap

logger = logging.getLogger(__name__)


@dataclass
class Gap:
    """Represents a knowledge gap"""
    gap_id: str
    gap_description: str
    gap_type: str  # "context", "detail", "domain", "evidence"
    priority: float
    suggested_questions: List[str]
    timestamp: str


class ProactiveGapFilling:
    """
    Identifies knowledge gaps in conversations.
    
    Asks follow-up questions to fill gaps, builds toward complete knowledge,
    and ensures gnosis accumulation.
    """
    
    def __init__(self):
        """Initialize proactive gap filling."""
        self.curiosity_engine = CuriosityEngine()
        logger.info("Proactive Gap Filling initialized")
    
    def identify_knowledge_gaps(self, conversation: Dict[str, Any]) -> List[Gap]:
        """
        Identify knowledge gaps in conversation.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            List of identified knowledge gaps
        """
        gaps = []
        
        query = conversation.get("query", "")
        response = conversation.get("response", "")
        metadata = conversation.get("metadata", {})
        
        # Identify context gaps
        if len(query.split()) < 5:
            gap = Gap(
                gap_id=f"gap_{len(gaps)}_{datetime.utcnow().timestamp()}",
                gap_description="Query lacks sufficient context",
                gap_type="context",
                priority=0.7,
                suggested_questions=[
                    "Can you provide more context?",
                    "What specific aspect are you interested in?",
                    "What is the background of this question?"
                ],
                timestamp=datetime.utcnow().isoformat()
            )
            gaps.append(gap)
        
        # Identify detail gaps
        if "what" in query.lower() or "how" in query.lower():
            gap = Gap(
                gap_id=f"gap_{len(gaps)}_{datetime.utcnow().timestamp()}",
                gap_description="Query may need more detail",
                gap_type="detail",
                priority=0.6,
                suggested_questions=[
                    "Can you provide more details?",
                    "What specific information are you looking for?",
                    "What level of detail do you need?"
                ],
                timestamp=datetime.utcnow().isoformat()
            )
            gaps.append(gap)
        
        # Identify domain gaps
        domains = metadata.get("domains", [])
        if not domains:
            gap = Gap(
                gap_id=f"gap_{len(gaps)}_{datetime.utcnow().timestamp()}",
                gap_description="Query lacks domain specification",
                gap_type="domain",
                priority=0.5,
                suggested_questions=[
                    "What domain does this relate to?",
                    "What field of study is this about?",
                    "What context are you working in?"
                ],
                timestamp=datetime.utcnow().isoformat()
            )
            gaps.append(gap)
        
        # Use curiosity engine to detect gaps
        try:
            curiosity_gaps = self.curiosity_engine.detect_knowledge_gaps(query, response)
            for curiosity_gap in curiosity_gaps:
                gap = Gap(
                    gap_id=f"gap_{len(gaps)}_{datetime.utcnow().timestamp()}",
                    gap_description=curiosity_gap.gap_description,
                    gap_type="curiosity",
                    priority=curiosity_gap.exploration_priority,
                    suggested_questions=curiosity_gap.suggested_queries,
                    timestamp=datetime.utcnow().isoformat()
                )
                gaps.append(gap)
        except Exception as e:
            logger.warning(f"Error using curiosity engine for gap detection: {e}")
        
        logger.info(f"Identified {len(gaps)} knowledge gaps in conversation")
        return gaps
    
    def generate_follow_up_questions(self, gaps: List[Gap]) -> List[str]:
        """
        Generate follow-up questions to fill gaps.
        
        Args:
            gaps: List of knowledge gaps
            
        Returns:
            List of follow-up questions
        """
        questions = []
        
        # Sort gaps by priority
        sorted_gaps = sorted(gaps, key=lambda g: g.priority, reverse=True)
        
        # Generate questions from top gaps
        for gap in sorted_gaps[:3]:  # Top 3 gaps
            questions.extend(gap.suggested_questions[:2])  # Top 2 questions per gap
        
        logger.info(f"Generated {len(questions)} follow-up questions for {len(gaps)} gaps")
        return questions
    
    def fill_gaps_proactively(self, gaps: List[Gap]) -> Dict[str, Any]:
        """
        Fill gaps proactively.
        
        Args:
            gaps: List of knowledge gaps
            
        Returns:
            Dictionary with gap filling results
        """
        result = {
            "gaps_identified": len(gaps),
            "gaps_filled": 0,
            "questions_generated": [],
            "filling_strategy": {}
        }
        
        # Generate follow-up questions
        questions = self.generate_follow_up_questions(gaps)
        result["questions_generated"] = questions
        
        # Determine filling strategy
        gap_types = [g.gap_type for g in gaps]
        result["filling_strategy"] = {
            "context_gaps": gap_types.count("context"),
            "detail_gaps": gap_types.count("detail"),
            "domain_gaps": gap_types.count("domain"),
            "curiosity_gaps": gap_types.count("curiosity")
        }
        
        logger.info(f"Filled {result['gaps_filled']} gaps proactively, generated {len(questions)} questions")
        return result
    
    def ensure_gnosis_accumulation(self, conversation: Dict[str, Any], accumulator) -> None:
        """
        Ensure conversation contributes to gnosis accumulation.
        
        Args:
            conversation: Conversation dictionary
            accumulator: UniversalKnowledgeAccumulator instance
        """
        # Extract insights
        from ..gnosis.conversation_insight_extractor import ConversationInsightExtractor
        insight_extractor = ConversationInsightExtractor()
        
        insights = insight_extractor.extract_insights(conversation)
        
        # Accumulate to gnosis
        accumulator.accumulate_to_gnosis(insights)
        
        logger.info(f"Ensured gnosis accumulation: {len(insights)} insights")

