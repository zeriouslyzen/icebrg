"""Reflex/Compression Agent - Compresses verbose responses and extracts key bullets."""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .linguistic_intelligence import (
    get_linguistic_engine,
    get_metaphor_generator,
    get_anticliche_detector,
    LinguisticStyle
)

logger = logging.getLogger(__name__)


@dataclass
class ReflexResponse:
    """Compressed response with 3-bullet preview."""
    preview: Dict[str, str]  # 3 bullets: core_insight, actionable_guidance, key_context
    full: str  # Full response (preserved)
    compressed: str  # Compressed version
    compression_ratio: float  # Ratio of compressed to original
    reflections: List[Dict[str, Any]]  # High-value reflections for knowledge tree


class ReflexAgent:
    """
    Reflex/Compression Agent that compresses verbose responses and extracts key bullets.
    
    Features:
    - Removes verbosity (hedging, filler, redundant explanations)
    - Extracts 3 bullets (Core Insight, Actionable Guidance, Key Context)
    - Preserves linguistic/psychological depth
    - Extracts high-value reflections for knowledge tree
    """
    
    def __init__(self):
        """Initialize Reflex Agent."""
        # Initialize linguistic intelligence components
        self.linguistic_engine = get_linguistic_engine()
        self.metaphor_generator = get_metaphor_generator()
        self.anticliche_detector = get_anticliche_detector()
        
        # Hedging patterns to remove
        self.hedging_patterns = [
            r'\bI think\b',
            r'\bI believe\b',
            r'\bI feel\b',
            r'\bI suppose\b',
            r'\bI guess\b',
            r'\bI would say\b',
            r'\bI would think\b',
            r'\bI would imagine\b',
            r'\bI would suggest\b',
            r'\bit seems\b',
            r'\bit appears\b',
            r'\bit looks like\b',
            r'\bit might be\b',
            r'\bit could be\b',
            r'\bit may be\b',
            r'\bperhaps\b',
            r'\bmaybe\b',
            r'\bpossibly\b',
            r'\bprobably\b',
            r'\blikely\b',
            r'\bpotentially\b',
            r'\bpresumably\b',
            r'\bapparently\b',
            r'\barguably\b',
            r'\bpresumably\b',
            r'\bwhat if\b',
            r'\bwhat about\b',
            r'\bhow about\b',
        ]
        
        # Filler phrases to remove
        self.filler_patterns = [
            r'\bas you can see\b',
            r'\bas we know\b',
            r'\bas mentioned\b',
            r'\bas stated\b',
            r'\bas noted\b',
            r'\bas discussed\b',
            r'\bin other words\b',
            r'\bto put it simply\b',
            r'\bto put it another way\b',
            r'\bthat is to say\b',
            r'\bin essence\b',
            r'\bin summary\b',
            r'\bto summarize\b',
            r'\bto conclude\b',
            r'\bultimately\b',
            r'\bfinally\b',
            r'\bin conclusion\b',
            r'\ball in all\b',
            r'\ball things considered\b',
        ]
        
        # Redundant explanation patterns
        self.redundant_patterns = [
            r'\.\s+In other words[^.]*\.',
            r'\.\s+That is[^.]*\.',
            r'\.\s+To clarify[^.]*\.',
            r'\.\s+To explain[^.]*\.',
            r'\.\s+What this means is[^.]*\.',
            r'\.\s+This means that[^.]*\.',
        ]
        
        # Compile patterns for efficiency
        self.hedging_regex = re.compile('|'.join(self.hedging_patterns), re.IGNORECASE)
        self.filler_regex = re.compile('|'.join(self.filler_patterns), re.IGNORECASE)
        self.redundant_regex = re.compile('|'.join(self.redundant_patterns), re.IGNORECASE)
    
    def compress_response(self, full_response: str) -> ReflexResponse:
        """
        Compress verbose response and extract 3 bullets.
        
        Args:
            full_response: Full ICEBURG response text
            
        Returns:
            ReflexResponse with preview bullets, compressed text, and reflections
        """
        if not full_response or not isinstance(full_response, str):
            logger.warning("Empty or invalid response provided to ReflexAgent")
            return ReflexResponse(
                preview={
                    "core_insight": "No response generated.",
                    "actionable_guidance": "Please try rephrasing your query.",
                    "key_context": ""
                },
                full=full_response or "",
                compressed=full_response or "",
                compression_ratio=1.0,
                reflections=[]
            )
        
        # Step 1: Remove verbosity using linguistic engine
        compressed = self._remove_verbosity(full_response)
        
        # Step 1.5: Apply linguistic enhancement (power words, anti-cliche)
        compressed = self._apply_linguistic_enhancement(compressed)
        
        # Step 2: Extract 3 bullets
        bullets = self._extract_3_bullets(compressed, full_response)
        
        # Step 2.5: Enhance bullets with metaphors where appropriate
        bullets = self._enhance_bullets_with_metaphors(bullets, full_response)
        
        # Step 3: Extract reflections for knowledge tree
        reflections = self._extract_reflections(compressed, full_response)
        
        # Calculate compression ratio
        original_length = len(full_response)
        compressed_length = len(compressed)
        compression_ratio = compressed_length / original_length if original_length > 0 else 1.0
        
        return ReflexResponse(
            preview=bullets,
            full=full_response,
            compressed=compressed,
            compression_ratio=compression_ratio,
            reflections=reflections
        )
    
    def _remove_verbosity(self, text: str) -> str:
        """
        Remove hedging, filler, and redundant explanations.
        
        Args:
            text: Original text
            
        Returns:
            Compressed text with verbosity removed
        """
        compressed = text
        
        # Remove hedging phrases (but preserve sentence structure)
        compressed = self.hedging_regex.sub('', compressed)
        
        # Remove filler phrases
        compressed = self.filler_regex.sub('', compressed)
        
        # Remove redundant explanations
        compressed = self.redundant_regex.sub('.', compressed)
        
        # Clean up multiple spaces and periods
        compressed = re.sub(r'\s+', ' ', compressed)  # Multiple spaces to single
        compressed = re.sub(r'\.\s*\.\s*\.', '.', compressed)  # Multiple periods to single
        compressed = re.sub(r'\s+\.', '.', compressed)  # Space before period
        compressed = re.sub(r'\.\s+\.', '.', compressed)  # Period space period
        
        # Remove leading/trailing whitespace
        compressed = compressed.strip()
        
        # Ensure sentences end properly
        if compressed and not compressed.endswith(('.', '!', '?')):
            # Add period if missing
            compressed += '.'
        
        return compressed
    
    def _extract_3_bullets(self, compressed: str, full: str) -> Dict[str, str]:
        """
        Extract 3 highest-value bullets from response.
        
        Bullets:
        1. Core Insight: Main finding/conclusion
        2. Actionable Guidance: What to do next
        3. Key Context: Important background/context
        
        Args:
            compressed: Compressed response text
            full: Full response text (for context)
            
        Returns:
            Dictionary with 3 bullets
        """
        # Split into sentences
        sentences = self._split_into_sentences(compressed)
        
        if not sentences:
            return {
                "core_insight": compressed[:200] if compressed else "No insight available.",
                "actionable_guidance": "Please review the full response for details.",
                "key_context": ""
            }
        
        # Extract core insight (first substantive sentence or summary)
        core_insight = self._extract_core_insight(sentences, full)
        
        # Extract actionable guidance (action verbs, directives)
        actionable_guidance = self._extract_actionable_guidance(sentences, full)
        
        # Extract key context (background, definitions, important details)
        key_context = self._extract_key_context(sentences, full)
        
        return {
            "core_insight": core_insight,
            "actionable_guidance": actionable_guidance,
            "key_context": key_context
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be enhanced with NLP)
        sentences = re.split(r'[.!?]+\s+', text)
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _extract_core_insight(self, sentences: List[str], full: str) -> str:
        """Extract core insight (main finding/conclusion)."""
        # Look for conclusion indicators
        conclusion_keywords = ['conclusion', 'finding', 'result', 'shows', 'demonstrates', 
                              'indicates', 'reveals', 'suggests', 'proves', 'confirms']
        
        # Check first few sentences (usually contains main point)
        for sentence in sentences[:3]:
            if any(keyword in sentence.lower() for keyword in conclusion_keywords):
                return sentence[:200]  # Limit length
        
        # If no conclusion found, use first substantive sentence
        for sentence in sentences:
            if len(sentence) > 20:  # Substantive sentence
                return sentence[:200]
        
        # Fallback: first sentence
        return sentences[0][:200] if sentences else "Core insight not available."
    
    def _extract_actionable_guidance(self, sentences: List[str], full: str) -> str:
        """Extract actionable guidance (what to do next)."""
        # Look for action verbs and directives
        action_keywords = ['should', 'must', 'need to', 'recommend', 'suggest', 'consider',
                          'focus on', 'prioritize', 'implement', 'apply', 'use', 'try',
                          'do', 'take', 'make', 'create', 'build', 'develop']
        
        # Check all sentences for actionable content
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in action_keywords):
                # Extract actionable part
                for keyword in action_keywords:
                    if keyword in sentence_lower:
                        idx = sentence_lower.find(keyword)
                        # Get sentence from keyword onwards
                        actionable = sentence[idx:].strip()
                        if len(actionable) > 10:
                            return actionable[:200]
        
        # If no action found, look for recommendations
        recommendation_keywords = ['recommendation', 'next step', 'action', 'guidance']
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in recommendation_keywords):
                return sentence[:200]
        
        # Fallback: look for sentences with "you" (directive)
        for sentence in sentences:
            if 'you' in sentence.lower() and len(sentence) > 20:
                return sentence[:200]
        
        return "Review the full response for actionable steps."
    
    def _extract_key_context(self, sentences: List[str], full: str) -> str:
        """Extract key context (important background/context)."""
        # Look for context indicators
        context_keywords = ['context', 'background', 'definition', 'meaning', 'refers to',
                           'is defined as', 'known as', 'called', 'term', 'concept']
        
        # Check sentences for context
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in context_keywords):
                return sentence[:200]
        
        # Look for definitions (X is Y, X means Y)
        definition_patterns = [
            r'is\s+(?:a|an|the)\s+[^.]{10,100}',
            r'means\s+[^.]{10,100}',
            r'refers to\s+[^.]{10,100}',
            r'defined as\s+[^.]{10,100}',
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, full, re.IGNORECASE)
            if matches:
                return matches[0][:200]
        
        # Fallback: use second or third sentence (often contains context)
        if len(sentences) > 1:
            return sentences[1][:200] if len(sentences) > 1 else sentences[0][:200]
        
        return ""
    
    def _extract_reflections(self, compressed: str, full: str) -> List[Dict[str, Any]]:
        """
        Extract high-value reflections for knowledge tree.
        
        Reflections include:
        - Core principles discovered
        - Novel insights
        - Cross-domain connections
        - Actionable patterns
        
        Args:
            compressed: Compressed response
            full: Full response
            
        Returns:
            List of reflection dictionaries
        """
        reflections = []
        
        # Extract core principles (sentences with "principle", "law", "rule", "pattern")
        principle_keywords = ['principle', 'law', 'rule', 'pattern', 'framework', 'model']
        for sentence in self._split_into_sentences(full):
            if any(keyword in sentence.lower() for keyword in principle_keywords):
                reflections.append({
                    "type": "principle",
                    "content": sentence[:300],
                    "source": "response"
                })
        
        # Extract novel insights (sentences with "discovery", "finding", "reveals")
        insight_keywords = ['discovery', 'finding', 'reveals', 'shows', 'demonstrates',
                           'indicates', 'suggests', 'proves', 'confirms']
        for sentence in self._split_into_sentences(full):
            if any(keyword in sentence.lower() for keyword in insight_keywords):
                reflections.append({
                    "type": "insight",
                    "content": sentence[:300],
                    "source": "response"
                })
        
        # Extract cross-domain connections (sentences with "relates to", "connects", "links")
        connection_keywords = ['relates to', 'connects', 'links', 'related to', 'associated with',
                              'similar to', 'analogous to', 'parallel to']
        for sentence in self._split_into_sentences(full):
            if any(keyword in sentence.lower() for keyword in connection_keywords):
                reflections.append({
                    "type": "connection",
                    "content": sentence[:300],
                    "source": "response"
                })
        
        # Extract actionable patterns (sentences with action verbs)
        action_patterns = [
            r'should\s+[^.]{10,100}',
            r'must\s+[^.]{10,100}',
            r'recommend\s+[^.]{10,100}',
            r'suggest\s+[^.]{10,100}',
        ]
        for pattern in action_patterns:
            matches = re.findall(pattern, full, re.IGNORECASE)
            for match in matches[:2]:  # Limit to 2 matches
                reflections.append({
                    "type": "actionable_pattern",
                    "content": match[:300],
                    "source": "response"
                })
        
        # Limit reflections to top 5 by length (prioritize substantial content)
        reflections.sort(key=lambda x: len(x.get("content", "")), reverse=True)
        return reflections[:5]
    
    def _apply_linguistic_enhancement(self, text: str) -> str:
        """
        Apply linguistic enhancement using linguistic intelligence system.
        
        Args:
            text: Input text
            
        Returns:
            Enhanced text with power words and cliche replacements
        """
        # Apply comprehensive linguistic enhancement
        enhancement = self.linguistic_engine.enhance_text(
            text,
            style=LinguisticStyle.INTELLIGENT,
            verbosity_reduction=0.2,  # Additional reduction
            power_enhancement=0.5
        )
        
        # Apply anti-cliche detection and replacement
        enhanced_text, cliche_replacements = self.anticliche_detector.detect_and_replace(
            enhancement.enhanced_text
        )
        
        return enhanced_text
    
    def _enhance_bullets_with_metaphors(self, bullets: Dict[str, str], full: str) -> Dict[str, str]:
        """
        Enhance bullets with metaphors where appropriate.
        
        Args:
            bullets: Dictionary of bullets
            full: Full response text for context
            
        Returns:
            Enhanced bullets with metaphors
        """
        enhanced_bullets = bullets.copy()
        
        # Add metaphors to core insight if complex
        core_insight = bullets.get("core_insight", "")
        if len(core_insight) > 100:  # Complex insight
            metaphor = self.metaphor_generator.generate_metaphor(
                core_insight[:50],  # Use first part as concept
                context=full[:200]  # Use beginning of full text as context
            )
            if metaphor and metaphor.clarity_score > 0.7:
                # Append metaphor as clarification
                enhanced_bullets["core_insight"] = f"{core_insight} ({metaphor.explanation})"
        
        # Add metaphors to key context if abstract
        key_context = bullets.get("key_context", "")
        if key_context and any(word in key_context.lower() for word in ['concept', 'principle', 'framework', 'model']):
            metaphor = self.metaphor_generator.generate_metaphor(
                key_context[:50],
                context=full[:200]
            )
            if metaphor and metaphor.clarity_score > 0.7:
                enhanced_bullets["key_context"] = f"{key_context} Think of it {metaphor.metaphor}."
        
        return enhanced_bullets
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            "hedging_patterns": len(self.hedging_patterns),
            "filler_patterns": len(self.filler_patterns),
            "redundant_patterns": len(self.redundant_patterns),
            "linguistic_enhancement": True,
            "metaphor_generation": True,
            "anticliche_detection": True
        }

