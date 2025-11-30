#!/usr/bin/env python3
"""
ICEBURG Semantic Pattern Matcher - Advanced Pattern Recognition
===============================================================

This module implements semantic similarity-based pattern matching using vector embeddings
to replace the problematic keyword-based matching system.

Based on research from:
- OpenAI GPT-4 semantic similarity approaches
- Google DeepMind pattern recognition methods
- Chinese AI labs (BAAI) domain classification techniques
- Anthropic Claude contextual understanding methods
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Domain classification types"""
    BIOLOGY = "biology"
    MEDICINE = "medicine"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    TECHNOLOGY = "technology"
    GENERAL = "general"

@dataclass
class DomainKeywords:
    """Domain-specific keywords for classification"""
    domain: DomainType
    keywords: List[str]
    weight: float = 1.0

class DomainClassifier:
    """Advanced domain classification using keyword analysis and context"""
    
    def __init__(self):
        self.domain_keywords = {
            DomainType.BIOLOGY: DomainKeywords(
                domain=DomainType.BIOLOGY,
                keywords=[
                    "protein", "dna", "rna", "cell", "molecular", "genetic", "genome",
                    "amino acid", "enzyme", "metabolism", "evolution", "organism",
                    "alphafold", "protein folding", "structure prediction", "biomolecule"
                ],
                weight=1.0
            ),
            DomainType.MEDICINE: DomainKeywords(
                domain=DomainType.MEDICINE,
                keywords=[
                    "cancer", "tumor", "oncology", "treatment", "therapy", "clinical",
                    "patient", "diagnosis", "disease", "cure", "medical", "health",
                    "pancreatic", "carcinoma", "malignancy", "chemotherapy"
                ],
                weight=1.0
            ),
            DomainType.PHYSICS: DomainKeywords(
                domain=DomainType.PHYSICS,
                keywords=[
                    "quantum", "electromagnetic", "thermodynamics", "mechanics",
                    "relativity", "field theory", "particle", "energy", "force",
                    "wave", "oscillation", "resonance"
                ],
                weight=1.0
            ),
            DomainType.CHEMISTRY: DomainKeywords(
                domain=DomainType.CHEMISTRY,
                keywords=[
                    "molecule", "reaction", "synthesis", "compound", "element",
                    "catalyst", "bond", "chemical", "organic", "inorganic"
                ],
                weight=1.0
            ),
            DomainType.TECHNOLOGY: DomainKeywords(
                domain=DomainType.TECHNOLOGY,
                keywords=[
                    "algorithm", "software", "hardware", "computing", "ai",
                    "machine learning", "neural network", "programming", "data",
                    "model", "training", "inference"
                ],
                weight=1.0
            )
        }
    
    def classify_domain(self, text: str) -> Tuple[DomainType, float]:
        """Classify text into domain with confidence score"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain_type, domain_info in self.domain_keywords.items():
            score = 0
            for keyword in domain_info.keywords:
                if keyword in text_lower:
                    # Weight by keyword specificity and frequency
                    keyword_weight = domain_info.weight
                    if len(keyword.split()) > 1:  # Multi-word keywords get higher weight
                        keyword_weight *= 1.5
                    score += keyword_weight
            
            domain_scores[domain_type] = score
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return DomainType.GENERAL, 0.0
        
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = min(domain_scores[best_domain] / 10.0, 1.0)  # Normalize to 0-1
        
        return best_domain, confidence

class SemanticPatternMatcher:
    """Semantic similarity-based pattern matching"""
    
    def __init__(self):
        self.similarity_threshold = 0.75  # High threshold for precision
        self.domain_classifier = DomainClassifier()
        
        # Try to load sentence transformers, fallback to simple similarity if not available
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
            logger.info("Loaded sentence transformers for semantic similarity")
        except ImportError:
            self.model = None
            self.use_embeddings = False
            logger.warning("Sentence transformers not available, using keyword-based similarity")
    
    def calculate_semantic_similarity(self, query: str, pattern: str) -> float:
        """Calculate semantic similarity between query and pattern"""
        if self.use_embeddings and self.model:
            return self._embedding_similarity(query, pattern)
        else:
            return self._keyword_similarity(query, pattern)
    
    def _embedding_similarity(self, query: str, pattern: str) -> float:
        """Calculate similarity using vector embeddings"""
        try:
            query_embedding = self.model.encode([query])
            pattern_embedding = self.model.encode([pattern])
            
            # Cosine similarity
            similarity = np.dot(query_embedding, pattern_embedding.T) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(pattern_embedding)
            )
            return float(similarity[0][0])
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}")
            return self._keyword_similarity(query, pattern)
    
    def _keyword_similarity(self, query: str, pattern: str) -> float:
        """Fallback keyword-based similarity calculation"""
        query_words = set(query.lower().split())
        pattern_words = set(pattern.lower().split())
        
        if not query_words or not pattern_words:
            return 0.0
        
        intersection = query_words.intersection(pattern_words)
        union = query_words.union(pattern_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def matches(self, query: str, pattern: str) -> bool:
        """Check if query matches pattern using semantic similarity and domain awareness"""
        
        # Step 1: Domain Classification
        query_domain, query_confidence = self.domain_classifier.classify_domain(query)
        pattern_domain, pattern_confidence = self.domain_classifier.classify_domain(pattern)
        
        # Must be in same domain with reasonable confidence
        if query_domain != pattern_domain:
            logger.debug(f"Domain mismatch: {query_domain} vs {pattern_domain}")
            return False
        
        if query_confidence < 0.3 or pattern_confidence < 0.3:
            logger.debug(f"Low domain confidence: query={query_confidence}, pattern={pattern_confidence}")
            return False
        
        # Step 2: Semantic Similarity
        similarity = self.calculate_semantic_similarity(query, pattern)
        
        # Step 3: Contextual Validation
        if similarity > self.similarity_threshold:
            return self._contextual_validation(query, pattern, query_domain)
        
        logger.debug(f"Low semantic similarity: {similarity} < {self.similarity_threshold}")
        return False
    
    def _contextual_validation(self, query: str, pattern: str, domain: DomainType) -> bool:
        """Additional contextual validation to prevent false positives"""
        
        # Domain-specific validation rules
        if domain == DomainType.BIOLOGY:
            # Ensure both are about biological concepts
            biology_terms = ["protein", "dna", "rna", "cell", "molecular", "genetic"]
            query_has_bio = any(term in query.lower() for term in biology_terms)
            pattern_has_bio = any(term in pattern.lower() for term in biology_terms)
            
            if not (query_has_bio and pattern_has_bio):
                return False
        
        elif domain == DomainType.MEDICINE:
            # Ensure both are about medical concepts
            medical_terms = ["cancer", "treatment", "therapy", "clinical", "medical"]
            query_has_med = any(term in query.lower() for term in medical_terms)
            pattern_has_med = any(term in pattern.lower() for term in medical_terms)
            
            if not (query_has_med and pattern_has_med):
                return False
        
        # Length similarity check (prevent very different length matches)
        length_ratio = min(len(query), len(pattern)) / max(len(query), len(pattern))
        if length_ratio < 0.3:  # Very different lengths
            return False
        
        return True

class AdvancedPatternMatcher:
    """Main pattern matcher that combines all techniques"""
    
    def __init__(self):
        self.semantic_matcher = SemanticPatternMatcher()
        self.domain_classifier = DomainClassifier()
        
    def matches(self, query: str, pattern: str) -> bool:
        """Advanced pattern matching with multiple validation layers"""
        
        # Quick domain check first
        query_domain, _ = self.domain_classifier.classify_domain(query)
        pattern_domain, _ = self.domain_classifier.classify_domain(pattern)
        
        if query_domain != pattern_domain:
            return False
        
        # Semantic similarity check
        return self.semantic_matcher.matches(query, pattern)
    
    def get_match_info(self, query: str, pattern: str) -> Dict[str, any]:
        """Get detailed information about the match attempt"""
        query_domain, query_conf = self.domain_classifier.classify_domain(query)
        pattern_domain, pattern_conf = self.domain_classifier.classify_domain(pattern)
        similarity = self.semantic_matcher.calculate_semantic_similarity(query, pattern)
        
        return {
            "query_domain": query_domain.value,
            "pattern_domain": pattern_domain.value,
            "domain_match": query_domain == pattern_domain,
            "query_confidence": query_conf,
            "pattern_confidence": pattern_conf,
            "semantic_similarity": similarity,
            "would_match": self.matches(query, pattern)
        }

# Global instance
advanced_pattern_matcher = AdvancedPatternMatcher()
