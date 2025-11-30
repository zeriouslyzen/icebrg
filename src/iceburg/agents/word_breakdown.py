"""
Word Breakdown Visualization System
Real-time morphological analysis, etymology, and semantic relationships
"""

from typing import Any, Dict, Optional, List, Tuple
import re
import time
from dataclasses import dataclass, field


@dataclass
class WordBreakdown:
    """Word breakdown analysis"""
    word: str
    morphological: Dict[str, Any]
    etymology: Dict[str, Any]
    semantic: Dict[str, Any]
    compression_hints: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlgorithmStep:
    """Algorithm pipeline step"""
    step_name: str
    input_data: Any
    output_data: Any
    processing_time: float
    status: str  # "processing", "complete", "error"


class WordBreakdownAnalyzer:
    """Analyzes words with morphological, etymological, and semantic breakdown"""
    
    def __init__(self):
        self.breakdown_cache: Dict[str, WordBreakdown] = {}
        
    def analyze_word(self, word: str) -> WordBreakdown:
        """Analyze a single word"""
        # Check cache
        if word.lower() in self.breakdown_cache:
            return self.breakdown_cache[word.lower()]
            
        # Morphological analysis
        morphological = self._analyze_morphology(word)
        
        # Etymology analysis (simplified - would use real etymology database)
        etymology = self._analyze_etymology(word)
        
        # Semantic analysis
        semantic = self._analyze_semantics(word)
        
        # Compression hints
        compression_hints = self._generate_compression_hints(word, morphological, etymology)
        
        breakdown = WordBreakdown(
            word=word,
            morphological=morphological,
            etymology=etymology,
            semantic=semantic,
            compression_hints=compression_hints
        )
        
        # Cache result
        self.breakdown_cache[word.lower()] = breakdown
        
        return breakdown
        
    def _analyze_morphology(self, word: str) -> Dict[str, Any]:
        """Analyze word morphology"""
        # Common prefixes
        prefixes = ["un", "re", "pre", "dis", "mis", "over", "under", "out", "in", "im", "non"]
        # Common suffixes
        suffixes = ["ing", "ed", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment", "ful", "less"]
        
        detected_prefix = None
        detected_suffix = None
        root = word
        
        for prefix in prefixes:
            if word.lower().startswith(prefix):
                detected_prefix = prefix
                root = word[len(prefix):]
                break
                
        for suffix in suffixes:
            if word.lower().endswith(suffix):
                detected_suffix = suffix
                root = word[:-len(suffix)]
                break
                
        return {
            "prefix": detected_prefix,
            "suffix": detected_suffix,
            "root": root,
            "length": len(word),
            "syllable_count": self._estimate_syllables(word)
        }
        
    def _estimate_syllables(self, word: str) -> int:
        """Estimate syllable count"""
        word = word.lower()
        if len(word) <= 3:
            return 1
        count = 0
        vowels = "aeiouy"
        prev_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        if word.endswith("e"):
            count -= 1
        return max(1, count)
        
    def _analyze_etymology(self, word: str) -> Dict[str, Any]:
        """Analyze word etymology with improved detection"""
        word_lower = word.lower()
        word_clean = re.sub(r'[^\w]', '', word_lower)
        
        # Common English words (Germanic origin)
        common_english_words = {
            "the", "is", "of", "to", "a", "and", "in", "it", "you", "that", "he", "was",
            "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this",
            "have", "from", "or", "one", "had", "by", "word", "but", "not", "what",
            "all", "were", "we", "when", "your", "can", "said", "there", "each",
            "which", "she", "do", "how", "their", "if", "will", "up", "other", "about",
            "out", "many", "then", "them", "these", "so", "some", "her", "would",
            "make", "like", "into", "him", "time", "has", "look", "two", "more",
            "write", "go", "see", "number", "no", "way", "could", "people", "my",
            "than", "first", "water", "been", "call", "who", "oil", "sit", "now",
            "find", "down", "day", "did", "get", "come", "made", "may", "part",
            "unknown", "nature", "consciousness"
        }
        
        # Latin indicators (suffixes and patterns)
        latin_indicators = [
            "tion", "sion", "ment", "able", "ible", "al", "ary", "ate", "ent", "ant",
            "ous", "ive", "ure", "ance", "ence", "ity", "ty", "cy", "fy", "ize"
        ]
        
        # Greek indicators
        greek_indicators = [
            "ology", "ism", "ist", "ic", "ical", "ph", "ch", "th", "y", "auto",
            "bio", "geo", "psych", "tech", "log", "graph", "scope", "meter"
        ]
        
        # Germanic indicators (English, German, Dutch)
        germanic_indicators = [
            "ing", "ed", "er", "est", "en", "ly", "ward", "wise", "ful", "less",
            "ness", "ship", "hood", "dom", "th", "gh", "ck", "sh", "ch"
        ]
        
        # French indicators
        french_indicators = [
            "eau", "eur", "eux", "tion", "sion", "que", "gue", "age", "ance", "ence"
        ]
        
        # Determine likely origin
        likely_origin = "unknown"
        linguistic_roots = []
        word_origins = {}
        
        # Check common English words first
        if word_clean in common_english_words:
            likely_origin = "english"
            linguistic_roots = ["germanic"]
            word_origins = {word: "english"}
        # Check Latin patterns
        elif any(ind in word_lower for ind in latin_indicators):
            likely_origin = "latin"
            linguistic_roots = ["latin"]
            word_origins = {word: "latin"}
        # Check Greek patterns
        elif any(ind in word_lower for ind in greek_indicators):
            likely_origin = "greek"
            linguistic_roots = ["greek"]
            word_origins = {word: "greek"}
        # Check French patterns
        elif any(ind in word_lower for ind in french_indicators):
            likely_origin = "french"
            linguistic_roots = ["french", "latin"]  # French often derived from Latin
            word_origins = {word: "french"}
        # Check Germanic patterns
        elif any(ind in word_lower for ind in germanic_indicators):
            likely_origin = "germanic"
            linguistic_roots = ["germanic"]
            word_origins = {word: "germanic"}
        # Default to English for short common words
        elif len(word_clean) <= 4:
            likely_origin = "english"
            linguistic_roots = ["germanic"]
            word_origins = {word: "english"}
            
        return {
            "likely_origin": likely_origin,
            "origin": likely_origin,  # Alias for compatibility
            "word_origins": word_origins,
            "linguistic_roots": linguistic_roots,
            "etymological_connections": f"Word '{word}' likely derived from {likely_origin} roots"
        }
        
    def _analyze_semantics(self, word: str) -> Dict[str, Any]:
        """Analyze semantic relationships"""
        # This would normally use semantic databases
        # For now, provide basic analysis
        return {
            "core_meaning": f"Core meaning of '{word}'",
            "relationships": [],
            "implicit_concepts": [],
            "semantic_field": "general"
        }
        
    def _generate_compression_hints(self, word: str, morphological: Dict[str, Any], etymology: Dict[str, Any]) -> List[str]:
        """Generate compression/retrieval optimization hints"""
        hints = []
        
        # Use root for compression
        if morphological.get("root"):
            hints.append(f"Use root '{morphological['root']}' for compression")
            
        # Use etymology for retrieval
        if etymology.get("likely_origin"):
            hints.append(f"Index by origin '{etymology['likely_origin']}' for faster retrieval")
            
        # Use morphological structure
        if morphological.get("prefix") or morphological.get("suffix"):
            hints.append("Use morphological structure for pattern matching")
            
        return hints
        
    def analyze_query(self, query: str) -> List[WordBreakdown]:
        """Analyze all words in a query"""
        # Tokenize query
        words = re.findall(r'\b\w+\b', query)
        
        breakdowns = []
        for word in words:
            breakdown = self.analyze_word(word)
            breakdowns.append(breakdown)
            
        return breakdowns
        
    def visualize_algorithm_pipeline(self, query: str) -> List[AlgorithmStep]:
        """Visualize the algorithm pipeline for word breakdown"""
        steps = []
        
        # Step 1: Tokenization
        start_time = time.time()
        words = re.findall(r'\b\w+\b', query)
        tokenization_time = time.time() - start_time
        steps.append(AlgorithmStep(
            step_name="tokenization",
            input_data=query,
            output_data=words,
            processing_time=tokenization_time,
            status="complete"
        ))
        
        # Step 2: Morphological Analysis
        start_time = time.time()
        morphological_results = []
        for word in words:
            morphological = self._analyze_morphology(word)
            morphological_results.append(morphological)
        morphology_time = time.time() - start_time
        steps.append(AlgorithmStep(
            step_name="morphological_analysis",
            input_data=words,
            output_data=morphological_results,
            processing_time=morphology_time,
            status="complete"
        ))
        
        # Step 3: Etymology Analysis
        start_time = time.time()
        etymology_results = []
        for word in words:
            etymology = self._analyze_etymology(word)
            etymology_results.append(etymology)
        etymology_time = time.time() - start_time
        steps.append(AlgorithmStep(
            step_name="etymology_analysis",
            input_data=words,
            output_data=etymology_results,
            processing_time=etymology_time,
            status="complete"
        ))
        
        # Step 4: Semantic Analysis
        start_time = time.time()
        semantic_results = []
        for word in words:
            semantic = self._analyze_semantics(word)
            semantic_results.append(semantic)
        semantic_time = time.time() - start_time
        steps.append(AlgorithmStep(
            step_name="semantic_analysis",
            input_data=words,
            output_data=semantic_results,
            processing_time=semantic_time,
            status="complete"
        ))
        
        # Step 5: Compression Optimization
        start_time = time.time()
        compression_hints = []
        for i, word in enumerate(words):
            hints = self._generate_compression_hints(word, morphological_results[i], etymology_results[i])
            compression_hints.append(hints)
        compression_time = time.time() - start_time
        steps.append(AlgorithmStep(
            step_name="compression_optimization",
            input_data=semantic_results,
            output_data=compression_hints,
            processing_time=compression_time,
            status="complete"
        ))
        
        return steps

