"""
ICEBURG Linguistic Intelligence System

Provides linguistic enhancement for agent communication:
- Verbosity reduction
- Powerful word selection
- Metaphorical explanations
- Anti-cliche detection
- Intelligence clichés and metaphors
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class LinguisticStyle(Enum):
    """Linguistic style preferences"""
    CONCISE = "concise"
    POWERFUL = "powerful"
    METAPHORICAL = "metaphorical"
    TECHNICAL = "technical"
    INTELLIGENT = "intelligent"


@dataclass
class WordReplacement:
    """Word replacement suggestion"""
    original: str
    replacement: str
    reason: str
    power_score: float  # 0-1, higher = more powerful


@dataclass
class Metaphor:
    """Metaphorical explanation"""
    concept: str
    metaphor: str
    explanation: str
    clarity_score: float  # 0-1, higher = clearer


@dataclass
class LinguisticEnhancement:
    """Result of linguistic enhancement"""
    original_text: str
    enhanced_text: str
    replacements: List[WordReplacement]
    metaphors: List[Metaphor]
    verbosity_reduction: float  # 0-1, higher = more reduction
    power_increase: float  # 0-1, higher = more powerful


class LinguisticReasoningEngine:
    """
    Linguistic reasoning engine for verbosity reduction and powerful word selection.
    
    Enhances agent communication by:
    - Identifying verbose phrases
    - Suggesting powerful alternatives
    - Maintaining technical accuracy
    - Preserving meaning while improving clarity
    """
    
    def __init__(self):
        self.verbose_patterns = self._load_verbose_patterns()
        self.powerful_words = self._load_powerful_words()
        self.weak_words = self._load_weak_words()
        self.intelligence_cliches = self._load_intelligence_cliches()
        self.metaphor_templates = self._load_metaphor_templates()
    
    def _load_verbose_patterns(self) -> List[Tuple[str, str]]:
        """Load patterns for verbose phrases and their concise alternatives"""
        return [
            (r'\b(it is important to note that|it should be noted that|it is worth noting that)\b', ''),
            (r'\b(due to the fact that|owing to the fact that)\b', 'because'),
            (r'\b(in order to|so as to)\b', 'to'),
            (r'\b(at this point in time|at the present time)\b', 'now'),
            (r'\b(for the purpose of|for purposes of)\b', 'for'),
            (r'\b(in the event that|in case)\b', 'if'),
            (r'\b(prior to|before the time that)\b', 'before'),
            (r'\b(subsequent to|after the time that)\b', 'after'),
            (r'\b(in spite of the fact that|despite the fact that)\b', 'although'),
            (r'\b(with regard to|with respect to|in relation to)\b', 'about'),
            (r'\b(in the process of|while in the act of)\b', 'while'),
            (r'\b(has the ability to|is able to)\b', 'can'),
            (r'\b(is in a position to|has the capacity to)\b', 'can'),
            (r'\b(a large number of|a great deal of)\b', 'many'),
            (r'\b(a small number of|a limited number of)\b', 'few'),
            (r'\b(take into consideration|take into account)\b', 'consider'),
            (r'\b(make use of|utilize)\b', 'use'),
            (r'\b(carry out|perform|conduct)\b', 'do'),
            (r'\b(come to a conclusion|reach a conclusion)\b', 'conclude'),
            (r'\b(put forward|put forth)\b', 'propose'),
        ]
    
    def _load_powerful_words(self) -> Dict[str, List[str]]:
        """Load powerful word alternatives"""
        return {
            'good': ['excellent', 'superior', 'outstanding', 'exceptional', 'remarkable'],
            'bad': ['deficient', 'inadequate', 'substandard', 'flawed', 'problematic'],
            'big': ['substantial', 'significant', 'considerable', 'extensive', 'comprehensive'],
            'small': ['minimal', 'negligible', 'marginal', 'limited', 'restricted'],
            'fast': ['rapid', 'swift', 'expeditious', 'accelerated', 'immediate'],
            'slow': ['deliberate', 'methodical', 'gradual', 'measured', 'systematic'],
            'important': ['critical', 'crucial', 'essential', 'vital', 'paramount'],
            'think': ['analyze', 'examine', 'evaluate', 'assess', 'scrutinize'],
            'show': ['demonstrate', 'reveal', 'illustrate', 'exemplify', 'manifest'],
            'help': ['facilitate', 'enable', 'empower', 'support', 'enhance'],
            'try': ['attempt', 'endeavor', 'strive', 'pursue', 'undertake'],
            'use': ['employ', 'utilize', 'leverage', 'harness', 'apply'],
            'make': ['generate', 'create', 'produce', 'construct', 'fabricate'],
            'get': ['obtain', 'acquire', 'retrieve', 'extract', 'derive'],
            'find': ['discover', 'identify', 'detect', 'uncover', 'reveal'],
            'know': ['understand', 'comprehend', 'grasp', 'perceive', 'recognize'],
            'see': ['observe', 'perceive', 'discern', 'detect', 'witness'],
            'say': ['state', 'declare', 'assert', 'proclaim', 'articulate'],
            'do': ['execute', 'perform', 'accomplish', 'implement', 'conduct'],
        }
    
    def _load_weak_words(self) -> List[str]:
        """Load weak words that should be replaced"""
        return [
            'very', 'really', 'quite', 'rather', 'pretty', 'somewhat',
            'kind of', 'sort of', 'a bit', 'a little', 'a lot',
            'maybe', 'perhaps', 'possibly', 'might', 'could',
            'seems', 'appears', 'looks like', 'sort of like',
            'thing', 'stuff', 'stuff like that', 'and stuff',
            'you know', 'I mean', 'like', 'um', 'uh',
        ]
    
    def _load_intelligence_cliches(self) -> List[str]:
        """Load intelligence clichés to avoid"""
        return [
            'think outside the box',
            'paradigm shift',
            'game changer',
            'disruptive innovation',
            'synergy',
            'leverage',
            'best practices',
            'low-hanging fruit',
            'circle back',
            'touch base',
            'deep dive',
            'drill down',
            'move the needle',
            'win-win',
            'think big',
            'push the envelope',
            'cutting edge',
            'state of the art',
            'next level',
            'take it to the next level',
            'raise the bar',
            'raise the stakes',
            'think strategically',
            'big picture',
            'connect the dots',
            'join the dots',
            'think differently',
            'challenge assumptions',
            'break the mold',
            'think creatively',
        ]
    
    def _load_metaphor_templates(self) -> Dict[str, List[str]]:
        """Load metaphor templates for common concepts"""
        return {
            'complexity': [
                'like a multi-layered onion',
                'like a Russian nesting doll',
                'like an intricate web',
                'like a complex ecosystem',
            ],
            'understanding': [
                'like peeling back layers',
                'like assembling a puzzle',
                'like navigating a maze',
                'like decoding a cipher',
            ],
            'process': [
                'like a well-oiled machine',
                'like a river flowing',
                'like a chain reaction',
                'like a domino effect',
            ],
            'growth': [
                'like a seed growing into a tree',
                'like a snowball rolling downhill',
                'like compound interest',
                'like a feedback loop',
            ],
            'connection': [
                'like threads in a tapestry',
                'like nodes in a network',
                'like links in a chain',
                'like branches of a tree',
            ],
            'emergence': [
                'like patterns emerging from chaos',
                'like order arising from complexity',
                'like a symphony from individual notes',
                'like a forest from individual trees',
            ],
        }
    
    def reduce_verbosity(self, text: str, target_reduction: float = 0.3) -> Tuple[str, List[WordReplacement]]:
        """
        Reduce verbosity in text while preserving meaning.
        
        Args:
            text: Input text
            target_reduction: Target verbosity reduction (0-1)
            
        Returns:
            Tuple of (reduced_text, replacements)
        """
        replacements = []
        reduced_text = text
        
        # Apply verbose pattern replacements
        for pattern, replacement in self.verbose_patterns:
            matches = re.finditer(pattern, reduced_text, re.IGNORECASE)
            for match in matches:
                original = match.group(0)
                if original.lower() != replacement.lower():
                    replacements.append(WordReplacement(
                        original=original,
                        replacement=replacement if replacement else '[removed]',
                        reason="Verbose phrase",
                        power_score=0.7
                    ))
                    reduced_text = re.sub(pattern, replacement, reduced_text, flags=re.IGNORECASE)
        
        # Remove weak words
        for weak_word in self.weak_words:
            pattern = r'\b' + re.escape(weak_word) + r'\b'
            if re.search(pattern, reduced_text, re.IGNORECASE):
                replacements.append(WordReplacement(
                    original=weak_word,
                    replacement='',
                    reason="Weak word removal",
                    power_score=0.5
                ))
                reduced_text = re.sub(pattern, '', reduced_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        reduced_text = re.sub(r'\s+', ' ', reduced_text)
        reduced_text = reduced_text.strip()
        
        return reduced_text, replacements
    
    def enhance_power_words(self, text: str, intensity: float = 0.5) -> Tuple[str, List[WordReplacement]]:
        """
        Replace weak words with powerful alternatives.
        
        Args:
            text: Input text
            intensity: Enhancement intensity (0-1)
            
        Returns:
            Tuple of (enhanced_text, replacements)
        """
        replacements = []
        enhanced_text = text
        
        # Find and replace weak words with powerful alternatives
        for weak_word, powerful_alternatives in self.powerful_words.items():
            pattern = r'\b' + re.escape(weak_word) + r'\b'
            matches = list(re.finditer(pattern, enhanced_text, re.IGNORECASE))
            
            if matches:
                # Select appropriate powerful alternative
                replacement = powerful_alternatives[0]  # Use first alternative
                
                for match in matches:
                    original = match.group(0)
                    replacements.append(WordReplacement(
                        original=original,
                        replacement=replacement,
                        reason="Power word enhancement",
                        power_score=0.8
                    ))
                    enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE, count=1)
        
        return enhanced_text, replacements
    
    def detect_cliches(self, text: str) -> List[str]:
        """
        Detect intelligence clichés in text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected clichés
        """
        detected = []
        text_lower = text.lower()
        
        for cliche in self.intelligence_cliches:
            if cliche.lower() in text_lower:
                detected.append(cliche)
        
        return detected
    
    def suggest_metaphor(self, concept: str, category: Optional[str] = None) -> Optional[Metaphor]:
        """
        Suggest a metaphor for a concept.
        
        Args:
            concept: Concept to explain
            category: Optional category hint
            
        Returns:
            Metaphor suggestion or None
        """
        # Try to match category
        if category and category in self.metaphor_templates:
            templates = self.metaphor_templates[category]
            if templates:
                metaphor_text = templates[0]
                explanation = f"{concept} is {metaphor_text}, revealing deeper layers of understanding"
                return Metaphor(
                    concept=concept,
                    metaphor=metaphor_text,
                    explanation=explanation,
                    clarity_score=0.8
                )
        
        # Default metaphor
        return Metaphor(
            concept=concept,
            metaphor="like a key unlocking a door",
            explanation=f"{concept} is like a key unlocking a door, revealing new possibilities",
            clarity_score=0.7
        )
    
    def enhance_text(self, text: str, style: LinguisticStyle = LinguisticStyle.INTELLIGENT,
                     verbosity_reduction: float = 0.3, power_enhancement: float = 0.5) -> LinguisticEnhancement:
        """
        Comprehensive text enhancement.
        
        Args:
            text: Input text
            style: Linguistic style preference
            verbosity_reduction: Target verbosity reduction (0-1)
            power_enhancement: Power word enhancement intensity (0-1)
            
        Returns:
            LinguisticEnhancement result
        """
        original_text = text
        enhanced_text = text
        all_replacements = []
        metaphors = []
        
        # Reduce verbosity
        if verbosity_reduction > 0:
            enhanced_text, verbosity_replacements = self.reduce_verbosity(enhanced_text, verbosity_reduction)
            all_replacements.extend(verbosity_replacements)
        
        # Enhance power words
        if power_enhancement > 0:
            enhanced_text, power_replacements = self.enhance_power_words(enhanced_text, power_enhancement)
            all_replacements.extend(power_replacements)
        
        # Detect clichés
        detected_cliches = self.detect_cliches(enhanced_text)
        
        # Calculate metrics
        original_length = len(original_text.split())
        enhanced_length = len(enhanced_text.split())
        verbosity_reduction_actual = max(0, (original_length - enhanced_length) / original_length) if original_length > 0 else 0
        
        power_increase = len([r for r in all_replacements if r.power_score > 0.7]) / max(1, len(all_replacements))
        
        return LinguisticEnhancement(
            original_text=original_text,
            enhanced_text=enhanced_text,
            replacements=all_replacements,
            metaphors=metaphors,
            verbosity_reduction=verbosity_reduction_actual,
            power_increase=power_increase
        )


class MetaphorGenerator:
    """
    Generates metaphorical explanations for complex concepts.
    
    Uses intelligence clichés and powerful metaphors to explain
    technical concepts in accessible ways.
    """
    
    def __init__(self):
        self.metaphor_database = self._load_metaphor_database()
        self.intelligence_metaphors = self._load_intelligence_metaphors()
    
    def _load_metaphor_database(self) -> Dict[str, List[str]]:
        """Load database of metaphors by category"""
        return {
            'computation': [
                'like a vast library where every book is instantly accessible',
                'like a neural network processing information',
                'like a quantum superposition of possibilities',
            ],
            'knowledge': [
                'like a web of interconnected insights',
                'like a tree of understanding branching into new domains',
                'like a constellation of ideas forming patterns',
            ],
            'discovery': [
                'like an archaeologist uncovering buried truths',
                'like a detective connecting clues',
                'like a scientist observing emergent patterns',
            ],
            'synthesis': [
                'like a composer weaving melodies into a symphony',
                'like a chef combining ingredients into a masterpiece',
                'like an architect designing from first principles',
            ],
            'emergence': [
                'like order emerging from chaos',
                'like patterns crystallizing from complexity',
                'like intelligence arising from simple rules',
            ],
        }
    
    def _load_intelligence_metaphors(self) -> List[str]:
        """Load intelligence-specific metaphors"""
        return [
            'like a mind expanding its horizons',
            'like consciousness awakening to new possibilities',
            'like intelligence crystallizing from information',
            'like understanding emerging from complexity',
            'like knowledge transcending its boundaries',
            'like insight piercing through complexity',
            'like wisdom distilled from experience',
            'like understanding unfolding like a flower',
        ]
    
    def generate_metaphor(self, concept: str, context: Optional[str] = None) -> Metaphor:
        """
        Generate a metaphor for a concept.
        
        Args:
            concept: Concept to explain
            context: Optional context hint
            
        Returns:
            Generated metaphor
        """
        # Try to match context category
        if context:
            for category, metaphors in self.metaphor_database.items():
                if category in context.lower():
                    if metaphors:
                        metaphor_text = metaphors[0]
                        explanation = f"{concept} is {metaphor_text}, revealing deeper understanding"
                        return Metaphor(
                            concept=concept,
                            metaphor=metaphor_text,
                            explanation=explanation,
                            clarity_score=0.8
                        )
        
        # Use intelligence metaphor
        metaphor_text = self.intelligence_metaphors[0]
        explanation = f"{concept} is {metaphor_text}, transcending conventional boundaries"
        
        return Metaphor(
            concept=concept,
            metaphor=metaphor_text,
            explanation=explanation,
            clarity_score=0.7
        )


class AntiClicheDetector:
    """
    Detects and suggests alternatives to intelligence clichés.
    
    Identifies overused phrases and suggests more original,
    powerful alternatives.
    """
    
    def __init__(self):
        self.cliche_alternatives = self._load_cliche_alternatives()
    
    def _load_cliche_alternatives(self) -> Dict[str, List[str]]:
        """Load alternatives for common clichés"""
        return {
            'think outside the box': [
                'transcend conventional boundaries',
                'break free from established patterns',
                'explore uncharted territories',
            ],
            'paradigm shift': [
                'fundamental transformation',
                'radical reconfiguration',
                'profound restructuring',
            ],
            'game changer': [
                'transformative innovation',
                'revolutionary breakthrough',
                'disruptive advancement',
            ],
            'synergy': [
                'collaborative amplification',
                'integrated enhancement',
                'unified optimization',
            ],
            'leverage': [
                'harness',
                'utilize',
                'employ',
            ],
            'best practices': [
                'proven methodologies',
                'validated approaches',
                'established techniques',
            ],
            'deep dive': [
                'comprehensive analysis',
                'thorough examination',
                'systematic investigation',
            ],
            'think strategically': [
                'apply strategic reasoning',
                'employ strategic thinking',
                'utilize strategic analysis',
            ],
        }
    
    def detect_and_replace(self, text: str) -> Tuple[str, List[WordReplacement]]:
        """
        Detect clichés and suggest replacements.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (enhanced_text, replacements)
        """
        replacements = []
        enhanced_text = text
        
        for cliche, alternatives in self.cliche_alternatives.items():
            pattern = r'\b' + re.escape(cliche) + r'\b'
            if re.search(pattern, enhanced_text, re.IGNORECASE):
                replacement = alternatives[0] if alternatives else cliche
                replacements.append(WordReplacement(
                    original=cliche,
                    replacement=replacement,
                    reason="Cliche replacement",
                    power_score=0.9
                ))
                enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
        
        return enhanced_text, replacements


# Global instances
_linguistic_engine: Optional[LinguisticReasoningEngine] = None
_metaphor_generator: Optional[MetaphorGenerator] = None
_anticliche_detector: Optional[AntiClicheDetector] = None


def get_linguistic_engine() -> LinguisticReasoningEngine:
    """Get or create global linguistic reasoning engine"""
    global _linguistic_engine
    if _linguistic_engine is None:
        _linguistic_engine = LinguisticReasoningEngine()
    return _linguistic_engine


def get_metaphor_generator() -> MetaphorGenerator:
    """Get or create global metaphor generator"""
    global _metaphor_generator
    if _metaphor_generator is None:
        _metaphor_generator = MetaphorGenerator()
    return _metaphor_generator


def get_anticliche_detector() -> AntiClicheDetector:
    """Get or create global anti-cliche detector"""
    global _anticliche_detector
    if _anticliche_detector is None:
        _anticliche_detector = AntiClicheDetector()
    return _anticliche_detector

