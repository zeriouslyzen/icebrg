"""
Decoder Agent - Symbol, pattern, and esoteric analysis.
Analyzes symbolic dimensions, timing, numerology, and hidden patterns.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

from ...config import IceburgConfig
from ...providers.factory import provider_factory

logger = logging.getLogger(__name__)


# === ESOTERIC KNOWLEDGE BASES ===

OCCULT_HOLIDAYS = {
    # Solstices/Equinoxes
    "winter_solstice": {"dates": ["12-21", "12-22"], "significance": "Rebirth, shortest day, Saturnalia"},
    "spring_equinox": {"dates": ["03-20", "03-21"], "significance": "Ostara, balance, new beginnings"},
    "summer_solstice": {"dates": ["06-20", "06-21"], "significance": "Litha, peak power, midsummer"},
    "fall_equinox": {"dates": ["09-22", "09-23"], "significance": "Mabon, harvest, balance"},
    
    # Major occult dates
    "walpurgis": {"dates": ["04-30"], "significance": "Witches' Night, Beltane eve"},
    "beltane": {"dates": ["05-01"], "significance": "Fire festival, fertility"},
    "samhain": {"dates": ["10-31", "11-01"], "significance": "Halloween, veil thinnest, death"},
    "imbolc": {"dates": ["02-01", "02-02"], "significance": "Candlemas, purification"},
    "lughnasadh": {"dates": ["08-01"], "significance": "First harvest"},
    
    # Numerological dates
    "911": {"dates": ["09-11"], "significance": "Emergency/crisis symbolism, Twin Towers"},
    "322": {"dates": ["03-22"], "significance": "Skull & Bones number"},
    "666_date": {"dates": ["06-06"], "significance": "Number of the beast"},
    "113": {"dates": ["01-13", "11-03"], "significance": "Dishonesty/betrayal in gematria"},
}

SIGNIFICANT_NUMBERS = {
    3: "Trinity, divine completion, Masonic",
    7: "Perfection, creation, luck",
    9: "Completion, judgment, enlightenment",
    11: "Master number, illumination, gateway",
    13: "Death/transformation, Templar, unlucky",
    22: "Master builder, Masonic degrees",
    33: "Highest Masonic degree, Christ age at death",
    39: "3x13, Masonic significance",
    42: "Answer to everything, 6x7",
    66: "Route 66, Revelation beast",
    93: "Thelema, 'Love' + 'Will'",
    322: "Skull & Bones, Genesis 3:22",
    666: "Number of the beast",
    777: "Divine completion, jackpot",
    911: "Emergency, Twin Towers date",
    1776: "Illuminati founding, US independence",
}

SYMBOL_DICTIONARY = {
    # Eye/Vision symbols
    "all_seeing_eye": {
        "description": "Eye in triangle/pyramid",
        "associations": ["Illuminati", "Freemasonry", "US dollar bill", "surveillance"],
        "meaning": "Divine providence, omniscience, control"
    },
    "eye_of_horus": {
        "description": "Egyptian eye symbol",
        "associations": ["Ancient Egypt", "protection", "royal power"],
        "meaning": "Protection, royal power, good health"
    },
    
    # Geometric symbols
    "pyramid": {
        "description": "Triangular structure",
        "associations": ["Egypt", "Illuminati", "corporate logos"],
        "meaning": "Hierarchy, power structure, ancient knowledge"
    },
    "obelisk": {
        "description": "Tall four-sided pillar",
        "associations": ["Washington Monument", "Vatican", "London", "Freemasonry"],
        "meaning": "Male principle, sun worship, power"
    },
    "hexagram": {
        "description": "Six-pointed star",
        "associations": ["Star of David", "Saturn", "alchemy"],
        "meaning": "As above so below, union of opposites"
    },
    "pentagram": {
        "description": "Five-pointed star",
        "associations": ["Wicca", "military", "inverted = Satanic"],
        "meaning": "Five elements, protection or inverted evil"
    },
    
    # Animal symbols
    "owl": {
        "description": "Nocturnal bird of prey",
        "associations": ["Bohemian Grove", "Minerva", "wisdom"],
        "meaning": "Hidden wisdom, secrecy, Moloch worship"
    },
    "phoenix": {
        "description": "Bird rising from flames",
        "associations": ["Rebirth", "Freemasonry", "alchemy"],
        "meaning": "Transformation, destruction and rebirth"
    },
    "serpent": {
        "description": "Snake/dragon",
        "associations": ["Eden", "kundalini", "medical symbol"],
        "meaning": "Knowledge, temptation, healing, evil"
    },
    
    # Hand symbols
    "hidden_hand": {
        "description": "Hand inside coat/jacket",
        "associations": ["Napoleon", "Masonic portraits"],
        "meaning": "Masonic allegiance, hidden knowledge"
    },
    "devil_horns": {
        "description": "Index and pinky extended",
        "associations": ["Rock music", "Texas Longhorns", "Italian curse"],
        "meaning": "Devil worship or protection from evil eye"
    },
    "666_hand": {
        "description": "OK sign forming 666",
        "associations": ["Celebrity photos", "politicians"],
        "meaning": "Allegiance to beast/Illuminati"
    },
}

SECRET_SOCIETIES = {
    "freemasonry": {
        "symbols": ["square_and_compass", "all_seeing_eye", "pyramid", "33"],
        "known_members": "Many US presidents, founding fathers",
        "influence_areas": ["government", "business", "entertainment"]
    },
    "skull_and_bones": {
        "symbols": ["skull_crossbones", "322", "coffin"],
        "known_members": "Bush family, Kerry, many Yale elites",
        "influence_areas": ["politics", "finance", "intelligence"]
    },
    "bilderberg": {
        "symbols": [],
        "known_members": "Politicians, CEOs, royalty",
        "influence_areas": ["policy", "media", "finance"]
    },
    "bohemian_grove": {
        "symbols": ["owl", "moloch"],
        "known_members": "US presidents, politicians, CEOs",
        "influence_areas": ["policy coordination"]
    },
    "cfr": {
        "name": "Council on Foreign Relations",
        "symbols": ["horse_rider"],
        "known_members": "Media heads, politicians, academics",
        "influence_areas": ["foreign policy", "media narrative"]
    },
    "trilateral_commission": {
        "symbols": [],
        "known_members": "Global elites, Rockefeller founded",
        "influence_areas": ["global governance", "trade policy"]
    },
}


@dataclass
class SymbolAnalysis:
    """Analysis of a detected symbol."""
    symbol_name: str
    description: str
    associations: List[str]
    meaning: str
    confidence: float
    context: str = ""


@dataclass
class TimingAnalysis:
    """Analysis of date/timing significance."""
    date: str
    matches: List[str]
    significance: str
    numerology: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DecoderReport:
    """Complete decoder analysis report."""
    query: str
    symbols_detected: List[SymbolAnalysis] = field(default_factory=list)
    timing_analysis: List[TimingAnalysis] = field(default_factory=list)
    numerological_patterns: List[Dict[str, Any]] = field(default_factory=list)
    society_connections: List[Dict[str, Any]] = field(default_factory=list)
    linguistic_patterns: List[str] = field(default_factory=list)
    hidden_meanings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "symbols_detected": [vars(s) for s in self.symbols_detected],
            "timing_analysis": [vars(t) for t in self.timing_analysis],
            "numerological_patterns": self.numerological_patterns,
            "society_connections": self.society_connections,
            "linguistic_patterns": self.linguistic_patterns,
            "hidden_meanings": self.hidden_meanings,
            "timestamp": self.timestamp.isoformat()
        }


class DecoderAgent:
    """
    Symbol and pattern decryption agent.
    
    Analyzes:
    - Symbols and imagery
    - Date/timing significance
    - Numerology and gematria
    - Secret society connections
    - Linguistic patterns and dog whistles
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.provider = None
    
    def _get_provider(self):
        if self.provider is None:
            self.provider = provider_factory(self.cfg)
        return self.provider
    
    def decode(
        self,
        query: str,
        content: str = "",
        dates: List[str] = None,
        thinking_callback: Optional[callable] = None
    ) -> DecoderReport:
        """
        Perform esoteric decoding on a topic.
        
        Args:
            query: The topic/query being researched
            content: Optional gathered content to analyze
            dates: Optional list of dates to analyze
            thinking_callback: Progress callback
            
        Returns:
            DecoderReport with analysis
        """
        report = DecoderReport(query=query)
        
        if thinking_callback:
            thinking_callback("🔮 Analyzing symbolic dimensions...")
        
        # Analyze for symbols
        report.symbols_detected = self._analyze_symbols(query, content)
        logger.info(f"🔍 Detected {len(report.symbols_detected)} symbols")
        
        if thinking_callback:
            thinking_callback("📅 Analyzing timing significance...")
        
        # Analyze dates/timing
        if dates:
            report.timing_analysis = self._analyze_timing(dates)
        else:
            # Try to extract dates from content
            report.timing_analysis = self._extract_and_analyze_timing(query, content)
        logger.info(f"📅 Found {len(report.timing_analysis)} timing patterns")
        
        if thinking_callback:
            thinking_callback("🔢 Analyzing numerological patterns...")
        
        # Numerology analysis
        report.numerological_patterns = self._analyze_numerology(query, content)
        
        if thinking_callback:
            thinking_callback("🏛️ Checking secret society connections...")
        
        # Society connections
        report.society_connections = self._check_society_connections(query, content)
        
        # Linguistic analysis for hidden meanings
        report.linguistic_patterns = self._analyze_linguistics(query, content)
        report.hidden_meanings = self._find_hidden_meanings(query, content)
        
        logger.info("✅ Decoding complete")
        return report
    
    def _analyze_symbols(self, query: str, content: str) -> List[SymbolAnalysis]:
        """Check for known symbols in query/content."""
        symbols = []
        combined = f"{query} {content}".lower()
        
        for symbol_key, symbol_info in SYMBOL_DICTIONARY.items():
            # Check if symbol name or associations appear
            keywords = [symbol_key.replace("_", " ")] + symbol_info.get("associations", [])
            
            for keyword in keywords:
                if keyword.lower() in combined:
                    symbols.append(SymbolAnalysis(
                        symbol_name=symbol_key,
                        description=symbol_info["description"],
                        associations=symbol_info["associations"],
                        meaning=symbol_info["meaning"],
                        confidence=0.7,
                        context=f"Matched keyword: {keyword}"
                    ))
                    break
        
        return symbols
    
    def _analyze_timing(self, dates: List[str]) -> List[TimingAnalysis]:
        """Analyze dates for esoteric significance."""
        analyses = []
        
        for date_str in dates:
            matches = []
            significance_parts = []
            
            # Try to parse date
            try:
                # Handle various formats
                if "-" in date_str:
                    parts = date_str.split("-")
                    if len(parts) >= 2:
                        month_day = f"{parts[1].zfill(2)}-{parts[2].zfill(2)}" if len(parts) >= 3 else date_str
                elif "/" in date_str:
                    parts = date_str.split("/")
                    if len(parts) >= 2:
                        month_day = f"{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                else:
                    month_day = date_str
                
                # Check against occult holidays
                for holiday_name, holiday_info in OCCULT_HOLIDAYS.items():
                    if month_day in holiday_info["dates"]:
                        matches.append(holiday_name)
                        significance_parts.append(holiday_info["significance"])
                
                if matches:
                    analyses.append(TimingAnalysis(
                        date=date_str,
                        matches=matches,
                        significance="; ".join(significance_parts)
                    ))
            except Exception as e:
                logger.debug(f"Could not parse date {date_str}: {e}")
        
        return analyses
    
    def _extract_and_analyze_timing(self, query: str, content: str) -> List[TimingAnalysis]:
        """Extract dates from content and analyze them."""
        analyses = []
        
        # Use LLM to extract dates
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Extract any dates mentioned in this text.
Return a JSON array of date strings in MM-DD format:
["01-15", "03-22", ...]

Text:
{query}
{content[:1500]}

Return ONLY valid JSON array. If no dates found, return []."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Extract dates accurately. Return valid JSON only.",
                temperature=0.1,
                options={"max_tokens": 200}
            )
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            dates = json.loads(response)
            if dates:
                analyses = self._analyze_timing(dates)
                
        except Exception as e:
            logger.debug(f"Date extraction failed: {e}")
        
        return analyses
    
    def _analyze_numerology(self, query: str, content: str) -> List[Dict[str, Any]]:
        """Find numerological patterns."""
        patterns = []
        combined = f"{query} {content}"
        
        # Check for significant numbers
        import re
        numbers_found = re.findall(r'\b(\d+)\b', combined)
        
        for num_str in numbers_found:
            try:
                num = int(num_str)
                if num in SIGNIFICANT_NUMBERS:
                    patterns.append({
                        "number": num,
                        "significance": SIGNIFICANT_NUMBERS[num],
                        "context": "Direct mention"
                    })
            except ValueError:
                pass
        
        # Check digit sums for hidden meanings
        for num_str in numbers_found:
            if len(num_str) >= 3:
                try:
                    digit_sum = sum(int(d) for d in num_str)
                    if digit_sum in SIGNIFICANT_NUMBERS:
                        patterns.append({
                            "number": f"{num_str} (digit sum = {digit_sum})",
                            "significance": SIGNIFICANT_NUMBERS[digit_sum],
                            "context": "Digit sum analysis"
                        })
                except ValueError:
                    pass
        
        return patterns[:10]  # Limit
    
    def _check_society_connections(self, query: str, content: str) -> List[Dict[str, Any]]:
        """Check for secret society connections."""
        connections = []
        combined = f"{query} {content}".lower()
        
        for society_key, society_info in SECRET_SOCIETIES.items():
            # Check if society name appears
            if society_key.replace("_", " ") in combined:
                connections.append({
                    "society": society_key,
                    "match_type": "direct_mention",
                    "symbols": society_info.get("symbols", []),
                    "influence_areas": society_info.get("influence_areas", [])
                })
            else:
                # Check if their symbols appear
                for symbol in society_info.get("symbols", []):
                    if str(symbol).lower() in combined:
                        connections.append({
                            "society": society_key,
                            "match_type": f"symbol_match: {symbol}",
                            "symbols": society_info.get("symbols", []),
                            "influence_areas": society_info.get("influence_areas", [])
                        })
                        break
        
        return connections
    
    def _analyze_linguistics(self, query: str, content: str) -> List[str]:
        """Analyze for linguistic patterns and dog whistles."""
        patterns = []
        
        # Common propaganda/manipulation patterns
        manipulation_phrases = [
            "experts say", "studies show", "it's been proven",
            "conspiracy theory", "misinformation", "fact checked",
            "debunked", "baseless claims", "without evidence"
        ]
        
        combined = f"{query} {content}".lower()
        
        for phrase in manipulation_phrases:
            if phrase in combined:
                patterns.append(f"Manipulation indicator: '{phrase}'")
        
        return patterns
    
    def _find_hidden_meanings(self, query: str, content: str) -> List[str]:
        """Use LLM to find potential hidden meanings."""
        meanings = []
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Analyze this topic for potential hidden meanings, symbolic significance, or esoteric connections.
Be specific and grounded. Only mention patterns that actually appear in the content.

Topic: {query}
Content: {content[:1500]}

Return a JSON array of potential hidden meanings:
["meaning 1", "meaning 2", ...]

Return ONLY valid JSON array."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="You are an esoteric analyst. Find hidden symbolic meanings. Be specific, not generic.",
                temperature=0.4,
                options={"max_tokens": 400}
            )
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            meanings = json.loads(response)
            
        except Exception as e:
            logger.debug(f"Hidden meanings analysis failed: {e}")
        
        return meanings[:5]


def decode_symbols(cfg: IceburgConfig, query: str, content: str = "") -> DecoderReport:
    """Convenience function for decoding."""
    agent = DecoderAgent(cfg)
    return agent.decode(query, content)
