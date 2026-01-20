"""
Source Scorer - Credibility and bias scoring for sources.
Evaluates source trustworthiness for intelligence gathering.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


@dataclass
class SourceScore:
    """Credibility and bias score for a source."""
    url: str
    domain: str
    credibility_score: float  # 0.0 to 1.0
    bias_score: float  # -1.0 (left) to 1.0 (right), 0.0 = center
    bias_label: str  # 'far-left', 'left', 'center-left', 'center', 'center-right', 'right', 'far-right'
    source_type: str  # 'mainstream', 'alternative', 'academic', 'government', 'corporate', 'unknown'
    flags: List[str] = field(default_factory=list)  # Warning flags
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "domain": self.domain,
            "credibility_score": self.credibility_score,
            "bias_score": self.bias_score,
            "bias_label": self.bias_label,
            "source_type": self.source_type,
            "flags": self.flags,
            "notes": self.notes
        }


class SourceScorer:
    """
    Source credibility and bias scorer.
    
    Uses a database of known sources plus heuristics for unknown sources.
    """
    
    # Known source database with credibility and bias ratings
    # Credibility: 0.0 (unreliable) to 1.0 (highly reliable)
    # Bias: -1.0 (far-left) to 1.0 (far-right)
    KNOWN_SOURCES = {
        # Wire services (high credibility, low bias)
        "reuters.com": {"credibility": 0.9, "bias": 0.0, "type": "mainstream", "label": "center"},
        "apnews.com": {"credibility": 0.9, "bias": 0.0, "type": "mainstream", "label": "center"},
        "afp.com": {"credibility": 0.85, "bias": 0.0, "type": "mainstream", "label": "center"},
        
        # Mainstream (varying bias)
        "nytimes.com": {"credibility": 0.8, "bias": -0.3, "type": "mainstream", "label": "center-left"},
        "washingtonpost.com": {"credibility": 0.8, "bias": -0.3, "type": "mainstream", "label": "center-left"},
        "wsj.com": {"credibility": 0.85, "bias": 0.2, "type": "mainstream", "label": "center-right"},
        "bbc.com": {"credibility": 0.85, "bias": -0.1, "type": "mainstream", "label": "center"},
        "bbc.co.uk": {"credibility": 0.85, "bias": -0.1, "type": "mainstream", "label": "center"},
        "theguardian.com": {"credibility": 0.75, "bias": -0.4, "type": "mainstream", "label": "left"},
        "foxnews.com": {"credibility": 0.6, "bias": 0.6, "type": "mainstream", "label": "right"},
        "cnn.com": {"credibility": 0.65, "bias": -0.4, "type": "mainstream", "label": "left"},
        "msnbc.com": {"credibility": 0.6, "bias": -0.5, "type": "mainstream", "label": "left"},
        
        # Alternative/independent (varying credibility)
        "theintercept.com": {"credibility": 0.75, "bias": -0.5, "type": "alternative", "label": "left"},
        "zerohedge.com": {"credibility": 0.5, "bias": 0.4, "type": "alternative", "label": "right"},
        "mintpressnews.com": {"credibility": 0.55, "bias": -0.5, "type": "alternative", "label": "left"},
        "grayzone.com": {"credibility": 0.6, "bias": -0.6, "type": "alternative", "label": "far-left"},
        "unlimitedhangout.com": {"credibility": 0.5, "bias": 0.0, "type": "alternative", "label": "center"},
        "corbettreport.com": {"credibility": 0.45, "bias": 0.3, "type": "alternative", "label": "right"},
        "breitbart.com": {"credibility": 0.4, "bias": 0.8, "type": "alternative", "label": "far-right"},
        "infowars.com": {"credibility": 0.2, "bias": 0.9, "type": "alternative", "label": "far-right"},
        "motherjones.com": {"credibility": 0.65, "bias": -0.6, "type": "alternative", "label": "far-left"},
        "dailykos.com": {"credibility": 0.5, "bias": -0.7, "type": "alternative", "label": "far-left"},
        
        # Academic (high credibility)
        "arxiv.org": {"credibility": 0.95, "bias": 0.0, "type": "academic", "label": "center"},
        "nature.com": {"credibility": 0.95, "bias": 0.0, "type": "academic", "label": "center"},
        "science.org": {"credibility": 0.95, "bias": 0.0, "type": "academic", "label": "center"},
        "pubmed.ncbi.nlm.nih.gov": {"credibility": 0.95, "bias": 0.0, "type": "academic", "label": "center"},
        "jstor.org": {"credibility": 0.9, "bias": 0.0, "type": "academic", "label": "center"},
        
        # Government (official but potentially biased)
        "whitehouse.gov": {"credibility": 0.7, "bias": 0.0, "type": "government", "label": "center", "flags": ["official_source"]},
        "state.gov": {"credibility": 0.7, "bias": 0.0, "type": "government", "label": "center", "flags": ["official_source"]},
        "gov.uk": {"credibility": 0.75, "bias": 0.0, "type": "government", "label": "center", "flags": ["official_source"]},
        "kremlin.ru": {"credibility": 0.5, "bias": 0.0, "type": "government", "label": "center", "flags": ["state_media"]},
        "rt.com": {"credibility": 0.4, "bias": 0.0, "type": "government", "label": "center", "flags": ["state_media", "propaganda_risk"]},
        
        # Fact-checkers (high credibility, some bias concerns)
        "snopes.com": {"credibility": 0.75, "bias": -0.2, "type": "fact_checker", "label": "center-left"},
        "politifact.com": {"credibility": 0.75, "bias": -0.2, "type": "fact_checker", "label": "center-left"},
        "factcheck.org": {"credibility": 0.8, "bias": -0.1, "type": "fact_checker", "label": "center"},
        
        # Wikipedia (good for background, needs verification)
        "wikipedia.org": {"credibility": 0.7, "bias": -0.1, "type": "encyclopedia", "label": "center-left", "flags": ["verify_claims"]},
        "en.wikipedia.org": {"credibility": 0.7, "bias": -0.1, "type": "encyclopedia", "label": "center-left", "flags": ["verify_claims"]},
    }
    
    # Domain patterns for type detection
    DOMAIN_PATTERNS = {
        "academic": [".edu", "arxiv", "researchgate", "academia.edu", "springer", "wiley", "elsevier"],
        "government": [".gov", ".mil", "state.gov", "whitehouse", "congress.gov"],
        "corporate": ["pr.newswire", "businesswire", "globenewswire"],
    }
    
    def __init__(self):
        """Initialize source scorer."""
        pass
    
    def score(self, url: str) -> SourceScore:
        """
        Score a source URL.
        
        Args:
            url: URL to score
            
        Returns:
            SourceScore with credibility and bias ratings
        """
        domain = self._extract_domain(url)
        
        # Check known sources
        for known_domain, data in self.KNOWN_SOURCES.items():
            if known_domain in domain:
                return SourceScore(
                    url=url,
                    domain=domain,
                    credibility_score=data["credibility"],
                    bias_score=data["bias"],
                    bias_label=data["label"],
                    source_type=data["type"],
                    flags=data.get("flags", []),
                    notes=f"Known source: {known_domain}"
                )
        
        # Unknown source - use heuristics
        return self._score_unknown(url, domain)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www.
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return url
    
    def _score_unknown(self, url: str, domain: str) -> SourceScore:
        """Score an unknown source using heuristics."""
        credibility = 0.5  # Default
        source_type = "unknown"
        flags = []
        
        # Check domain patterns
        for stype, patterns in self.DOMAIN_PATTERNS.items():
            if any(p in domain for p in patterns):
                source_type = stype
                if stype == "academic":
                    credibility = 0.8
                elif stype == "government":
                    credibility = 0.65
                    flags.append("official_source")
                elif stype == "corporate":
                    credibility = 0.5
                    flags.append("pr_release")
                break
        
        # Check for suspicious patterns
        suspicious_patterns = [
            "news24", "daily", "patriot", "truth", "real", "free",
            "exposed", "uncensored", "leak"
        ]
        if any(p in domain.lower() for p in suspicious_patterns):
            credibility = min(credibility, 0.4)
            flags.append("unverified_source")
        
        # Check for social media
        social_domains = ["twitter.com", "x.com", "facebook.com", "reddit.com", "youtube.com"]
        if any(s in domain for s in social_domains):
            source_type = "social_media"
            credibility = 0.3
            flags.append("social_media")
        
        return SourceScore(
            url=url,
            domain=domain,
            credibility_score=credibility,
            bias_score=0.0,  # Unknown bias
            bias_label="unknown",
            source_type=source_type,
            flags=flags,
            notes="Unknown source - using heuristics"
        )
    
    def score_batch(self, urls: List[str]) -> List[SourceScore]:
        """Score multiple URLs."""
        return [self.score(url) for url in urls]
    
    def get_credibility_label(self, score: float) -> str:
        """Convert credibility score to label."""
        if score >= 0.8:
            return "HIGH"
        elif score >= 0.6:
            return "MEDIUM"
        elif score >= 0.4:
            return "LOW"
        else:
            return "VERY LOW"


def score_source(url: str) -> SourceScore:
    """Convenience function for scoring a source."""
    scorer = SourceScorer()
    return scorer.score(url)
