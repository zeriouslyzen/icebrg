"""
COLOSSUS Risk Scoring

ML-powered risk assessment for entities.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk levels."""
    CRITICAL = "critical"  # 90-100
    HIGH = "high"          # 70-89
    MEDIUM = "medium"      # 40-69
    LOW = "low"            # 10-39
    MINIMAL = "minimal"    # 0-9


@dataclass
class RiskFactor:
    """Individual risk factor."""
    name: str
    category: str  # sanctions, pep, adverse_media, network, behavioral
    score: float  # 0-100
    weight: float  # Importance multiplier
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Complete risk assessment for an entity."""
    entity_id: str
    entity_name: str
    overall_score: float  # 0-100
    risk_level: RiskLevel
    factors: List[RiskFactor] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.now)


class RiskScorer:
    """
    Calculate risk scores for entities.
    
    Factors:
    - Sanctions presence and severity
    - PEP (Politically Exposed Person) status
    - Adverse media mentions
    - Network connections to high-risk entities
    - Behavioral patterns (transaction anomalies)
    - Jurisdictional risk
    """
    
    def __init__(
        self,
        use_llm_analysis: bool = True,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen2.5:14b",
    ):
        """
        Initialize risk scorer.
        
        Args:
            use_llm_analysis: Use LLM for narrative analysis
            ollama_host: Ollama API endpoint
            model: Model for analysis
        """
        self.use_llm_analysis = use_llm_analysis
        self.ollama_host = ollama_host
        self.model = model
        
        # Risk weights by category
        self.category_weights = {
            "sanctions": 1.0,     # Highest weight
            "pep": 0.7,
            "adverse_media": 0.6,
            "network": 0.5,
            "jurisdictional": 0.4,
            "behavioral": 0.3,
        }
        
        # Sanction list severity (how serious is each list)
        self.sanction_severity = {
            "us_ofac_sdn": 100,
            "us_ofac_cons": 90,
            "eu_sanctions": 95,
            "un_sanctions": 100,
            "gb_hmt_sanctions": 90,
            "au_dfat": 85,
            "ca_sema": 85,
            "ch_seco": 80,
            "ua_nsdc": 75,
            "default": 70,
        }
        
        # High-risk jurisdictions
        self.high_risk_jurisdictions = {
            "ir": 100,  # Iran
            "kp": 100,  # North Korea
            "sy": 95,   # Syria
            "ru": 80,   # Russia
            "by": 80,   # Belarus
            "cu": 75,   # Cuba
            "ve": 70,   # Venezuela
            "mm": 70,   # Myanmar
        }
    
    def assess(
        self,
        entity: Dict[str, Any],
        network_data: Optional[Dict[str, Any]] = None,
        media_data: Optional[List[str]] = None,
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            entity: Entity data
            network_data: Entity's network connections
            media_data: Adverse media mentions
            
        Returns:
            Complete risk assessment
        """
        factors = []
        
        # 1. Sanctions risk
        sanctions_factor = self._assess_sanctions(entity)
        if sanctions_factor:
            factors.append(sanctions_factor)
        
        # 2. PEP risk
        pep_factor = self._assess_pep(entity)
        if pep_factor:
            factors.append(pep_factor)
        
        # 3. Jurisdictional risk
        jurisdiction_factor = self._assess_jurisdiction(entity)
        if jurisdiction_factor:
            factors.append(jurisdiction_factor)
        
        # 4. Network risk
        if network_data:
            network_factor = self._assess_network(entity, network_data)
            if network_factor:
                factors.append(network_factor)
        
        # 5. Adverse media risk
        if media_data:
            media_factor = self._assess_media(entity, media_data)
            if media_factor:
                factors.append(media_factor)
        
        # Calculate weighted overall score
        overall_score = self._calculate_overall_score(factors)
        risk_level = self._score_to_level(overall_score)
        
        # Generate summary and recommendations
        summary = self._generate_summary(entity, factors, overall_score)
        recommendations = self._generate_recommendations(factors, risk_level)
        
        return RiskAssessment(
            entity_id=entity.get("entity_id", ""),
            entity_name=entity.get("name", ""),
            overall_score=overall_score,
            risk_level=risk_level,
            factors=factors,
            summary=summary,
            recommendations=recommendations,
        )
    
    def _assess_sanctions(self, entity: Dict[str, Any]) -> Optional[RiskFactor]:
        """Assess sanctions risk."""
        sanctions = entity.get("sanctions", []) or entity.get("datasets", [])
        
        if not sanctions:
            return None
        
        # Calculate score based on number and severity of sanctions
        max_severity = 0
        evidence = []
        
        for sanction in sanctions:
            sanction_key = sanction.lower().replace("-", "_")
            severity = self.sanction_severity.get(sanction_key, self.sanction_severity["default"])
            max_severity = max(max_severity, severity)
            evidence.append(f"Listed on {sanction}")
        
        # Boost for multiple sanctions
        count_boost = min(len(sanctions) * 5, 20)
        score = min(100, max_severity + count_boost)
        
        return RiskFactor(
            name="Sanctions Exposure",
            category="sanctions",
            score=score,
            weight=self.category_weights["sanctions"],
            evidence=evidence,
            sources=sanctions,
        )
    
    def _assess_pep(self, entity: Dict[str, Any]) -> Optional[RiskFactor]:
        """Assess PEP (Politically Exposed Person) risk."""
        datasets = entity.get("datasets", [])
        properties = entity.get("properties", {})
        
        # Check for PEP indicators
        pep_indicators = [d for d in datasets if "pep" in d.lower()]
        positions = properties.get("position", [])
        
        if not pep_indicators and not positions:
            return None
        
        score = 50  # Base PEP score
        evidence = []
        
        if pep_indicators:
            score += 20
            evidence.extend([f"Listed in {d}" for d in pep_indicators])
        
        if positions:
            # High-risk positions
            high_risk_keywords = ["president", "minister", "governor", "judge", "military"]
            for position in positions:
                if any(k in position.lower() for k in high_risk_keywords):
                    score += 15
                    evidence.append(f"Position: {position}")
        
        return RiskFactor(
            name="Politically Exposed Person",
            category="pep",
            score=min(100, score),
            weight=self.category_weights["pep"],
            evidence=evidence,
        )
    
    def _assess_jurisdiction(self, entity: Dict[str, Any]) -> Optional[RiskFactor]:
        """Assess jurisdictional risk."""
        countries = entity.get("countries", [])
        
        if not countries:
            return None
        
        max_risk = 0
        evidence = []
        
        for country in countries:
            country_lower = country.lower()
            risk = self.high_risk_jurisdictions.get(country_lower, 0)
            if risk > 0:
                max_risk = max(max_risk, risk)
                evidence.append(f"High-risk jurisdiction: {country.upper()}")
        
        if max_risk == 0:
            return None
        
        return RiskFactor(
            name="Jurisdictional Risk",
            category="jurisdictional",
            score=max_risk,
            weight=self.category_weights["jurisdictional"],
            evidence=evidence,
        )
    
    def _assess_network(
        self,
        entity: Dict[str, Any],
        network_data: Dict[str, Any]
    ) -> Optional[RiskFactor]:
        """Assess network connection risk."""
        nodes = network_data.get("nodes", [])
        
        high_risk_connections = 0
        evidence = []
        
        for node in nodes:
            if node.get("id") == entity.get("entity_id"):
                continue
            
            sanctions_count = node.get("sanctions_count", 0)
            if sanctions_count > 0:
                high_risk_connections += 1
                evidence.append(f"Connected to sanctioned entity: {node.get('name')}")
        
        if high_risk_connections == 0:
            return None
        
        # Score based on number of risky connections
        score = min(100, high_risk_connections * 15)
        
        return RiskFactor(
            name="Network Connections",
            category="network",
            score=score,
            weight=self.category_weights["network"],
            evidence=evidence[:5],  # Limit evidence
        )
    
    def _assess_media(
        self,
        entity: Dict[str, Any],
        media_data: List[str]
    ) -> Optional[RiskFactor]:
        """Assess adverse media risk."""
        if not media_data:
            return None
        
        # Keyword analysis
        negative_keywords = [
            "fraud", "corruption", "money laundering", "bribery",
            "investigation", "charged", "convicted", "lawsuit",
            "sanction", "embargo", "criminal"
        ]
        
        negative_count = 0
        evidence = []
        
        for article in media_data:
            article_lower = article.lower()
            for keyword in negative_keywords:
                if keyword in article_lower:
                    negative_count += 1
                    if len(evidence) < 3:
                        evidence.append(f"Media mention: '{keyword}' found")
                    break
        
        if negative_count == 0:
            return None
        
        score = min(100, negative_count * 20)
        
        return RiskFactor(
            name="Adverse Media",
            category="adverse_media",
            score=score,
            weight=self.category_weights["adverse_media"],
            evidence=evidence,
        )
    
    def _calculate_overall_score(self, factors: List[RiskFactor]) -> float:
        """Calculate weighted overall risk score."""
        if not factors:
            return 0.0
        
        weighted_sum = sum(f.score * f.weight for f in factors)
        weight_sum = sum(f.weight for f in factors)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert score to risk level."""
        if score >= 90:
            return RiskLevel.CRITICAL
        elif score >= 70:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MEDIUM
        elif score >= 10:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _generate_summary(
        self,
        entity: Dict[str, Any],
        factors: List[RiskFactor],
        score: float
    ) -> str:
        """Generate risk summary."""
        name = entity.get("name", "Unknown")
        level = self._score_to_level(score)
        
        summary = f"{name} has a {level.value.upper()} risk level (score: {score:.0f}/100)."
        
        if factors:
            top_factor = max(factors, key=lambda f: f.score * f.weight)
            summary += f" Primary concern: {top_factor.name} ({top_factor.score:.0f})."
        
        return summary
    
    def _generate_recommendations(
        self,
        factors: List[RiskFactor],
        level: RiskLevel
    ) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []
        
        if level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("Conduct enhanced due diligence before proceeding")
            recommendations.append("Escalate to compliance team for review")
        
        for factor in factors:
            if factor.category == "sanctions" and factor.score >= 80:
                recommendations.append("Verify sanctions status with official sources")
            elif factor.category == "pep" and factor.score >= 50:
                recommendations.append("Obtain PEP declaration and source of wealth information")
            elif factor.category == "network" and factor.score >= 50:
                recommendations.append("Map full network connections and assess each")
        
        return recommendations[:5]  # Limit recommendations
