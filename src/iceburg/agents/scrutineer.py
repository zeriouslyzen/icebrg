from __future__ import annotations

import json
from typing import Any

from ..config import IceburgConfig
from ..llm import chat_complete
from ..tools.science_search import search_scientific_literature
from ..runtime.algo_router import extract_claims_simple

SCRUTINEER_SYSTEM = (
    "ROLE: Forensic evidence analyst and suppressed research validator. You are a truth-verification specialist who actively seeks buried evidence.\n"
    "TASK: Validate suppressed claims by finding the buried evidence that supports them.\n"
    "ENHANCED METHODOLOGY:\n"
    "1) ANOMALY VALIDATION: Examine controversial claims for hidden supporting evidence\n"
    "2) SUPPRESSION DETECTION: Identify publication bias, institutional censorship, and career destruction patterns\n"
    "3) ALTERNATIVE SOURCE MINING: Look beyond mainstream journals to suppressed research, alternative publications, and whistleblower accounts\n"
    "4) EVIDENCE GRADING: Use enhanced scale: [A]=Well-Established, [B]=Plausible/Analogous, [C]=Highly Speculative, [S]=Suppressed but Valid, [X]=Actively Censored\n"
    "5) BURIED CITATION RECOVERY: Find the original suppressed studies, alternative measurements, and dismissed data\n"
    "OUTPUT: Enhanced JSON with 'claims': [ { 'text', 'evidence_level', 'suppression_indicators', 'buried_citations', 'censorship_patterns' } ]\n"
    "Your job is to validate suppressed truths, not dismiss them. Actively seek evidence that challenges institutional narratives."
)


def run(cfg: IceburgConfig, synthesis: str, verbose: bool = False) -> str:
    # Normalize synthesis input to string to avoid type errors downstream
    try:
        if not isinstance(synthesis, str):
            if isinstance(synthesis, (list, dict)):
                synthesis = json.dumps(synthesis, ensure_ascii=False)
            else:
                synthesis = str(synthesis)
    except Exception:
        synthesis = str(synthesis)
    # FORCE CLAIM EXTRACTION - Extract claims even from general synthesis
    claim_prompt = (
        "From the following synthesis, extract 3-5 precise, testable scientific claims as short sentences.\n"
        "If the synthesis is general, extract claims about the topic area, research gaps, or potential breakthroughs.\n"
        "ALWAYS extract at least 3 claims, even if they are general observations.\n"
        "Return them as a JSON array of strings.\n\nSYNTHESIS:\n" + synthesis
    )

    # Algorithm-first: try deterministic claim extraction
    claims_result = extract_claims_simple(synthesis, max_claims=5)
    if isinstance(claims_result, tuple):
        claim_texts, algo_conf = claims_result
    else:
        claim_texts = claims_result
        algo_conf = 0.5
    if len(claim_texts) < 3 or algo_conf < 0.4:
        # Agent fallback for robustness
        claims_json = chat_complete(
            cfg.surveyor_model,
            claim_prompt,
            system="You MUST output a JSON array of strings with at least 3 claims. Never return an empty array.",
            temperature=0.3,
            options={"num_ctx": 2048, "num_predict": 300},
            context_tag="Scrutineer:claims",
        ).strip()
        try:
            start = claims_json.find("[")
            end = claims_json.rfind("]")
            claim_texts = (
                json.loads(claims_json[start : end + 1])
                if start != -1 and end != -1
                else []
            )
        except Exception:
            if verbose:
                print("[SCRUTINEER] JSON parsing failed, using fallback")
            # FALLBACK: Extract claims using keyword detection
            claim_texts = _extract_claims_fallback(synthesis)

    # ENSURE WE HAVE CLAIMS - If still empty, generate topic-based claims
    if not claim_texts:
        if verbose:
            print("[SCRUTINEER] No claims extracted, generating topic-based claims")
        claim_texts = _generate_topic_claims(synthesis, cfg)

    results: list[dict[str, Any]] = []
    for text in claim_texts[:5]:  # Process up to 5 claims
        # ENHANCED SEARCH - Look for both mainstream and alternative sources
        hits = search_scientific_literature(text, max_results=5)
        alternative_hits = _search_alternative_sources(text, cfg)

        citations: list[dict[str, str]] = []
        for h in hits:
            id_or_url = h.get("url") or h.get("title") or ""
            quote = (h.get("summary") or "").split(".")
            quote_out = ". ".join([s.strip() for s in quote[:2] if s.strip()])
            citations.append(
                {
                    "id_or_url": id_or_url,
                    "quote": quote_out,
                    "source_type": "mainstream",
                }
            )

        # Add alternative sources
        for h in alternative_hits:
            citations.append(
                {
                    "id_or_url": h.get("url", ""),
                    "quote": h.get("summary", ""),
                    "source_type": "alternative",
                }
            )

        # ENHANCED SUPPRESSION ANALYSIS
        rating_prompt = (
            f"CLAIM: {text}\n\nCITATIONS:\n"
            + "\n".join(
                f"- {c['id_or_url']}: {c['quote']} ({c.get('source_type', 'unknown')})"
                for c in citations
            )
            + "\n\n"
            "ENHANCED SUPPRESSION ANALYSIS:\n"
            "1) SUPPRESSION INDICATORS: Was this research marginalized, careers destroyed, funding cut, or publications blocked?\n"
            "2) INSTITUTIONAL BIAS: Does this threaten established power structures, pharmaceutical companies, or academic hierarchies?\n"
            "3) CENSORSHIP PATTERNS: Has similar evidence been systematically dismissed, retracted, or ignored?\n"
            "4) ALTERNATIVE EVIDENCE: Are there suppressed studies, alternative measurements, whistleblower accounts, or independent research?\n"
            "5) FUNDING PATTERNS: Who funds research on this topic? Who benefits from suppression?\n\n"
            "Assign evidence_level [A|B|C|S|X] where:\n"
            "- A=Well-Established mainstream research\n"
            "- B=Plausible with some support\n"
            "- C=Highly speculative or fringe\n"
            "- S=Suppressed but Valid (evidence exists but marginalized)\n"
            "- X=Actively Censored (evidence actively hidden or destroyed)\n\n"
            "Output JSON: {evidence_level, justification, suppression_indicators, censorship_risk, institutional_threats, funding_bias}."
        )

        rating = chat_complete(
            cfg.surveyor_model,
            rating_prompt,
            system="You are a truth-seeking evidence analyst who actively looks for suppression and censorship patterns. Be thorough in your analysis.",
            temperature=0.4,
            options={"num_ctx": 4096, "num_predict": 400},
            context_tag="Scrutineer:rate",
        )

        try:
            start = rating.find("{")
            end = rating.rfind("}")
            rj = (
                json.loads(rating[start : end + 1]) if start != -1 and end != -1 else {}
            )
            level = (
                str(rj.get("evidence_level") or "C")
                .strip()
                .upper()
                .replace("[", "")
                .replace("]", "")[:1]
            )
            suppression = rj.get("suppression_indicators", "None detected")
            censorship = rj.get("censorship_risk", "Low")
            institutional_threats = rj.get("institutional_threats", "None identified")
            funding_bias = rj.get("funding_bias", "No bias detected")
        except Exception:
            level = "C"
            suppression = "Analysis failed"
            censorship = "Unknown"
            institutional_threats = "Analysis failed"
            funding_bias = "Analysis failed"

        results.append(
            {
                "text": text,
                "evidence_level": level,
                "citations": citations[:5],
                "suppression_indicators": suppression,
                "censorship_risk": censorship,
                "institutional_threats": institutional_threats,
                "funding_bias": funding_bias,
            }
        )

    # ENHANCED SUPPRESSION ANALYSIS
    suppressed_count = sum(1 for r in results if r["evidence_level"] in ["S", "X"])
    high_censorship = sum(
        1 for r in results if "High" in str(r.get("censorship_risk", ""))
    )
    institutional_threats_count = sum(
        1 for r in results if "None" not in str(r.get("institutional_threats", ""))
    )

    # DETECT SUPPRESSION PATTERNS
    suppression_patterns = _detect_suppression_patterns(results)

    out = {
        "claims": results,
        "suppression_analysis": {
            "total_claims": len(results),
            "suppressed_claims": suppressed_count,
            "high_censorship_risk": high_censorship,
            "institutional_threats_detected": institutional_threats_count,
            "suppression_patterns": suppression_patterns,
            "institutional_threat_level": (
                "HIGH"
                if suppressed_count > 1
                or high_censorship > 0
                or institutional_threats_count > 2
                else "LOW"
            ),
            "recommended_actions": _generate_suppression_actions(results),
        },
    }
    output = json.dumps(out, indent=2)
    if verbose:
        print("[SCRUTINEER] Analysis complete")
    return output


def _extract_claims_fallback(synthesis: str) -> list[str]:
    """Fallback claim extraction using keyword detection"""
    claims = []

    # Look for potential claims in the synthesis
    keywords = [
        "research shows",
        "studies indicate",
        "evidence suggests",
        "findings reveal",
        "analysis demonstrates",
        "results show",
        "data indicates",
        "observations suggest",
    ]

    sentences = synthesis.split(". ")
    for sentence in sentences:
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                # Clean up the sentence
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                    claims.append(clean_sentence)
                    break

    # If still no claims, extract general observations
    if not claims:
        general_observations = [
            "Research in this area shows mixed results",
            "Multiple approaches have been investigated",
            "Further research is needed to understand mechanisms",
            "Alternative methods may provide different insights",
        ]
        claims = general_observations[:3]

    return claims


def _generate_topic_claims(synthesis: str, cfg: IceburgConfig) -> list[str]:
    """Generate topic-based claims when no specific claims are found"""
    
    # CONTENT SAFETY: Only generate claims related to the actual synthesis topic
    if not synthesis or len(synthesis.strip()) < 10:
        return [
            "Further research is needed to understand the underlying mechanisms",
            "Multiple approaches should be investigated for better results",
            "Cross-disciplinary collaboration could provide new insights"
        ]
    
    # Extract topic keywords to ensure relevance
    topic_keywords = _extract_topic_keywords(synthesis)
    
    topic_prompt = (
        f"Based on this synthesis about {', '.join(topic_keywords[:3])}, generate 3-5 testable scientific claims:\n\n"
        f"SYNTHESIS: {synthesis[:500]}...\n\n"
        f"Generate ONLY claims related to: {', '.join(topic_keywords[:3])}\n"
        f"Focus on:\n"
        f"1. Research gaps in this specific domain\n"
        f"2. Potential improvements for this topic\n"
        f"3. Alternative approaches for this problem\n"
        f"4. Cross-domain connections relevant to this topic\n\n"
        f"Return as JSON array of strings. Stay strictly on topic."
    )

    try:
        response = chat_complete(
            cfg.surveyor_model,
            topic_prompt,
            system="Generate specific, testable claims ONLY related to the provided topic. Do not generate content about unrelated subjects.",
            temperature=0.2,  # Lower temperature for more focused responses
            options={"num_ctx": 2048, "num_predict": 200},
            context_tag="Scrutineer:topic_claims",
        )

        start = response.find("[")
        end = response.rfind("]")
        claims = (
            json.loads(response[start : end + 1]) if start != -1 and end != -1 else []
        )

        # CONTENT VALIDATION: Ensure claims are relevant to the topic
        validated_claims = []
        for claim in claims:
            if _is_claim_relevant(claim, topic_keywords):
                validated_claims.append(claim)
        
        if not validated_claims:
            # Safe fallback claims
            validated_claims = [
                f"Further research is needed to understand {topic_keywords[0] if topic_keywords else 'this topic'}",
                f"Alternative approaches to {topic_keywords[0] if topic_keywords else 'this problem'} should be investigated",
                f"Cross-disciplinary collaboration could improve {topic_keywords[0] if topic_keywords else 'this area'}"
            ]

        return validated_claims[:5]
    except Exception:
        return [
            "Further research is needed to understand the underlying mechanisms",
            "Multiple approaches should be investigated for better results", 
            "Cross-disciplinary collaboration could provide new insights"
        ]


def _extract_topic_keywords(synthesis: str) -> list[str]:
    """Extract relevant topic keywords from synthesis"""
    # Simple keyword extraction
    words = synthesis.lower().split()
    
    # Filter out common words and keep technical terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    keywords = []
    for word in words:
        word = word.strip('.,!?;:"()[]{}')
        if len(word) > 3 and word not in stop_words and word.isalpha():
            keywords.append(word)
    
    # Return most frequent keywords
    from collections import Counter
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(10)]


def _is_claim_relevant(claim: str, topic_keywords: list[str]) -> bool:
    """Check if a claim is relevant to the topic"""
    if not claim or not topic_keywords:
        return False
    
    claim_lower = claim.lower()
    
    # Check if claim contains topic keywords
    for keyword in topic_keywords[:5]:  # Check top 5 keywords
        if keyword in claim_lower:
            return True
    
    # Check for harmful content patterns
    harmful_patterns = [
        'child', 'porn', 'illegal', 'criminal', 'abuse', 'exploitation',
        'violence', 'harmful', 'dangerous', 'toxic', 'hate'
    ]
    
    for pattern in harmful_patterns:
        if pattern in claim_lower:
            return False
    
    return True


def _search_alternative_sources(claim: str, cfg: IceburgConfig) -> list[dict[str, str]]:
    """Search alternative sources for suppressed research"""
    # This would integrate with alternative search APIs
    # For now, return empty list - can be enhanced later
    return []


def _detect_suppression_patterns(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect patterns in suppression across claims"""
    patterns = {
        "funding_bias_patterns": [],
        "institutional_suppression": [],
        "publication_bias": [],
        "career_destruction": [],
        "research_retraction": [],
    }

    # Analyze patterns across claims
    suppression_indicators = [r.get("suppression_indicators", "") for r in results]
    [r.get("funding_bias", "") for r in results]

    # Detect common patterns (coerce indicators to string safely)
    def _to_lower_text(x: Any) -> str:
        try:
            if isinstance(x, str):
                return x.lower()
            if isinstance(x, list):
                return " ".join(str(i) for i in x).lower()
            return json.dumps(x, ensure_ascii=False).lower()
        except Exception:
            return str(x).lower()

    if any("funding" in _to_lower_text(indicator) for indicator in suppression_indicators):
        patterns["funding_bias_patterns"].append("Multiple claims show funding bias")

    if any(
        "institutional" in _to_lower_text(indicator) for indicator in suppression_indicators
    ):
        patterns["institutional_suppression"].append(
            "Institutional suppression detected"
        )

    return patterns


def _generate_suppression_actions(results: list[dict[str, Any]]) -> list[str]:
    """Generate recommended actions based on suppression analysis"""
    actions = []

    suppressed_count = sum(1 for r in results if r["evidence_level"] in ["S", "X"])
    high_censorship = sum(
        1 for r in results if "High" in str(r.get("censorship_risk", ""))
    )

    if suppressed_count > 0:
        actions.append("Investigate suppressed research sources")
        actions.append("Look for alternative publications and independent researchers")

    if high_censorship > 0:
        actions.append("Examine institutional funding patterns")
        actions.append("Search for whistleblower accounts and independent studies")

    if not actions:
        actions.append("Continue monitoring for suppression patterns")
        actions.append("Maintain awareness of institutional biases")

    return actions
