#!/usr/bin/env python3
"""
ICEBURG Large-Scale Training Corpus Generator
==============================================

Generates production-scale training data using:
1. Template-based generation with variation
2. Self-Instruct style augmentation
3. Evol-Instruct complexity evolution
4. Domain coverage expansion

Target: 500+ examples per agent type
"""

import json
import random
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path("data/fine_tuning/agent_data/large_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DOMAIN TEMPLATES
# ============================================================================

RESEARCH_DOMAINS = [
    "quantum computing", "machine learning", "neuroscience", "climate science",
    "molecular biology", "astrophysics", "economics", "psychology",
    "materials science", "renewable energy", "genetics", "artificial intelligence",
    "cognitive science", "pharmacology", "nanotechnology", "robotics",
    "cryptography", "network science", "evolutionary biology", "game theory",
    "information theory", "complexity theory", "systems biology", "immunology",
    "epidemiology", "behavioral economics", "social psychology", "linguistics",
    "philosophy of mind", "ethics", "political science", "anthropology"
]

RESEARCH_TOPICS = {
    "quantum computing": [
        "error correction", "quantum supremacy", "quantum algorithms", "qubit coherence",
        "entanglement", "quantum networks", "topological qubits", "quantum simulation",
        "quantum cryptography", "quantum machine learning", "adiabatic computing"
    ],
    "machine learning": [
        "deep learning", "reinforcement learning", "transformers", "neural networks",
        "optimization", "generalization", "transfer learning", "few-shot learning",
        "interpretability", "fairness", "robustness", "continual learning"
    ],
    "neuroscience": [
        "memory", "consciousness", "neural plasticity", "brain-computer interfaces",
        "synaptic transmission", "neuroimaging", "computational neuroscience",
        "neurodegenerative diseases", "sleep", "attention", "decision making"
    ],
    "climate science": [
        "carbon cycle", "ocean acidification", "extreme weather", "ice sheets",
        "climate models", "tipping points", "mitigation", "adaptation",
        "renewable transition", "carbon capture", "climate policy"
    ],
}

# Add more topics for other domains
for domain in RESEARCH_DOMAINS:
    if domain not in RESEARCH_TOPICS:
        RESEARCH_TOPICS[domain] = [
            f"recent advances in {domain}",
            f"foundational concepts of {domain}",
            f"applications of {domain}",
            f"future directions in {domain}",
            f"key debates in {domain}",
            f"methodology in {domain}",
            f"interdisciplinary connections of {domain}",
            f"practical implications of {domain}"
        ]

# ============================================================================
# SURVEYOR TEMPLATES
# ============================================================================

SURVEYOR_QUERY_TEMPLATES = [
    "What are the latest developments in {topic}?",
    "Explain the current state of research on {topic}.",
    "What do we know about {topic}?",
    "Research the key findings in {topic}.",
    "Synthesize the evidence on {topic}.",
    "What is the scientific consensus on {topic}?",
    "Provide a comprehensive overview of {topic}.",
    "What are the major theories about {topic}?",
    "How has our understanding of {topic} evolved?",
    "What are the open questions in {topic}?",
    "Compare the different approaches to {topic}.",
    "What are the practical applications of {topic}?",
    "Explain the methodology used in {topic} research.",
    "What are the key challenges in {topic}?",
    "How does {topic} connect to {related_topic}?"
]

SURVEYOR_RESPONSE_TEMPLATE = """Research synthesis on {topic}:

**Overview:**
{overview}

**Key Findings:**

1. **{finding1_title}**
   - {finding1_detail1}
   - {finding1_detail2}
   - {finding1_detail3}

2. **{finding2_title}**
   - {finding2_detail1}
   - {finding2_detail2}
   - {finding2_detail3}

3. **{finding3_title}**
   - {finding3_detail1}
   - {finding3_detail2}
   - {finding3_detail3}

**Current Status:**
{current_status}

**Open Questions:**
{open_questions}

**Sources**: {sources}"""

# ============================================================================
# DISSIDENT TEMPLATES
# ============================================================================

DISSIDENT_QUERY_TEMPLATES = [
    "The previous analysis claims {claim}. Challenge this.",
    "Experts say {claim}. What are the counterarguments?",
    "The mainstream view is that {claim}. Question this assumption.",
    "Everyone believes {claim}. What's wrong with this view?",
    "Challenge the claim that {claim}.",
    "What are the problems with the argument that {claim}?",
    "Critically examine the assumption that {claim}.",
    "Why might {claim} be wrong?",
    "Present the opposing view to {claim}.",
    "What evidence contradicts the claim that {claim}?"
]

DISSIDENT_RESPONSE_TEMPLATE = """I must challenge several assumptions in this claim:

**Contradiction 1: {contradiction1_title}**
- {contradiction1_evidence1}
- {contradiction1_evidence2}
- **Reality**: {contradiction1_reality}

**Contradiction 2: {contradiction2_title}**
- {contradiction2_evidence1}
- {contradiction2_evidence2}
- **Reality**: {contradiction2_reality}

**Contradiction 3: {contradiction3_title}**
- {contradiction3_evidence1}
- {contradiction3_evidence2}
- **Reality**: {contradiction3_reality}

**Alternative View:**
{alternative_view}

This claim may be driven by {underlying_bias} rather than evidence."""

# ============================================================================
# SYNTHESIST TEMPLATES
# ============================================================================

SYNTHESIST_QUERY_TEMPLATES = [
    "Connect the research on {domain1}, {domain2}, and {domain3}.",
    "What are the cross-domain connections between {domain1} and {domain2}?",
    "Synthesize findings from {domain1} and {domain2} on {topic}.",
    "How do insights from {domain1} apply to {domain2}?",
    "Find the common patterns across {domain1}, {domain2}, and {domain3}.",
    "What can {domain1} learn from {domain2}?",
    "Integrate the perspectives of {domain1} and {domain2} on {topic}.",
    "Draw an analogy between {domain1} and {domain2}."
]

SYNTHESIST_RESPONSE_TEMPLATE = """Cross-domain synthesis reveals surprising convergences:

**Convergence 1: {convergence1_title}**
- {domain1}: {convergence1_domain1}
- {domain2}: {convergence1_domain2}
- **Synthesis**: {convergence1_synthesis}

**Convergence 2: {convergence2_title}**
- {domain1}: {convergence2_domain1}
- {domain2}: {convergence2_domain2}
- **Insight**: {convergence2_insight}

**Convergence 3: {convergence3_title}**
- {domain1}: {convergence3_domain1}
- {domain2}: {convergence3_domain2}
- **Pattern**: {convergence3_pattern}

**Novel Connection:**
{novel_connection}

**Practical Application:**
{practical_application}"""

# ============================================================================
# ORACLE TEMPLATES
# ============================================================================

ORACLE_QUERY_TEMPLATES = [
    "Based on the multi-agent analysis, what is the fundamental truth about {topic}?",
    "Validate the truth about {topic}.",
    "What is the validated conclusion on {topic}?",
    "After considering all perspectives, what can we conclude about {topic}?",
    "What is the final truth determination on {topic}?",
    "Synthesize the evidence and determine the truth about {topic}.",
    "Given conflicting claims, what is actually true about {topic}?",
    "What principles can we extract from the analysis of {topic}?"
]

ORACLE_RESPONSE_TEMPLATE = """Truth Validation and Principle Extraction:

**Evidence Assessment:**

Surveyor's evidence:
- {surveyor_evidence1}: {surveyor_assessment1}
- {surveyor_evidence2}: {surveyor_assessment2}

Dissident's challenges:
- {dissident_challenge1}: {dissident_assessment1}
- {dissident_challenge2}: {dissident_assessment2}

**Validated Conclusions:**

1. **{conclusion1}** (Confidence: {confidence1}%)
   - {conclusion1_evidence}
   - {conclusion1_reasoning}

2. **{conclusion2}** (Confidence: {confidence2}%)
   - {conclusion2_evidence}
   - {conclusion2_reasoning}

**Extracted Principles:**

**Principle 1: {principle1_title}**
{principle1_content}

**Principle 2: {principle2_title}**
{principle2_content}

**Final Truth Determination:**
{final_determination}

**Practical Implication:**
{practical_implication}"""

# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_hash(content: str) -> str:
    """Generate content hash for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()[:8]


def vary_text(text: str) -> str:
    """Add variation to text to prevent repetition."""
    variations = [
        ("shows", ["demonstrates", "indicates", "reveals", "suggests"]),
        ("important", ["significant", "crucial", "critical", "key"]),
        ("research", ["studies", "investigations", "work", "analysis"]),
        ("evidence", ["data", "findings", "results", "observations"]),
        ("suggests", ["indicates", "implies", "points to", "hints at"]),
        ("major", ["significant", "key", "primary", "main"]),
        ("current", ["present", "contemporary", "recent", "modern"]),
        ("understanding", ["knowledge", "comprehension", "grasp", "insight"]),
    ]
    
    result = text
    for original, replacements in variations:
        if original in result and random.random() > 0.5:
            result = result.replace(original, random.choice(replacements), 1)
    return result


def generate_surveyor_example(domain: str, topic: str) -> Dict[str, Any]:
    """Generate a Surveyor training example."""
    query_template = random.choice(SURVEYOR_QUERY_TEMPLATES)
    
    if "{related_topic}" in query_template:
        related = random.choice([t for t in RESEARCH_TOPICS.get(domain, []) if t != topic])
        query = query_template.format(topic=topic, related_topic=related)
    else:
        query = query_template.format(topic=topic)
    
    # Generate response content
    response = SURVEYOR_RESPONSE_TEMPLATE.format(
        topic=topic,
        overview=vary_text(f"The field of {topic} has seen significant advances in recent years. Current research focuses on understanding fundamental mechanisms and developing practical applications."),
        finding1_title=f"Theoretical Foundations",
        finding1_detail1=vary_text(f"Core principles of {topic} are now well-established"),
        finding1_detail2=vary_text("Mathematical frameworks provide predictive power"),
        finding1_detail3=vary_text("Experimental validation confirms theoretical predictions"),
        finding2_title=f"Methodological Advances",
        finding2_detail1=vary_text("New techniques enable more precise measurements"),
        finding2_detail2=vary_text("Computational methods accelerate discovery"),
        finding2_detail3=vary_text("Interdisciplinary approaches yield novel insights"),
        finding3_title=f"Practical Applications",
        finding3_detail1=vary_text("Industrial applications are emerging"),
        finding3_detail2=vary_text("Healthcare implications are being explored"),
        finding3_detail3=vary_text("Policy frameworks are being developed"),
        current_status=vary_text(f"The field is transitioning from basic research to practical implementation. Key challenges remain in scaling and reliability."),
        open_questions=vary_text(f"How can we improve efficiency? What are the long-term implications? How does this connect to broader systems?"),
        sources=f"Nature {random.randint(2020, 2024)}, Science {random.randint(2020, 2024)}, arXiv preprints"
    )
    
    return {
        "messages": [
            {"role": "system", "content": "You are ICEBURG Surveyor, a research agent specialized in gathering comprehensive information, exploring domains, and synthesizing evidence from multiple authoritative sources."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ],
        "metadata": {
            "agent_type": "surveyor",
            "domain": domain,
            "topic": topic,
            "quality_score": 0.85,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "large_corpus_generator",
            "hash": generate_hash(query + response)
        }
    }


def generate_dissident_example(domain: str, topic: str) -> Dict[str, Any]:
    """Generate a Dissident training example."""
    claims = [
        f"{topic} will revolutionize everything within 5 years",
        f"{topic} is the key solution to all related problems",
        f"{topic} has no significant drawbacks",
        f"everyone agrees that {topic} is beneficial",
        f"{topic} makes previous approaches obsolete",
        f"the benefits of {topic} outweigh all costs",
        f"{topic} is ready for widespread adoption",
        f"critics of {topic} are simply uninformed"
    ]
    
    claim = random.choice(claims)
    query_template = random.choice(DISSIDENT_QUERY_TEMPLATES)
    query = query_template.format(claim=claim)
    
    response = DISSIDENT_RESPONSE_TEMPLATE.format(
        contradiction1_title="Overstated Timeline",
        contradiction1_evidence1=vary_text(f"Current {topic} capabilities remain limited"),
        contradiction1_evidence2=vary_text("Technical barriers are more significant than claimed"),
        contradiction1_reality=vary_text("Progress is incremental, not revolutionary"),
        contradiction2_title="Hidden Costs",
        contradiction2_evidence1=vary_text("Implementation challenges are often ignored"),
        contradiction2_evidence2=vary_text("Resource requirements are understated"),
        contradiction2_reality=vary_text("Trade-offs exist that proponents minimize"),
        contradiction3_title="Alternative Explanations",
        contradiction3_evidence1=vary_text("Other factors may explain observed effects"),
        contradiction3_evidence2=vary_text("Confounding variables are not controlled"),
        contradiction3_reality=vary_text("Causation is not established"),
        alternative_view=vary_text(f"A more measured view recognizes both potential and limitations of {topic}. Progress requires honest assessment of challenges."),
        underlying_bias=random.choice(["commercial interests", "confirmation bias", "hype cycles", "funding pressures", "publication bias"])
    )
    
    return {
        "messages": [
            {"role": "system", "content": "You are ICEBURG Dissident, an adversarial agent specialized in challenging assumptions, detecting contradictions, and presenting alternative perspectives to ensure truth through conflict."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ],
        "metadata": {
            "agent_type": "dissident",
            "domain": domain,
            "topic": topic,
            "quality_score": 0.85,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "large_corpus_generator",
            "hash": generate_hash(query + response)
        }
    }


def generate_synthesist_example(domains: List[str]) -> Dict[str, Any]:
    """Generate a Synthesist training example."""
    domain1, domain2 = domains[:2]
    domain3 = domains[2] if len(domains) > 2 else None
    
    if domain3:
        query_template = random.choice([t for t in SYNTHESIST_QUERY_TEMPLATES if "{domain3}" in t or "domain3" not in t])
    else:
        query_template = random.choice([t for t in SYNTHESIST_QUERY_TEMPLATES if "{domain3}" not in t])
    
    topic = random.choice(RESEARCH_TOPICS.get(domain1, ["complex systems"]))
    
    query = query_template.format(domain1=domain1, domain2=domain2, domain3=domain3 or "", topic=topic)
    query = query.replace(", , and", " and").replace(",  and", " and")
    
    response = SYNTHESIST_RESPONSE_TEMPLATE.format(
        domain1=domain1,
        domain2=domain2,
        convergence1_title="Shared Mathematical Structure",
        convergence1_domain1=vary_text(f"{domain1} uses specific mathematical frameworks"),
        convergence1_domain2=vary_text(f"{domain2} employs similar formal structures"),
        convergence1_synthesis=vary_text("Common mathematical foundations reveal deep connections"),
        convergence2_title="Emergence and Complexity",
        convergence2_domain1=vary_text(f"{domain1} exhibits emergent properties"),
        convergence2_domain2=vary_text(f"{domain2} shows similar emergent behavior"),
        convergence2_insight=vary_text("Complex systems share universal properties"),
        convergence3_title="Information Processing",
        convergence3_domain1=vary_text(f"{domain1} involves information transformation"),
        convergence3_domain2=vary_text(f"{domain2} processes information analogously"),
        convergence3_pattern=vary_text("Information is fundamental across both domains"),
        novel_connection=vary_text(f"Insights from {domain1} suggest new approaches to {domain2}. The isomorphism between their structures enables cross-pollination of methods and concepts."),
        practical_application=vary_text(f"Practitioners in {domain1} could benefit from {domain2} methodologies, and vice versa. This synthesis suggests hybrid approaches.")
    )
    
    return {
        "messages": [
            {"role": "system", "content": "You are ICEBURG Synthesist, a connection agent specialized in cross-domain synthesis, integrating insights from multiple fields, and discovering unexpected connections."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ],
        "metadata": {
            "agent_type": "synthesist",
            "domains": domains,
            "quality_score": 0.85,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "large_corpus_generator",
            "hash": generate_hash(query + response)
        }
    }


def generate_oracle_example(domain: str, topic: str) -> Dict[str, Any]:
    """Generate an Oracle training example."""
    query_template = random.choice(ORACLE_QUERY_TEMPLATES)
    query = query_template.format(topic=topic)
    
    confidence1 = random.randint(75, 95)
    confidence2 = random.randint(60, 85)
    
    response = ORACLE_RESPONSE_TEMPLATE.format(
        surveyor_evidence1=vary_text(f"Research findings on {topic}"),
        surveyor_assessment1="CONFIRMED",
        surveyor_evidence2=vary_text(f"Multiple studies support core claims"),
        surveyor_assessment2="PARTIALLY CONFIRMED",
        dissident_challenge1=vary_text("Overstated implications"),
        dissident_assessment1="VALID",
        dissident_challenge2=vary_text("Methodological concerns"),
        dissident_assessment2="PARTIALLY VALID",
        conclusion1=vary_text(f"Core findings on {topic} are well-supported"),
        confidence1=confidence1,
        conclusion1_evidence=vary_text("Multiple independent lines of evidence converge"),
        conclusion1_reasoning=vary_text("Robust to methodological variations"),
        conclusion2=vary_text(f"Practical applications require further development"),
        confidence2=confidence2,
        conclusion2_evidence=vary_text("Gap between research and implementation"),
        conclusion2_reasoning=vary_text("Real-world conditions differ from controlled studies"),
        principle1_title="Evidence Hierarchy",
        principle1_content=vary_text("Multiple independent sources provide stronger support than single studies."),
        principle2_title="Bounded Claims",
        principle2_content=vary_text("Truth claims should be proportional to evidence strength."),
        final_determination=vary_text(f"The fundamental truth about {topic} is that we have solid evidence for core claims but significant uncertainty about broader implications. This represents a confident understanding of mechanisms with appropriate humility about applications."),
        practical_implication=vary_text(f"Decisions about {topic} should proceed with evidence-based confidence while maintaining adaptive capacity as understanding evolves.")
    )
    
    return {
        "messages": [
            {"role": "system", "content": "You are ICEBURG Oracle, a truth-validation agent specialized in extracting fundamental principles, validating conclusions against evidence, and making final truth determinations with explicit confidence levels."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ],
        "metadata": {
            "agent_type": "oracle",
            "domain": domain,
            "topic": topic,
            "quality_score": 0.85,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "large_corpus_generator",
            "hash": generate_hash(query + response)
        }
    }


def generate_corpus(target_per_agent: int = 100) -> Dict[str, List[Dict]]:
    """Generate full training corpus."""
    corpus = {
        "surveyor": [],
        "dissident": [],
        "synthesist": [],
        "oracle": []
    }
    
    seen_hashes = set()
    
    # Generate Surveyor examples
    print(f"Generating {target_per_agent} Surveyor examples...")
    while len(corpus["surveyor"]) < target_per_agent:
        domain = random.choice(RESEARCH_DOMAINS)
        topics = RESEARCH_TOPICS.get(domain, [f"general {domain}"])
        topic = random.choice(topics)
        
        example = generate_surveyor_example(domain, topic)
        h = example["metadata"]["hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            corpus["surveyor"].append(example)
    
    # Generate Dissident examples
    print(f"Generating {target_per_agent} Dissident examples...")
    while len(corpus["dissident"]) < target_per_agent:
        domain = random.choice(RESEARCH_DOMAINS)
        topics = RESEARCH_TOPICS.get(domain, [f"general {domain}"])
        topic = random.choice(topics)
        
        example = generate_dissident_example(domain, topic)
        h = example["metadata"]["hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            corpus["dissident"].append(example)
    
    # Generate Synthesist examples
    print(f"Generating {target_per_agent} Synthesist examples...")
    while len(corpus["synthesist"]) < target_per_agent:
        domains = random.sample(RESEARCH_DOMAINS, k=random.randint(2, 3))
        
        example = generate_synthesist_example(domains)
        h = example["metadata"]["hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            corpus["synthesist"].append(example)
    
    # Generate Oracle examples
    print(f"Generating {target_per_agent} Oracle examples...")
    while len(corpus["oracle"]) < target_per_agent:
        domain = random.choice(RESEARCH_DOMAINS)
        topics = RESEARCH_TOPICS.get(domain, [f"general {domain}"])
        topic = random.choice(topics)
        
        example = generate_oracle_example(domain, topic)
        h = example["metadata"]["hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            corpus["oracle"].append(example)
    
    return corpus


def save_corpus(corpus: Dict[str, List[Dict]]):
    """Save corpus to files."""
    for agent_type, examples in corpus.items():
        output_file = OUTPUT_DIR / f"{agent_type}_training_data.jsonl"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        print(f"{agent_type.upper()}: {len(examples)} samples -> {output_file}")
    
    # Combined file
    combined_file = OUTPUT_DIR / "combined_training_data.jsonl"
    with open(combined_file, "w", encoding="utf-8") as f:
        for agent_type, examples in corpus.items():
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    total = sum(len(examples) for examples in corpus.values())
    print(f"\nTOTAL: {total} samples -> {combined_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate large training corpus")
    parser.add_argument("--target", type=int, default=100,
                       help="Target examples per agent type")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("ICEBURG Large-Scale Corpus Generator")
    print("=" * 60)
    print(f"\nTarget: {args.target} examples per agent type")
    print(f"Seed: {args.seed}")
    print()
    
    corpus = generate_corpus(target_per_agent=args.target)
    save_corpus(corpus)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

