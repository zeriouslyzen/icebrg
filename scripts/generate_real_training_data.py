#!/usr/bin/env python3
"""
Generate REAL Training Data Using ICEBURG's Actual Agents
==========================================================

Uses the actual ICEBURG system to generate high-quality training examples
by running real agent conversations.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Real research topics with depth
REAL_RESEARCH_QUERIES = [
    # Quantum Computing - specific questions
    "What are the specific technical challenges preventing practical quantum error correction, and what approaches show the most promise?",
    "Explain the difference between superconducting qubits, trapped ion qubits, and photonic qubits. Which has the best path to scale?",
    "What did Google's Willow chip actually achieve, and what are the legitimate criticisms of their quantum supremacy claims?",
    
    # AI/ML - real substance
    "How do modern LLMs like GPT-4 and Claude actually work at a technical level? What are transformers doing?",
    "What is the alignment problem in AI, and why do researchers like Stuart Russell think it's existentially important?",
    "Explain chain-of-thought prompting and why it improves reasoning. What are its limitations?",
    
    # Neuroscience - substantive
    "What does current neuroscience actually know about how memories are formed and consolidated?",
    "Explain the free energy principle and predictive processing. How does this theory explain perception and action?",
    "What is the evidence for and against the idea that consciousness requires quantum effects in microtubules?",
    
    # Climate - specific
    "What are the specific tipping points in the climate system, and which ones might we be closest to crossing?",
    "Explain the AMOC (Atlantic Meridional Overturning Circulation) and why its potential collapse matters.",
    "What does the latest IPCC report say about remaining carbon budgets for 1.5C and 2C targets?",
    
    # Biology - technical
    "How does CRISPR-Cas9 actually work at a molecular level, and what are the off-target effects?",
    "Explain the endosymbiotic origin of mitochondria and chloroplasts. What's the evidence?",
    "What are the leading theories for the origin of life, and what evidence supports each?",
    
    # Economics - substantive
    "What actually caused the 2008 financial crisis? Walk through the mechanism from subprime to systemic collapse.",
    "Explain Modern Monetary Theory. What are its core claims, and what are the strongest criticisms?",
    "What does the empirical evidence say about the effects of minimum wage increases on employment?",
    
    # Physics - real depth
    "Explain the measurement problem in quantum mechanics. Why hasn't it been solved after 100 years?",
    "What is dark matter, what's the evidence for it, and why have we failed to detect it directly?",
    "What did the detection of gravitational waves actually prove, and what new questions did it open?",
    
    # Philosophy - substantive
    "What is the hard problem of consciousness, and why do philosophers think it's different from 'easy' problems?",
    "Explain the Chinese Room argument against strong AI. What are the best responses to it?",
    "What is the difference between correlation and causation, and how do we establish causal claims?",
]

REAL_CHALLENGE_QUERIES = [
    # Challenge real claims
    "Everyone says AI will transform everything. What are the strongest reasons to be skeptical?",
    "Quantum computing is often hyped as revolutionary. What are the realistic limitations?",
    "The tech industry claims to be solving climate change. Challenge this narrative.",
    "Experts say the economy is doing well. Why might ordinary people disagree?",
    "Meditation is promoted as scientifically proven. What's wrong with this claim?",
    "Organic food is marketed as healthier. Challenge the evidence.",
    "Social media is blamed for mental health problems. What's the counterargument?",
    "Nuclear power is called too dangerous. Challenge this conventional wisdom.",
    "IQ tests are used to measure intelligence. What's problematic about this?",
    "Free markets are said to be most efficient. What are the strongest objections?",
]

REAL_SYNTHESIS_QUERIES = [
    "How do concepts from evolutionary biology apply to understanding economic markets?",
    "What can AI researchers learn from how the brain processes information?",
    "How does game theory connect to evolutionary biology and economics?",
    "What parallels exist between thermodynamics and information theory?",
    "How do ideas from network science apply to understanding both epidemics and social movements?",
    "What can climate science learn from complex systems theory?",
    "How do concepts from linguistics inform our understanding of AI language models?",
    "What connections exist between quantum mechanics and consciousness theories?",
]

REAL_ORACLE_QUERIES = [
    "Given conflicting claims about AI capabilities, what can we actually conclude with confidence?",
    "After examining the evidence on climate change, what is definitively established vs uncertain?",
    "What fundamental truths can we extract from the debate about consciousness?",
    "Synthesizing evidence on vaccines, what is the validated scientific consensus?",
    "After considering all perspectives on economic growth, what principles are robust?",
    "What can we definitively conclude about the effectiveness of psychotherapy?",
    "Given the research on social media, what truth claims are actually supported?",
    "What fundamental principles emerge from the study of complex adaptive systems?",
]


def run_agent(agent_type: str, query: str) -> str:
    """Run an actual ICEBURG agent and get its response."""
    try:
        from iceburg.config import load_config
        
        # Load config
        cfg = load_config()
        
        if agent_type == "surveyor":
            # Surveyor needs vector store
            from iceburg.vectorstore import VectorStore
            vs = VectorStore(cfg)
            from iceburg.agents.surveyor import run as surveyor_run
            return surveyor_run(cfg, vs, query)
            
        elif agent_type == "dissident":
            # Dissident needs surveyor output - run surveyor first
            from iceburg.vectorstore import VectorStore
            vs = VectorStore(cfg)
            from iceburg.agents.surveyor import run as surveyor_run
            from iceburg.agents.dissident import run as dissident_run
            surveyor_output = surveyor_run(cfg, vs, query)
            return dissident_run(cfg, query, surveyor_output)
            
        elif agent_type == "synthesist":
            # Use the actual synthesist agent
            from iceburg.agents.synthesist import run as synthesist_run
            return synthesist_run(cfg, query)
            
        elif agent_type == "oracle":
            # Use secretary for oracle-like responses  
            from iceburg.agents.secretary import run as secretary_run
            return secretary_run(cfg, query)
        else:
            return None
            
    except Exception as e:
        print(f"  Error running {agent_type}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_training_sample(agent_type: str, query: str, response: str) -> dict:
    """Create a properly formatted training sample."""
    system_prompts = {
        "surveyor": "You are ICEBURG Surveyor, a research agent specialized in gathering comprehensive information, exploring domains, and synthesizing evidence from multiple authoritative sources. Provide detailed, accurate, well-sourced responses.",
        "dissident": "You are ICEBURG Dissident, an adversarial agent specialized in challenging assumptions, detecting contradictions, and presenting alternative perspectives. Be rigorous, cite counter-evidence, and expose weaknesses in arguments.",
        "synthesist": "You are ICEBURG Synthesist, a connection agent specialized in cross-domain synthesis, integrating insights from multiple fields, and discovering unexpected connections between disparate areas of knowledge.",
        "oracle": "You are ICEBURG Oracle, a truth-validation agent specialized in extracting fundamental principles, assessing evidence quality, and making final truth determinations with explicit confidence levels."
    }
    
    return {
        "messages": [
            {"role": "system", "content": system_prompts[agent_type]},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ],
        "metadata": {
            "agent_type": agent_type,
            "quality_score": 0.95,  # Real agent output
            "timestamp": datetime.utcnow().isoformat(),
            "source": "real_iceburg_agents",
            "generation_method": "live_agent_call"
        }
    }


def generate_real_data():
    """Generate training data using actual ICEBURG agents."""
    output_dir = Path("data/fine_tuning/agent_data/real_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "surveyor": (REAL_RESEARCH_QUERIES, "surveyor"),
        "dissident": (REAL_CHALLENGE_QUERIES, "dissident"),
        "synthesist": (REAL_SYNTHESIS_QUERIES, "synthesist"),
        "oracle": (REAL_ORACLE_QUERIES, "oracle"),
    }
    
    for agent_type, (queries, agent_name) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Generating {agent_type.upper()} data ({len(queries)} queries)")
        print('='*60)
        
        samples = []
        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] {query[:60]}...")
            
            response = run_agent(agent_name, query)
            
            if response and len(response) > 100:
                sample = create_training_sample(agent_type, query, response)
                samples.append(sample)
                print(f"  OK: {len(response)} chars")
            else:
                print(f"  SKIP: No valid response")
        
        # Save
        output_file = output_dir / f"{agent_type}_training_data.jsonl"
        with open(output_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"\n{agent_type.upper()}: {len(samples)} samples -> {output_file}")
    
    # Combined
    combined_file = output_dir / "combined_training_data.jsonl"
    with open(combined_file, "w") as f:
        for agent_type in datasets:
            agent_file = output_dir / f"{agent_type}_training_data.jsonl"
            if agent_file.exists():
                with open(agent_file) as af:
                    f.write(af.read())
    
    print(f"\nAll data saved to: {output_dir}")


if __name__ == "__main__":
    print("="*60)
    print("ICEBURG Real Training Data Generator")
    print("Using actual ICEBURG agents for quality data")
    print("="*60)
    
    generate_real_data()

