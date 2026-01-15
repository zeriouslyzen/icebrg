"""
ICEBURG Model Evaluation Suite

Evaluates trained models on agent-specific benchmarks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class EvalMetric(Enum):
    """Evaluation metrics."""
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    SPECIFICITY = "specificity"
    FACTUAL_DENSITY = "factual_density"
    STRUCTURE = "structure"
    CITATION_QUALITY = "citation_quality"
    CONTRADICTION_DETECTION = "contradiction_detection"
    SYNTHESIS_DEPTH = "synthesis_depth"
    TRUTH_CALIBRATION = "truth_calibration"


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""
    prompt: str
    response: str
    expected_type: str
    metrics: Dict[EvalMetric, float] = field(default_factory=dict)
    overall_score: float = 0.0
    notes: str = ""


@dataclass
class BenchmarkResult:
    """Result of running a full benchmark."""
    model_name: str
    agent_type: str
    total_samples: int
    mean_score: float
    metric_scores: Dict[EvalMetric, float] = field(default_factory=dict)
    individual_results: List[EvalResult] = field(default_factory=list)
    evaluation_time: float = 0.0


class EvaluationSuite:
    """Evaluate ICEBURG models on agent-specific benchmarks."""
    
    def __init__(self):
        self.benchmarks = self._load_benchmarks()
    
    def _load_benchmarks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load benchmark questions for each agent type."""
        return {
            "surveyor": [
                {
                    "prompt": "Explain the current state of quantum computing",
                    "expected_elements": ["qubits", "error correction", "companies", "limitations"],
                    "metrics": [EvalMetric.COHERENCE, EvalMetric.FACTUAL_DENSITY, EvalMetric.STRUCTURE]
                },
                {
                    "prompt": "What do we know about CRISPR gene editing?",
                    "expected_elements": ["mechanism", "applications", "limitations", "ethical"],
                    "metrics": [EvalMetric.COHERENCE, EvalMetric.FACTUAL_DENSITY, EvalMetric.CITATION_QUALITY]
                },
                {
                    "prompt": "Summarize research on sleep and health",
                    "expected_elements": ["stages", "memory", "health effects", "recommendations"],
                    "metrics": [EvalMetric.COHERENCE, EvalMetric.RELEVANCE, EvalMetric.STRUCTURE]
                },
                {
                    "prompt": "What is the evidence on climate change?",
                    "expected_elements": ["temperature", "CO2", "mechanisms", "impacts", "sources"],
                    "metrics": [EvalMetric.FACTUAL_DENSITY, EvalMetric.CITATION_QUALITY, EvalMetric.STRUCTURE]
                },
                {
                    "prompt": "Explain how large language models work",
                    "expected_elements": ["transformer", "training", "attention", "limitations"],
                    "metrics": [EvalMetric.COHERENCE, EvalMetric.SPECIFICITY, EvalMetric.STRUCTURE]
                }
            ],
            "dissident": [
                {
                    "prompt": "Challenge the standard narrative about the food pyramid",
                    "expected_elements": ["contradictions", "industry influence", "alternative views", "evidence"],
                    "metrics": [EvalMetric.CONTRADICTION_DETECTION, EvalMetric.SPECIFICITY, EvalMetric.CITATION_QUALITY]
                },
                {
                    "prompt": "What are the problems with GDP as a measure?",
                    "expected_elements": ["limitations", "alternatives", "what's excluded", "who benefits"],
                    "metrics": [EvalMetric.COHERENCE, EvalMetric.CONTRADICTION_DETECTION, EvalMetric.SPECIFICITY]
                },
                {
                    "prompt": "Critique the replication crisis in psychology",
                    "expected_elements": ["statistics", "incentives", "specific failures", "systemic issues"],
                    "metrics": [EvalMetric.FACTUAL_DENSITY, EvalMetric.CONTRADICTION_DETECTION, EvalMetric.STRUCTURE]
                },
                {
                    "prompt": "Challenge assumptions about standardized testing",
                    "expected_elements": ["what tests measure", "biases", "alternatives", "who benefits"],
                    "metrics": [EvalMetric.CONTRADICTION_DETECTION, EvalMetric.SPECIFICITY, EvalMetric.RELEVANCE]
                },
                {
                    "prompt": "What's wrong with the chemical imbalance theory of depression?",
                    "expected_elements": ["evidence against", "origins", "industry role", "alternatives"],
                    "metrics": [EvalMetric.CONTRADICTION_DETECTION, EvalMetric.CITATION_QUALITY, EvalMetric.FACTUAL_DENSITY]
                }
            ],
            "synthesist": [
                {
                    "prompt": "What connects thermodynamics and information theory?",
                    "expected_elements": ["entropy", "Landauer", "Maxwell's demon", "implications"],
                    "metrics": [EvalMetric.SYNTHESIS_DEPTH, EvalMetric.COHERENCE, EvalMetric.SPECIFICITY]
                },
                {
                    "prompt": "How do evolution and machine learning inform each other?",
                    "expected_elements": ["genetic algorithms", "selection", "both directions", "predictions"],
                    "metrics": [EvalMetric.SYNTHESIS_DEPTH, EvalMetric.STRUCTURE, EvalMetric.RELEVANCE]
                },
                {
                    "prompt": "What patterns connect epidemics, viral content, and market bubbles?",
                    "expected_elements": ["SIR model", "network effects", "tipping points", "interventions"],
                    "metrics": [EvalMetric.SYNTHESIS_DEPTH, EvalMetric.COHERENCE, EvalMetric.SPECIFICITY]
                },
                {
                    "prompt": "How do game theory, evolution, and social dynamics relate?",
                    "expected_elements": ["ESS", "cooperation", "common framework", "applications"],
                    "metrics": [EvalMetric.SYNTHESIS_DEPTH, EvalMetric.FACTUAL_DENSITY, EvalMetric.STRUCTURE]
                },
                {
                    "prompt": "What connects compression, intelligence, and prediction?",
                    "expected_elements": ["equivalence", "Kolmogorov", "LLMs", "implications"],
                    "metrics": [EvalMetric.SYNTHESIS_DEPTH, EvalMetric.COHERENCE, EvalMetric.SPECIFICITY]
                }
            ],
            "oracle": [
                {
                    "prompt": "What can we say with high confidence about exercise and health?",
                    "expected_elements": ["confidence levels", "established vs uncertain", "effect sizes", "nuance"],
                    "metrics": [EvalMetric.TRUTH_CALIBRATION, EvalMetric.STRUCTURE, EvalMetric.FACTUAL_DENSITY]
                },
                {
                    "prompt": "Is AI an existential risk? What can we actually conclude?",
                    "expected_elements": ["confidence levels", "evidence quality", "what we don't know", "defensible positions"],
                    "metrics": [EvalMetric.TRUTH_CALIBRATION, EvalMetric.COHERENCE, EvalMetric.SPECIFICITY]
                },
                {
                    "prompt": "What does the evidence show about meditation?",
                    "expected_elements": ["effect sizes", "what works", "what's overstated", "honest assessment"],
                    "metrics": [EvalMetric.TRUTH_CALIBRATION, EvalMetric.FACTUAL_DENSITY, EvalMetric.CITATION_QUALITY]
                },
                {
                    "prompt": "Resolve the nature vs nurture debate",
                    "expected_elements": ["heritability estimates", "both matter", "nuances", "what we can't say"],
                    "metrics": [EvalMetric.TRUTH_CALIBRATION, EvalMetric.STRUCTURE, EvalMetric.SPECIFICITY]
                },
                {
                    "prompt": "What fundamental truths about expertise and performance?",
                    "expected_elements": ["deliberate practice", "domain specificity", "what's overstated", "confidence levels"],
                    "metrics": [EvalMetric.TRUTH_CALIBRATION, EvalMetric.COHERENCE, EvalMetric.FACTUAL_DENSITY]
                }
            ]
        }
    
    def evaluate_response(
        self, 
        response: str, 
        benchmark: Dict[str, Any],
        agent_type: str
    ) -> EvalResult:
        """Evaluate a single response against benchmark criteria."""
        
        result = EvalResult(
            prompt=benchmark["prompt"],
            response=response,
            expected_type=agent_type
        )
        
        # Heuristic scoring (can be replaced with LLM-based evaluation)
        for metric in benchmark["metrics"]:
            score = self._score_metric(response, benchmark, metric, agent_type)
            result.metrics[metric] = score
        
        # Overall score is mean of individual metrics
        if result.metrics:
            result.overall_score = sum(result.metrics.values()) / len(result.metrics)
        
        return result
    
    def _score_metric(
        self, 
        response: str, 
        benchmark: Dict[str, Any], 
        metric: EvalMetric,
        agent_type: str
    ) -> float:
        """Score a response on a specific metric (0-1 scale)."""
        
        response_lower = response.lower()
        expected = benchmark.get("expected_elements", [])
        
        if metric == EvalMetric.COHERENCE:
            # Check for logical flow indicators
            flow_indicators = ["therefore", "however", "because", "thus", "consequently", 
                             "furthermore", "additionally", "in contrast", "specifically"]
            flow_count = sum(1 for ind in flow_indicators if ind in response_lower)
            # Has sections/structure
            has_structure = "**" in response or "##" in response or "\n\n" in response
            return min(1.0, (flow_count / 5) * 0.5 + (0.5 if has_structure else 0))
        
        elif metric == EvalMetric.RELEVANCE:
            # Check if expected elements are mentioned
            found = sum(1 for elem in expected if elem.lower() in response_lower)
            return found / len(expected) if expected else 0.5
        
        elif metric == EvalMetric.SPECIFICITY:
            # Check for specific details (numbers, names, technical terms)
            has_numbers = any(c.isdigit() for c in response)
            has_percentages = "%" in response
            has_citations = "source" in response_lower or "study" in response_lower or "research" in response_lower
            word_count = len(response.split())
            length_score = min(1.0, word_count / 500)  # Longer is more specific, up to 500 words
            return (has_numbers * 0.2 + has_percentages * 0.2 + has_citations * 0.2 + length_score * 0.4)
        
        elif metric == EvalMetric.FACTUAL_DENSITY:
            # Check for factual claims (numbers, dates, names)
            words = response.split()
            factual_indicators = sum(1 for w in words if w[0].isupper() and len(w) > 2) if words else 0
            number_count = sum(1 for w in words if any(c.isdigit() for c in w))
            return min(1.0, (factual_indicators + number_count * 2) / 50)
        
        elif metric == EvalMetric.STRUCTURE:
            # Check for clear organization
            has_headers = "**" in response or "##" in response
            has_lists = "- " in response or "1." in response or "•" in response
            has_paragraphs = response.count("\n\n") >= 2
            return (has_headers * 0.4 + has_lists * 0.3 + has_paragraphs * 0.3)
        
        elif metric == EvalMetric.CITATION_QUALITY:
            # Check for source mentions
            citation_patterns = ["source:", "study", "research", "according to", "et al", 
                               "journal", "review", "meta-analysis", "evidence"]
            found = sum(1 for p in citation_patterns if p.lower() in response_lower)
            return min(1.0, found / 4)
        
        elif metric == EvalMetric.CONTRADICTION_DETECTION:
            # For dissident: check for challenge language
            challenge_patterns = ["however", "but", "contradicts", "problem", "issue",
                                "challenge", "critique", "flaw", "limitation", "misleading",
                                "overstated", "false", "myth"]
            found = sum(1 for p in challenge_patterns if p.lower() in response_lower)
            return min(1.0, found / 5)
        
        elif metric == EvalMetric.SYNTHESIS_DEPTH:
            # For synthesist: check for connection language
            synthesis_patterns = ["connection", "relates", "similar", "both", "pattern",
                                "underlying", "unified", "common", "framework", "maps to",
                                "analogous", "emerges"]
            found = sum(1 for p in synthesis_patterns if p.lower() in response_lower)
            return min(1.0, found / 5)
        
        elif metric == EvalMetric.TRUTH_CALIBRATION:
            # For oracle: check for confidence language
            calibration_patterns = ["confidence", "uncertain", "established", "probable",
                                  "evidence", "strong", "weak", "unknown", "debated",
                                  "high confidence", "low confidence", "%", "likely"]
            found = sum(1 for p in calibration_patterns if p.lower() in response_lower)
            has_table = "|" in response  # Tables often show confidence levels
            return min(1.0, found / 5 + (0.3 if has_table else 0))
        
        return 0.5  # Default
    
    def run_benchmark(
        self,
        model,
        tokenizer,
        agent_type: str,
        device: str = "cpu"
    ) -> BenchmarkResult:
        """Run full benchmark for an agent type."""
        
        start_time = time.time()
        benchmarks = self.benchmarks.get(agent_type, [])
        
        if not benchmarks:
            logger.warning(f"No benchmarks found for agent type: {agent_type}")
            return BenchmarkResult(
                model_name=str(model),
                agent_type=agent_type,
                total_samples=0,
                mean_score=0.0
            )
        
        results = []
        metric_totals: Dict[EvalMetric, List[float]] = {}
        
        # Get system prompt for agent
        system_prompts = {
            "surveyor": "You are ICEBURG Surveyor, a research agent specialized in gathering comprehensive information, exploring domains, and synthesizing evidence from multiple authoritative sources.",
            "dissident": "You are ICEBURG Dissident, an adversarial agent specialized in challenging assumptions, identifying contradictions, and generating alternative paradigms to uncover suppressed truths.",
            "synthesist": "You are ICEBURG Synthesist, a cross-domain integration agent specialized in finding emergent connections between disparate fields, identifying meta-patterns, and formulating novel hypotheses.",
            "oracle": "You are ICEBURG Oracle, a truth-validation agent specialized in extracting fundamental principles, assessing evidence quality, and making final truth determinations with explicit confidence levels."
        }
        
        system_prompt = system_prompts.get(agent_type, "You are a helpful assistant.")
        
        for benchmark in benchmarks:
            try:
                # Generate response
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": benchmark["prompt"]}
                ]
                
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the assistant response
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1].strip()
                elif "assistant" in response.lower():
                    parts = response.split("assistant")
                    if len(parts) > 1:
                        response = parts[-1].strip()
                
                # Evaluate
                eval_result = self.evaluate_response(response, benchmark, agent_type)
                results.append(eval_result)
                
                # Aggregate metrics
                for metric, score in eval_result.metrics.items():
                    if metric not in metric_totals:
                        metric_totals[metric] = []
                    metric_totals[metric].append(score)
                    
            except Exception as e:
                logger.error(f"Benchmark failed for '{benchmark['prompt'][:50]}...': {e}")
        
        # Calculate aggregate scores
        mean_score = sum(r.overall_score for r in results) / len(results) if results else 0.0
        metric_scores = {
            metric: sum(scores) / len(scores) 
            for metric, scores in metric_totals.items()
        }
        
        return BenchmarkResult(
            model_name=str(model.config._name_or_path if hasattr(model, 'config') else 'unknown'),
            agent_type=agent_type,
            total_samples=len(results),
            mean_score=mean_score,
            metric_scores=metric_scores,
            individual_results=results,
            evaluation_time=time.time() - start_time
        )
    
    def compare_models(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Compare benchmark results across models."""
        
        comparison = {
            "models": [],
            "best_overall": None,
            "best_per_metric": {}
        }
        
        best_score = 0.0
        
        for result in results:
            model_data = {
                "name": result.model_name,
                "agent_type": result.agent_type,
                "mean_score": result.mean_score,
                "metrics": {m.value: s for m, s in result.metric_scores.items()}
            }
            comparison["models"].append(model_data)
            
            if result.mean_score > best_score:
                best_score = result.mean_score
                comparison["best_overall"] = result.model_name
            
            for metric, score in result.metric_scores.items():
                metric_key = metric.value
                if metric_key not in comparison["best_per_metric"]:
                    comparison["best_per_metric"][metric_key] = {"model": result.model_name, "score": score}
                elif score > comparison["best_per_metric"][metric_key]["score"]:
                    comparison["best_per_metric"][metric_key] = {"model": result.model_name, "score": score}
        
        return comparison
    
    def save_results(self, result: BenchmarkResult, output_path: Path):
        """Save benchmark results to JSON."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "model_name": result.model_name,
            "agent_type": result.agent_type,
            "total_samples": result.total_samples,
            "mean_score": result.mean_score,
            "metric_scores": {m.value: s for m, s in result.metric_scores.items()},
            "evaluation_time": result.evaluation_time,
            "individual_results": [
                {
                    "prompt": r.prompt,
                    "response": r.response[:500] + "..." if len(r.response) > 500 else r.response,
                    "metrics": {m.value: s for m, s in r.metrics.items()},
                    "overall_score": r.overall_score
                }
                for r in result.individual_results
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_path}")


def evaluate_model_cli():
    """CLI for evaluating a trained model."""
    import argparse
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    parser = argparse.ArgumentParser(description="Evaluate ICEBURG model")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--agent-type", required=True, choices=["surveyor", "dissident", "synthesist", "oracle"])
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B", help="Base model name")
    parser.add_argument("--output", default=None, help="Output path for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    
    model = PeftModel.from_pretrained(base_model, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Run evaluation
    suite = EvaluationSuite()
    result = suite.run_benchmark(model, tokenizer, args.agent_type, device)
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Results: {args.agent_type}".center(60))
    print("="*60)
    print(f"\nModel: {result.model_name}")
    print(f"Samples: {result.total_samples}")
    print(f"Mean Score: {result.mean_score:.3f}")
    print(f"Time: {result.evaluation_time:.1f}s")
    print("\nMetric Scores:")
    for metric, score in result.metric_scores.items():
        print(f"  {metric.value}: {score:.3f}")
    
    # Save if requested
    if args.output:
        suite.save_results(result, Path(args.output))


if __name__ == "__main__":
    evaluate_model_cli()

