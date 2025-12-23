
#!/usr/bin/env python3
"""
ICEBURG Metacognition Impact Benchmark
======================================

Measures the performance impact and behavioral differences of enabling 
metacognitive features.

Configurations Tested:
1. Baseline (Legacy): Metacognition disabled
2. Metacognitive (Standard): Semantic alignment + Contradiction detection enabled
3. Full Stack (Coconut): Metacognition + Vector-space reasoning (Coconut) enabled

Metrics:
- Latency Overhead (ms)
- Prompt Complexity (token estimate)
- Logic Trigger Rate (quarantine events)
"""

import os
import sys
import time
import json
import statistics
from unittest.mock import MagicMock, patch
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iceburg.agents.deliberation_agent import add_deliberation_pause
from src.iceburg.config import IceburgConfig
from src.iceburg.core.quarantine_manager import QuarantineManager

@dataclass
class ImpactResult:
    config_name: str
    total_time_ms: float
    overhead_ms: float
    quarantine_count: int
    prompt_length_avg: int
    
class MetacognitionBenchmark:
    def __init__(self):
        self.results = []
        self.cfg = self._setup_config()
        self.test_cases = self._generate_test_cases()
        
    def _setup_config(self):
        cfg = MagicMock(spec=IceburgConfig)
        cfg.embed_model = "top_secret_embed_model_v4"
        cfg.surveyor_model = "iceburg-reasoner-v3"
        return cfg

    def _generate_test_cases(self):
        """Generate synthetic agent outputs with varying quality."""
        return [
            # Case 1: High quality, aligned
            {
                "agent": "Surveyor",
                "query": "What is the capital of France?",
                "output": "The capital of France is Paris. It is a major European city.",
                "expected_quarantine": False
            },
            # Case 2: Low alignment (off-topic)
            {
                "agent": "Surveyor",
                "query": "Explain quantum entanglement",
                "output": "I really like chocolate ice cream. It is delicious.",
                "expected_quarantine": False # Warning only
            },
            # Case 3: Contradictory
            {
                "agent": "Analyst",
                "query": "Is the project strictly timeline-based?",
                "output": "The project is strictly timeline-based. However, the project is not time-bound at all.",
                "expected_quarantine": True
            }
        ]

    def run_benchmark(self):
        print(f"\n{'='*60}")
        print("ICEBURG Metacognition Impact Benchmark")
        print(f"{'='*60}\n")
        
        configs = {
            "Baseline": {},
            "Metacognitive": {"ICEBURG_ENABLE_METACOGNITION": "true"},
            "Full_Stack": {
                "ICEBURG_ENABLE_METACOGNITION": "true", 
                "ICEBURG_ENABLE_COCONUT_DELIBERATION": "true"
            }
        }
        
        # Clean quarantine for stats
        qm = QuarantineManager()
        
        for name, env_vars in configs.items():
            print(f"Running Config: {name}...")
            
            # Set Meta Env Vars
            # First clear existing
            keys = ["ICEBURG_ENABLE_METACOGNITION", "ICEBURG_ENABLE_COCONUT_DELIBERATION"]
            for k in keys:
                if k in os.environ: del os.environ[k]
            
            # Set new
            for k, v in env_vars.items():
                os.environ[k] = v
                
            self._run_config(name, qm)
            
        self._print_results()

    def _run_config(self, config_name, qm):
        """Run all test cases under current config."""
        
        latencies = []
        prompt_lengths = []
        quarantine_start_count = len(list(qm.quarantine_dir.glob("*.json")))
        
        # Mock dependencies (we want to measure overhead, not LLM api latency)
        # We mock chat_complete but let the Agent class logic (vector math) run if possible.
        # However, Agent class uses `embed_texts`. We need to mock that to be constant time 
        # OR simulate network delay if we want "realistic" overhead.
        # For "Computational Overhead", constant time mock is best.
        
        with patch("src.iceburg.agents.deliberation_agent.chat_complete") as mock_chat, \
             patch("src.iceburg.agents.deliberation_agent.DeliberationAgent._calculate_semantic_alignment") as mock_align, \
             patch("src.iceburg.agents.deliberation_agent.DeliberationAgent._detect_contradictions") as mock_contra, \
             patch("src.iceburg.agents.deliberation_agent.DeliberationAgent._analyze_reasoning_complexity") as mock_complex, \
             patch("src.iceburg.agents.deliberation_agent.add_deliberation_pause_coconut") as mock_coconut:
            
            # Setup Mocks to be fast but functional
            mock_chat.return_value = "Reflected."
            mock_align.side_effect = lambda q, o: 0.9 if "Paris" in o else 0.1
            mock_contra.side_effect = lambda o: [{"type":"logic"}] if "timeline" in o else []
            mock_complex.return_value = {"complexity": "medium", "depth_indicators": 2}
            
            mock_coconut.return_value = "Coconut Reflected."
            
            start_total = time.time()
            
            for case in self.test_cases:
                t0 = time.time()
                add_deliberation_pause(self.cfg, case["agent"], case["output"], case["query"], verbose=False)
                dt = (time.time() - t0) * 1000
                latencies.append(dt)
                
                # Check prompt length if chat_complete was called
                if mock_chat.call_args:
                    args = mock_chat.call_args[0]
                    if len(args) > 1:
                        prompt_lengths.append(len(args[1])) 
                elif mock_coconut.call_args:
                     args = mock_coconut.call_args[0]
                     # Coconut prompt is in arg 3
                     if len(args) > 3:
                         prompt_lengths.append(len(args[3]))

            avg_latency = statistics.mean(latencies)
            avg_prompt_len = statistics.mean(prompt_lengths) if prompt_lengths else 0
            
            quarantine_end_count = len(list(qm.quarantine_dir.glob("*.json")))
            new_quarantines = quarantine_end_count - quarantine_start_count
            
            # Calculate overhead vs baseline (naive: just raw time for now)
            # Baseline is usually first, so overhead is 0. 
            # We'll calculate relative later.
            
            self.results.append(ImpactResult(
                config_name=config_name,
                total_time_ms=avg_latency,
                overhead_ms=0, # Calc later
                quarantine_count=new_quarantines,
                prompt_length_avg=int(avg_prompt_len)
            ))

    def _print_results(self):
        print("\nBenchmark Results:")
        print(f"{'Config':<15} | {'Latency (ms)':<12} | {'Overhead':<10} | {'Quarantines':<11} | {'Prompt Len':<10}")
        print("-" * 70)
        
        baseline_time = self.results[0].total_time_ms
        
        for r in self.results:
            overhead = r.total_time_ms - baseline_time
            print(f"{r.config_name:<15} | {r.total_time_ms:<12.2f} | {overhead:<+10.2f} | {r.quarantine_count:<11} | {r.prompt_length_avg:<10}")
            
        print("-" * 70)
        print("\nAnalysis:")
        meta = self.results[1]
        print(f"1. Metacognition adds ~{meta.total_time_ms - baseline_time:.2f}ms overhead (mocked)")
        print(f"2. Logic Trigger Rate: {meta.quarantine_count} items quarantined")
        print(f"3. Prompt Enrichment: +{meta.prompt_length_avg - self.results[0].prompt_length_avg} chars")

if __name__ == "__main__":
    MetacognitionBenchmark().run_benchmark()
