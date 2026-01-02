"""
Prompt Tuner
Optimizes system prompts based on telemetry data
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PromptTuner:
    """Analyzes prompt metrics and suggests optimizations"""
    
    def __init__(self, telemetry_dir: str = "data/telemetry"):
        self.telemetry_dir = Path(telemetry_dir)
        self.metrics_file = self.telemetry_dir / "prompt_metrics.jsonl"
        
    def analyze_latency_bottlenecks(self, threshold_ms: float = 5000) -> List[Dict[str, Any]]:
        """Identify agents/prompts causing high latency"""
        bottlenecks = []
        
        if not self.metrics_file.exists():
            return []
            
        try:
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    try:
                        metric = json.loads(line)
                        duration = metric.get("response_time", 0) * 1000 # to ms
                        
                        if duration > threshold_ms:
                            bottlenecks.append({
                                "prompt_id": metric.get("prompt_id"),
                                "latency_ms": duration,
                                "token_count": metric.get("token_count"),
                                "model": metric.get("model_used"),
                                "suggestion": "Consider reducing system prompt length or enabling quantization"
                            })
                    except:
                        continue
        except Exception as e:
            logger.error(f"Failed to analyze metrics: {e}")
            
        return sorted(bottlenecks, key=lambda x: x["latency_ms"], reverse=True)
        
    def suggest_optimizations(self):
        """Print optimization suggestions"""
        bottlenecks = self.analyze_latency_bottlenecks()
        if not bottlenecks:
            print("No significant latency bottlenecks found.")
            return
            
        print(f"Found {len(bottlenecks)} high-latency interactions:")
        for b in bottlenecks[:5]:
            print(f"- {b['model']} took {b['latency_ms']:.0f}ms ({b['token_count']} tokens)")
            print(f"  Suggestion: {b['suggestion']}")

if __name__ == "__main__":
    tuner = PromptTuner()
    tuner.suggest_optimizations()
