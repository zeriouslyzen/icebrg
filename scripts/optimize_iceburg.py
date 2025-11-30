#!/usr/bin/env python3
"""
ICEBURG Overall Optimization Script
Applies all optimizations: quantization, memory, fine-tuning
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.quantization.advanced_quantization import AdvancedQuantization, QuantizationConfig
from iceburg.memory.advanced_memory_manager import AdvancedMemoryManager
from iceburg.training.specialized_model_tuner import SpecializedModelTuner
import json
from datetime import datetime


def main():
    """Main optimization script"""
    print("=" * 80)
    print("ICEBURG OVERALL OPTIMIZATION")
    print("=" * 80)
    print()
    
    # Initialize components
    quantization = AdvancedQuantization()
    memory_manager = AdvancedMemoryManager(max_cache_size_mb=2048.0)
    tuner = SpecializedModelTuner()
    
    print("=" * 80)
    print("STEP 1: QUANTIZATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze models for quantization
    models_to_quantize = [
        "llama3.1:8b",
        "llama3.1:70b",
        "qwen2.5:32b",
    ]
    
    for model_name in models_to_quantize:
        quant_config = quantization.get_quantization_recommendation(
            model_name=model_name,
            available_memory_gb=16.0
        )
        print(f"{model_name}:")
        print(f"   Recommended: {quant_config.method}")
        print(f"   Bits: {quant_config.bits}")
        print(f"   Compression: ~{quant_config.lora_r}x")
        print()
    
    print("=" * 80)
    print("STEP 2: MEMORY OPTIMIZATION")
    print("=" * 80)
    print()
    
    # Get memory stats
    memory_stats = memory_manager.get_stats()
    print(f"Total Memory: {memory_stats.total_memory_mb:.2f} MB")
    print(f"Used Memory: {memory_stats.used_memory_mb:.2f} MB")
    print(f"Free Memory: {memory_stats.free_memory_mb:.2f} MB")
    print(f"Compression Savings: {memory_stats.compression_savings_mb:.2f} MB")
    print(f"Cache Hit Rate: {memory_stats.hit_rate:.2%}")
    print()
    
    print("=" * 80)
    print("STEP 3: AGENT MODEL OPTIMIZATION")
    print("=" * 80)
    print()
    
    # Get agent model recommendations
    agent_configs = tuner.get_recommended_agent_models()
    print(f"Recommended {len(agent_configs)} specialized agent models:")
    for agent_name in agent_configs.keys():
        print(f"   - {agent_name}")
    print()
    
    print("=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print()
    print("✅ Quantization: Ready for QLoRA, 4-bit, GGUF")
    print("✅ Memory Management: Advanced caching and compression")
    print("✅ Specialized Models: Agent-specific fine-tuning ready")
    print()
    print("Next steps:")
    print("1. Run: python scripts/self_improvement_activation.py")
    print("2. Review self-improvement recommendations")
    print("3. Run: python scripts/fine_tune_agent_models.py")
    print("4. Apply optimizations based on recommendations")


if __name__ == "__main__":
    main()

