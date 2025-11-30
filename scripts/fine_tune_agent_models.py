#!/usr/bin/env python3
"""
Fine-tune Specialized Models for ICEBURG Agents
Creates optimized models for each agent type
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.training.specialized_model_tuner import SpecializedModelTuner, AgentModelConfig
from iceburg.quantization.advanced_quantization import AdvancedQuantization, QuantizationConfig
import json


def main():
    """Main fine-tuning script"""
    print("=" * 80)
    print("ICEBURG AGENT MODEL FINE-TUNING")
    print("=" * 80)
    print()
    
    tuner = SpecializedModelTuner()
    quantization = AdvancedQuantization()
    
    # Get recommended agent models
    agent_configs = tuner.get_recommended_agent_models()
    
    print("=" * 80)
    print("RECOMMENDED AGENT MODELS")
    print("=" * 80)
    print()
    
    for agent_name, config in agent_configs.items():
        print(f"{agent_name.upper()}:")
        print(f"   Base Model: {config.base_model}")
        print(f"   Target Model: {config.target_model}")
        print(f"   Task Type: {config.task_type}")
        print(f"   Quantization: {config.quantization_config}")
        print(f"   LoRA Config: {config.lora_config}")
        print()
    
    print("=" * 80)
    print("QUANTIZATION RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Get quantization recommendations for each model
    for agent_name, config in agent_configs.items():
        quant_config = quantization.get_quantization_recommendation(
            model_name=config.base_model,
            available_memory_gb=16.0  # M4 has 16GB
        )
        print(f"{agent_name.upper()}:")
        print(f"   Recommended Method: {quant_config.method}")
        print(f"   Bits: {quant_config.bits}")
        print(f"   LoRA Rank: {quant_config.lora_r}")
        print(f"   LoRA Alpha: {quant_config.lora_alpha}")
        print()
    
    print("=" * 80)
    print("FINE-TUNING INSTRUCTIONS")
    print("=" * 80)
    print()
    print("To fine-tune agent models:")
    print("1. Create training data for each agent")
    print("2. Run: tuner.fine_tune_agent_model(agent_config)")
    print("3. Apply quantization if needed")
    print("4. Deploy optimized models")
    print()
    print("Example:")
    print("  config = agent_configs['surveyor']")
    print("  result = tuner.fine_tune_agent_model(config)")
    print("  print(result)")


if __name__ == "__main__":
    main()

