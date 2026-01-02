#!/usr/bin/env python3
"""
ICEBURG Model Fine-Tuning Starter
Starts the fine-tuning process for ICEBURG agent models
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.training.specialized_model_tuner import SpecializedModelTuner, AgentModelConfig
from iceburg.data_collection.export_fine_tuning_data import export_for_llama_factory
from iceburg.data_collection.fine_tuning_logger import FineTuningLogger
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def check_training_data():
    """Check if training data exists, export if needed"""
    logger.info("Checking for training data...")
    
    fine_tuning_logger = FineTuningLogger()
    stats = fine_tuning_logger.get_stats()
    
    print("\n" + "=" * 80)
    print("FINE-TUNING DATA STATUS")
    print("=" * 80)
    print(f"Data Collection Enabled: {stats['enabled']}")
    print(f"Data Directory: {stats['data_dir']}")
    print(f"Conversations: {stats['conversations']}")
    print(f"Reasoning Chains: {stats['reasoning_chains']}")
    print(f"Quality Metrics: {stats['quality_metrics']}")
    print(f"Agent Generations: {stats['agent_generations']}")
    print()
    
    # Check if we have data to export
    if stats['conversations'] > 0:
        logger.info(f"Found {stats['conversations']} conversations, exporting...")
        export_dir = Path("data/fine_tuning/export")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        count = export_for_llama_factory(
            output_dir=export_dir,
            min_quality=0.7,
            min_conversations=1
        )
        
        if count > 0:
            logger.info(f"✅ Exported {count} conversations for fine-tuning")
            return True
        else:
            logger.warning("No conversations met quality threshold")
    else:
        logger.warning("No conversation data found. Enable data collection first:")
        logger.warning("  export ICEBURG_ENABLE_FINE_TUNING_DATA=1")
        logger.warning("  Then use ICEBURG to generate conversations")
    
    return False


def create_sample_training_data():
    """Create sample training data for testing"""
    logger.info("Creating sample training data...")
    
    training_dir = Path("data/training")
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data for each agent type
    sample_data = {
        "surveyor": [
            {
                "input": "What is the current state of research on quantum consciousness?",
                "output": "Research on quantum consciousness explores the hypothesis that quantum processes in microtubules within neurons may be fundamental to consciousness. Key researchers include Penrose and Hameroff. Current state: theoretical models exist, experimental validation is ongoing. Recent work focuses on quantum coherence in biological systems.",
                "task_type": "survey"
            }
        ],
        "dissident": [
            {
                "input": "Analyze this claim: 'All AI systems are deterministic and cannot be creative.'",
                "output": "This claim contains contradictions. While AI systems are deterministic at the algorithmic level, they can exhibit emergent behaviors that appear creative. The contradiction lies in conflating determinism with lack of creativity. Creative outputs can emerge from deterministic processes through complexity and randomness in training data.",
                "task_type": "contradiction"
            }
        ],
        "synthesist": [
            {
                "input": "Synthesize insights from quantum mechanics, biology, and consciousness research.",
                "output": "Cross-domain synthesis reveals: 1) Quantum coherence may enable biological information processing beyond classical limits. 2) Consciousness may emerge from quantum-classical interfaces in neural microtubules. 3) Biological systems exploit quantum effects for efficiency. This suggests consciousness is not purely classical computation but involves quantum information processing.",
                "task_type": "synthesis"
            }
        ],
        "oracle": [
            {
                "input": "Should we prioritize quantum computing research or biological computing research?",
                "output": "Both are essential and complementary. Quantum computing offers exponential speedups for specific problems. Biological computing offers energy efficiency and self-organization. Recommendation: Pursue both in parallel, with quantum for optimization problems and biological for pattern recognition and adaptive systems. The synthesis of both may yield hybrid systems superior to either alone.",
                "task_type": "decision"
            }
        ]
    }
    
    tuner = SpecializedModelTuner()
    created = 0
    
    for agent_name, examples in sample_data.items():
        output_path = training_dir / f"{agent_name}_training.jsonl"
        if tuner.create_agent_training_data(agent_name, examples, str(output_path)):
            logger.info(f"✅ Created sample training data for {agent_name}: {output_path}")
            created += 1
    
    return created > 0


def start_fine_tuning(agent_name: str = None):
    """Start fine-tuning process"""
    print("\n" + "=" * 80)
    print("ICEBURG MODEL FINE-TUNING")
    print("=" * 80)
    print()
    
    tuner = SpecializedModelTuner()
    agent_configs = tuner.get_recommended_agent_models()
    
    # Check/create training data
    training_dir = Path("data/training")
    has_data = False
    
    for name, config in agent_configs.items():
        data_path = Path(config.training_data_path)
        if data_path.exists():
            logger.info(f"✅ Found training data for {name}: {data_path}")
            has_data = True
        else:
            logger.warning(f"⚠️  Missing training data for {name}: {data_path}")
    
    if not has_data:
        logger.info("Creating sample training data...")
        if create_sample_training_data():
            logger.info("✅ Sample training data created")
        else:
            logger.error("Failed to create sample training data")
            return
    
    # Start fine-tuning for specified agent or all
    if agent_name:
        if agent_name not in agent_configs:
            logger.error(f"Unknown agent: {agent_name}")
            logger.info(f"Available agents: {', '.join(agent_configs.keys())}")
            return
        
        config = agent_configs[agent_name]
        logger.info(f"Starting fine-tuning for {agent_name}...")
        result = tuner.fine_tune_agent_model(config)
        
        if result.success:
            logger.info(f"✅ Fine-tuning completed for {agent_name}")
            logger.info(f"   Model: {result.model_path}")
            logger.info(f"   Training samples: {result.training_samples}")
        else:
            logger.error(f"❌ Fine-tuning failed for {agent_name}: {result.error_message}")
    else:
        logger.info("Starting fine-tuning for all agents...")
        print("\n" + "=" * 80)
        print("FINE-TUNING ALL AGENTS")
        print("=" * 80)
        print()
        
        for name, config in agent_configs.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Fine-tuning {name.upper()}")
            logger.info(f"{'='*80}")
            
            result = tuner.fine_tune_agent_model(config)
            
            if result.success:
                logger.info(f"✅ {name}: {result.model_path} ({result.training_samples} samples)")
            else:
                logger.error(f"❌ {name}: {result.error_message}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start ICEBURG model fine-tuning")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["surveyor", "dissident", "synthesist", "oracle", "biological_lab", "quantum_lab"],
        help="Fine-tune specific agent (default: all)"
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check training data status and export if available"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample training data"
    )
    
    args = parser.parse_args()
    
    if args.check_data:
        check_training_data()
    elif args.create_samples:
        create_sample_training_data()
    else:
        start_fine_tuning(args.agent)


if __name__ == "__main__":
    main()

