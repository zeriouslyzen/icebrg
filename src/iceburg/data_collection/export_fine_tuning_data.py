"""
Export Fine-Tuning Data
Exports collected data in formats suitable for fine-tuning open-source LLMs
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from .fine_tuning_logger import FineTuningLogger

logger = logging.getLogger(__name__)


def export_for_llama_factory(
    output_dir: Path = Path("data/fine_tuning/export"),
    min_quality: float = 0.8,
    min_conversations: int = 2
) -> int:
    """
    Export data for llama-factory fine-tuning.
    
    Args:
        output_dir: Output directory path
        min_quality: Minimum quality score (0.0-1.0)
        min_conversations: Minimum number of messages in conversation
        
    Returns:
        Number of exported conversations
    """
    logger.info("Exporting data for llama-factory fine-tuning...")
    
    fine_tuning_logger = FineTuningLogger()
    
    # Export in ChatML format (Llama, Mistral, Qwen)
    output_file = output_dir / "chatml_format.jsonl"
    count = fine_tuning_logger.export_for_fine_tuning(
        output_file=output_file,
        format="chatml",
        min_quality=min_quality,
        min_conversations=min_conversations
    )
    
    if count > 0:
        logger.info(f"✅ Exported {count} conversations to {output_file}")
        logger.info(f"   Use with llama-factory: llama-factory train --dataset {output_file}")
    else:
        logger.warning("No conversations exported (check quality threshold and data collection)")
    
    return count


def export_for_unsloth(
    output_dir: Path = Path("data/fine_tuning/export"),
    min_quality: float = 0.8,
    min_conversations: int = 2
) -> int:
    """
    Export data for unsloth fine-tuning.
    
    Args:
        output_dir: Output directory path
        min_quality: Minimum quality score (0.0-1.0)
        min_conversations: Minimum number of messages in conversation
        
    Returns:
        Number of exported conversations
    """
    logger.info("Exporting data for unsloth fine-tuning...")
    
    fine_tuning_logger = FineTuningLogger()
    
    # Export in ChatML format (works with unsloth)
    output_file = output_dir / "unsloth_format.jsonl"
    count = fine_tuning_logger.export_for_fine_tuning(
        output_file=output_file,
        format="chatml",
        min_quality=min_quality,
        min_conversations=min_conversations
    )
    
    if count > 0:
        logger.info(f"✅ Exported {count} conversations to {output_file}")
        logger.info(f"   Use with unsloth: from unsloth import FastLanguageModel")
    else:
        logger.warning("No conversations exported (check quality threshold and data collection)")
    
    return count


def export_for_alpaca(
    output_dir: Path = Path("data/fine_tuning/export"),
    min_quality: float = 0.8,
    min_conversations: int = 2
) -> int:
    """
    Export data for Alpaca format fine-tuning.
    
    Args:
        output_dir: Output directory path
        min_quality: Minimum quality score (0.0-1.0)
        min_conversations: Minimum number of messages in conversation
        
    Returns:
        Number of exported conversations
    """
    logger.info("Exporting data for Alpaca format fine-tuning...")
    
    fine_tuning_logger = FineTuningLogger()
    
    # Export in Alpaca format
    output_file = output_dir / "alpaca_format.jsonl"
    count = fine_tuning_logger.export_for_fine_tuning(
        output_file=output_file,
        format="alpaca",
        min_quality=min_quality,
        min_conversations=min_conversations
    )
    
    if count > 0:
        logger.info(f"✅ Exported {count} conversations to {output_file}")
        logger.info(f"   Use with instruction-tuning frameworks")
    else:
        logger.warning("No conversations exported (check quality threshold and data collection)")
    
    return count


def export_for_sharegpt(
    output_dir: Path = Path("data/fine_tuning/export"),
    min_quality: float = 0.8,
    min_conversations: int = 2
) -> int:
    """
    Export data for ShareGPT format fine-tuning.
    
    Args:
        output_dir: Output directory path
        min_quality: Minimum quality score (0.0-1.0)
        min_conversations: Minimum number of messages in conversation
        
    Returns:
        Number of exported conversations
    """
    logger.info("Exporting data for ShareGPT format fine-tuning...")
    
    fine_tuning_logger = FineTuningLogger()
    
    # Export in ShareGPT format
    output_file = output_dir / "sharegpt_format.jsonl"
    count = fine_tuning_logger.export_for_fine_tuning(
        output_file=output_file,
        format="sharegpt",
        min_quality=min_quality,
        min_conversations=min_conversations
    )
    
    if count > 0:
        logger.info(f"✅ Exported {count} conversations to {output_file}")
        logger.info(f"   Use with conversation-tuning frameworks")
    else:
        logger.warning("No conversations exported (check quality threshold and data collection)")
    
    return count


def main():
    """Main entry point for export script."""
    parser = argparse.ArgumentParser(description="Export fine-tuning data")
    parser.add_argument(
        "--format",
        type=str,
        choices=["chatml", "alpaca", "sharegpt", "all"],
        default="all",
        help="Export format (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fine_tuning/export"),
        help="Output directory (default: data/fine_tuning/export)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.8,
        help="Minimum quality score (default: 0.8)"
    )
    parser.add_argument(
        "--min-conversations",
        type=int,
        default=2,
        help="Minimum number of messages in conversation (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Export based on format
    if args.format == "all":
        export_for_llama_factory(args.output_dir, args.min_quality, args.min_conversations)
        export_for_unsloth(args.output_dir, args.min_quality, args.min_conversations)
        export_for_alpaca(args.output_dir, args.min_quality, args.min_conversations)
        export_for_sharegpt(args.output_dir, args.min_quality, args.min_conversations)
    elif args.format == "chatml":
        export_for_llama_factory(args.output_dir, args.min_quality, args.min_conversations)
    elif args.format == "alpaca":
        export_for_alpaca(args.output_dir, args.min_quality, args.min_conversations)
    elif args.format == "sharegpt":
        export_for_sharegpt(args.output_dir, args.min_quality, args.min_conversations)
    
    # Print stats
    fine_tuning_logger = FineTuningLogger()
    stats = fine_tuning_logger.get_stats()
    print("\n📊 Fine-Tuning Data Collection Stats:")
    print(f"   Enabled: {stats['enabled']}")
    print(f"   Data Directory: {stats['data_dir']}")
    print(f"   Conversations: {stats['conversations']}")
    print(f"   Reasoning Chains: {stats['reasoning_chains']}")
    print(f"   Quality Metrics: {stats['quality_metrics']}")
    print(f"   Agent Generations: {stats['agent_generations']}")


if __name__ == "__main__":
    main()

