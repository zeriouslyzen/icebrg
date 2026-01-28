#!/usr/bin/env python3
"""
ICEBURG Model Training CLI
==========================

Command-line interface for training ICEBURG-specialized LLMs.

Usage:
    python scripts/train_iceburg_model.py --model-type surveyor --base-model llama3.2:3b
    python scripts/train_iceburg_model.py --model-type oracle --base-model qwen2.5:7b
    python scripts/train_iceburg_model.py --detect-hardware
    python scripts/train_iceburg_model.py --export-only models/iceburg/iceburg-surveyor-20251231
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.training import get_available_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_hardware():
    """Detect and display hardware configuration."""
    from iceburg.training import get_m4_optimizer
    
    print("\n" + "="*60)
    print("ICEBURG Hardware Detection")
    print("="*60 + "\n")
    
    optimizer = get_m4_optimizer()
    config = optimizer.detect_hardware()
    
    print(f"Device: {config.device_name}")
    print(f"Device Type: {config.device.value}")
    print(f"Available Memory: {config.max_memory_gb:.1f} GB")
    print(f"MPS Available: {config.mps_available}")
    print(f"MLX Available: {config.mlx_available}")
    print(f"Recommended Batch Size (3B): {config.recommended_batch_size}")
    print(f"Gradient Checkpointing: {config.gradient_checkpointing}")
    print(f"Mixed Precision: {config.mixed_precision}")
    
    print("\n" + "-"*60)
    print("Training Arguments for 3B Model:")
    print("-"*60)
    
    args = optimizer.configure_training_args(model_size="3b")
    for key, value in args.items():
        print(f"  {key}: {value}")
        
    print("\n" + "-"*60)
    print("LoRA Configuration for 3B Model:")
    print("-"*60)
    
    lora_config = optimizer.get_lora_config(model_size="3b")
    for key, value in lora_config.items():
        print(f"  {key}: {value}")
        
    print()


def show_features():
    """Display available features."""
    print("\n" + "="*60)
    print("ICEBURG Training Features")
    print("="*60 + "\n")
    
    features = get_available_features()
    
    for feature, available in features.items():
        status = "Available" if available else "Not Available"
        symbol = "+" if available else "-"
        print(f"  [{symbol}] {feature}: {status}")
        
    print()


def show_recommended_configs():
    """Display recommended configurations."""
    from iceburg.training import get_recommended_config, ModelType
    
    print("\n" + "="*60)
    print("Recommended Configurations")
    print("="*60 + "\n")
    
    for model_type in ["base", "surveyor", "dissident", "synthesist", "oracle"]:
        config = get_recommended_config(model_type)
        print(f"{model_type.upper()}:")
        print(f"  Base Model: {config.base_model}")
        print(f"  LoRA Rank: {config.lora_r}")
        print(f"  LoRA Alpha: {config.lora_alpha}")
        print()


def train_model(args):
    """Train an ICEBURG model."""
    from iceburg.training import (
        ICEBURGFineTuner,
        TrainingConfig,
        ModelType,
        get_recommended_config
    )
    
    print("\n" + "="*60)
    print("ICEBURG Model Training")
    print("="*60 + "\n")
    
    # Get recommended config as base
    config = get_recommended_config(args.model_type)
    
    # Override with command line args
    if args.base_model:
        config.base_model = args.base_model
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.data_path:
        config.data_path = Path(args.data_path)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.output_name:
        config.output_name = args.output_name
        
    # Quality filtering options
    if args.min_quality:
        config.min_quality_score = args.min_quality
    if args.no_truth_filter:
        config.use_truth_filter = False
    if args.no_emergence:
        config.use_emergence_weighting = False
    if args.curriculum:
        config.curriculum_strategy = args.curriculum
        
    # Export options
    config.export_ollama = not args.no_ollama_export
    config.export_huggingface = not args.no_hf_export
    
    print("Configuration:")
    print(f"  Model Type: {config.model_type.value}")
    print(f"  Base Model: {config.base_model}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size or 'auto'}")
    print(f"  LoRA Rank: {config.lora_r}")
    print(f"  Truth Filter: {config.use_truth_filter}")
    print(f"  Emergence Weighting: {config.use_emergence_weighting}")
    print(f"  Curriculum: {config.curriculum_strategy}")
    print()
    
    # Create tuner and train
    tuner = ICEBURGFineTuner(config)
    
    print("Starting training...")
    print("-"*60)
    
    result = tuner.train()
    
    print("\n" + "-"*60)
    print("Training Complete!")
    print("-"*60)
    
    if result.success:
        print(f"\nModel: {result.model_name}")
        print(f"Path: {result.model_path}")
        print(f"Training Time: {result.training_time_seconds:.1f}s")
        print(f"Samples: {result.filtered_samples}/{result.total_samples} used")
        print(f"Final Loss: {result.final_loss:.4f}")
        
        if result.export_paths:
            print("\nExported to:")
            for fmt, path in result.export_paths.items():
                print(f"  {fmt}: {path}")
    else:
        print(f"\nTraining failed: {result.error_message}")
        return 1
        
    # Save result
    result_path = config.output_dir / config.output_name / "training_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nResult saved to: {result_path}")
    
    return 0


def export_model(args):
    """Export an existing model."""
    from iceburg.training import ModelExporter, ExportFormat
    
    print("\n" + "="*60)
    print("ICEBURG Model Export")
    print("="*60 + "\n")
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return 1
        
    model_name = args.output_name or model_path.name
    
    # Determine formats
    formats = []
    if args.formats:
        format_mapping = {
            "huggingface": ExportFormat.HUGGINGFACE,
            "hf": ExportFormat.HUGGINGFACE,
            "ollama": ExportFormat.OLLAMA,
            "gguf": ExportFormat.GGUF,
            "onnx": ExportFormat.ONNX
        }
        for fmt in args.formats:
            if fmt.lower() in format_mapping:
                formats.append(format_mapping[fmt.lower()])
    else:
        formats = [ExportFormat.HUGGINGFACE, ExportFormat.OLLAMA]
        
    print(f"Model: {model_path}")
    print(f"Output Name: {model_name}")
    print(f"Formats: {[f.value for f in formats]}")
    print()
    
    exporter = ModelExporter()
    results = exporter.export(model_path, model_name, formats)
    
    print("Export Results:")
    print("-"*60)
    
    for fmt, result in results.items():
        status = "Success" if result.success else "Failed"
        print(f"\n{fmt.value}: {status}")
        if result.success:
            print(f"  Path: {result.output_path}")
            print(f"  Size: {result.file_size_mb:.2f} MB")
            print(f"  Time: {result.export_time_seconds:.2f}s")
        else:
            print(f"  Error: {result.error_message}")
            
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ICEBURG Model Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect hardware configuration
  python scripts/train_iceburg_model.py --detect-hardware
  
  # Show available features
  python scripts/train_iceburg_model.py --features
  
  # Show recommended configs
  python scripts/train_iceburg_model.py --configs
  
  # Train a Surveyor model
  python scripts/train_iceburg_model.py --model-type surveyor
  
  # Train with custom settings
  python scripts/train_iceburg_model.py \\
      --model-type oracle \\
      --base-model qwen2.5:7b \\
      --epochs 5 \\
      --data-path data/fine_tuning/oracle_data.jsonl
  
  # Export an existing model
  python scripts/train_iceburg_model.py --export-only models/my-model --formats ollama gguf
"""
    )
    
    # Mode selection
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--detect-hardware",
        action="store_true",
        help="Detect and display hardware configuration"
    )
    mode_group.add_argument(
        "--features",
        action="store_true",
        help="Display available training features"
    )
    mode_group.add_argument(
        "--configs",
        action="store_true",
        help="Display recommended configurations"
    )
    mode_group.add_argument(
        "--export-only",
        metavar="MODEL_PATH",
        help="Export an existing model instead of training"
    )
    
    # Training options
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--model-type",
        choices=["base", "surveyor", "dissident", "synthesist", "oracle"],
        default="base",
        help="Type of ICEBURG model to train (default: base)"
    )
    train_group.add_argument(
        "--base-model",
        help="Base model to fine-tune (e.g., mistral:7b, llama3.2:3b)"
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (auto-detected if not specified)"
    )
    train_group.add_argument(
        "--data-path",
        help="Path to training data (JSONL)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        default="models/iceburg",
        help="Output directory for trained models"
    )
    output_group.add_argument(
        "--output-name",
        help="Name for the output model"
    )
    
    # Quality filtering
    quality_group = parser.add_argument_group("Quality Filtering")
    quality_group.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum quality score for training data (default: 0.7)"
    )
    quality_group.add_argument(
        "--no-truth-filter",
        action="store_true",
        help="Disable truth-based filtering"
    )
    quality_group.add_argument(
        "--no-emergence",
        action="store_true",
        help="Disable emergence weighting"
    )
    quality_group.add_argument(
        "--curriculum",
        choices=["weighted", "novel_first", "progressive"],
        default="weighted",
        help="Curriculum strategy (default: weighted)"
    )
    
    # Export options
    export_group = parser.add_argument_group("Export")
    export_group.add_argument(
        "--formats",
        nargs="+",
        help="Export formats (huggingface, ollama, gguf, onnx)"
    )
    export_group.add_argument(
        "--no-ollama-export",
        action="store_true",
        help="Skip Ollama Modelfile export"
    )
    export_group.add_argument(
        "--no-hf-export",
        action="store_true",
        help="Skip HuggingFace export"
    )
    
    args = parser.parse_args()
    
    # Check features are available
    features = get_available_features()
    if not features.get("iceburg_finetuner"):
        print("Error: ICEBURG Fine-Tuning framework not available.")
        print("Some dependencies may be missing. Check import errors.")
        return 1
    
    # Execute based on mode
    if args.detect_hardware:
        detect_hardware()
        return 0
    elif args.features:
        show_features()
        return 0
    elif args.configs:
        show_recommended_configs()
        return 0
    elif args.export_only:
        args.model_path = args.export_only
        return export_model(args)
    else:
        return train_model(args)


if __name__ == "__main__":
    sys.exit(main())

