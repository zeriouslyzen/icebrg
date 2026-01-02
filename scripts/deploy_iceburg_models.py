#!/usr/bin/env python3
"""
ICEBURG Model Deployment CLI
=============================

Deploys trained ICEBURG models to Ollama for local inference.

Usage:
    python scripts/deploy_iceburg_models.py --list
    python scripts/deploy_iceburg_models.py --deploy-all
    python scripts/deploy_iceburg_models.py --deploy surveyor
    python scripts/deploy_iceburg_models.py --test surveyor
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.training.model_registry import get_model_registry, ModelStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_models():
    """List all registered models."""
    registry = get_model_registry()
    models = registry.list_models()
    
    print("\n" + "="*70)
    print("ICEBURG Model Registry")
    print("="*70 + "\n")
    
    if not models:
        print("No models registered.")
        print("\nRun training first:")
        print("  python scripts/train_iceburg_model.py --model-type surveyor")
        return
        
    # Group by type
    by_type = {}
    for model in models:
        if model.model_type not in by_type:
            by_type[model.model_type] = []
        by_type[model.model_type].append(model)
        
    for model_type, type_models in sorted(by_type.items()):
        print(f"\n{model_type.upper()} Models:")
        print("-" * 60)
        
        for model in sorted(type_models, key=lambda m: m.final_loss):
            status_icon = {
                ModelStatus.AVAILABLE: "[  ]",
                ModelStatus.DEPLOYED: "[OK]",
                ModelStatus.UNAVAILABLE: "[--]",
                ModelStatus.TRAINING: "[..]"
            }.get(model.status, "[??]")
            
            print(f"  {status_icon} {model.name}")
            print(f"       Base: {model.base_model}")
            print(f"       Loss: {model.final_loss:.4f}")
            print(f"       Samples: {model.training_samples}")
            if model.ollama_name:
                print(f"       Ollama: {model.ollama_name}")
            print()
            
    print("\nStatus Legend:")
    print("  [OK] = Deployed to Ollama")
    print("  [  ] = Available (not deployed)")
    print("  [--] = Unavailable (files missing)")
    print()


def deploy_model(model_type: str):
    """Deploy the best model of a type to Ollama."""
    registry = get_model_registry()
    
    # Get best model of this type
    model = registry.get_best_model(model_type)
    if not model:
        print(f"\nNo {model_type} model found.")
        print(f"\nTrain one first:")
        print(f"  python scripts/train_iceburg_model.py --model-type {model_type}")
        return False
        
    print(f"\nDeploying {model.name} to Ollama...")
    print(f"  Type: {model.model_type}")
    print(f"  Loss: {model.final_loss:.4f}")
    print(f"  Path: {model.path}")
    
    success, message = registry.deploy_to_ollama(model.name)
    
    if success:
        print(f"\n  SUCCESS: {message}")
        return True
    else:
        print(f"\n  FAILED: {message}")
        return False


def deploy_all():
    """Deploy best model of each type."""
    print("\n" + "="*70)
    print("Deploying All ICEBURG Models")
    print("="*70)
    
    results = {}
    for model_type in ["surveyor", "dissident", "synthesist", "oracle"]:
        print(f"\n--- {model_type.upper()} ---")
        results[model_type] = deploy_model(model_type)
        
    print("\n" + "="*70)
    print("Deployment Summary")
    print("="*70)
    
    for model_type, success in results.items():
        status = "DEPLOYED" if success else "SKIPPED"
        print(f"  {model_type}: {status}")
        
    return all(results.values())


def test_model(model_type: str):
    """Test a deployed model with a sample query."""
    registry = get_model_registry()
    ollama_name = registry.get_ollama_model_name(model_type)
    
    if not ollama_name:
        print(f"\nNo {model_type} model found.")
        return False
        
    print(f"\nTesting {ollama_name}...")
    
    # Sample prompts by type
    test_prompts = {
        "surveyor": "What are the key findings in quantum computing research?",
        "dissident": "Challenge the claim that AI will solve all problems.",
        "synthesist": "Connect the concepts of evolution, economics, and AI.",
        "oracle": "What is the validated truth about climate change?"
    }
    
    prompt = test_prompts.get(model_type, "Hello, what can you do?")
    
    try:
        result = subprocess.run(
            ["ollama", "run", ollama_name, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"\nPrompt: {prompt}")
            print(f"\nResponse:\n{result.stdout}")
            return True
        else:
            print(f"\nError: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("\nOllama not found. Install from https://ollama.ai/")
        return False
    except subprocess.TimeoutExpired:
        print("\nTimeout waiting for response.")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False


def check_ollama():
    """Check if Ollama is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="ICEBURG Model Deployment")
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered models"
    )
    parser.add_argument(
        "--deploy",
        metavar="TYPE",
        choices=["surveyor", "dissident", "synthesist", "oracle", "base"],
        help="Deploy best model of specified type"
    )
    parser.add_argument(
        "--deploy-all",
        action="store_true",
        help="Deploy best model of each type"
    )
    parser.add_argument(
        "--test",
        metavar="TYPE",
        choices=["surveyor", "dissident", "synthesist", "oracle", "base"],
        help="Test a deployed model"
    )
    
    args = parser.parse_args()
    
    # Check Ollama for deployment/test commands
    if args.deploy or args.deploy_all or args.test:
        if not check_ollama():
            print("\nOllama is not running or not installed.")
            print("Start Ollama first: ollama serve")
            print("Or install from: https://ollama.ai/")
            return 1
    
    if args.list:
        list_models()
    elif args.deploy:
        success = deploy_model(args.deploy)
        return 0 if success else 1
    elif args.deploy_all:
        success = deploy_all()
        return 0 if success else 1
    elif args.test:
        success = test_model(args.test)
        return 0 if success else 1
    else:
        # Default: list
        list_models()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

