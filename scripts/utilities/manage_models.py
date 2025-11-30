#!/usr/bin/env python3
"""
ICEBURG Model Management Script

Helps identify and delete unused models to save disk space.
"""

import subprocess
import json
import sys
from typing import List, Dict, Tuple

# Models currently used by ICEBURG
USED_MODELS = {
    # Core models (required)
    "llama3.1:8b": {"size": "4.9 GB", "priority": "HIGH", "used_by": "Surveyor (primary)"},
    "mistral:7b-instruct": {"size": "4.4 GB", "priority": "HIGH", "used_by": "Dissident"},
    "mistral:7b": {"size": "4.4 GB", "priority": "MEDIUM", "used_by": "Dissident (alternative)"},
    "nomic-embed-text": {"size": "~100 MB", "priority": "HIGH", "used_by": "Embeddings"},
    
    # Alternative models (recommended)
    "llama3:8b": {"size": "4.7 GB", "priority": "MEDIUM", "used_by": "Alternative"},
    "qwen2.5:7b": {"size": "4.7 GB", "priority": "MEDIUM", "used_by": "Alternative"},
    "phi3:mini": {"size": "2.2 GB", "priority": "MEDIUM", "used_by": "Fast mode"},
    
    # Code generation models (if code generation enabled)
    "deepseek-coder:1.3b": {"size": "776 MB", "priority": "MEDIUM", "used_by": "Code generation"},
    "deepseek-coder:6.7b": {"size": "3.8 GB", "priority": "LOW", "used_by": "Code generation (better)"},
    "codellama:7b": {"size": "3.8 GB", "priority": "LOW", "used_by": "Code generation"},
    
    # Vision models (if multimodal enabled)
    "moondream:1.8b": {"size": "1.7 GB", "priority": "MEDIUM", "used_by": "Vision processing"},
    
    # Large models (optional, only if you have space)
    "llama3.1:70b": {"size": "42 GB", "priority": "OPTIONAL", "used_by": "Oracle/Synthesist (best quality)"},
    "mixtral:8x7b": {"size": "26 GB", "priority": "OPTIONAL", "used_by": "Alternative large model"},
}

# Models that can be deleted
UNUSED_MODELS = {
    "glm-4.6:cloud": {"size": "Unknown", "reason": "Cloud model, not local"},
    "bakllava:7b": {"size": "4.7 GB", "reason": "Large vision model, moondream:1.8b is sufficient"},
    "codellama:7b-instruct": {"size": "3.8 GB", "reason": "Duplicate of codellama:7b"},
    "gemma:7b": {"size": "5.0 GB", "reason": "Alternative model, not required"},
    "yi:6b": {"size": "3.5 GB", "reason": "Alternative model, not required"},
}

def get_installed_models() -> List[Dict[str, str]]:
    """Get list of installed models from Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"Error running ollama list: {result.stderr}")
            return []
        
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return []
        
        models = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                size = parts[2] if len(parts) > 2 else "Unknown"
                models.append({"name": name, "size": size})
        
        return models
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def analyze_models() -> Tuple[List[str], List[str], List[str]]:
    """Analyze models and categorize them"""
    installed = get_installed_models()
    installed_names = [m["name"] for m in installed]
    
    used = []
    unused = []
    optional = []
    
    for model_name in installed_names:
        if model_name in USED_MODELS:
            priority = USED_MODELS[model_name]["priority"]
            if priority == "HIGH":
                used.append(model_name)
            elif priority == "MEDIUM":
                used.append(model_name)
            elif priority == "OPTIONAL":
                optional.append(model_name)
        elif model_name in UNUSED_MODELS:
            unused.append(model_name)
        else:
            # Unknown model - ask user
            unused.append(model_name)
    
    return used, unused, optional

def delete_model(model_name: str) -> bool:
    """Delete a model from Ollama"""
    try:
        print(f"Deleting {model_name}...")
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✅ Deleted {model_name}")
            return True
        else:
            print(f"❌ Failed to delete {model_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error deleting {model_name}: {e}")
        return False

def main():
    """Main model management interface"""
    print("=" * 80)
    print("ICEBURG Model Management")
    print("=" * 80)
    print()
    
    # Analyze models
    used, unused, optional = analyze_models()
    
    print("📊 Model Analysis")
    print("-" * 80)
    print(f"✅ Used Models (Keep): {len(used)}")
    for model in used:
        info = USED_MODELS.get(model, {})
        print(f"   • {model} ({info.get('size', 'Unknown')}) - {info.get('used_by', 'Unknown')}")
    
    print()
    print(f"⚠️  Optional Models (Keep if you have space): {len(optional)}")
    for model in optional:
        info = USED_MODELS.get(model, {})
        print(f"   • {model} ({info.get('size', 'Unknown')}) - {info.get('used_by', 'Unknown')}")
    
    print()
    print(f"❌ Unused Models (Can Delete): {len(unused)}")
    total_space = 0
    for model in unused:
        info = UNUSED_MODELS.get(model, {})
        size_str = info.get("size", "Unknown")
        reason = info.get("reason", "Not in used models list")
        print(f"   • {model} ({size_str}) - {reason}")
        
        # Try to parse size for total
        if "GB" in size_str:
            try:
                size_gb = float(size_str.replace(" GB", ""))
                total_space += size_gb
            except:
                pass
    
    print()
    print(f"💾 Estimated Space Savings: ~{total_space:.1f} GB")
    print()
    
    # Show all installed models
    installed = get_installed_models()
    print(f"📦 Total Installed Models: {len(installed)}")
    print()
    
    # Ask user what to do
    print("Options:")
    print("  1. Show detailed model information")
    print("  2. Delete unused models (recommended)")
    print("  3. Delete specific model")
    print("  4. Exit")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print()
        print("Detailed Model Information:")
        print("-" * 80)
        for model in installed:
            name = model["name"]
            size = model["size"]
            if name in USED_MODELS:
                info = USED_MODELS[name]
                print(f"✅ {name} ({size})")
                print(f"   Priority: {info['priority']}")
                print(f"   Used by: {info['used_by']}")
            elif name in UNUSED_MODELS:
                info = UNUSED_MODELS[name]
                print(f"❌ {name} ({size})")
                print(f"   Reason: {info['reason']}")
            else:
                print(f"❓ {name} ({size})")
                print(f"   Status: Unknown (not in ICEBURG model list)")
            print()
    
    elif choice == "2":
        print()
        print("⚠️  WARNING: This will delete unused models!")
        print(f"Models to delete: {', '.join(unused)}")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm == "yes":
            deleted = 0
            failed = 0
            for model in unused:
                if delete_model(model):
                    deleted += 1
                else:
                    failed += 1
            
            print()
            print(f"✅ Deleted {deleted} models")
            if failed > 0:
                print(f"❌ Failed to delete {failed} models")
        else:
            print("Cancelled.")
    
    elif choice == "3":
        print()
        print("Available models to delete:")
        for i, model in enumerate(unused, 1):
            info = UNUSED_MODELS.get(model, {})
            print(f"  {i}. {model} ({info.get('size', 'Unknown')})")
        
        try:
            idx = int(input("Enter model number: ").strip()) - 1
            if 0 <= idx < len(unused):
                model_to_delete = unused[idx]
                confirm = input(f"Delete {model_to_delete}? (yes/no): ").strip().lower()
                if confirm == "yes":
                    delete_model(model_to_delete)
                else:
                    print("Cancelled.")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")
    
    elif choice == "4":
        print("Exiting...")
        sys.exit(0)
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()

