# ICEBURG Internal Fine-Tuning Framework

## Overview

The ICEBURG Internal Fine-Tuning Framework is a custom-built solution for creating ICEBURG-specialized LLMs. Unlike external frameworks (Unsloth, Llama-Factory, MLX), this framework is deeply integrated with ICEBURG's unique intelligence systems.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ICEBURG Internal Fine-Tuning Framework                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │  M4 Optimizer   │    │  Truth Filter   │    │   Emergence     │        │
│   │                 │    │                 │    │   Processor     │        │
│   │ - MPS Detection │    │ - Quality Score │    │ - Novelty Score │        │
│   │ - Batch Auto    │    │ - Duplicate Rem │    │ - Curriculum    │        │
│   │ - Memory Mgmt   │    │ - Truth System  │    │ - Weighting     │        │
│   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘        │
│            │                      │                      │                  │
│            └──────────────────────┴──────────────────────┘                  │
│                                   │                                         │
│                                   ▼                                         │
│                      ┌─────────────────────────┐                           │
│                      │   ICEBURG Fine-Tuner    │                           │
│                      │                         │                           │
│                      │  - Training Pipeline    │                           │
│                      │  - LoRA/QLoRA Config    │                           │
│                      │  - Progress Tracking    │                           │
│                      └────────────┬────────────┘                           │
│                                   │                                         │
│                                   ▼                                         │
│                      ┌─────────────────────────┐                           │
│                      │    Model Exporter       │                           │
│                      │                         │                           │
│                      │  - Ollama Modelfile     │                           │
│                      │  - HuggingFace Format   │                           │
│                      │  - GGUF Quantized       │                           │
│                      └─────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. M4 Optimizer (`m4_optimizer.py`)

Optimizes training for Apple M4 Mac with:
- PyTorch MPS (Metal Performance Shaders) detection
- Automatic batch size calculation based on available memory
- LoRA configuration recommendations per model size
- Gradient checkpointing for memory efficiency

```python
from iceburg.training import get_m4_optimizer

optimizer = get_m4_optimizer()
config = optimizer.detect_hardware()
print(f"Device: {config.device_name}")
print(f"Memory: {config.max_memory_gb} GB")
print(f"Recommended batch size: {config.recommended_batch_size}")
```

### 2. Truth Filter (`truth_filter.py`)

Filters training data using ICEBURG's InstantTruthSystem:
- Quality-based filtering (configurable threshold)
- Duplicate detection and removal
- Agent-specific scoring (Surveyor, Dissident, etc.)
- Integration with InstantTruthSystem for pattern matching

```python
from iceburg.training import TruthFilter

filter = TruthFilter(min_quality_score=0.7)
filtered_data, stats = filter.filter(training_data, agent_type="surveyor")
print(f"Passed: {stats.final_output}/{stats.total_input}")
```

### 3. Emergence Processor (`emergence_processor.py`)

Scores and weights training data by emergence patterns:
- Novelty detection using embedding distance
- Keyword-based emergence scoring
- Curriculum generation (weighted, novel_first, progressive)
- Integration with ICEBURG's EmergenceDetector

```python
from iceburg.training import EmergenceProcessor

processor = EmergenceProcessor()
emergence_data, stats = processor.process(training_data)
curriculum = processor.create_curriculum(emergence_data, strategy="weighted")
```

### 4. ICEBURG Fine-Tuner (`iceburg_fine_tuner.py`)

Main orchestrator combining all components:
- 6-phase training pipeline
- Model type specialization (Surveyor, Dissident, Synthesist, Oracle)
- LoRA/QLoRA training with PEFT
- Automatic configuration based on hardware

```python
from iceburg.training import ICEBURGFineTuner, TrainingConfig, ModelType

config = TrainingConfig(
    base_model="llama3.2:3b",
    model_type=ModelType.SURVEYOR,
    epochs=3,
    use_truth_filter=True,
    use_emergence_weighting=True
)

tuner = ICEBURGFineTuner(config)
result = tuner.train()
```

### 5. Model Exporter (`model_exporter.py`)

Exports trained models in multiple formats:
- **HuggingFace**: For sharing and further fine-tuning
- **Ollama Modelfile**: For local deployment with Ollama
- **GGUF**: Quantized format via llama.cpp
- **ONNX**: For cross-platform deployment

## CLI Usage

```bash
# Detect hardware
python scripts/train_iceburg_model.py --detect-hardware

# Show available features
python scripts/train_iceburg_model.py --features

# Show recommended configs
python scripts/train_iceburg_model.py --configs

# Train a Surveyor model
python scripts/train_iceburg_model.py --model-type surveyor

# Train with custom settings
python scripts/train_iceburg_model.py \
    --model-type oracle \
    --base-model qwen2.5:7b \
    --epochs 5 \
    --data-path data/fine_tuning/oracle_data.jsonl

# Export existing model
python scripts/train_iceburg_model.py --export-only models/my-model --formats ollama gguf
```

## Model Types

| Model Type | Purpose | Recommended Base | LoRA Config |
|------------|---------|------------------|-------------|
| `iceburg-base` | General truth-seeking | mistral:7b | r=32, alpha=64 |
| `iceburg-surveyor` | Research & information gathering | llama3.2:3b | r=16, alpha=32 |
| `iceburg-dissident` | Contradiction detection | llama3.2:3b | r=16, alpha=32 |
| `iceburg-synthesist` | Cross-domain synthesis | mistral:7b | r=32, alpha=64 |
| `iceburg-oracle` | Truth validation | qwen2.5:7b | r=64, alpha=128 |

## Installation Requirements

```bash
# Core requirements
pip install torch transformers peft datasets

# Optional for quantization
pip install bitsandbytes  # CUDA only
pip install llama-cpp-python  # For GGUF

# For MLX support (optional)
pip install mlx mlx-lm
```

## Data Format

Training data should be in JSONL format with ChatML structure:

```json
{
    "messages": [
        {"role": "system", "content": "You are ICEBURG Surveyor..."},
        {"role": "user", "content": "Research quantum computing."},
        {"role": "assistant", "content": "Based on my research..."}
    ],
    "quality_score": 0.9,
    "emergence_score": 0.6
}
```

## Integration with ICEBURG Systems

The framework integrates with:
- **InstantTruthSystem**: For verified pattern matching in data filtering
- **EmergenceDetector**: For novelty scoring and emergence-aware curricula
- **Agent Pipeline**: Training data can be collected from Surveyor/Dissident/Synthesist/Oracle outputs
- **FineTuningLogger**: For automatic training data collection

## Hardware Recommendations

| Model Size | Minimum RAM | Recommended RAM | Training Time (3 epochs, 1000 samples) |
|------------|-------------|-----------------|----------------------------------------|
| 1B | 8GB | 16GB | ~10 min |
| 3B | 16GB | 32GB | ~30 min |
| 7B | 32GB | 64GB | ~2 hours |
| 14B | 64GB | 128GB | ~6 hours |

## Future Enhancements

- [ ] Integration with Self-Redesign Engine for architecture optimization
- [ ] Swarm-based hyperparameter optimization via MicroAgentSwarm
- [ ] Memory-integrated learning with UnifiedMemory
- [ ] Curiosity-driven training data generation
- [ ] Multi-GPU/distributed training support

