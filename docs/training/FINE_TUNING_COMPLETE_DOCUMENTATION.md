# ICEBURG Fine-Tuning: Complete Documentation

**Date**: January 2026  
**Status**: Phase 1 Complete - Infrastructure Built, Quality Data Needed

---

## Executive Summary

We built a complete internal fine-tuning framework for creating ICEBURG-specialized LLMs on Apple M4 Macs. The infrastructure works, but the training data quality is insufficient for production use. This document covers what was built, what works, what doesn't, and the path forward.

---

## What Was Built

### 1. Core Training Framework

| File | Purpose | Status |
|------|---------|--------|
| `src/iceburg/training/iceburg_fine_tuner.py` | Main orchestrator - 6-phase training pipeline | Working |
| `src/iceburg/training/m4_optimizer.py` | Apple Silicon MPS/MLX detection and optimization | Working |
| `src/iceburg/training/truth_filter.py` | Data quality filtering via InstantTruthSystem | Working |
| `src/iceburg/training/emergence_processor.py` | Emergence-aware curriculum learning | Working |
| `src/iceburg/training/model_exporter.py` | Export to HuggingFace and Ollama formats | Working |
| `src/iceburg/training/model_registry.py` | Model discovery, versioning, registry | Working |

### 2. CLI Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/train_iceburg_model.py` | Main training CLI | Working |
| `scripts/deploy_iceburg_models.py` | Model registry and deployment | Working |
| `scripts/run_iceburg_model.py` | Inference with trained models | Working |
| `scripts/generate_training_corpus.py` | Hand-crafted examples (60 samples) | Working |
| `scripts/generate_training_corpus_large.py` | Template-based generation (600 samples) | Working but LOW QUALITY |
| `scripts/generate_real_training_data.py` | Real agent data collection | Blocked by ChromaDB issue |

### 3. Trained Models

```
models/iceburg/
├── iceburg-surveyor-20251231_170132/    # Best: Loss 2.84, 150 samples
├── iceburg-dissident-20251231_171213/   # Loss 3.69, 150 samples  
├── iceburg-synthesist-20251231_171213/  # Loss 2.95, 150 samples
├── iceburg-oracle-20251231_171213/      # Loss 3.22, 150 samples
└── ... (15+ model versions total)
```

### 4. Training Data

```
data/fine_tuning/agent_data/
├── surveyor_training_data.jsonl      # 26 hand-crafted examples
├── dissident_training_data.jsonl     # 16 hand-crafted examples
├── synthesist_training_data.jsonl    # 10 hand-crafted examples
├── oracle_training_data.jsonl        # 8 hand-crafted examples
└── large_corpus/
    ├── surveyor_training_data.jsonl  # 150 template-generated
    ├── dissident_training_data.jsonl # 150 template-generated
    ├── synthesist_training_data.jsonl# 150 template-generated
    └── oracle_training_data.jsonl    # 150 template-generated
```

---

## What Actually Works

### Training Pipeline
- **LoRA fine-tuning** with PyTorch + PEFT + Transformers
- **M4 Mac MPS acceleration** for 5-10x faster training vs CPU
- **Automatic device detection** (MPS/CUDA/CPU)
- **Memory optimization** for consumer hardware
- **Model export** to HuggingFace and Ollama Modelfile formats
- **Model registry** with versioning and discovery

### Inference
- **Model loading** from trained LoRA adapters
- **Generation** with configurable temperature, tokens
- **Interactive mode** for testing

### Example Working Command:
```bash
cd /Users/jackdanger/Desktop/Projects/iceburg

# Train a model
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 scripts/train_iceburg_model.py \
    --model-type surveyor \
    --base-model Qwen/Qwen2.5-0.5B \
    --data-path data/fine_tuning/agent_data/surveyor_training_data.jsonl \
    --epochs 5

# List models
python3 scripts/deploy_iceburg_models.py --list

# Run inference
python3 scripts/run_iceburg_model.py --model surveyor \
    --prompt "Explain quantum computing"
```

---

## What Does NOT Work (Honestly)

### 1. Training Data Quality is Poor

**The Problem**:
```
TEMPLATE DATA (what we trained on):
"The field of {TOPIC} has seen significant advances...
 Core principles are now well-established...
 Industrial applications are emerging..."

SAME TEMPLATE, DIFFERENT TOPICS = MODEL LEARNS TEMPLATES, NOT KNOWLEDGE
```

**Evidence**:
- Surveyor produces generic responses
- Synthesist outputs repetitive patterns
- Dissident echoes template structure
- Oracle lacks substantive reasoning

### 2. Real Data Collection Blocked

**The Problem**: Generating training data from actual ICEBURG agents requires:
- VectorStore (ChromaDB) - currently corrupted/unstable
- Multi-agent orchestration - complex dependencies
- API calls to external LLMs - requires active providers

### 3. Model Size Too Small

**The Problem**: Using Qwen2.5-0.5B (500M parameters)
- Fast to train but limited capability
- Cannot hold complex knowledge
- Prone to repetition

### 4. No Evaluation Suite

**The Problem**: No automated way to measure:
- Response quality
- Factual accuracy
- Agent-specific capabilities
- Regression between versions

---

## Honest Assessment: What Did We Actually Accomplish?

### Accomplished (Infrastructure):
1. **Complete training pipeline** that runs end-to-end
2. **M4 Mac optimization** for Apple Silicon
3. **Model registry** for versioning and discovery
4. **Export formats** for Ollama and HuggingFace
5. **CLI tools** for training and inference
6. **Integration points** with ICEBURG systems (TruthFilter, EmergenceProcessor)

### NOT Accomplished (Quality):
1. **No production-quality models** - outputs are generic/repetitive
2. **No real training data** - only synthetic templates
3. **No evaluation metrics** - can't measure improvement
4. **No alignment** - no RLHF/DPO implemented
5. **No integration** - models not wired into ICEBURG agents

### Percentage Complete:
- **Infrastructure**: 90% complete
- **Data Quality**: 10% complete (templates are junk)
- **Model Quality**: 5% complete (outputs are poor)
- **Production Ready**: 0%

---

## Root Cause Analysis

### Why Did We Train With Bad Data?

1. **Speed over Quality**: Prioritized getting something running fast
2. **Template Trap**: Easy to generate volume, hard to generate quality
3. **No Validation Loop**: Didn't check data quality before training
4. **Blocked Dependencies**: Real agent data requires VectorStore, which is broken

### What Should Have Happened:

1. **Fix VectorStore first** - Enable real agent data collection
2. **Quality over Quantity** - 50 excellent examples > 600 templates
3. **Human Curation** - Hand-write high-quality examples
4. **Incremental Validation** - Test after each batch of data

---

## Technical Debt Created

| Issue | Impact | Fix Effort |
|-------|--------|------------|
| ChromaDB corruption | Blocks real data collection | Medium - may need DB rebuild |
| Template training data | Models learned wrong patterns | High - need new data |
| No evaluation suite | Can't measure progress | Medium - build benchmarks |
| Small base model | Limited capability ceiling | Low - change config |
| No preference data | Can't do DPO/RLHF | High - need collection system |

---

## Future Roadmap

### Phase 2: Data Quality (Priority: CRITICAL)

**Option A: Human Curation (Recommended)**
1. Write 50-100 high-quality examples per agent type
2. Include real research, real sources, real reasoning
3. Review and iterate based on model output
4. Estimated time: 2-4 hours per agent type

**Option B: Fix Real Data Collection**
1. Rebuild/fix ChromaDB VectorStore
2. Enable `ICEBURG_ENABLE_FINE_TUNING_DATA=1` in production
3. Collect from actual ICEBURG usage
4. Estimated time: 1-2 days for infrastructure

**Option C: LLM-Assisted Generation**
1. Use Claude/GPT-4 to generate quality examples
2. Human review and curation
3. Fine-tune on curated outputs
4. Estimated time: 4-8 hours

### Phase 3: Model Quality

1. **Larger Base Model**: Switch to 3B-7B parameters
2. **Longer Training**: 10-20 epochs with quality data
3. **Hyperparameter Optimization**: Systematic search
4. **Regularization**: Prevent overfitting to templates

### Phase 4: Evaluation

1. **Custom Benchmarks**: Agent-specific test suites
2. **Automated Testing**: Run after each training
3. **Human Evaluation**: Periodic quality checks
4. **A/B Testing**: Compare model versions

### Phase 5: Alignment

1. **Collect Preference Data**: Good vs bad responses
2. **Implement DPO**: Direct Preference Optimization
3. **Use ICEBURG Signals**: TruthFilter, EmergenceDetector as rewards
4. **Constitutional AI**: Principle-based self-critique

### Phase 6: Integration

1. **Wire Models to Agents**: Replace API calls with local models
2. **Hybrid Routing**: Use local for common, API for complex
3. **Continuous Training**: Weekly updates from usage data
4. **Monitoring**: Track quality in production

---

## Files Reference

### Core Training Code
```
src/iceburg/training/
├── __init__.py              # Module exports
├── iceburg_fine_tuner.py    # Main orchestrator (29KB)
├── m4_optimizer.py          # Apple Silicon optimization (12KB)
├── truth_filter.py          # Data quality filtering (17KB)
├── emergence_processor.py   # Curriculum learning (18KB)
├── model_exporter.py        # Export formats (20KB)
├── model_registry.py        # Model versioning (12KB)
└── specialized_model_tuner.py # Agent configs (9KB)
```

### Scripts
```
scripts/
├── train_iceburg_model.py           # Training CLI
├── deploy_iceburg_models.py         # Registry CLI
├── run_iceburg_model.py             # Inference CLI
├── generate_training_corpus.py      # Quality examples
├── generate_training_corpus_large.py # Template examples (BAD)
├── generate_real_training_data.py   # Real agent data (BLOCKED)
└── collect_training_data.py         # Usage collection
```

### Documentation
```
docs/training/
├── ICEBURG_INTERNAL_FINETUNING.md   # Framework docs
├── ICEBURG_LAB_ROADMAP.md           # Lab practices
└── FINE_TUNING_COMPLETE_DOCUMENTATION.md # This file
```

---

## Recommendations

### Immediate (This Week)
1. **Delete template-trained models** - they're teaching wrong patterns
2. **Write 20 quality examples per agent** - by hand, with real content
3. **Retrain on quality data** - even 20 good examples beats 600 templates

### Short-term (This Month)
1. **Fix ChromaDB** - Enable real data collection
2. **Enable production logging** - Collect from actual usage
3. **Build evaluation suite** - Measure what matters

### Medium-term (This Quarter)
1. **Implement DPO** - Preference-based training
2. **Scale to larger models** - 3B+ parameters
3. **Integrate into ICEBURG** - Replace API calls

---

## Lessons Learned

1. **Data quality > Data quantity**: 50 good examples > 600 templates
2. **Validate before training**: Check data makes sense
3. **Infrastructure is not the goal**: Working pipeline means nothing with bad data
4. **Be honest about progress**: Don't confuse activity with achievement
5. **Fix dependencies first**: ChromaDB should have been fixed before data generation

---

## Conclusion

We built a solid fine-tuning infrastructure for ICEBURG on M4 Macs. The training pipeline, optimization, export, and registry all work correctly. However, the models produced are not useful because the training data is low-quality template generation.

**The infrastructure works. The data doesn't.**

Next step: Generate or curate high-quality training examples before any more model training.

