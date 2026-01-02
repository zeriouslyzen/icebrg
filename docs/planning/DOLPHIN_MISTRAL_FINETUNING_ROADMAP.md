# Fine-Tuning Dolphin-Mistral with ICEBURG Data - Roadmap

**Last Updated**: December 30, 2025  
**Target Model**: `dolphin-mistral:7b` (uncensored Mistral 7B)  
**Goal**: Create an ICEBURG-optimized version trained on high-quality conversation data

---

## Overview

This roadmap outlines the process for fine-tuning dolphin-mistral using ICEBURG's collected conversation data, reasoning chains, and agent outputs to create a specialized model optimized for ICEBURG's use cases.

---

## Current Status

### ✅ What Exists
- **Fine-Tuning Logger**: `src/iceburg/data_collection/fine_tuning_logger.py`
  - Collects conversations in ChatML format
  - Logs reasoning chains and agent generations
  - Quality filtering (min 0.8 quality score)
  
- **Data Export**: `src/iceburg/data_collection/export_fine_tuning_data.py`
  - Exports in ChatML, Alpaca, ShareGPT formats
  - Compatible with llama-factory, unsloth, etc.
  
- **Training Infrastructure**: 
  - `src/iceburg/training/fine_tuning_pipeline.py` (skeleton)
  - `src/iceburg/training/specialized_model_tuner.py` (placeholder)
  - MLX support mentioned in walkthrough.md

### ❌ What's Missing
- Actual training implementation
- Data collection enabled by default
- Integration with fine-tuning frameworks
- Evaluation metrics
- Deployment pipeline

---

## Phase 1: Data Collection & Preparation (Week 1)

### 1.1 Enable Data Collection
```bash
# Enable fine-tuning data collection
export ICEBURG_ENABLE_FINE_TUNING_DATA=1

# Start ICEBURG and generate conversations
./scripts/start_iceburg.sh
```

### 1.2 Collect Training Data
**Target**: 1,000+ high-quality conversations

**Sources**:
- User conversations (ChatML format)
- Multi-agent research outputs (Surveyor → Dissident → Synthesist → Oracle)
- Reasoning chains from deliberation agent
- High-quality agent generations (quality score ≥ 0.8)

**Data Collection Strategy**:
1. Use ICEBURG normally for 1-2 weeks
2. Focus on diverse query types:
   - Research questions
   - Code generation
   - Explanations
   - Analysis tasks
   - Cross-domain synthesis
3. Monitor data quality in `data/fine_tuning/conversations.jsonl`

### 1.3 Export Training Data
```bash
# Export in ChatML format (compatible with Mistral)
python3 -m src.iceburg.data_collection.export_fine_tuning_data \
    --format chatml \
    --min-quality 0.8 \
    --min-conversations 2 \
    --output-dir data/fine_tuning/export
```

**Expected Output**:
- `data/fine_tuning/export/chatml_format.jsonl`
- Format: `{"messages": [{"role": "system/user/assistant", "content": "..."}]}`

### 1.4 Data Validation
- Check conversation count: `wc -l data/fine_tuning/export/chatml_format.jsonl`
- Verify format: Sample 10 conversations
- Check quality distribution
- Remove duplicates if any

---

## Phase 2: Training Setup (Week 2)

### 2.1 Choose Fine-Tuning Framework

**Option A: Unsloth (Recommended for M4 Mac)**
- Fast training with QLoRA
- Optimized for Apple Silicon
- Easy integration

**Option B: Llama-Factory**
- Comprehensive features
- Multiple quantization options
- Good documentation

**Option C: MLX (Native Apple)**
- Best performance on M4
- Native Metal acceleration
- Requires more setup

### 2.2 Install Dependencies

**For Unsloth**:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
```

**For Llama-Factory**:
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

**For MLX**:
```bash
pip install mlx mlx-lm
```

### 2.3 Prepare Training Configuration

**QLoRA Configuration** (Recommended):
```python
{
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

**Training Hyperparameters**:
- Learning rate: 2e-4
- Batch size: 4 (adjust for memory)
- Epochs: 3
- Warmup steps: 100
- Max sequence length: 2048

---

## Phase 3: Fine-Tuning Execution (Week 2-3)

### 3.1 Training Script (Unsloth Example)

Create `scripts/finetune_dolphin_mistral.py`:
```python
from unsloth import FastLanguageModel
import torch

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/dolphin-2.9-mistral-7b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# Load training data
from datasets import load_dataset
dataset = load_dataset("json", data_files="data/fine_tuning/export/chatml_format.jsonl")

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="messages",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs/iceburg-dolphin-mistral",
        optim="adamw_torch",
    ),
)

trainer.train()

# Save model
model.save_pretrained("models/iceburg-dolphin-mistral-7b")
```

### 3.2 Training Execution
```bash
# Run training
python3 scripts/finetune_dolphin_mistral.py

# Monitor training
# - Watch loss decrease
# - Check for overfitting
# - Monitor GPU/CPU usage
```

### 3.3 Training Checkpoints
- Save checkpoints every epoch
- Keep best model based on validation loss
- Store training logs for analysis

---

## Phase 4: Evaluation & Testing (Week 3)

### 4.1 Create Evaluation Dataset
- 100 test queries covering:
  - Research questions
  - Code generation
  - Explanations
  - Current events (if applicable)
  - Cross-domain synthesis

### 4.2 Run Evaluation
```python
# Compare base vs fine-tuned model
base_model = load_model("dolphin-mistral:7b")
fine_tuned = load_model("models/iceburg-dolphin-mistral-7b")

# Test on evaluation set
for query in test_queries:
    base_response = base_model.generate(query)
    tuned_response = fine_tuned.generate(query)
    # Compare quality, accuracy, relevance
```

### 4.3 Metrics to Track
- **Accuracy**: Correct answers on factual queries
- **Relevance**: Response matches query intent
- **Coherence**: Logical flow and structure
- **ICEBURG-specific**: 
  - Multi-agent reasoning quality
  - Cross-domain synthesis ability
  - Contradiction detection
  - Truth-seeking behavior

### 4.4 A/B Testing
- Deploy fine-tuned model alongside base model
- Route 10% of traffic to fine-tuned version
- Collect user feedback
- Compare metrics

---

## Phase 5: Deployment (Week 4)

### 5.1 Model Conversion for Ollama
```bash
# Convert to Ollama format
# (Ollama supports custom models via Modelfile)

# Create Modelfile
cat > Modelfile << EOF
FROM models/iceburg-dolphin-mistral-7b
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create Ollama model
ollama create iceburg-dolphin-mistral -f Modelfile
```

### 5.2 Integration with ICEBURG
Update `config/iceburg_unified.yaml`:
```yaml
secretary:
  model: "iceburg-dolphin-mistral"  # Use fine-tuned model
  # or keep base as fallback
  # model: "dolphin-mistral:7b"
```

### 5.3 Gradual Rollout
1. **Week 1**: 10% of traffic to fine-tuned model
2. **Week 2**: 50% if metrics are good
3. **Week 3**: 100% if no issues

---

## Phase 6: Continuous Improvement (Ongoing)

### 6.1 Continuous Data Collection
- Keep `ICEBURG_ENABLE_FINE_TUNING_DATA=1` enabled
- Regularly export new high-quality conversations
- Retrain every 1-2 months with new data

### 6.2 Model Iteration
- Version models: `iceburg-dolphin-mistral-v1`, `v2`, etc.
- Track performance improvements
- A/B test new versions

### 6.3 Specialized Variants
Consider creating specialized models:
- `iceburg-dolphin-mistral-research`: Optimized for research queries
- `iceburg-dolphin-mistral-code`: Optimized for code generation
- `iceburg-dolphin-mistral-synthesis`: Optimized for cross-domain synthesis

---

## Expected Outcomes

### Performance Improvements
- **Accuracy**: +10-15% on ICEBURG-specific tasks
- **Relevance**: Better understanding of multi-agent context
- **Speed**: Similar to base model (QLoRA adds minimal overhead)
- **Quality**: More aligned with ICEBURG's truth-seeking methodology

### Model Characteristics
- Maintains uncensored nature of dolphin-mistral
- Better at multi-step reasoning
- Improved cross-domain synthesis
- Enhanced contradiction detection
- More aligned with ICEBURG's research methodology

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Data Collection** | Week 1 | 1,000+ conversations exported |
| **Phase 2: Training Setup** | Week 2 | Framework installed, config ready |
| **Phase 3: Fine-Tuning** | Week 2-3 | Trained model checkpoint |
| **Phase 4: Evaluation** | Week 3 | Evaluation metrics, A/B test results |
| **Phase 5: Deployment** | Week 4 | Model deployed in Ollama, integrated |
| **Phase 6: Continuous** | Ongoing | Regular retraining, improvements |

**Total Initial Timeline**: 4 weeks  
**Ongoing**: Monthly retraining cycles

---

## Resources & References

### Documentation
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Llama-Factory Documentation](https://github.com/hiyouga/LLaMA-Factory)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

### ICEBURG Components
- `src/iceburg/data_collection/fine_tuning_logger.py` - Data collection
- `src/iceburg/data_collection/export_fine_tuning_data.py` - Data export
- `scripts/start_fine_tuning.py` - Fine-tuning starter script

### Data Locations
- `data/fine_tuning/conversations.jsonl` - Collected conversations
- `data/fine_tuning/export/` - Exported training data
- `data/training/` - Agent-specific training data

---

## Next Steps

1. **Enable data collection**: `export ICEBURG_ENABLE_FINE_TUNING_DATA=1`
2. **Start using ICEBURG** to generate training data
3. **Review this roadmap** and adjust based on your setup
4. **Choose fine-tuning framework** (Unsloth recommended for M4 Mac)
5. **Begin Phase 1** data collection

---

**Questions or Issues?**  
- Check existing fine-tuning scripts in `scripts/`
- Review data collection code in `src/iceburg/data_collection/`
- See CHANGELOG.md for recent fine-tuning additions

