# Local Frameworks Used in ICEBURG

**Last Updated**: December 30, 2025

This document lists all local (on-device) frameworks and libraries used in ICEBURG for AI/ML operations, fine-tuning, and model management.

---

## Core AI/ML Frameworks

### 1. **Ollama** (Primary LLM Framework)
- **Purpose**: Local LLM inference and model management
- **Version**: `>=0.2.1`
- **Usage**: 
  - Running local models (dolphin-mistral, llama3.1:8b, qwen2.5, etc.)
  - Model inference for all agents
  - Base framework for local AI operations
- **Location**: `src/iceburg/providers/ollama_provider.py`

### 2. **Transformers** (Hugging Face)
- **Purpose**: Model loading, tokenization, training
- **Usage**: 
  - Fine-tuning pipeline (`src/iceburg/training/fine_tuning_pipeline.py`)
  - Model architecture definitions
  - Tokenizer management
- **Components Used**:
  - `AutoModelForCausalLM`
  - `AutoTokenizer`
  - `TrainingArguments`
  - `Trainer`

### 3. **PEFT** (Parameter-Efficient Fine-Tuning)
- **Purpose**: LoRA/QLoRA fine-tuning
- **Usage**: 
  - QLoRA configuration for efficient fine-tuning
  - LoRA adapters for specialized models
- **Components Used**:
  - `LoraConfig`
  - `get_peft_model`

### 4. **PyTorch** (Torch)
- **Purpose**: Deep learning backend
- **Version**: `>=2.0.0,<3.0.0`
- **Usage**:
  - Neural network implementations
  - Quantum-classical hybrid models
  - Reinforcement learning agents
  - Computer vision
  - Voice processing
- **Location**: Used across multiple modules:
  - `src/iceburg/quantum/` (quantum ML)
  - `src/iceburg/rl/` (reinforcement learning)
  - `src/iceburg/vision/` (computer vision)
  - `src/iceburg/voice/` (voice processing)

---

## Fine-Tuning Frameworks (Mentioned/Planned)

### 5. **MLX** (Apple Silicon Native)
- **Purpose**: Native Apple Silicon acceleration for training
- **Status**: Mentioned in walkthrough.md, recommended for M4 Mac
- **Usage**: 
  - Fine-tuning on Apple M4 Neural Engine
  - Student-teacher distillation loop
- **Location**: Referenced in `frontend/public/research_data/walkthrough.md`
- **Note**: Not yet fully integrated, but recommended in roadmap

### 6. **Unsloth** (Fast Fine-Tuning)
- **Purpose**: Fast QLoRA training optimized for Apple Silicon
- **Status**: Recommended in fine-tuning roadmap
- **Usage**: 
  - Fast training with QLoRA
  - Optimized for M4 Mac
- **Location**: `docs/planning/DOLPHIN_MISTRAL_FINETUNING_ROADMAP.md`

### 7. **Llama-Factory**
- **Purpose**: Comprehensive fine-tuning framework
- **Status**: Mentioned as alternative option
- **Usage**: 
  - Multiple quantization options
  - Comprehensive training features
- **Location**: Referenced in fine-tuning roadmap

---

## Vector & Search Frameworks

### 8. **ChromaDB**
- **Purpose**: Local vector database
- **Version**: `>=0.5.5`
- **Usage**: 
  - Semantic search
  - Vector storage for embeddings
  - Knowledge base retrieval
- **Location**: `src/iceburg/vectorstore.py`

### 9. **Sentence Transformers**
- **Purpose**: Semantic embeddings
- **Version**: `>=2.2.0`
- **Usage**: 
  - Text embeddings for search
  - Semantic similarity
  - RAG (Retrieval-Augmented Generation)
- **Location**: Used in search pipeline

### 10. **Rank-BM25**
- **Purpose**: Lexical search (BM25 algorithm)
- **Version**: `>=0.2.2`
- **Usage**: 
  - Hybrid search (BM25 + semantic)
  - Fast keyword matching
  - Secretary V2 search engine
- **Location**: `src/iceburg/search/`

---

## Quantum Computing Frameworks

### 11. **PennyLane**
- **Purpose**: Quantum machine learning
- **Version**: `>=0.43.0`
- **Usage**: 
  - Quantum circuits
  - Quantum-classical hybrid models
  - Quantum optimization
- **Location**: `src/iceburg/quantum/`

### 12. **Qiskit** (Optional)
- **Purpose**: Quantum computing framework
- **Version**: `>=1.0.0` (optional dependency)
- **Usage**: 
  - Quantum circuit simulation
  - Lab features
- **Location**: Optional lab dependency

### 13. **Cirq** (Optional)
- **Purpose**: Quantum circuit framework
- **Version**: `>=1.3.0` (optional dependency)
- **Usage**: 
  - Quantum algorithms
  - Lab features
- **Location**: Optional lab dependency

---

## Reinforcement Learning Frameworks

### 14. **Stable-Baselines3**
- **Purpose**: RL algorithms
- **Version**: `>=2.0.0`
- **Usage**: 
  - PPO (Proximal Policy Optimization)
  - SAC (Soft Actor-Critic)
  - Trading agent training
- **Location**: `src/iceburg/rl/`

### 15. **Gymnasium**
- **Purpose**: RL environments
- **Version**: `>=1.0.0`
- **Usage**: 
  - Trading environments
  - Agent training environments
- **Location**: `src/iceburg/rl/`

### 16. **Ray RLlib**
- **Purpose**: Distributed RL training
- **Version**: `>=2.0.0`
- **Usage**: 
  - Distributed training
  - Multi-agent RL
- **Location**: Used in RL optimization

### 17. **PettingZoo**
- **Purpose**: Multi-agent RL environments
- **Version**: `>=1.20.0`
- **Usage**: 
  - Multi-agent coordination
  - Swarm training
- **Location**: Multi-agent RL scenarios

---

## Traditional ML Frameworks

### 18. **Scikit-Learn**
- **Purpose**: Traditional ML algorithms
- **Version**: `>=1.0.0`
- **Usage**: 
  - Classification
  - Regression
  - Feature engineering
- **Location**: Used in various ML components

### 19. **XGBoost**
- **Purpose**: Gradient boosting
- **Version**: `>=1.5.0`
- **Usage**: 
  - Financial predictions
  - Feature importance
- **Location**: Financial AI components

### 20. **LightGBM**
- **Purpose**: Fast gradient boosting
- **Version**: `>=3.3.0`
- **Usage**: 
  - Fast training
  - Large datasets
- **Location**: Financial AI components

---

## Data Processing Frameworks

### 21. **NumPy**
- **Purpose**: Numerical computing
- **Version**: `>=1.20.0,<3.0.0` (elite financial) / `>=1.26.0` (core)
- **Usage**: 
  - Array operations
  - Mathematical computations
  - Data manipulation
- **Location**: Used throughout codebase

### 22. **Pandas**
- **Purpose**: Data manipulation
- **Version**: `>=1.3.0`
- **Usage**: 
  - DataFrames
  - Financial data processing
  - Time series analysis
- **Location**: Financial and data processing modules

### 23. **SciPy**
- **Purpose**: Scientific computing
- **Version**: `>=1.7.0`
- **Usage**: 
  - Statistical functions
  - Optimization
  - Signal processing
- **Location**: Scientific computing modules

---

## Visualization Frameworks

### 24. **Matplotlib**
- **Purpose**: Plotting
- **Version**: `>=3.5.0` (elite financial) / `>=3.8.0` (lab)
- **Usage**: 
  - Charts and graphs
  - Data visualization
- **Location**: Visualization components

### 25. **Plotly**
- **Purpose**: Interactive visualization
- **Version**: `>=5.0.0` (elite financial) / `>=5.18.0` (lab)
- **Usage**: 
  - Interactive charts
  - Dashboard visualizations
- **Location**: Admin dashboard, financial dashboards

### 26. **Seaborn**
- **Purpose**: Statistical visualization
- **Version**: `>=0.11.0`
- **Usage**: 
  - Statistical plots
  - Data analysis visualization
- **Location**: Data analysis modules

---

## Specialized Frameworks

### 27. **NetworkX**
- **Purpose**: Graph/network analysis
- **Version**: `>=3.2.1`
- **Usage**: 
  - Knowledge graphs
  - Agent network analysis
  - Relationship mapping
- **Location**: Graph-based components

### 28. **Redis**
- **Purpose**: Caching and fast storage
- **Version**: `>=4.0.0`
- **Usage**: 
  - KV cache
  - Fast memory storage
  - Session management
- **Location**: `src/iceburg/caching/`

---

## Summary by Category

### Fine-Tuning & Training
- **Currently Used**: Transformers, PEFT, PyTorch
- **Planned/Recommended**: MLX, Unsloth, Llama-Factory

### Model Inference
- **Primary**: Ollama
- **Supporting**: Transformers (for custom models)

### Vector Search & RAG
- ChromaDB (vector storage)
- Sentence Transformers (embeddings)
- Rank-BM25 (lexical search)

### Quantum Computing
- PennyLane (primary)
- Qiskit, Cirq (optional lab features)

### Reinforcement Learning
- Stable-Baselines3
- Gymnasium
- Ray RLlib
- PettingZoo

### Traditional ML
- Scikit-Learn
- XGBoost
- LightGBM

### Data Processing
- NumPy
- Pandas
- SciPy

---

## Framework Selection Rationale

### Why Ollama?
- **Local-first**: All models run locally
- **Easy model management**: Simple CLI for model operations
- **Fast inference**: Optimized for local hardware
- **Uncensored models**: Access to dolphin-mistral and other uncensored variants

### Why Transformers + PEFT?
- **Industry standard**: Widely used and well-documented
- **Efficient fine-tuning**: QLoRA allows fine-tuning on consumer hardware
- **Flexibility**: Works with many model architectures

### Why MLX (Planned)?
- **Apple Silicon native**: Best performance on M4 Mac
- **Metal acceleration**: Direct GPU access
- **Fast training**: Optimized for Apple hardware

### Why ChromaDB?
- **Local vector DB**: No external dependencies
- **Fast**: Optimized for semantic search
- **Simple**: Easy to integrate

---

## Installation Notes

### Core Dependencies
```bash
# Core ICEBURG
pip install -r requirements/requirements_elite_financial.txt

# Or from pyproject.toml
pip install -e .
```

### Fine-Tuning (Optional)
```bash
# For Unsloth (recommended for M4 Mac)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# For MLX (Apple Silicon)
pip install mlx mlx-lm

# For Llama-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e .
```

### Optional Lab Features
```bash
pip install -e ".[lab]"  # Includes Qiskit, Cirq
```

---

## Framework Status

| Framework | Status | Integration Level |
|-----------|--------|-------------------|
| Ollama | ✅ Active | Fully integrated |
| Transformers | ✅ Active | Used in fine-tuning pipeline |
| PEFT | ✅ Active | Used for LoRA |
| PyTorch | ✅ Active | Used in RL, quantum, vision |
| ChromaDB | ✅ Active | Vector storage |
| Sentence Transformers | ✅ Active | Search embeddings |
| Rank-BM25 | ✅ Active | Hybrid search |
| PennyLane | ✅ Active | Quantum ML |
| Stable-Baselines3 | ✅ Active | RL training |
| MLX | 📋 Planned | Recommended for M4 |
| Unsloth | 📋 Planned | Recommended for fine-tuning |
| Llama-Factory | 📋 Planned | Alternative option |

---

## References

- **Fine-Tuning Roadmap**: `docs/planning/DOLPHIN_MISTRAL_FINETUNING_ROADMAP.md`
- **Requirements**: `requirements/requirements_elite_financial.txt`
- **Project Config**: `pyproject.toml`
- **Walkthrough**: `frontend/public/research_data/walkthrough.md`

---

**Last Updated**: December 30, 2025

