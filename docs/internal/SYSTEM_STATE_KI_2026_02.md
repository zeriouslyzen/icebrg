# Knowledge Item Artifact: System State & Routing Evolution (Feb 2026)

## 📌 Context
This document tracks the critical architectural transition of the ICEBURG system on the M4 Mac Mini environment, focusing on latency reduction and workspace hygiene.

## 🧹 Workspace Hygiene (The Great Purge)
On Feb 5, 2026, we performed a major cleanup to reclaim ~9GB of disk space.
- **Removed**: 
    - Broken legacy environments: `environments/sovereign_library_env`, `elite_financial_env`, `venv2` (pointing to old user `deshonjackson`).
    - Redundant Ollama models: `glm-4.6:cloud`, `bakllava:7b`, `codellama:7b`, `deepseek-coder:1.3b`.
    - Build artifacts: `.venv_broken`, `dist`, `.pytest_cache`, `.ruff_cache`.
- **Primary Environment**: All work is now consolidated in the root `venv/` (Python 3.9+).

## ⚡ Routing Evolution: MoE (Mixture of Experts)
We transitioned from a basic `request_router.py` to a more advanced `moe_router.py` to solve a 13s+ latency issue in chat mode.

### Active Expert Mapping:
| Expert Domain | Model | Rationale |
| :--- | :--- | :--- |
| **Simple Chat** | `phi3:mini` | Ultra-fast responses for greetings/basic turns. |
| **Fast Chat** | `dolphin-mistral` | Default for general conversation. |
| **Deep Research** | `llama3:70b-instruct` | Full reasoning capabilities for the Surveyor swarm. |
| **Philosophical** | `llama3.1:8b` | Specialized for abstract consciousness queries. |
| **Code** | `deepseek-v2:16b` | M4-optimized expert for software development. |

### Hardware Flag:
- `M4_OPTIMIZATION=True` is now the default in configuration, forcing MLX-optimized local models wherever possible to leverage the Unified Memory architecture.

## 🚨 Ongoing "Trap" Warning
- **ChromaDB**: Currently running in **Mock Mode**. Vector indexing is simulated because the Rust bindings for `chromadb` often fail on this specific M4 environment. Do not rely on persistent embeddings until this is resolved.
- **Matrix Store**: Always use `src/iceburg/matrix/` tools for DB access; direct SQL is prone to locking the 1.5M+ entity store.
