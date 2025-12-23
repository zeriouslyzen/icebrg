# ICEBURG: Multi-Agent Research Platform

**Local-first AI research system powered by Ollama with multi-agent deliberation.**

Built for [katanx.com](https://www.katanx.com) - a self-development platform for practitioners of "The Nine Arts."

## What Works Now ✅

ICEBURG currently provides:

- **Secretary Chat Agent**: Fast conversational AI using local Ollama models (llama3.1:8b, qwen2.5, etc.)
- **Multi-Agent Research Protocol**: Surveyor → Dissident → Synthesist → Oracle for deep analysis
- **Web Frontend**: Mobile-first UI with real-time SSE streaming (http://localhost:3000)
- **Conversation Memory**: Context-aware follow-up questions
- **Conversation Memory**: Context-aware follow-up questions
- **Knowledge Base**: 233-entry Celestial Encyclopedia on bioelectricity & consciousness
- **Metacognition**: Self-correction and contradiction detection (v3.4)
- **Quarantine System**: Safe storage for novel/contradictory ideas

## Quick Start

### Prerequisites
- **Ollama** installed ([ollama.com](https://ollama.com))
- **Python 3.9+**
- **Node.js 18+** (for frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/iceburg.git
cd iceburg

# Install Python dependencies
pip install -r requirements/requirements_elite_financial.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Running ICEBURG

```bash
# Start the system (backend + frontend)
./scripts/start_iceburg.sh

# Or start components separately:
# Backend: uvicorn src.iceburg.api.server:app --host 0.0.0.0 --port 8000 --reload
# Frontend: cd frontend && npm run dev
```

**Access the UI**: http://localhost:3000

### Stop ICEBURG

```bash
./scripts/stop_iceburg.sh
```

## Usage Examples

### Chat Mode (Fast)
Ask simple questions, get direct answers:
```
User: What is consciousness?
ICEBURG: [Provides philosophical answer with sources]
```

### Research Mode (Deep)
Generate multi-agent analysis:
```
User: Analyze the connection between quantum mechanics and bioelectricity
ICEBURG: [Generates comprehensive multi-perspective report with Surveyor/Dissident/Synthesist/Oracle]
```

Example research outputs: [`data/research_outputs/`](data/research_outputs/)

## Architecture

```
┌─────────────────────────────────────┐
│      Frontend (Vite + Vanilla JS)   │  ← http://localhost:3000
│      SSE Streaming, Mobile-First    │
└─────────────────┬───────────────────┘
                  │ HTTP/SSE
┌─────────────────▼───────────────────┐
│    FastAPI Server (Port 8000)       │
│    - /api/query (SSE streaming)     │
│    - Conversation memory            │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Secretary Agent                │
│      - Simple Q&A (fast path)       │
│      - Multi-agent research         │
│      - Memory retrieval             │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Ollama (Local LLMs)            │
│      - llama3.1:8b (default)        │
│      - qwen2.5, mistral, etc.       │
└─────────────────────────────────────┘
```

## Project Structure

```
iceburg/
├── src/iceburg/           # Core Python package
│   ├── agents/            # Secretary and research agents
│   ├── api/               # FastAPI server
│   ├── providers/         # Ollama integration
│   └── config.py          # Configuration
├── frontend/              # Vite web application
│   ├── main.js            # Frontend logic
│   ├── index.html         # UI
│   └── styles.css         # Styling
├── data/                  # Knowledge base & conversation logs
│   ├── celestial_encyclopedia.json
│   ├── conversation_logs/
│   └── research_outputs/
├── scripts/               # Utility scripts
│   ├── start_iceburg.sh
│   └── stop_iceburg.sh
├── docs/                  # Documentation (organized)
│   ├── INDEX.md           # Master documentation index
│   ├── guides/            # User/dev guides
│   ├── architecture/      # System design
│   ├── status/            # Status reports
│   └── testing/           # Test documentation
└── tests/                 # Test suites
```

## Documentation

- **[Current State](CURRENT_STATE.md)** - What's operational vs planned
- **[Full Documentation Index](docs/INDEX.md)** - Complete docs organized by category
- **[Changelog](CHANGELOG.md)** - Version history and recent updates
- **[Contributing](CONTRIBUTING.md)** - How to contribute

## Configuration

Main configuration file: `config/iceburg_unified.yaml`

Key settings:
- `default_mode`: "chat" (fast) or "research" (deep)
- `primary_model`: Ollama model to use (default: "llama3.1:8b")
- `enable_memory`: Enable conversation history

## For katanx.com Integration

ICEBURG powers the research capabilities for [katanx.com](https://www.katanx.com)'s self-development platform:

- **Multi-perspective research**: Generate insights from multiple viewpoints
- **Evidence tracking**: Track sources and detect contradictions
- **Suppression detection**: Identify patterns in suppressed research
- **Cross-domain synthesis**: Connect ideas across "The Nine Arts"

## Development Status

**Version**: 3.4.0  
**Status**: Chat & Research features operational  
**Last Updated**: December 23, 2025

### Recent Updates (v3.4.0)
- ✅ **Metacognition**: Integrated semantic alignment & contradiction detection
- ✅ **Safety**: Added Quarantine System for flagged outputs
- ✅ **Documentation**: 100% feature coverage in [COMPLETE_FEATURE_REFERENCE.md](docs/COMPLETE_FEATURE_REFERENCE.md)

See [CHANGELOG.md](CHANGELOG.md) for detailed history.

## Roadmap

See [docs/planning/ROADMAP_TO_10_OUT_OF_10.md](docs/planning/ROADMAP_TO_10_OUT_OF_10.md) for future plans.

**Priorities:**
1. Improve conversation memory reliability
2. Enhanced multi-agent coordination
3. Better error handling and resilience
4. Performance optimizations

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick start:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [docs/INDEX.md](docs/INDEX.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/iceburg/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/iceburg/discussions)

---

**Built with ❤️ for truth-seekers and practitioners of The Nine Arts.**