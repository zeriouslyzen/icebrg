# ICEBURG AI Constitution
## Quick Reference for AI Assistants

**Purpose**: This document provides instant understanding of how ICEBURG components work together, reducing context requirements for AI-assisted development.

**Last Updated**: January 28, 2026  
**Version**: 3.4.0

---

## 🎯 Core Concept

ICEBURG is a **local-first multi-agent research platform** powered by Ollama. Think of it as:
- **Frontend**: Futuristic web UI (Vite + Vanilla JS)
- **Backend**: FastAPI server orchestrating 37+ specialized AI agents
- **Intelligence**: Multi-agent deliberation for deep research
- **Data**: SQLite + ChromaDB + JSONL for persistence

---

## 📁 Critical File Locations

### Must-Know Files
```
/src/iceburg/api/server.py          # FastAPI server, all routes
/src/iceburg/agents/secretary.py    # Main chat orchestrator
/src/iceburg/protocol_fixed.py      # Multi-agent research protocol
/frontend/main.js                   # Frontend logic (9300 lines!)
/frontend/app.html                  # Main UI
/config/iceburg_unified.yaml        # System configuration
```

### Key Directories
```
/src/iceburg/
├── agents/          # 59 agent files - specialized AI workers
├── api/             # FastAPI routes and WebSocket handlers
├── colossus/        # Graph intelligence (Pegasus visualization)
├── matrix/          # SQLite data store (1.5M+ entities)
├── memory/          # ChromaDB vector store + persistence
├── tools/           # Web search, PDF processing, OSINT
├── middleware/      # Hallucination detection, emergence tracking
└── core/            # System integrator, load balancer

/frontend/
├── main.js          # 9300 lines - handles everything
├── app.html         # Main chat interface
├── pegasus.html     # Network visualization
└── styles.css       # 143KB of styling
```

---

## 🔄 How Data Flows

### User Query → Response (Simplified)
```
1. User types in frontend (app.html)
2. main.js sends via WebSocket to /ws
3. server.py routes to Secretary agent
4. Secretary detects mode (chat/research)
5. Agents execute (Surveyor → Dissident → Synthesist → Oracle)
6. Response streams back via WebSocket
7. main.js renders markdown + charts
```

### Key Integration Points
- **WebSocket**: `/ws` endpoint for real-time streaming
- **HTTP**: `/api/query` for synchronous requests
- **Memory**: ChromaDB for semantic search, SQLite for structured data
- **LLM**: Ollama (localhost:11434) for all model calls

---

## 🤖 Agent System (37 Total)

### Core Research Agents (5)
| Agent | Purpose | When Used |
|-------|---------|-----------|
| **Secretary** | Chat orchestrator, routing | Every query |
| **Surveyor** | Information gathering | Research mode |
| **Dissident** | Challenge assumptions | Research mode |
| **Synthesist** | Integrate findings | Research mode |
| **Oracle** | Pattern detection | Research mode |

### Specialized Agents (32+)
- **Reflex Agent**: Fast responses (chat mode)
- **Visual Architect**: UI generation
- **Weaver**: Code generation
- **Scrutineer**: Quality control
- **Archaeologist**: Historical research
- See `/docs/COMPLETE_FEATURE_REFERENCE.md` for full list

### Agent Coordination
- **Load Balancer**: `src/iceburg/distributed/load_balancer.py`
- **System Integrator**: `src/iceburg/core/system_integrator.py`
- **Capability Registry**: `src/iceburg/agents/capability_registry.py`

---

## 💾 Data Storage

### Three-Layer Architecture
```
1. ChromaDB (Vector Store)
   - Location: /data/chroma/
   - Purpose: Semantic search, embeddings
   - Status: Currently in mock mode (Rust binding issue)

2. SQLite (Structured Data)
   - Location: /data/iceburg_unified.db
   - Purpose: Agent states, metrics, config
   - Tables: conversations, agents, metrics, etc.

3. JSONL (Event Streaming)
   - Location: /data/telemetry/
   - Purpose: Emergence events, breakthroughs
   - Format: One JSON object per line
```

---

## 🌐 API Endpoints

### Essential Routes
```python
# Core
GET  /health                    # Health check
POST /api/query                 # Synchronous query
WS   /ws                        # WebSocket streaming

# Knowledge
GET  /api/encyclopedia          # Celestial encyclopedia
POST /api/knowledge/submit      # Submit suppressed knowledge

# Admin
GET  /api/status                # System status
GET  /api/middleware/stats      # Middleware metrics
```

See `/src/iceburg/api/server.py` for all routes.

---

## ⚙️ Configuration

### Environment Variables (Key Ones)
```bash
# Models
ICEBURG_SURVEYOR_MODEL=llama3.1:8b
ICEBURG_ORACLE_MODEL=llama3:70b-instruct

# Features
ICEBURG_INSTANT_TRUTH=true          # Fast path optimization
ICEBURG_ENABLE_WEB=0                # Web search (disabled by default)
ICEBURG_FIELD_AWARE=0               # Consciousness interface

# Paths
ICEBURG_DATA_DIR=./data
ICEBURG_TOKEN_USAGE_DB=./data/token_usage.db
```

Full list: `/docs/COMPLETE_FEATURE_REFERENCE.md`

---

## 🔧 Making Changes

### Frontend Changes
1. **UI/UX**: Edit `/frontend/app.html` or `/frontend/styles.css`
2. **Logic**: Edit `/frontend/main.js` (careful - 9300 lines!)
3. **Test**: Refresh browser (Vite hot-reload)

### Backend Changes
1. **Routes**: Edit `/src/iceburg/api/server.py`
2. **Agents**: Edit files in `/src/iceburg/agents/`
3. **Protocol**: Edit `/src/iceburg/protocol_fixed.py`
4. **Test**: Server auto-reloads with `--reload` flag

### Adding New Agent
1. Create `/src/iceburg/agents/my_agent.py`
2. Register in `/src/iceburg/core/system_integrator.py`
3. Add to capability registry
4. Update config if needed

---

## 🚨 Common Pitfalls

### Frontend
- **WebSocket race conditions**: Connection may not be ready
- **Massive main.js**: Use search, don't scroll
- **Streaming**: Chunks arrive out of order sometimes

### Backend
- **ChromaDB disabled**: Currently in mock mode
- **Ollama required**: Must be running on localhost:11434
- **Agent timeouts**: Long research queries can timeout
- **Memory leaks**: WebSocket connections need cleanup

### Data
- **SQLite locking**: Concurrent writes can fail
- **ChromaDB Rust**: Binding issues on some systems
- **Large files**: Frontend has size limits

---

## 📊 System Architecture (Visual)

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│  Frontend (Vite) → WebSocket → FastAPI Server          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  SECRETARY AGENT                         │
│  Mode Detection → Route to Appropriate Agents           │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌──────────────────┐              ┌──────────────────┐
│   CHAT MODE      │              │  RESEARCH MODE   │
│  Reflex Agent    │              │  Multi-Agent     │
│  (Fast)          │              │  Protocol        │
└──────────────────┘              └──────────────────┘
                                           ↓
                          ┌────────────────┴────────────────┐
                          ↓                ↓                ↓
                    ┌──────────┐    ┌──────────┐    ┌──────────┐
                    │ Surveyor │ →  │Dissident │ →  │Synthesist│
                    └──────────┘    └──────────┘    └──────────┘
                                                            ↓
                                                     ┌──────────┐
                                                     │  Oracle  │
                                                     └──────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  DATA LAYER                              │
│  ChromaDB (vectors) + SQLite (structured) + JSONL       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  OLLAMA (Local LLMs)                     │
│  localhost:11434 → llama3.1, mistral, qwen2.5          │
└─────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Concepts

### Multi-Agent Protocol
- **Surveyor**: Gathers information, checks lost knowledge DB
- **Dissident**: Challenges assumptions, finds alternatives
- **Synthesist**: Integrates findings, creates synthesis
- **Oracle**: Detects patterns, makes predictions

### Fast Path vs Deep Path
- **Fast**: Reflex agent, instant truth system (\<30s)
- **Deep**: Full multi-agent protocol (2-5 min)

### Emergence Detection
- System detects breakthrough discoveries
- Tracks novelty, surprise, compression gain
- Stores in emergence database

### Hallucination Prevention
- Middleware layer checks all responses
- Grounding in knowledge base
- Contradiction detection
- Source validation

---

## 📚 Documentation Map

### For Quick Reference
- **This file**: Instant system understanding
- `/README.md`: Project overview, quick start
- `/CHANGELOG.md`: Version history

### For Deep Dives
- `/docs/architecture/COMPLETE_ICEBURG_ARCHITECTURE.md`: Full architecture (679 lines)
- `/docs/COMPLETE_FEATURE_REFERENCE.md`: All features (401 lines)
- `/docs/INTERNAL_TECH_STACKS.md`: Technology details

### For Specific Tasks
- `/docs/guides/`: How-to guides
- `/docs/examples/`: Code examples
- `/docs/reports/`: System audits and analyses

---

## 🔍 Quick Debugging

### Frontend Not Loading
```bash
# Check Vite server
cd frontend && npm run dev

# Check console for errors
# Open browser DevTools → Console
```

### Backend Not Responding
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check API server
curl http://localhost:8000/health

# Check logs
tail -f logs/api_server.log
```

### WebSocket Issues
```javascript
// In browser console
ws = new WebSocket('ws://localhost:8000/ws')
ws.onmessage = (e) => console.log(e.data)
```

---

## 💡 Pro Tips for AI Assistants

1. **Start with this file** - Don't read 679-line architecture docs first
2. **Use grep** - `grep -r "function_name" src/` is your friend
3. **Check server.py** - All routes are defined there
4. **main.js is huge** - Use search, don't try to understand it all
5. **Test locally** - Always verify changes work before committing
6. **ChromaDB is mocked** - Don't rely on vector search currently
7. **Ollama required** - System won't work without it
8. **WebSocket is primary** - HTTP is fallback only

---

## 🎯 Decision Tree for Changes

```
Need to change UI?
  → Edit /frontend/app.html or /frontend/styles.css

Need to change logic?
  → Frontend: /frontend/main.js
  → Backend: /src/iceburg/api/server.py or /src/iceburg/agents/

Need to add feature?
  → New agent: /src/iceburg/agents/
  → New route: /src/iceburg/api/server.py
  → New UI: /frontend/

Need to fix bug?
  → Check logs: /logs/
  → Check console: Browser DevTools
  → Check server: curl http://localhost:8000/health

Need to understand system?
  → Read this file first
  → Then /docs/architecture/COMPLETE_ICEBURG_ARCHITECTURE.md
  → Then specific component docs
```

---

## 📝 Version History

- **3.4.0** (Dec 2025): Metacognition, safety, documentation
- **3.0.0** (Jan 2025): Complete architecture overhaul
- **2.0.0**: Multi-agent protocol stabilized
- **1.0.0**: Initial release

See `/CHANGELOG.md` for detailed history.

---

**Remember**: This is a living document. Update it when you make significant architectural changes!

**For Humans**: This document is optimized for AI assistants to quickly understand the system. For human-friendly docs, see `/README.md` and `/docs/`.
